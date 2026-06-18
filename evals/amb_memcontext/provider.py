"""MemContext provider adapter for agent-memory-benchmark.

The adapter intentionally lives in this repository. It imports AMB's provider
types when AMB is installed or supplied by the runner, but keeps small fallback
types so product-side tests can run without mutating or vendoring AMB.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memcontext.consolidate import consolidate_facts
from memcontext.digests import build_session_digest, store_digest
from memcontext.event_frames import assemble_event_frames
from memcontext.extractors import LLMExtractor
from memcontext.importance import recompute_all_importance
from memcontext.life_events import detect_life_events, store_life_events
from memcontext.on_new_turn import TurnResult, on_new_turn
from memcontext.retrieval import (
    BGE_M3_MODEL_ID,
    EmbeddingClient,
    backfill_event_frame_embeddings,
    episode_embedder,
    retrieve_memory_across,
)
from memcontext.schema import Speaker, open_database
from memcontext.serving import session_briefing
from memcontext.supersession_semantic import SemanticSupersession

from .router_llm import TOKENROUTER_BASE_URL, TOKENROUTER_EXTRACTOR_MODEL

try:  # pragma: no cover - exercised only with a real AMB checkout installed.
    from memory_bench.memory.base import MemoryProvider
    from memory_bench.models import Document
except Exception:  # noqa: BLE001

    @dataclass
    class Document:  # type: ignore[no-redef]
        id: str
        content: str
        user_id: str | None = None
        messages: list[dict] | None = None
        timestamp: str | None = None
        context: str | None = None

    class MemoryProvider:  # type: ignore[no-redef]
        name: str
        description: str
        kind: str
        provider: str | None = None
        variant: str | None = None
        concurrency: int = 1

        def initialize(self) -> None:
            return None

        def cleanup(self) -> None:
            return None


DEFAULT_EXTRACTOR_MODEL = TOKENROUTER_EXTRACTOR_MODEL


@dataclass(frozen=True, slots=True)
class _IngestedTurn:
    speaker: Speaker
    text: str
    metadata: dict[str, Any]


def _doc_value(doc: Any, name: str, default: Any = None) -> Any:
    if isinstance(doc, dict):
        return doc.get(name, default)
    return getattr(doc, name, default)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _speaker_from_role(role: Any) -> Speaker:
    normalized = str(role or "user").strip().lower()
    if normalized in {"assistant", "ai", "bot", "model"}:
        return Speaker.ASSISTANT
    if normalized in {"system", "developer", "tool"}:
        return Speaker.SYSTEM
    return Speaker.USER


def _message_content(message: Any) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return _safe_text(message)
    for key in ("content", "text", "message", "value", "utterance"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(_safe_text(item.get("text") or item.get("content")))
                else:
                    parts.append(_safe_text(item))
            return "\n".join(p for p in parts if p.strip())
    return _safe_text(message)


def _json_turns(content: str) -> list[Any] | None:
    try:
        loaded = json.loads(content)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict):
        for key in ("messages", "turns", "conversation"):
            value = loaded.get(key)
            if isinstance(value, list):
                return value
    return None


def _parse_timestamp_ns(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 10**18:
            return int(number)
        if number > 10**12:
            return int(number * 1_000_000)
        return int(number * 1_000_000_000)
    text = str(value).strip()
    if not text:
        return None
    try:
        return _parse_timestamp_ns(float(text))
    except ValueError:
        pass
    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(text, fmt).replace(tzinfo=UTC)
                break
            except ValueError:
                dt = None  # type: ignore[assignment]
        if dt is None:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000_000)


def _iso_from_ns(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1_000_000_000, tz=UTC).isoformat()


def _claim_text(row: sqlite3.Row | None) -> str | None:
    if row is None:
        return None
    keys = row.keys()
    text = row["text"] if "text" in keys else None
    if text:
        return str(text)
    return f"{row['subject']} {row['predicate']} {row['value']}".strip()


class MemContextFullProvider(MemoryProvider):
    """AMB-compatible provider backed by MemContext's full memory substrate."""

    name = "memcontext-full"
    description = "MemContext full substrate: episodes, claims, supersession, summaries, events"
    kind = "local"
    provider = "memcontext"
    variant = "full"
    concurrency = 1

    def __init__(
        self,
        *,
        extractor: Any | None = None,
        embedder: EmbeddingClient | None = None,
        semantic: SemanticSupersession | None = None,
        require_full: bool = True,
        retrieval_depth: int = 50,
        context_token_budget: int = 16_384,
    ) -> None:
        self._conn: sqlite3.Connection | None = None
        self._store_dir: Path | None = None
        self._db_path: Path | None = None
        self._extractor = extractor
        self._embedder = embedder
        self._semantic = semantic
        self._require_full = require_full
        self._retrieval_depth = retrieval_depth
        self._context_token_budget = context_token_budget
        self._sessions_by_namespace: dict[str, set[str]] = {}
        self._doc_sources: dict[str, dict[str, Any]] = {}

    def initialize(self) -> None:
        return None

    def cleanup(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("MemContextFullProvider.prepare() must be called first")
        return self._conn

    def prepare(
        self,
        store_dir: Path,
        unit_ids: set[str] | None = None,
        reset: bool = True,
    ) -> None:
        self.cleanup()
        self._validate_full_dependencies()
        self._store_dir = Path(store_dir)
        if reset and self._store_dir.exists():
            shutil.rmtree(self._store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._store_dir / "memcontext.db"
        self._conn = open_database(self._db_path)
        self._sessions_by_namespace = {str(uid): set() for uid in sorted(unit_ids or set())}
        self._doc_sources = {}

        if self._extractor is None:
            self._extractor = LLMExtractor(
                backend=os.environ.get("MEMCONTEXT_EXTRACTOR_BACKEND", "openrouter"),
                model=os.environ.get("MEMCONTEXT_EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL),
            )
        if self._embedder is None:
            self._embedder = episode_embedder()
        if self._semantic is None and self._embedder is not None:
            self._semantic = SemanticSupersession(self._embedder)

        self._write_config_snapshot(unit_ids=unit_ids, reset=reset)

    def ingest(self, documents: list[Document]) -> None:
        for doc in documents:
            self._ingest_one(doc)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        namespace = str(user_id) if user_id is not None else None
        session_ids = self._session_ids(namespace)
        if not session_ids:
            return [], {
                "provider": self.name,
                "query": query,
                "user_id": user_id,
                "session_ids": [],
                "hits": [],
                "summary": None,
            }
        valid_at_ts = _parse_timestamp_ns(query_timestamp)
        explain: dict[str, dict[str, float]] = {}
        hits = retrieve_memory_across(
            self.conn,
            session_ids=session_ids,
            query=query,
            top_k=max(k, self._retrieval_depth),
            per_session_k=3,
            valid_at_ts=valid_at_ts,
            embedding_client=self._embedder,
            explain=explain,
            include_superseded=valid_at_ts is not None,
        )

        documents: list[Document] = []
        raw_hits: list[dict[str, Any]] = []
        char_budget = max(1_000, self._context_token_budget * 4)
        used_chars = 0

        summary = None if valid_at_ts is not None else self._summary_text(namespace)
        if summary:
            summary_doc = Document(
                id=f"memcontext-summary:{namespace or 'all'}",
                content=summary,
                user_id=user_id,
                timestamp=query_timestamp,
                context="MemContext profile/event summary",
            )
            documents.append(summary_doc)
            used_chars += len(summary)

        for idx, (hit, score) in enumerate(hits, start=1):
            row = self._source_row(hit.source_turn_id)
            ts_ns = row["ts"] if row is not None else None
            source = self._doc_sources.get(hit.source_turn_id, {})
            claim_row = self._claim_row(hit.id) if hit.kind == "fact" else None
            body = self._format_hit_content(
                rank=idx,
                kind=hit.kind,
                score=score,
                session_id=row["session_id"] if row is not None else "",
                turn_id=hit.source_turn_id,
                claim_id=hit.id if hit.kind == "fact" else None,
                timestamp_ns=ts_ns,
                text=_claim_text(claim_row) if hit.kind == "fact" else hit.text,
                source=source,
            )
            if used_chars + len(body) > char_budget and documents:
                break
            documents.append(
                Document(
                    id=f"memcontext-{hit.kind}:{hit.id}",
                    content=body,
                    user_id=user_id,
                    timestamp=_iso_from_ns(ts_ns),
                    context="MemContext retrieval hit",
                )
            )
            used_chars += len(body)
            raw_hits.append(
                {
                    "rank": idx,
                    "kind": hit.kind,
                    "id": hit.id,
                    "source_turn_id": hit.source_turn_id,
                    "session_id": row["session_id"] if row is not None else None,
                    "score": round(float(score), 8),
                    "timestamp": _iso_from_ns(ts_ns),
                    "source": source,
                    "signals": explain.get(hit.id),
                }
            )

        raw = {
            "provider": self.name,
            "query": query,
            "user_id": user_id,
            "query_timestamp": query_timestamp,
            "valid_at_ts": valid_at_ts,
            "session_ids": session_ids,
            "summary": summary,
            "hits": raw_hits,
            "context_token_budget": self._context_token_budget,
        }
        return documents[: max(k, len(documents))], raw

    def _validate_full_dependencies(self) -> None:
        if not self._require_full:
            return
        backend = os.environ.get("MEMCONTEXT_EXTRACTOR_BACKEND", "openrouter").strip().lower()
        if backend != "openrouter":
            raise RuntimeError(
                "AMB full-product runs require MEMCONTEXT_EXTRACTOR_BACKEND=openrouter "
                "with TokenRouter's OpenAI-compatible endpoint."
            )
        endpoint = os.environ.get(
            "MEMCONTEXT_EXTRACTOR_ENDPOINT", f"{TOKENROUTER_BASE_URL}/chat/completions"
        )
        if "tokenrouter.com" not in endpoint:
            raise RuntimeError(
                "AMB full-product runs require MEMCONTEXT_EXTRACTOR_ENDPOINT to point "
                "at TokenRouter."
            )
        if not os.environ.get("MEMCONTEXT_EXTRACTOR_API_KEY"):
            raise RuntimeError(
                "AMB full-product runs require MEMCONTEXT_EXTRACTOR_API_KEY "
                "from the TokenRouter extractor secret."
            )
        model = os.environ.get("MEMCONTEXT_EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL)
        if model != DEFAULT_EXTRACTOR_MODEL:
            raise RuntimeError(
                "AMB full-product runs should use MEMCONTEXT_EXTRACTOR_MODEL="
                f"{DEFAULT_EXTRACTOR_MODEL} for comparable results."
            )
        if os.environ.get("MEMCONTEXT_EMBED_EPISODES", "1") == "0":
            raise RuntimeError("AMB full-product runs require MEMCONTEXT_EMBED_EPISODES=1.")
        if BGE_M3_MODEL_ID != "BAAI/bge-m3":
            raise RuntimeError(
                "AMB full-product runs require MEMCONTEXT_EMBED_MODEL=BAAI/bge-m3. "
                "Use a separately named ablation for other embedders."
            )
        if os.environ.get("MODAL_BGE_M3_URL"):
            return
        try:
            import sentence_transformers  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "AMB full-product runs require either MODAL_BGE_M3_URL or local "
                "sentence-transformers support (`pip install -e .[embeddings]`)."
            ) from exc

    def _write_config_snapshot(self, *, unit_ids: set[str] | None, reset: bool) -> None:
        assert self._store_dir is not None
        snapshot = {
            "provider": self.name,
            "db_path": str(self._db_path),
            "reset": reset,
            "unit_count": len(unit_ids or ()),
            "unit_ids": sorted(unit_ids or ()),
            "extractor_backend": os.environ.get("MEMCONTEXT_EXTRACTOR_BACKEND", "openrouter"),
            "extractor_model": os.environ.get(
                "MEMCONTEXT_EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL
            ),
            "extractor_endpoint": os.environ.get(
                "MEMCONTEXT_EXTRACTOR_ENDPOINT", f"{TOKENROUTER_BASE_URL}/chat/completions"
            ),
            "embed_episodes": os.environ.get("MEMCONTEXT_EMBED_EPISODES", "1"),
            "embed_model": BGE_M3_MODEL_ID,
            "modal_bge_m3_url_set": bool(os.environ.get("MODAL_BGE_M3_URL")),
            "retrieval_depth": self._retrieval_depth,
            "context_token_budget": self._context_token_budget,
        }
        (self._store_dir / "memcontext_run_config.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _ingest_one(self, doc: Any) -> None:
        doc_id = str(_doc_value(doc, "id", "document"))
        namespace = str(_doc_value(doc, "user_id", None) or "__default__")
        session_id = doc_id
        timestamp_ns = _parse_timestamp_ns(_doc_value(doc, "timestamp", None))
        context = _safe_text(_doc_value(doc, "context", ""))
        turns = self._turns_from_document(doc, context=context)
        self._sessions_by_namespace.setdefault(namespace, set()).add(session_id)

        for ordinal, item in enumerate(turns):
            turn_id = f"amb:{doc_id}:{ordinal}"
            result = on_new_turn(
                self.conn,
                session_id=session_id,
                speaker=item.speaker,
                text=item.text,
                extractor=self._extractor,
                semantic=self._semantic,
                embedder=self._embedder,
                namespace=namespace,
                turn_id=turn_id,
            )
            if not result.admitted:
                continue
            source_ts = (timestamp_ns or result.turn.ts) + ordinal * 1_000_000_000
            self._retime_result(result, source_ts)
            metadata = {
                "amb_document_id": doc_id,
                "amb_user_id": _doc_value(doc, "user_id", None),
                "amb_timestamp": _doc_value(doc, "timestamp", None),
                "amb_context": context or None,
                "turn_ordinal": ordinal,
                **item.metadata,
            }
            self.conn.execute(
                "UPDATE turns SET source_metadata = ? WHERE turn_id = ?",
                (json.dumps(metadata, ensure_ascii=False, sort_keys=True), turn_id),
            )
            self._doc_sources[turn_id] = metadata

        self._force_enrichment(session_id=session_id, namespace=namespace)

    def _turns_from_document(self, doc: Any, *, context: str) -> list[_IngestedTurn]:
        messages = _doc_value(doc, "messages", None)
        content = _safe_text(_doc_value(doc, "content", ""))
        raw_turns: list[Any] | None = messages if isinstance(messages, list) else None
        if raw_turns is None:
            raw_turns = _json_turns(content)
        if raw_turns is None:
            text = self._with_context(content, context=context, role="document")
            return [_IngestedTurn(Speaker.USER, text, {"source_shape": "content"})]

        parsed: list[_IngestedTurn] = []
        for idx, message in enumerate(raw_turns):
            role = message.get("role") if isinstance(message, dict) else None
            role = role or (message.get("speaker") if isinstance(message, dict) else None)
            content_text = _message_content(message)
            if not content_text.strip():
                continue
            speaker = _speaker_from_role(role)
            text = self._with_context(
                content_text,
                context=context if idx == 0 else "",
                role=speaker.value,
            )
            parsed.append(
                _IngestedTurn(
                    speaker,
                    text,
                    {"source_shape": "messages", "message_role": str(role or speaker.value)},
                )
            )
        if parsed:
            return parsed
        return [
            _IngestedTurn(
                Speaker.USER,
                self._with_context(content, context=context, role="document"),
                {},
            )
        ]

    def _with_context(self, text: str, *, context: str, role: str) -> str:
        pieces = []
        if context.strip():
            pieces.append(f"Document context: {context.strip()}")
        pieces.append(f"{role.capitalize()} turn: {text.strip()}")
        return "\n".join(pieces)

    def _retime_result(self, result: TurnResult, ts_ns: int) -> None:
        if result.turn is None:
            return
        self.conn.execute("UPDATE turns SET ts = ? WHERE turn_id = ?", (ts_ns, result.turn.turn_id))
        for claim in result.created_claims:
            self.conn.execute(
                "UPDATE claims SET created_ts = ?, valid_from_ts = ?, event_ts = ?"
                " WHERE claim_id = ?",
                (ts_ns, ts_ns, ts_ns, claim.claim_id),
            )
        for edge in result.supersession_edges:
            self.conn.execute(
                "UPDATE supersession_edges SET created_ts = ? WHERE edge_id = ?",
                (ts_ns, edge.edge_id),
            )
            self.conn.execute(
                "UPDATE claims SET valid_until_ts = ? WHERE claim_id = ?",
                (ts_ns, edge.old_claim_id),
            )

    def _force_enrichment(self, *, session_id: str, namespace: str) -> None:
        try:
            recompute_all_importance(self.conn)
            store_digest(self.conn, build_session_digest(self.conn, session_id))
            assemble_event_frames(self.conn, session_id)
            events = detect_life_events(self.conn, "user", namespace=namespace, min_predicates=2)
            store_life_events(self.conn, events)
            consolidate_facts(self.conn, min_sessions=3)
            if self._embedder is not None:
                backfill_event_frame_embeddings(self.conn, session_id, client=self._embedder)
        except Exception:  # noqa: BLE001
            # Enrichment failures should not poison the AMB ingestion lifecycle.
            return

    def _session_ids(self, namespace: str | None) -> list[str]:
        if namespace is not None:
            return sorted(self._sessions_by_namespace.get(namespace, set()))
        all_ids: set[str] = set()
        for ids in self._sessions_by_namespace.values():
            all_ids.update(ids)
        return sorted(all_ids)

    def _summary_text(self, namespace: str | None) -> str | None:
        pieces: list[str] = []
        briefing = session_briefing(self.conn, namespace=namespace)
        if briefing:
            pieces.append("MemContext briefing:\n" + briefing)
        if namespace is not None:
            life_events = detect_life_events(
                self.conn, "user", namespace=namespace, min_predicates=2
            )
        else:
            life_events = detect_life_events(self.conn, "user", min_predicates=2)
        if life_events:
            lines = [
                f"- {event.summary_text} (significance={event.significance:.2f})"
                for event in life_events[:8]
            ]
            pieces.append("Life/event digest:\n" + "\n".join(lines))
        if not pieces:
            return None
        return "\n\n".join(pieces)

    def _source_row(self, turn_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()

    def _claim_row(self, claim_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()

    def _format_hit_content(
        self,
        *,
        rank: int,
        kind: str,
        score: float,
        session_id: str,
        turn_id: str,
        claim_id: str | None,
        timestamp_ns: int | None,
        text: str | None,
        source: dict[str, Any],
    ) -> str:
        provenance = {
            "rank": rank,
            "kind": kind,
            "score": round(float(score), 8),
            "session_id": session_id,
            "turn_id": turn_id,
            "claim_id": claim_id,
            "timestamp": _iso_from_ns(timestamp_ns),
            "amb_document_id": source.get("amb_document_id"),
            "amb_user_id": source.get("amb_user_id"),
            "amb_timestamp": source.get("amb_timestamp"),
        }
        return (
            "MemContext retrieval hit\n"
            f"Provenance: {json.dumps(provenance, ensure_ascii=False, sort_keys=True)}\n"
            f"Content: {text or ''}"
        )

    def debug_state(self) -> dict[str, Any]:
        """Small test/debug hook; not used by AMB."""
        config_path = self._store_dir / "memcontext_run_config.json" if self._store_dir else None
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
        return {
            "store_dir": str(self._store_dir) if self._store_dir else None,
            "db_path": str(self._db_path) if self._db_path else None,
            "config": config,
            "sessions_by_namespace": {
                key: sorted(value) for key, value in self._sessions_by_namespace.items()
            },
        }


def document_to_dict(doc: Document) -> dict[str, Any]:
    """Return a JSON-friendly view of either AMB's or our fallback Document."""
    if is_dataclass(doc):
        return asdict(doc)
    return {
        "id": _doc_value(doc, "id"),
        "content": _doc_value(doc, "content"),
        "user_id": _doc_value(doc, "user_id"),
        "messages": _doc_value(doc, "messages"),
        "timestamp": _doc_value(doc, "timestamp"),
        "context": _doc_value(doc, "context"),
    }
