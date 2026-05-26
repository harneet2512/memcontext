"""MemContext adapter for the Agent Memory Benchmark (AMB) harness.

Implements the MemoryProvider interface from
https://github.com/vectorize-io/agent-memory-benchmark

Two required methods:
  - ingest(documents) — parse conversation turns, extract claims, embed
  - retrieve(query, k, user_id, query_timestamp) — hybrid retrieval, return Documents

Usage:
  Copy this file into the AMB repo at src/memory_bench/memory/memcontext.py,
  or symlink it. Then run:

    uv run amb run --dataset longmemeval --domain S --memory memcontext
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from memcontext.claims import now_ns
from memcontext.extractors import PassthroughExtractor, auto_extractor
from memcontext.on_new_turn import on_new_turn
from memcontext.retrieval import EmbeddingClient, backfill_embeddings, retrieve_hybrid
from memcontext.schema import Speaker, Turn, open_database


class MemContextProvider:
    """AMB MemoryProvider adapter for MemContext.

    Stores claims in SQLite, retrieves via multi-signal hybrid search
    (semantic + BM25 + RRF). Fully deterministic core — no LLM calls
    in the storage/retrieval path.
    """

    name = "memcontext"
    description = (
        "Deterministic memory context layer. Extracts structured claims "
        "(subject-predicate-value triples) with provenance, temporal validity, "
        "and supersession tracking. Multi-signal hybrid retrieval via RRF."
    )
    kind = "local"
    concurrency = 1

    def __init__(self):
        self._conn: sqlite3.Connection | None = None
        self._embedding_client: EmbeddingClient | None = None
        self._extractor = None
        self._store_dir: Path | None = None
        self._db_path: str = ":memory:"

    def initialize(self) -> None:
        if not os.environ.get("ACTIVE_PACK"):
            os.environ["ACTIVE_PACK"] = "personal_assistant"
            from memcontext.predicate_packs import active_pack
            active_pack.cache_clear()

        self._extractor = auto_extractor()
        self._embedding_client = EmbeddingClient()

    def cleanup(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def prepare(
        self,
        store_dir: Path,
        unit_ids: set[str] | None = None,
        reset: bool = True,
    ) -> None:
        self._store_dir = store_dir
        if reset:
            self.cleanup()
            self._conn = open_database(self._db_path)

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = open_database(self._db_path)
        return self._conn

    def ingest(self, documents: list) -> None:
        """Ingest AMB Documents into MemContext.

        Each Document contains conversation turns as JSON in doc.content.
        We parse turns, extract claims, run supersession, and embed.
        """
        conn = self._ensure_conn()
        if self._extractor is None:
            self._extractor = auto_extractor()
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient()

        for doc in documents:
            user_id = getattr(doc, "user_id", None) or "default"
            session_id = f"amb_{user_id}_{getattr(doc, 'id', 'unknown')}"

            turns = _parse_document_turns(doc)

            for role, text in turns:
                sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
                claims_data = _extract_claims(self._extractor, session_id, sp, text)

                if claims_data:
                    pt = PassthroughExtractor(claims_data)
                    on_new_turn(
                        conn,
                        session_id=session_id,
                        speaker=sp,
                        text=text,
                        extractor=pt,
                    )

            backfill_embeddings(conn, session_id, client=self._embedding_client)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list, dict | None]:
        """Retrieve top-k relevant context for a query.

        Returns (list_of_document_like_objects, None).
        Each returned object has .id and .content attributes.
        """
        conn = self._ensure_conn()
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient()

        session_ids = _get_session_ids(conn, user_id)

        all_results = []
        for sid in session_ids:
            results = retrieve_hybrid(
                conn,
                session_id=sid,
                query=query,
                top_k=k,
                embedding_client=self._embedding_client,
                weights=(0.5, 0.2, 0.1, 0.2),
            )
            all_results.extend(results)

        all_results.sort(key=lambda x: (-x[1], x[0].claim_id))
        top = all_results[:k]

        from memcontext.claims import get_turn

        result_docs = []
        seen_turns: set[str] = set()
        for claim, score in top:
            if claim.source_turn_id in seen_turns:
                continue
            seen_turns.add(claim.source_turn_id)

            turn = get_turn(conn, claim.source_turn_id)
            content = turn.text if turn else claim.value

            result_docs.append(
                _ResultDoc(
                    id=claim.claim_id,
                    content=content,
                    user_id=user_id,
                )
            )

        return result_docs, None


class _ResultDoc:
    """Minimal document-like object returned from retrieve()."""

    def __init__(self, id: str, content: str, user_id: str | None = None):
        self.id = id
        self.content = content
        self.user_id = user_id


def _parse_document_turns(doc) -> list[tuple[str, str]]:
    """Parse conversation turns from an AMB Document."""
    messages = getattr(doc, "messages", None)
    if messages and isinstance(messages, list):
        return [
            (m.get("role", "user"), m.get("content", ""))
            for m in messages
            if isinstance(m, dict) and m.get("content", "").strip()
        ]

    content = getattr(doc, "content", "")
    if not content:
        return []

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [
                (m.get("role", "user"), m.get("content", ""))
                for m in parsed
                if isinstance(m, dict) and m.get("content", "").strip()
            ]
    except (json.JSONDecodeError, TypeError):
        pass

    if content.strip():
        return [("user", content)]
    return []


def _extract_claims(extractor, session_id, speaker, text):
    """Extract claims from a single turn using the configured extractor."""
    turn = Turn(
        turn_id=f"tu_amb_{now_ns()}",
        session_id=session_id,
        speaker=speaker,
        text=text,
        ts=now_ns(),
    )
    try:
        claims = extractor(turn)
        return [
            {
                "subject": c.subject,
                "predicate": c.predicate,
                "value": c.value,
                "confidence": c.confidence,
            }
            for c in claims
        ]
    except Exception:
        return []


def _get_session_ids(conn: sqlite3.Connection, user_id: str | None) -> list[str]:
    """Get all session IDs, optionally filtered by user_id prefix."""
    if user_id:
        prefix = f"amb_{user_id}_%"
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM claims WHERE session_id LIKE ?",
            (prefix,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM claims"
        ).fetchall()

    return [r["session_id"] if isinstance(r, sqlite3.Row) else r[0] for r in rows]
