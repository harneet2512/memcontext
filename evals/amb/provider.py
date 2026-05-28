"""MemContext adapter for the Agent Memory Benchmark (AMB) harness.

Implements the MemoryProvider interface from
https://github.com/vectorize-io/agent-memory-benchmark
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .base import MemoryProvider
from ..models import Document

from memcontext.claims import now_ns
from memcontext.extractors import PassthroughExtractor, auto_extractor
from memcontext.on_new_turn import on_new_turn
from memcontext.retrieval import EmbeddingClient, backfill_embeddings, retrieve_hybrid
from memcontext.schema import Speaker, Turn, open_database

logger = logging.getLogger(__name__)


class MemContextProvider(MemoryProvider):
    name = "memcontext"
    description = (
        "Deterministic memory context layer. Extracts structured claims "
        "with provenance and supersession. Multi-signal hybrid retrieval via RRF."
    )
    kind = "local"
    concurrency = 1

    def __init__(self):
        self._conn: sqlite3.Connection | None = None
        self._embedding_client: EmbeddingClient | None = None
        self._extractor = None
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
        if reset:
            self.cleanup()
            self._conn = open_database(self._db_path)
            self._conn.row_factory = sqlite3.Row

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = open_database(self._db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def ingest(self, documents: list[Document]) -> None:
        conn = self._ensure_conn()
        if self._extractor is None:
            self._extractor = auto_extractor()
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient()

        extractor = self._extractor
        all_work: list[tuple[str, str, str]] = []

        first_user_id = documents[0].user_id if documents else "default"
        unified_session = f"amb_{first_user_id}"

        for doc in documents:
            turns = _parse_document_turns(doc)
            for role, text in turns:
                all_work.append((unified_session, role, text))

        extracted: list[tuple[str, str, str, list[dict]]] = []

        def _extract_one(item: tuple[str, str, str]) -> tuple[str, str, str, list[dict]]:
            sid, role, text = item
            sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
            claims = _extract_claims(extractor, sid, sp, text)
            return (sid, role, text, claims)

        _workers = int(os.environ.get("MEMCONTEXT_EXTRACTION_WORKERS", "32"))
        with ThreadPoolExecutor(max_workers=_workers) as pool:
            futures = {pool.submit(_extract_one, w): w for w in all_work}
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    extracted.append(fut.result())
                except Exception:
                    w = futures[fut]
                    extracted.append((w[0], w[1], w[2], []))
                if done % 100 == 0:
                    logger.info(f"Extracted {done}/{len(all_work)} turns")

        if done > 0:
            logger.info(f"Extracted {done}/{len(all_work)} turns")

        by_session: dict[str, list[tuple[str, str, list[dict]]]] = {}
        for sid, role, text, claims_data in extracted:
            by_session.setdefault(sid, []).append((role, text, claims_data))

        for sid in sorted(by_session.keys()):
            for role, text, claims_data in by_session[sid]:
                if not claims_data:
                    claims_data = [{
                        "subject": "user" if role == "user" else "assistant",
                        "predicate": "user_fact",
                        "value": text[:500],
                        "confidence": 0.3,
                    }]

                sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
                pt = PassthroughExtractor(claims_data)
                on_new_turn(
                    conn,
                    session_id=sid,
                    speaker=sp,
                    text=text,
                    extractor=pt,
                )

        backfill_embeddings(conn, unified_session, client=self._embedding_client)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        conn = self._ensure_conn()
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient()

        unified_session = f"amb_{user_id}" if user_id else _get_any_session(conn)

        results = retrieve_hybrid(
            conn,
            session_id=unified_session,
            query=query,
            top_k=k * 5,
            embedding_client=self._embedding_client,
        )
        top = results[:k]

        from memcontext.claims import get_turn

        result_docs = []
        seen_turns: set[str] = set()
        for claim, score in top:
            if claim.source_turn_id in seen_turns:
                continue
            seen_turns.add(claim.source_turn_id)

            turn = get_turn(conn, claim.source_turn_id)
            content = turn.text if turn else claim.value

            result_docs.append(Document(
                id=claim.claim_id,
                content=content,
                user_id=user_id,
            ))

        return result_docs, None


def _parse_document_turns(doc: Document) -> list[tuple[str, str]]:
    messages = doc.messages
    if messages and isinstance(messages, list):
        return [
            (m.get("role", "user"), m.get("content", ""))
            for m in messages
            if isinstance(m, dict) and m.get("content", "").strip()
        ]

    content = doc.content
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


def _get_any_session(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT DISTINCT session_id FROM claims LIMIT 1").fetchone()
    if row:
        return row["session_id"] if isinstance(row, sqlite3.Row) else row[0]
    return "amb_default"
