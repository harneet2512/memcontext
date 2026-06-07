"""Tier-1 episode tests.

Episodes (turns / tool-call results / browser observations) are the always-on,
zero-LLM, synchronous floor: stored at ingest, embedded with a local model (no
LLM), and immediately retrievable via the hybrid retrieval signals — with no
structured fields and no fact extraction required.
"""
from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from memcontext.claims import get_turn, insert_turn, new_turn_id, now_ns
from memcontext.retrieval import (
    EmbeddingClient,
    backfill_episode_embeddings,
    embed_and_store_episode,
    retrieve_episodes,
)
from memcontext.schema import ExtractionStatus, SourceType, Speaker, Turn, open_database

DIM = 64


class _StubEmbedder:
    """Deterministic token-hashing embedder — no model load, no LLM, no network.

    Each text becomes a normalised bag-of-words vector over a fixed hashing
    space, so cosine similarity reflects token overlap. Uses a stable (non-salted)
    hash so the on-disk embedding cache stays content-consistent across runs.
    Lets episode retrieval be asserted without downloading all-MiniLM-L6-v2.
    """

    model_version = "test-model"

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim
        self.embed_calls = 0

    @staticmethod
    def _bucket(token: str, dim: int) -> int:
        acc = 0
        for ch in token:
            acc = (acc * 31 + ord(ch)) % dim
        return acc

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self._dim
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                v[self._bucket(tok, self._dim)] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def _embedder() -> tuple[_StubEmbedder, EmbeddingClient]:
    """Return the stub and the same object typed as EmbeddingClient for the API."""
    stub = _StubEmbedder()
    return stub, cast(EmbeddingClient, stub)


@pytest.fixture(autouse=True)
def _isolate_embed_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the on-disk embedding cache at a fresh dir per test.

    The cache is keyed on (model_version, text); without isolation a vector
    cached under "test-model" by one test would leak into another, making
    embed-call counts and the semantic channel non-deterministic.
    """
    monkeypatch.setenv("SUBSTRATE_EMBED_CACHE_DIR", str(tmp_path / "emb_cache"))


def _conn() -> sqlite3.Connection:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _episode(
    conn: sqlite3.Connection,
    sid: str,
    text: str,
    *,
    source_type: SourceType = SourceType.CONVERSATION,
    ts: int | None = None,
    speaker: Speaker = Speaker.USER,
) -> Turn:
    turn = Turn(
        turn_id=new_turn_id(),
        session_id=sid,
        speaker=speaker,
        text=text,
        ts=ts if ts is not None else now_ns(),
        source_type=source_type,
    )
    insert_turn(conn, turn)
    return turn


# --- Test 1: episode store is synchronous, zero-LLM, immediately retrievable ---


def test_episode_store_is_synchronous_and_zero_llm():
    conn = _conn()
    sid = "s1"
    emb, client = _embedder()

    turn = _episode(conn, sid, "the deploy script lives in tools/deploy.sh")
    # Embedding the episode is a single local model call — never an LLM call.
    embed_and_store_episode(conn, turn, client=client)

    # The episode row and its embedding both exist synchronously.
    assert get_turn(conn, turn.turn_id) is not None
    row = conn.execute(
        "SELECT embedding_model_version FROM turn_embeddings WHERE turn_id = ?",
        (turn.turn_id,),
    ).fetchone()
    assert row is not None
    assert row["embedding_model_version"] == "test-model"
    assert emb.embed_calls == 1

    # No facts/claims were created — episodes need no extraction.
    n_claims = conn.execute(
        "SELECT COUNT(*) AS n FROM claims WHERE session_id = ?", (sid,)
    ).fetchone()["n"]
    assert n_claims == 0


def test_episode_immediately_retrievable_after_store():
    conn = _conn()
    sid = "s2"
    _, client = _embedder()

    deploy = _episode(conn, sid, "the deploy script lives in tools/deploy.sh")
    coffee = _episode(conn, sid, "I had coffee with Priya this morning")
    budget = _episode(conn, sid, "the quarterly budget review is next Tuesday")
    for t in (deploy, coffee, budget):
        embed_and_store_episode(conn, t, client=client)

    results = retrieve_episodes(
        conn, session_id=sid, query="where is the deploy script", embedding_client=client
    )
    assert results, "episodes must be retrievable immediately after store"
    assert results[0][0].turn_id == deploy.turn_id


# --- Test: all three source_types are first-class retrievable episodes ---------


def test_all_source_types_are_retrievable():
    conn = _conn()
    sid = "s3"
    _, client = _embedder()

    conv = _episode(
        conn, sid, "remember that my passport number is on the kitchen table",
        source_type=SourceType.CONVERSATION,
    )
    tool = _episode(
        conn, sid, "tool result: ls returned report_2026.pdf and notes.txt",
        source_type=SourceType.TOOL_CALL, speaker=Speaker.ASSISTANT,
    )
    browser = _episode(
        conn, sid, "page title Acme pricing shows the enterprise tier is 499 usd",
        source_type=SourceType.BROWSER, speaker=Speaker.ASSISTANT,
    )
    for t in (conv, tool, browser):
        embed_and_store_episode(conn, t, client=client)

    # source_type round-trips through storage.
    tool_back = get_turn(conn, tool.turn_id)
    browser_back = get_turn(conn, browser.turn_id)
    assert tool_back is not None and tool_back.source_type is SourceType.TOOL_CALL
    assert browser_back is not None and browser_back.source_type is SourceType.BROWSER

    by_query = {
        conv.turn_id: "where is my passport number",
        tool.turn_id: "what did the ls tool return",
        browser.turn_id: "what is the enterprise tier price",
    }
    for expected_id, query in by_query.items():
        results = retrieve_episodes(
            conn, session_id=sid, query=query, embedding_client=client
        )
        assert results[0][0].turn_id == expected_id, f"query {query!r} mis-ranked"


# --- Test: retrieval degrades to BM25 + recency when no embeddings exist -------


def test_retrieve_episodes_degrades_without_embeddings():
    conn = _conn()
    sid = "s4"
    emb, client = _embedder()

    # Deliberately store episodes WITHOUT embedding them (the async/floor case).
    _episode(conn, sid, "the deploy script lives in tools/deploy.sh")
    _episode(conn, sid, "I had coffee with Priya this morning")

    results = retrieve_episodes(
        conn, session_id=sid, query="deploy script location", embedding_client=client
    )
    assert results, "episodes must retrieve via BM25 even with no embeddings"
    assert "deploy" in results[0][0].text
    # No turn_embeddings were written, so the semantic channel never fired.
    assert emb.embed_calls == 0


def test_backfill_episode_embeddings_fills_missing():
    conn = _conn()
    sid = "s5"
    _, client = _embedder()

    _episode(conn, sid, "alpha episode about migrations")
    _episode(conn, sid, "beta episode about retrieval")

    n = backfill_episode_embeddings(conn, sid, client=client)
    assert n == 2
    count = conn.execute("SELECT COUNT(*) AS n FROM turn_embeddings").fetchone()["n"]
    assert count == 2
    # Idempotent: a second pass embeds nothing new.
    assert backfill_episode_embeddings(conn, sid, client=client) == 0


def test_extraction_status_defaults_pending():
    conn = _conn()
    sid = "s6"
    turn = _episode(conn, sid, "an episode awaiting async fact extraction")
    back = get_turn(conn, turn.turn_id)
    assert back is not None and back.extraction_status is ExtractionStatus.PENDING
