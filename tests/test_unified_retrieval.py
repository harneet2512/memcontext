"""Unified retrieve_memory: facts AND episodes, source-tagged, NL-only-safe.

Ranks via BM25 over NL text (no embeddings → zero model downloads). Verifies the
merge tags each hit, that episodes carry retrieval when no facts exist (the
Tier-1 floor), that NL-only facts rank with no structured field, and that a fact
outranks the episode it came from.
"""
from __future__ import annotations

import math
import sqlite3
from typing import cast

from memcontext.claims import insert_fact, insert_turn, new_turn_id, now_ns
from memcontext.retrieval import EmbeddingClient, retrieve_memory
from memcontext.schema import Speaker, Turn, open_database


class _StubEmbedder:
    """model_version only; embed() is never called (no embeddings written)."""

    model_version = "test-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 / math.sqrt(8)] * 8 for _ in texts]


def _client() -> EmbeddingClient:
    return cast(EmbeddingClient, _StubEmbedder())


def _conn() -> sqlite3.Connection:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _episode(conn: sqlite3.Connection, sid: str, text: str) -> Turn:
    turn = Turn(
        turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER,
        text=text, ts=now_ns(),
    )
    insert_turn(conn, turn)
    return turn


def test_retrieve_memory_merges_and_tags_both_kinds():
    conn = _conn()
    sid = "s1"
    # An episode whose text is about the deploy script, and a structured fact too.
    ep = _episode(conn, sid, "the deploy script lives in tools/deploy.sh on prod")
    insert_fact(
        conn, session_id=sid, source_turn_id=ep.turn_id, confidence=0.9,
        subject="deploy_script", predicate="user_fact", value="tools/deploy.sh",
    )
    # An unrelated episode so ranking has to discriminate.
    _episode(conn, sid, "we had ramen for lunch near the office")

    results = retrieve_memory(
        conn, session_id=sid, query="where is the deploy script", top_k=10,
        embedding_client=_client(),
    )
    assert results
    kinds = {hit.kind for hit, _ in results}
    assert "fact" in kinds and "episode" in kinds, kinds
    # Top hit is about the deploy script (either kind), not lunch.
    assert "deploy" in results[0][0].text


def test_episodes_carry_retrieval_when_no_facts_exist():
    """Tier-1 floor: with extraction disabled (no facts), episodes still rank."""
    conn = _conn()
    sid = "s2"
    _episode(conn, sid, "the staging database password rotates every ninety days")
    _episode(conn, sid, "lunch options near the office are thai and ramen")

    results = retrieve_memory(
        conn, session_id=sid, query="how often does the staging password rotate",
        top_k=10, embedding_client=_client(),
    )
    assert results, "episodes must carry retrieval with zero facts"
    assert all(hit.kind == "episode" for hit, _ in results)
    assert "staging database password" in results[0][0].text


def test_nl_only_fact_appears_in_unified_ranking():
    conn = _conn()
    sid = "s3"
    ep = _episode(conn, sid, "context turn")
    insert_fact(
        conn, session_id=sid, source_turn_id=ep.turn_id, confidence=0.8,
        text="the quarterly board meeting is on the first tuesday of march",
    )
    results = retrieve_memory(
        conn, session_id=sid, query="when is the quarterly board meeting",
        top_k=10, embedding_client=_client(),
    )
    assert results
    fact_hits = [h for h, _ in results if h.kind == "fact"]
    assert fact_hits, "NL-only fact must appear in unified retrieval"
    assert "board meeting" in fact_hits[0].text


def test_fact_outranks_the_episode_it_came_from():
    conn = _conn()
    sid = "s4"
    ep = _episode(conn, sid, "my preferred deployment target is dynamodb")
    insert_fact(
        conn, session_id=sid, source_turn_id=ep.turn_id, confidence=0.9,
        subject="user", predicate="user_preference", value="dynamodb deployment target",
    )
    results = retrieve_memory(
        conn, session_id=sid, query="preferred deployment target dynamodb",
        top_k=10, embedding_client=_client(),
    )
    # The fact and its source episode both match; the fact (higher-information)
    # ranks above the episode it came from.
    order = [hit.kind for hit, _ in results]
    assert order and order[0] == "fact", order
