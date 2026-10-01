"""Cycle B (temporal truth & freshness): superseded facts are excluded by default
but surface when the query asks about the past.

The differentiator is ONLY the history intent of the query — same matching value,
opposite inclusion — so it proves the temporal behavior, not retrieval luck.
"""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim
from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.retrieval import detect_history_intent
from memcontext.schema import ClaimStatus, Speaker, open_database


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def test_detect_history_intent():
    assert detect_history_intent("what did I use to like")
    assert detect_history_intent("my address before the move")
    assert detect_history_intent("what was my plan previously")
    assert not detect_history_intent("what is my current address")
    assert not detect_history_intent("where do I live")


def test_history_mode_surfaces_superseded_only_on_past_intent():
    from memcontext.mcp_tools import handle_memory_query

    conn = _conn()
    # one active fact + a superseded prior value for the same slot
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="I live in Munich",
        extractor=PassthroughExtractor(
            [{"subject": "user", "predicate": "user_fact", "value": "munich", "confidence": 0.9}]),
    )
    turn = conn.execute("SELECT source_turn_id FROM claims LIMIT 1").fetchone()["source_turn_id"]
    insert_claim(
        conn, session_id="s1", subject="user", predicate="user_fact",
        value="berlin", confidence=0.9, source_turn_id=turn,
        status=ClaimStatus.SUPERSEDED,
    )

    # Same matching token ("berlin"), but NO history intent -> superseded excluded.
    res = handle_memory_query(conn, query="tell me about berlin", session_id="s1", top_k=10)
    assert "berlin" not in {c["value"] for c in res["claims"]}, "superseded excluded by default"

    # WITH history intent ("previously") -> the superseded fact surfaces.
    res_h = handle_memory_query(
        conn, query="what about berlin previously", session_id="s1", top_k=10)
    assert "berlin" in {c["value"] for c in res_h["claims"]}, "history mode surfaces superseded"
