"""LIPI integration: the capability cycles compose on a single DB without
conflict -- Phase 1 ranking + B temporal + C consolidation + F retention + D
working context all coexist and behave correctly together.
"""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim
from memcontext.consolidate import consolidate_facts
from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import handle_memory_query
from memcontext.on_new_turn import on_new_turn
from memcontext.retention import demote_low_utility
from memcontext.schema import ClaimStatus, Speaker, open_database
from memcontext.working_context import build_working_context


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _say(conn, session, subject, predicate, value):
    on_new_turn(
        conn, session_id=session, speaker=Speaker.USER, text=f"{subject} {predicate} {value}",
        extractor=PassthroughExtractor(
            [{"subject": subject, "predicate": predicate, "value": value, "confidence": 0.9}]),
    )


def test_cycles_compose_on_one_db():
    conn = _conn()
    # C: same preference recurs across 3 sessions; plus a unique low-value fact in s1
    for s in ("s1", "s2", "s3"):
        _say(conn, s, "user", "user_preference", "darkmode")
    _say(conn, "s1", "user", "user_fact", "trivia")

    # C: the recurring preference graduates to a durable consolidated fact
    assert consolidate_facts(conn, min_sessions=3) == 1

    # Phase 1 + C: the door serves the consolidated fact with full ranking debug
    res = handle_memory_query(conn, query="darkmode preference", top_k=10, debug=True)
    assert any(c.get("consolidated") for c in res["claims"]), "C: consolidated fact surfaced"
    assert "token_report" in res and res["ranking"], "Phase 1: token report + ranking debug"
    cid = res["claims"][0]["claim_id"]
    assert {"importance", "usage", "final"} <= set(res["ranking"][cid]), "Phase 1: live signals"

    # F: an ancient low-utility fact is demoted out of active retrieval
    tcid = conn.execute("SELECT claim_id FROM claims WHERE value='trivia'").fetchone()["claim_id"]
    conn.execute("UPDATE claims SET created_ts=1000 WHERE claim_id=?", (tcid,))
    conn.execute("UPDATE claim_metadata SET importance_score=0.02, access_count=0 WHERE claim_id=?", (tcid,))
    assert demote_low_utility(conn, threshold=0.35, min_age_days=1.0) >= 1
    served = {c["claim_id"] for c in handle_memory_query(conn, query="trivia", session_id="s1")["claims"]}
    assert tcid not in served, "F: demoted fact gone from retrieval"

    # B: a superseded prior value surfaces ONLY on past-intent (composes with the rest)
    turn = conn.execute("SELECT source_turn_id FROM claims WHERE session_id='s1' LIMIT 1").fetchone()["source_turn_id"]
    insert_claim(
        conn, session_id="s1", subject="user", predicate="user_preference",
        value="lightmode", confidence=0.9, source_turn_id=turn, status=ClaimStatus.SUPERSEDED)
    lcid = conn.execute("SELECT claim_id FROM claims WHERE value='lightmode'").fetchone()["claim_id"]
    now = {c["claim_id"] for c in handle_memory_query(conn, query="lightmode", session_id="s1")["claims"]}
    hist = {c["claim_id"] for c in handle_memory_query(conn, query="lightmode previously", session_id="s1")["claims"]}
    assert lcid not in now and lcid in hist, "B: history mode toggles superseded inclusion"

    # D: a working context over the same DB is budget-bounded
    ctx = build_working_context(conn, "s1", token_budget=10)
    assert ctx.tokens_used <= 10 and ctx.total_active >= 1, "D: budget-bounded working set"
