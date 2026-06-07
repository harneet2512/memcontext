"""Cycle E (procedural memory, EXPERIMENTAL): recurring ordered action sequences
across sessions graduate to a procedure with provenance; off by default.
"""
from __future__ import annotations

import sqlite3

from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.procedural import detect_procedures, procedural_enabled
from memcontext.schema import Speaker, open_database


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _seq(conn, session, predicates):
    """Lay down an ordered action sequence (controlled predicates) for a session."""
    on_new_turn(conn, session_id=session, speaker=Speaker.USER,
                text=f"{session} workflow has several meaningful steps to perform today",
                extractor=PassthroughExtractor([]))
    tid = conn.execute(
        "SELECT turn_id FROM turns WHERE session_id=? ORDER BY ts DESC LIMIT 1", (session,)
    ).fetchone()["turn_id"]
    for i, p in enumerate(predicates):
        conn.execute(
            "INSERT INTO claims (claim_id, session_id, subject, predicate, value,"
            " confidence, source_turn_id, status, created_ts)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (f"cl_{session}_{i}", session, "user", p, f"v{i}", 0.9, tid, "active", 1000 + i),
        )


def test_detect_recurring_procedure_with_provenance():
    conn = _conn()
    _seq(conn, "s1", ["plan", "build", "ship"])
    _seq(conn, "s2", ["plan", "build", "ship"])
    _seq(conn, "s3", ["plan", "rest"])  # the 3-step sequence does not recur here

    procs = detect_procedures(conn, min_sessions=2, min_steps=3)
    match = [p for p in procs if p.steps == ("plan", "build", "ship")]
    assert match, "recurring 3-step sequence detected"
    p = match[0]
    assert p.recurrence == 2 and p.trigger == "plan"
    assert set(p.sessions) == {"s1", "s2"}
    assert len(p.source_claim_ids) >= 3, "provenance to source claims"


def test_procedural_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MEMCONTEXT_EXPERIMENTAL_PROCEDURAL", raising=False)
    assert procedural_enabled() is False
    monkeypatch.setenv("MEMCONTEXT_EXPERIMENTAL_PROCEDURAL", "1")
    assert procedural_enabled() is True


def test_procedural_door_respects_flag(monkeypatch):
    from memcontext.mcp_tools import handle_memory_procedures

    conn = _conn()
    _seq(conn, "s1", ["plan", "build", "ship"])
    _seq(conn, "s2", ["plan", "build", "ship"])

    monkeypatch.delenv("MEMCONTEXT_EXPERIMENTAL_PROCEDURAL", raising=False)
    off = handle_memory_procedures(conn)
    assert off["enabled"] is False and off["procedures"] == []

    monkeypatch.setenv("MEMCONTEXT_EXPERIMENTAL_PROCEDURAL", "1")
    on = handle_memory_procedures(conn)
    assert on["enabled"] is True
    assert any(tuple(p["steps"]) == ("plan", "build", "ship") for p in on["procedures"])
