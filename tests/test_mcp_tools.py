from __future__ import annotations

import sqlite3

import pytest

from memcontext.mcp_tools import (
    handle_memory_correct,
    handle_memory_query,
    handle_memory_store,
    handle_memory_trace,
)
from memcontext.schema import open_database


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def test_store_with_passthrough_claims(conn):
    result = handle_memory_store(
        conn,
        text="I live in Toronto and prefer dark mode",
        session_id="s1",
        claims=[
            {"subject": "user", "predicate": "user_fact", "value": "lives in Toronto", "confidence": 0.9},
            {"subject": "user", "predicate": "user_preference", "value": "prefers dark mode", "confidence": 0.8},
        ],
    )
    assert result["admitted"] is True
    assert result["claims_created"] == 2
    assert len(result["claim_ids"]) == 2


def test_store_with_raw_text(conn):
    result = handle_memory_store(
        conn,
        text="I prefer using Python for backend development",
        session_id="s1",
    )
    assert result["admitted"] is True
    assert result["claims_created"] >= 1


def test_store_returns_structure(conn):
    result = handle_memory_store(conn, text="I am a developer", session_id="s1")
    assert "turn_id" in result
    assert "claims_created" in result
    assert "claim_ids" in result
    assert "session_id" in result
    assert "supersessions" in result


def test_store_auto_session(conn):
    result = handle_memory_store(conn, text="I am a developer")
    assert result["session_id"].startswith("session_")


def test_query_finds_stored(conn):
    handle_memory_store(
        conn, text="I really prefer using dark mode everywhere", session_id="s1",
        claims=[{"subject": "user", "predicate": "user_preference", "value": "prefers dark mode", "confidence": 0.9}],
    )
    result = handle_memory_query(conn, query="dark mode", session_id="s1")
    assert result["total"] >= 1
    assert len(result["claims"]) >= 1
    values = [c["value"] for c in result["claims"]]
    assert any("dark" in v.lower() for v in values)


def test_query_empty_session(conn):
    result = handle_memory_query(conn, query="anything", session_id="empty")
    assert result["claims"] == []
    assert result["total"] == 0


def test_trace_returns_provenance(conn):
    store_result = handle_memory_store(
        conn, text="I live in Toronto", session_id="s1",
        claims=[{"subject": "user", "predicate": "user_fact", "value": "lives in Toronto", "confidence": 0.9}],
    )
    claim_id = store_result["claim_ids"][0]
    trace = handle_memory_trace(conn, claim_id=claim_id)
    assert "claim" in trace
    assert trace["claim"]["claim_id"] == claim_id
    assert "source_turn" in trace
    assert trace["source_turn"] is not None
    assert "supersession_chain" in trace


def test_trace_nonexistent_claim(conn):
    result = handle_memory_trace(conn, claim_id="cl_nonexistent")
    assert "error" in result


def test_correct_dismiss(conn):
    store_result = handle_memory_store(
        conn, text="I live in Toronto", session_id="s1",
        claims=[{"subject": "user", "predicate": "user_fact", "value": "lives in Toronto", "confidence": 0.9}],
    )
    claim_id = store_result["claim_ids"][0]
    result = handle_memory_correct(conn, claim_id=claim_id, action="dismiss")
    assert result["action"] == "dismissed"
    assert result["claim_id"] == claim_id


def test_correct_correction(conn):
    store_result = handle_memory_store(
        conn, text="I live in Toronto", session_id="s1",
        claims=[{"subject": "user", "predicate": "user_fact", "value": "lives in Toronto", "confidence": 0.9}],
    )
    claim_id = store_result["claim_ids"][0]
    result = handle_memory_correct(conn, claim_id=claim_id, action="correct", new_value="lives in Vancouver")
    assert result["action"] == "corrected"
    assert result["new_claim_id"] is not None
    assert result["new_value"] == "lives in Vancouver"


def test_correct_missing_value(conn):
    store_result = handle_memory_store(
        conn, text="I live in Toronto", session_id="s1",
        claims=[{"subject": "user", "predicate": "user_fact", "value": "lives in Toronto", "confidence": 0.9}],
    )
    claim_id = store_result["claim_ids"][0]
    result = handle_memory_correct(conn, claim_id=claim_id, action="correct")
    assert "error" in result


def test_correct_nonexistent(conn):
    result = handle_memory_correct(conn, claim_id="cl_nonexistent", action="dismiss")
    assert "error" in result


def test_memory_observe(conn):
    from memcontext.mcp_tools import handle_memory_observe

    result = handle_memory_observe(
        conn,
        url="https://example.com/dashboard",
        title="Dashboard",
        accessibility_tree={
            "role": "heading",
            "name": "Project Status",
            "children": [],
        },
        session_id="obs_test",
    )
    assert result["claims_stored"] >= 1
    assert result["turn_id"] is not None
    assert result["session_id"] == "obs_test"


def test_memory_query_serves_episodes_when_session_has_no_facts(
    conn: sqlite3.Connection,
):
    """Served Tier-1 floor: a session with episodes but no facts returns episodes
    (not an empty result), with no structured field required."""
    from memcontext.claims import insert_turn, new_turn_id, now_ns
    from memcontext.schema import Speaker, Turn

    sid = "floor"
    turn = Turn(
        turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER,
        text="the deploy runbook lives in the ops wiki under release",
        ts=now_ns(),
    )
    insert_turn(conn, turn)  # an episode, with NO facts extracted

    result = handle_memory_query(conn, query="where is the deploy runbook", session_id=sid)
    assert result["claims"] == []
    assert result["episodes"], "episodes must be served when no facts exist"
    assert any("deploy runbook" in e["text"] for e in result["episodes"])
    assert result["episodes"][0]["source_type"] == "conversation"


def test_memory_store_defers_extraction_to_queue(conn: sqlite3.Connection):
    """handle_memory_store routes a deferrable extractor through the queue: the
    episode is stored immediately, facts appear only after the queue drains."""
    from memcontext.extraction_queue import InlineQueue
    from memcontext.on_new_turn import ExtractedClaim

    class _FakeDeferrable:
        is_deferrable = True

        def __call__(self, turn):
            return [ExtractedClaim("user", "user_preference", "dark mode", 0.9)]

    ext = _FakeDeferrable()
    q = InlineQueue(conn, extractor=ext)
    handle_memory_store(
        conn, text="I prefer dark mode", session_id="d", extractor=ext, queue=q,
    )
    # Episode stored; facts deferred.
    n0 = conn.execute("SELECT COUNT(*) AS n FROM claims WHERE session_id='d'").fetchone()["n"]
    assert n0 == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM turns WHERE session_id='d'").fetchone()["n"] == 1
    q.drain()
    n1 = conn.execute("SELECT COUNT(*) AS n FROM claims WHERE session_id='d'").fetchone()["n"]
    assert n1 == 1
