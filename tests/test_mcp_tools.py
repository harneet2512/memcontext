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
