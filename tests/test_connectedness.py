"""Connectedness: each substrate capability reaches a production door, and the
previously write-only columns are now read. Guards against dormant surface.
"""
from __future__ import annotations

import sqlite3
import time

from memcontext.claims import insert_claim
from memcontext.consolidate import consolidate_facts
from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import handle_memory_query
from memcontext.on_new_turn import on_new_turn
from memcontext.retention import compute_utility
from memcontext.retrieval import retrieve_memory
from memcontext.schema import ClaimStatus, Speaker, open_database


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _say(conn, value, session="s1"):
    on_new_turn(
        conn, session_id=session, speaker=Speaker.USER, text=f"user likes {value}",
        extractor=PassthroughExtractor(
            [{"subject": "user", "predicate": "user_fact", "value": value, "confidence": 0.9}]),
    )


def test_history_mode_is_universal_at_retrieve_memory_layer():
    """retrieve_memory itself auto-detects past-intent, so cli query (and any
    direct caller) gets temporal truth -- not just the MCP door."""
    conn = _conn()
    _say(conn, "munich")
    turn = conn.execute("SELECT source_turn_id FROM claims LIMIT 1").fetchone()["source_turn_id"]
    insert_claim(
        conn, session_id="s1", subject="user", predicate="user_fact",
        value="berlin", confidence=0.9, source_turn_id=turn, status=ClaimStatus.SUPERSEDED)
    bcid = conn.execute("SELECT claim_id FROM claims WHERE value='berlin'").fetchone()["claim_id"]

    ids = {h.id for h, _ in retrieve_memory(conn, session_id="s1", query="tell me about berlin")}
    assert bcid not in ids, "superseded excluded by default"
    ids_h = {h.id for h, _ in retrieve_memory(conn, session_id="s1", query="what about berlin previously")}
    assert bcid in ids_h, "history mode is universal (no MCP door involved)"


def test_last_accessed_ts_feeds_utility():
    """compute_utility now reads last_accessed_ts (was write-only)."""
    conn = _conn()
    _say(conn, "coffee")
    cid = conn.execute("SELECT claim_id FROM claims").fetchone()["claim_id"]
    conn.execute("UPDATE claims SET created_ts=1000 WHERE claim_id=?", (cid,))  # ancient
    conn.execute("UPDATE claim_metadata SET importance_score=0.5, access_count=1 WHERE claim_id=?", (cid,))
    u_stale = compute_utility(conn, cid)
    conn.execute("UPDATE claim_metadata SET last_accessed_ts=? WHERE claim_id=?", (time.time_ns(), cid))
    u_fresh = compute_utility(conn, cid)
    assert u_fresh > u_stale, "a recently-accessed claim is fresher -> higher utility"


def test_door_surfaces_consolidated_marker():
    """handle_memory_query exposes the consolidation marker on served facts."""
    conn = _conn()
    for s in ("s1", "s2", "s3"):
        _say(conn, "coffee", session=s)
    assert consolidate_facts(conn, min_sessions=3) == 1
    res = handle_memory_query(conn, query="coffee", top_k=10)  # cross-session door
    assert res["claims"], "the durable consolidated fact is served"
    assert any(c.get("consolidated") for c in res["claims"]), "consolidation marker surfaced"
