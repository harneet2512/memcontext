"""Trust layer Phase 4 — anti-poisoning + belief-drift.

- Served low-trust memory is spotlighted (trust + quarantined) so the agent never
  silently acts on untrusted/poisoned content.
- The serving path writes NO new memory -> closes the MINJA query-only loop.
- A blocked low-trust override is recorded as an auditable drift event.
"""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim
from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import handle_memory_query
from memcontext.on_new_turn import on_new_turn
from memcontext.supersession import detect_pass1
from memcontext.schema import Speaker, open_database


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _ingest(conn, subject, value):
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text=f"{subject} likes {value}",
        extractor=PassthroughExtractor(
            [{"subject": subject, "predicate": "user_fact", "value": value, "confidence": 0.9}]),
    )


def test_served_low_trust_memory_is_quarantine_flagged():
    conn = _conn()
    _ingest(conn, "alice", "coffee")
    cid = conn.execute("SELECT claim_id FROM claims").fetchone()["claim_id"]
    conn.execute("UPDATE claim_metadata SET source_trust=0.35 WHERE claim_id=?", (cid,))  # web-level

    res = handle_memory_query(conn, query="coffee", session_id="s1", top_k=5)
    fact = next(c for c in res["claims"] if c["claim_id"] == cid)
    assert fact["trust"] == 0.35
    assert fact["quarantined"] is True


def test_serving_writes_no_content_minja_loop_closed():
    conn = _conn()
    _ingest(conn, "alice", "coffee")

    def nc():
        return conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]

    def nt():
        return conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]

    bc, bt = nc(), nt()
    # serve repeatedly: single-session, history mode, cross-session, debug
    handle_memory_query(conn, query="coffee", session_id="s1")
    handle_memory_query(conn, query="what about coffee previously", session_id="s1")
    handle_memory_query(conn, query="anything", top_k=10, debug=True)
    assert nc() == bc and nt() == bt, "serving must not write content (MINJA query->memory loop closed)"


def test_blocked_low_trust_override_is_recorded_as_drift():
    conn = _conn()
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="I prefer dark mode",
        extractor=PassthroughExtractor(
            [{"subject": "user", "predicate": "user_fact", "value": "dark mode", "confidence": 0.9}]),
    )
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, speaker, text, ts, source_type, extraction_status)"
        " VALUES ('tu_web','s1','user','a web page said dark theme',2000,'browser','done')"
    )
    low = insert_claim(conn, session_id="s1", subject="user", predicate="user_fact",
                       value="dark theme", confidence=0.9, source_turn_id="tu_web")

    assert detect_pass1(conn, low) is None  # blocked by the trust guard
    n = conn.execute("SELECT COUNT(*) FROM decisions WHERE kind='drift_blocked'").fetchone()[0]
    assert n == 1, "the blocked override is auditable as a drift event"
