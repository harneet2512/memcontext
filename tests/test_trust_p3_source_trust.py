"""Trust layer Phase 3 — source-trust tiering (provenance-aware trust scoring).

Memory is ranked by WHERE it came from (user > tool > browser > inferred), and a
markedly lower-trust claim cannot silently supersede a higher-trust one — the
named memory-poisoning defense applied to the substrate.
"""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim
from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.retrieval import retrieve_hybrid
from memcontext.source_trust import (
    EXTERNAL_WEB,
    TRUSTED_USER,
    trust_for_source,
)
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


def test_trust_for_source_tiers():
    assert trust_for_source("conversation", "user") == TRUSTED_USER
    assert trust_for_source("browser", "user") == EXTERNAL_WEB
    assert trust_for_source("tool_call", "user") == 0.7
    assert trust_for_source("conversation", "assistant") == 0.5


def test_source_trust_written_at_ingest():
    conn = _conn()
    _ingest(conn, "user", "coffee")  # a user conversation turn -> fully trusted
    t = conn.execute("SELECT source_trust FROM claim_metadata").fetchone()[0]
    assert t == TRUSTED_USER


def test_trusted_memory_outranks_untrusted():
    conn = _conn()
    _ingest(conn, "alice", "coffee")
    _ingest(conn, "bob", "coffee")
    rows = {r["subject"]: r["claim_id"] for r in conn.execute("SELECT subject, claim_id FROM claims").fetchall()}
    a, b = rows["alice"], rows["bob"]
    conn.execute("UPDATE claims SET created_ts=1000, valid_from_ts=1000")
    conn.execute("UPDATE claim_metadata SET source_trust=0.95 WHERE claim_id=?", (a,))
    conn.execute("UPDATE claim_metadata SET source_trust=0.10 WHERE claim_id=?", (b,))

    ex: dict[str, dict[str, float]] = {}
    hits = retrieve_hybrid(conn, session_id="s1", query="coffee", top_k=10, explain=ex)
    assert ex[a]["source_trust"] > ex[b]["source_trust"]
    ids = [c.claim_id for c, _ in hits]
    assert ids.index(a) < ids.index(b)  # higher trust ranks first, all else equal


def test_low_trust_cannot_supersede_high_trust():
    conn = _conn()
    # a high-trust fact the user stated
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="I prefer dark mode",
        extractor=PassthroughExtractor(
            [{"subject": "user", "predicate": "user_fact", "value": "dark mode", "confidence": 0.9}]),
    )
    high = conn.execute("SELECT claim_id, status FROM claims WHERE value='dark mode'").fetchone()["claim_id"]

    # a low-trust value scraped from a browsed page (different source turn, overlapping value)
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, speaker, text, ts, source_type, extraction_status)"
        " VALUES ('tu_web','s1','user','a web page said dark theme',2000,'browser','done')"
    )
    low = insert_claim(conn, session_id="s1", subject="user", predicate="user_fact",
                       value="dark theme", confidence=0.9, source_turn_id="tu_web")
    # trust was derived from the browser source at insert
    assert conn.execute(
        "SELECT source_trust FROM claim_metadata WHERE claim_id=?", (low.claim_id,)
    ).fetchone()[0] == EXTERNAL_WEB

    # Pass-1 finds the match but the guard blocks the low-trust override
    assert detect_pass1(conn, low) is None
    assert conn.execute("SELECT status FROM claims WHERE claim_id=?", (high,)).fetchone()[0] in (
        "active", "confirmed", "audited")
