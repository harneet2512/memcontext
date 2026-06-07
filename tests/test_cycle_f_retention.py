"""Cycle F (utility-weighted retention): low-utility, old claims are demoted out
of active retrieval (provenance preserved, reversible) — bounding the active set.
"""
from __future__ import annotations

import sqlite3

from memcontext.claims import get_claim
from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.retention import compute_utility, demote_low_utility
from memcontext.retrieval import retrieve_hybrid
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


def test_utility_is_monotonic_in_importance_and_usage():
    conn = _conn()
    _ingest(conn, "alice", "coffee")
    cid = conn.execute("SELECT claim_id FROM claims").fetchone()["claim_id"]
    conn.execute("UPDATE claim_metadata SET importance_score=0.1, access_count=0 WHERE claim_id=?", (cid,))
    low = compute_utility(conn, cid)
    conn.execute("UPDATE claim_metadata SET importance_score=0.95, access_count=20 WHERE claim_id=?", (cid,))
    high = compute_utility(conn, cid)
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert high > low, "utility rises with importance + usage"


def test_demote_removes_from_active_retrieval_but_keeps_traceable():
    conn = _conn()
    _ingest(conn, "alice", "coffee")
    cid = conn.execute("SELECT claim_id FROM claims").fetchone()["claim_id"]
    # make it ancient + low importance/usage -> low utility
    conn.execute("UPDATE claims SET created_ts=1000 WHERE claim_id=?", (cid,))
    conn.execute("UPDATE claim_metadata SET importance_score=0.05, access_count=0 WHERE claim_id=?", (cid,))
    assert compute_utility(conn, cid) < 0.35

    def _retrieved(**kw):
        return any(c.claim_id == cid for c, _ in
                   retrieve_hybrid(conn, session_id="s1", query="coffee", top_k=10, **kw))

    assert _retrieved(), "retrievable before demotion"

    n = demote_low_utility(conn, threshold=0.35, min_age_days=1.0)
    assert n == 1

    assert not _retrieved(), "demoted -> out of active retrieval"
    assert get_claim(conn, cid) is not None, "claim + provenance preserved (not deleted)"
    assert _retrieved(include_demoted=True), "still traceable / reinstatable with include_demoted"


def test_demote_spares_recent_or_high_utility():
    conn = _conn()
    _ingest(conn, "alice", "coffee")  # recent (created now) -> spared by min_age
    cid = conn.execute("SELECT claim_id FROM claims").fetchone()["claim_id"]
    conn.execute("UPDATE claim_metadata SET importance_score=0.01, access_count=0 WHERE claim_id=?", (cid,))
    # low utility but RECENT -> not demoted (age guard)
    assert demote_low_utility(conn, threshold=0.9, min_age_days=30.0) == 0
