"""Cycle C (episodic -> semantic consolidation): a fact recurring across sessions
graduates into one durable consolidated fact with full provenance; a contested
(multi-valued) slot does not.
"""
from __future__ import annotations

import json
import sqlite3

from memcontext.claims import get_claim
from memcontext.consolidate import consolidate_facts
from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.schema import Speaker, open_database


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _say(conn, session, subject, value):
    on_new_turn(
        conn, session_id=session, speaker=Speaker.USER, text=f"{subject} likes {value}",
        extractor=PassthroughExtractor(
            [{"subject": subject, "predicate": "user_fact", "value": value, "confidence": 0.9}]),
    )


def test_recurring_fact_graduates_with_provenance():
    conn = _conn()
    for s in ("s1", "s2", "s3"):
        _say(conn, s, "user", "coffee")
    active = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE status IN ('active','confirmed','audited')"
    ).fetchone()[0]
    assert active == 3, f"expected 3 cross-session actives, got {active}"

    n = consolidate_facts(conn, min_sessions=3)
    assert n == 1

    canon = conn.execute(
        "SELECT claim_id, consolidated_sources, importance_score"
        " FROM claim_metadata WHERE consolidated=1"
    ).fetchone()
    assert canon is not None
    sources = json.loads(canon["consolidated_sources"])
    assert len(sources) == 3, "provenance to all source claims"
    assert canon["importance_score"] >= 0.9, "consolidated fact is durable (high importance)"
    # the 2 redundant duplicates leave active retrieval
    assert conn.execute("SELECT COUNT(*) FROM claim_metadata WHERE demoted=1").fetchone()[0] == 2
    # every source still exists (never deleted)
    assert all(get_claim(conn, cid) is not None for cid in sources)


def test_contested_slot_not_consolidated():
    conn = _conn()
    _say(conn, "s1", "user", "berlin")
    _say(conn, "s2", "user", "berlin")
    _say(conn, "s3", "user", "munich")  # different value -> contested slot
    assert consolidate_facts(conn, min_sessions=2) == 0
