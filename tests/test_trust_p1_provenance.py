"""Trust layer Phase 1 — provenance completeness.

Closes the audit's one blind spot: session_digests now carries a queryable
source_claim_ids column (was only buried in digest_data JSON), so every served
summary is traceable to its source claims (and cascade-deletable in Phase 2).
"""
from __future__ import annotations

import json
import sqlite3

from memcontext.claims import get_claim
from memcontext.digests import build_session_digest, store_digest
from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.schema import SCHEMA_VERSION, Speaker, open_database


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


def test_v8_session_digests_has_source_claim_ids():
    conn = _conn()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(session_digests)").fetchall()}
    assert "source_claim_ids" in cols
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_digest_carries_traceable_source_claim_ids():
    conn = _conn()
    _ingest(conn, "alice", "coffee")
    _ingest(conn, "bob", "tea")
    store_digest(conn, build_session_digest(conn, "s1"))

    row = conn.execute(
        "SELECT source_claim_ids FROM session_digests WHERE session_id='s1'"
    ).fetchone()
    assert row["source_claim_ids"], "digest links to its source claims (structural column)"
    cids = set(json.loads(row["source_claim_ids"]))
    assert cids
    # every linked claim genuinely exists — provenance is real, not decorative
    assert all(get_claim(conn, cid) is not None for cid in cids)


def test_every_served_summary_table_can_name_its_source_claims():
    """Provenance invariant: no served/derived summary table lacks a claim link."""
    conn = _conn()

    def cols(t):
        return {r["name"] for r in conn.execute(f"PRAGMA table_info({t})").fetchall()}

    assert "source_claim_ids" in cols("session_digests"), "digests traceable (v8 closes the gap)"
    assert "claim_ids" in cols("life_events")
    assert "source_claim_ids" in cols("output_sentences")
    assert {"event_id", "claim_id"} <= cols("event_frame_claims")
