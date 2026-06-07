"""Cardinality-aware Pass-1 supersession.

A single-valued (subject, predicate) slot holds one current value, so a new value
supersedes the prior one deterministically — no token overlap required, no LLM,
no manual correction. Multi-valued / undeclared predicates keep the additive
token-overlap gate.
"""
from __future__ import annotations

from memcontext.claims import find_same_identity_claim, get_claim, list_active_claims
from memcontext.mcp_tools import handle_memory_store
from memcontext.predicate_packs import active_pack
from memcontext.schema import open_database


def _store(conn, subject, predicate, value, session="s"):
    # Full sentence so the admission filter (needs >2 content words) admits the turn.
    return handle_memory_store(
        conn,
        text=f"The team decided the {subject} for the project is {value} going forward.",
        session_id=session,
        claims=[{"subject": subject, "predicate": predicate, "value": value, "confidence": 0.9}],
    )


def test_single_valued_supersedes_disjoint_value(monkeypatch):
    """Postgres -> ClickHouse (no shared tokens) supersedes via cardinality."""
    monkeypatch.setenv("ACTIVE_PACK", "developer")
    active_pack.cache_clear()
    try:
        conn = open_database(":memory:")
        r1 = _store(conn, "datastore", "decision_made", "PostgreSQL")
        r2 = _store(conn, "datastore", "decision_made", "ClickHouse")
        head = find_same_identity_claim(
            conn, session_id="s", subject="datastore", predicate="decision_made"
        )
        assert head is not None and head.value == "ClickHouse"
        assert get_claim(conn, r1["claim_ids"][0]).status.value == "superseded"
        assert r2["supersessions"] == 1
    finally:
        active_pack.cache_clear()


def test_single_valued_restatement_is_noop(monkeypatch):
    """Re-stating the same value does not create a spurious supersession."""
    monkeypatch.setenv("ACTIVE_PACK", "developer")
    active_pack.cache_clear()
    try:
        conn = open_database(":memory:")
        _store(conn, "datastore", "decision_made", "ClickHouse")
        r2 = _store(conn, "datastore", "decision_made", "ClickHouse")
        assert r2["supersessions"] == 0
    finally:
        active_pack.cache_clear()


def test_multi_valued_not_clobbered(monkeypatch):
    """A non-single-valued predicate (todo) stays additive."""
    monkeypatch.setenv("ACTIVE_PACK", "developer")
    active_pack.cache_clear()
    try:
        conn = open_database(":memory:")
        _store(conn, "project", "todo", "write tests")
        _store(conn, "project", "todo", "add CI")
        vals = {c.value for c in list_active_claims(conn, "s") if c.predicate == "todo"}
        assert vals == {"write tests", "add CI"}
    finally:
        active_pack.cache_clear()
