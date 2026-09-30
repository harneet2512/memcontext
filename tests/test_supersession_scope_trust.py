"""Pass-1 supersession: project (namespace) scope, source-trust direction, ordering.

Regressions covered:
- Supersession compared only claims in the same session_id. Every Claude Code
  session has a new id, so a decision changed in a later session never retired
  the old one (both stayed "current"), while two tenants sharing a session_id
  could supersede each other's facts.
- A lower-trust ASSISTANT claim silently superseded a USER-stated fact (the trust
  guard skipped ASSISTANT_CONFIRM edges), and a USER correcting an assistant claim
  produced a status-neutral CONTRADICTS edge, leaving the stale value active.
- Two near-concurrent inserts could each treat the other as "prior" (the candidate
  query had no created_ts bound), leaving the OLDEST value as the only active one.
"""
from __future__ import annotations

import sqlite3

import pytest

from memcontext.claims import get_claim, insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.predicate_packs import active_pack
from memcontext.schema import ClaimStatus, EdgeType, Speaker, Turn
from memcontext.supersession import detect_pass1


@pytest.fixture(autouse=True)
def developer_pack(monkeypatch):
    monkeypatch.setenv("ACTIVE_PACK", "general,developer")
    active_pack.cache_clear()
    yield
    active_pack.cache_clear()


def _turn(db, session_id, speaker=Speaker.USER, text="x", namespace="default") -> Turn:
    t = Turn(turn_id=new_turn_id(), session_id=session_id, speaker=speaker, text=text, ts=now_ns())
    insert_turn(db, t, namespace=namespace)
    return t


def _claim(db, session_id, value, *, speaker=Speaker.USER, namespace="default",
           subject="orders service", predicate="decision_made", detect=True):
    t = _turn(db, session_id, speaker, f"The {subject} will {value}.", namespace)
    c = insert_claim(db, session_id=session_id, subject=subject, predicate=predicate,
                     value=value, confidence=0.9, source_turn_id=t.turn_id)
    edge = detect_pass1(db, c) if detect else None
    return c, edge


def _status(db, claim) -> ClaimStatus:
    return get_claim(db, claim.claim_id).status


# ── scope: namespace (project), not session ─────────────────────────────────

def test_decision_changed_in_a_later_session_supersedes(db: sqlite3.Connection):
    old, _ = _claim(db, "claude-session-1", "use PostgreSQL")
    new, edge = _claim(db, "claude-session-2", "use SQLite")
    assert edge is not None and edge.old_claim_id == old.claim_id
    assert _status(db, old) is ClaimStatus.SUPERSEDED
    assert _status(db, new) is ClaimStatus.ACTIVE


def test_same_session_id_in_another_namespace_never_supersedes(db: sqlite3.Connection):
    a, _ = _claim(db, "shared", "use PostgreSQL", namespace="tenantA")
    _, edge = _claim(db, "shared", "use SQLite", namespace="tenantB")
    assert edge is None
    assert _status(db, a) is ClaimStatus.ACTIVE


# ── trust direction ─────────────────────────────────────────────────────────

def test_assistant_claim_cannot_override_a_user_fact(db: sqlite3.Connection):
    user, _ = _claim(db, "s1", "use PostgreSQL")
    assistant, edge = _claim(db, "s1", "use SQLite", speaker=Speaker.ASSISTANT)
    # The user's fact stays current; the disagreement is surfaced, not applied.
    assert edge is not None and edge.edge_type is EdgeType.CONTRADICTS
    assert _status(db, user) is ClaimStatus.ACTIVE
    assert _status(db, assistant) is ClaimStatus.ACTIVE
    blocked = db.execute("SELECT COUNT(*) FROM decisions WHERE kind = 'drift_blocked'").fetchone()[0]
    assert blocked == 1


def test_user_correction_retires_an_assistant_claim(db: sqlite3.Connection):
    assistant, _ = _claim(db, "s1", "use PostgreSQL", speaker=Speaker.ASSISTANT)
    _, edge = _claim(db, "s1", "use SQLite")
    assert edge is not None and edge.edge_type is EdgeType.USER_CORRECTION
    assert _status(db, assistant) is ClaimStatus.SUPERSEDED


def test_assistant_can_revise_its_own_claim(db: sqlite3.Connection):
    first, _ = _claim(db, "s1", "use PostgreSQL", speaker=Speaker.ASSISTANT)
    _, edge = _claim(db, "s1", "use SQLite", speaker=Speaker.ASSISTANT)
    assert edge is not None and edge.edge_type is EdgeType.ASSISTANT_CONFIRM
    assert _status(db, first) is ClaimStatus.SUPERSEDED


# ── ordering ────────────────────────────────────────────────────────────────

def test_an_older_claim_never_supersedes_a_newer_one(db: sqlite3.Connection):
    # Both inserted before either runs detection (near-concurrent writers).
    older, _ = _claim(db, "s1", "use PostgreSQL", detect=False)
    newer, _ = _claim(db, "s1", "use SQLite", detect=False)

    assert detect_pass1(db, older) is None  # must not treat the newer claim as prior
    assert detect_pass1(db, newer) is not None
    assert _status(db, older) is ClaimStatus.SUPERSEDED
    assert _status(db, newer) is ClaimStatus.ACTIVE


# ── Pass 2 shares the scope and ordering rules ──────────────────────────────

class _SameVector:
    """Every text embeds identically: cosine 1.0, so only scope/order decide."""

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _semantic(db, claim):
    from memcontext.supersession_semantic import SemanticSupersession

    return SemanticSupersession(_SameVector()).detect(db, claim)


def test_pass2_structured_scope_is_the_namespace(db: sqlite3.Connection):
    old, _ = _claim(db, "s1", "use PostgreSQL", subject="db", predicate="context", detect=False)
    other_ns, _ = _claim(db, "s2", "use MySQL", subject="db", predicate="context",
                         namespace="tenantB", detect=False)
    new, _ = _claim(db, "s2", "use SQLite", subject="db", predicate="context", detect=False)
    edge = _semantic(db, new)
    assert edge is not None and edge.old_claim_id == old.claim_id
    assert _status(db, other_ns) is ClaimStatus.ACTIVE


def test_pass2_older_claim_never_replaces_a_newer_one(db: sqlite3.Connection):
    older, _ = _claim(db, "s1", "use PostgreSQL", subject="db", predicate="context", detect=False)
    newer, _ = _claim(db, "s1", "use SQLite", subject="db", predicate="context", detect=False)
    assert _semantic(db, older) is None
    assert _status(db, newer) is ClaimStatus.ACTIVE


# ── duplicate copies of the replaced value ──────────────────────────────────

def test_restating_the_same_fact_keeps_both_copies(db: sqlite3.Connection):
    # Repeats are recurrence evidence (consolidation counts them) — not collapsed.
    first, _ = _claim(db, "s1", "use SQLite")
    second, edge = _claim(db, "s2", "use SQLite")
    assert edge is None
    assert _status(db, first) is ClaimStatus.ACTIVE and _status(db, second) is ClaimStatus.ACTIVE


def test_a_change_retires_every_copy_of_the_old_value(db: sqlite3.Connection):
    first, _ = _claim(db, "s1", "use SQLite")
    second, _ = _claim(db, "s2", "use SQLite")
    changed, _ = _claim(db, "s3", "use PostgreSQL")
    assert _status(db, first) is ClaimStatus.SUPERSEDED
    assert _status(db, second) is ClaimStatus.SUPERSEDED
    assert _status(db, changed) is ClaimStatus.ACTIVE
    edges = db.execute(
        "SELECT COUNT(*) FROM supersession_edges WHERE new_claim_id = ?", (changed.claim_id,)
    ).fetchone()[0]
    assert edges == 2  # each retired copy stays traceable to the change


def test_lower_trust_change_retires_no_copy(db: sqlite3.Connection):
    first, _ = _claim(db, "s1", "use SQLite")
    second, _ = _claim(db, "s2", "use SQLite")
    _claim(db, "s3", "use PostgreSQL", speaker=Speaker.ASSISTANT)
    assert _status(db, first) is ClaimStatus.ACTIVE and _status(db, second) is ClaimStatus.ACTIVE
