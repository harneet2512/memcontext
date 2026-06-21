"""event_ts EVENT-GUARD for Pass-1 supersession.

A new claim must NOT supersede an old same-(subject, predicate) candidate when
BOTH carry an explicit `event_ts` and the two timestamps DIFFER — those are two
distinct dated occurrences and superseding one with the other deletes valid
history. With no `event_ts`, or equal `event_ts`, behavior is unchanged.

All tests use `:memory:` SQLite + the default NullEmbedder path (Pass-1 is purely
structural, so no embedding model is loaded — zero downloads in CI).
"""
from __future__ import annotations

import sqlite3

import pytest

from memcontext.claims import get_claim, insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.predicate_packs import active_pack
from memcontext.schema import ClaimStatus, Speaker, Turn
from memcontext.supersession import _event_blocks, detect_pass1


def _make_turn(db: sqlite3.Connection, session_id: str, text: str) -> Turn:
    turn = Turn(
        turn_id=new_turn_id(),
        session_id=session_id,
        speaker=Speaker.USER,
        text=text,
        ts=now_ns(),
        asr_confidence=None,
    )
    insert_turn(db, turn)
    return turn


# --- predicate unit tests (no DB) --------------------------------------------

class _Stub:
    def __init__(self, event_ts):
        self.event_ts = event_ts


def test_event_blocks_both_dated_differ():
    assert _event_blocks(_Stub(100), _Stub(200)) is True


def test_event_blocks_both_dated_equal():
    assert _event_blocks(_Stub(100), _Stub(100)) is False


def test_event_blocks_one_undated():
    assert _event_blocks(_Stub(None), _Stub(200)) is False
    assert _event_blocks(_Stub(100), _Stub(None)) is False


def test_event_blocks_both_undated():
    assert _event_blocks(_Stub(None), _Stub(None)) is False


# --- (i) distinct event_ts => both stay active -------------------------------

def test_distinct_event_ts_keeps_both_active(
    db: sqlite3.Connection, session_id: str,
):
    """Two same-(subject, predicate) claims with DIFFERENT event_ts are distinct
    dated events: NO supersession, both remain active. (temporal-preservation)."""
    t1 = _make_turn(db, session_id, "On Jan 5 I ran a 5K")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="ran a 5K race today morning", confidence=0.9,
        source_turn_id=t1.turn_id, event_ts=1_000,
    )
    t2 = _make_turn(db, session_id, "On Feb 9 I ran a 5K again")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="ran a 5K race today evening", confidence=0.9,
        source_turn_id=t2.turn_id, event_ts=2_000,
    )

    edge = detect_pass1(db, c2)
    assert edge is None
    assert get_claim(db, c1.claim_id).status == ClaimStatus.ACTIVE
    assert get_claim(db, c2.claim_id).status == ClaimStatus.ACTIVE


def test_attribute_slot_distinct_event_ts_keeps_both(
    db: sqlite3.Connection, session_id: str,
):
    """Attribute-slot path: two DATED relocation events stay distinct."""
    t1 = _make_turn(db, session_id, "In 2019 I lived in NYC")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="lives in NYC", confidence=0.9,
        source_turn_id=t1.turn_id, event_ts=1_000,
    )
    t2 = _make_turn(db, session_id, "In 2023 I moved to Boston")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="moved to Boston", confidence=0.9,
        source_turn_id=t2.turn_id, event_ts=2_000,
    )

    edge = detect_pass1(db, c2)
    assert edge is None
    assert get_claim(db, c1.claim_id).status == ClaimStatus.ACTIVE


# --- (ii) no/equal event_ts => existing behavior preserved -------------------

def test_no_event_ts_still_supersedes(
    db: sqlite3.Connection, session_id: str,
):
    """Undated state supersession is unchanged (jaccard path, regression lock)."""
    t1 = _make_turn(db, session_id, "I take medication in the morning")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="takes medication daily morning", confidence=0.9,
        source_turn_id=t1.turn_id,
    )
    t2 = _make_turn(db, session_id, "Now I take it at night")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="takes medication daily night", confidence=0.9,
        source_turn_id=t2.turn_id,
    )

    edge = detect_pass1(db, c2)
    assert edge is not None
    assert edge.old_claim_id == c1.claim_id
    assert get_claim(db, c1.claim_id).status == ClaimStatus.SUPERSEDED


def test_equal_event_ts_still_supersedes(
    db: sqlite3.Connection, session_id: str,
):
    """Equal event_ts is NOT a distinct event => supersession unchanged."""
    t1 = _make_turn(db, session_id, "I take medication in the morning")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="takes medication daily morning", confidence=0.9,
        source_turn_id=t1.turn_id, event_ts=5_000,
    )
    t2 = _make_turn(db, session_id, "Now I take it at night")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="takes medication daily night", confidence=0.9,
        source_turn_id=t2.turn_id, event_ts=5_000,
    )

    edge = detect_pass1(db, c2)
    assert edge is not None
    assert get_claim(db, c1.claim_id).status == ClaimStatus.SUPERSEDED


def test_mixed_one_dated_one_undated_still_supersedes(
    db: sqlite3.Connection, session_id: str,
):
    """Guard fires ONLY when BOTH dated: a new dated claim vs an old undated one
    still supersedes (documented semantics)."""
    t1 = _make_turn(db, session_id, "I take medication in the morning")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="takes medication daily morning", confidence=0.9,
        source_turn_id=t1.turn_id,  # undated
    )
    t2 = _make_turn(db, session_id, "Now I take it at night")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="takes medication daily night", confidence=0.9,
        source_turn_id=t2.turn_id, event_ts=2_000,
    )

    edge = detect_pass1(db, c2)
    assert edge is not None
    assert get_claim(db, c1.claim_id).status == ClaimStatus.SUPERSEDED


# --- (iii) single_valued update still supersedes -----------------------------

def test_single_valued_update_still_supersedes(
    db: sqlite3.Connection, session_id: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """single_valued cardinality path (developer pack `decision_made`): an undated
    new value supersedes the prior one — regression lock on existing behavior."""
    monkeypatch.setenv("ACTIVE_PACK", "developer")
    active_pack.cache_clear()
    assert "decision_made" in active_pack().single_valued

    t1 = _make_turn(db, session_id, "We will use Postgres")
    c1 = insert_claim(
        db, session_id=session_id, subject="team", predicate="decision_made",
        value="use Postgres", confidence=0.9, source_turn_id=t1.turn_id,
    )
    t2 = _make_turn(db, session_id, "Switching to DynamoDB")
    c2 = insert_claim(
        db, session_id=session_id, subject="team", predicate="decision_made",
        value="use DynamoDB", confidence=0.9, source_turn_id=t2.turn_id,
    )

    edge = detect_pass1(db, c2)
    assert edge is not None
    assert edge.old_claim_id == c1.claim_id
    assert get_claim(db, c1.claim_id).status == ClaimStatus.SUPERSEDED


def test_single_valued_distinct_event_ts_keeps_both(
    db: sqlite3.Connection, session_id: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """single_valued path also honours the guard: two DATED decisions stay distinct."""
    monkeypatch.setenv("ACTIVE_PACK", "developer")
    active_pack.cache_clear()

    t1 = _make_turn(db, session_id, "Sprint 1: use Postgres")
    c1 = insert_claim(
        db, session_id=session_id, subject="team", predicate="decision_made",
        value="use Postgres", confidence=0.9, source_turn_id=t1.turn_id,
        event_ts=1_000,
    )
    t2 = _make_turn(db, session_id, "Sprint 9: use DynamoDB")
    c2 = insert_claim(
        db, session_id=session_id, subject="team", predicate="decision_made",
        value="use DynamoDB", confidence=0.9, source_turn_id=t2.turn_id,
        event_ts=2_000,
    )

    edge = detect_pass1(db, c2)
    assert edge is None
    assert get_claim(db, c1.claim_id).status == ClaimStatus.ACTIVE


# --- determinism / idempotency ----------------------------------------------

def test_guard_idempotent_no_double_write(
    db: sqlite3.Connection, session_id: str,
):
    """Re-running detect_pass1 on a guarded pair never writes an edge."""
    t1 = _make_turn(db, session_id, "On Jan 5 I ran a 5K")
    insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="ran a 5K race today morning", confidence=0.9,
        source_turn_id=t1.turn_id, event_ts=1_000,
    )
    t2 = _make_turn(db, session_id, "On Feb 9 I ran a 5K again")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="ran a 5K race today evening", confidence=0.9,
        source_turn_id=t2.turn_id, event_ts=2_000,
    )

    assert detect_pass1(db, c2) is None
    assert detect_pass1(db, c2) is None
    n = db.execute("SELECT COUNT(*) FROM supersession_edges").fetchone()[0]
    assert n == 0
