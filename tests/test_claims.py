from __future__ import annotations

import sqlite3

import pytest

from memcontext.claims import (
    ClaimValidationError,
    _normalise_subject,
    _temporal_bin,
    find_same_identity_claim,
    get_claim,
    insert_claim,
    insert_turn,
    list_active_claims,
    new_turn_id,
    now_ns,
    set_claim_status,
)
from memcontext.schema import Claim, ClaimStatus, Speaker, Turn


def test_insert_and_get_claim(db: sqlite3.Connection, session_id: str, sample_turn: Turn):
    claim = insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_fact",
        value="lives in Toronto",
        confidence=0.85,
        source_turn_id=sample_turn.turn_id,
    )
    assert claim.claim_id.startswith("cl_")
    assert claim.subject == "user"
    assert claim.predicate == "user_fact"
    assert claim.value == "lives in Toronto"
    assert claim.status == ClaimStatus.ACTIVE

    retrieved = get_claim(db, claim.claim_id)
    assert retrieved is not None
    assert retrieved.claim_id == claim.claim_id
    assert retrieved.value == "lives in Toronto"


def test_insert_claim_validates_predicate(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    with pytest.raises(ClaimValidationError, match="predicate.*not in allowed"):
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="invalid_predicate_xyz",
            value="something",
            confidence=0.5,
            source_turn_id=sample_turn.turn_id,
        )


def test_insert_claim_validates_confidence_bounds(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    with pytest.raises(ClaimValidationError, match="confidence"):
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="user_fact",
            value="something",
            confidence=1.5,
            source_turn_id=sample_turn.turn_id,
        )
    with pytest.raises(ClaimValidationError, match="confidence"):
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="user_fact",
            value="something",
            confidence=-0.1,
            source_turn_id=sample_turn.turn_id,
        )


def test_insert_claim_validates_source_turn(db: sqlite3.Connection, session_id: str):
    with pytest.raises(ClaimValidationError, match="does not reference any turn"):
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="user_fact",
            value="something",
            confidence=0.5,
            source_turn_id="nonexistent_turn_id",
        )


def test_insert_claim_validates_span_consistency(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    with pytest.raises(ClaimValidationError, match="both be set or both be None"):
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="user_fact",
            value="something",
            confidence=0.5,
            source_turn_id=sample_turn.turn_id,
            char_start=0,
            char_end=None,
        )


def test_now_ns_monotonic():
    values = [now_ns() for _ in range(1000)]
    for i in range(1, len(values)):
        assert values[i] > values[i - 1]


def test_list_active_claims(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    c1 = insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_fact",
        value="fact1",
        confidence=0.9,
        source_turn_id=sample_turn.turn_id,
    )
    c2 = insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_preference",
        value="pref1",
        confidence=0.8,
        source_turn_id=sample_turn.turn_id,
    )
    set_claim_status(db, c1.claim_id, ClaimStatus.SUPERSEDED)

    active = list_active_claims(db, session_id)
    active_ids = {c.claim_id for c in active}
    assert c2.claim_id in active_ids
    assert c1.claim_id not in active_ids


def test_set_claim_status(db: sqlite3.Connection, sample_claim: Claim):
    set_claim_status(db, sample_claim.claim_id, ClaimStatus.CONFIRMED)
    updated = get_claim(db, sample_claim.claim_id)
    assert updated is not None
    assert updated.status == ClaimStatus.CONFIRMED


def test_normalise_subject():
    assert _normalise_subject("  John  Doe ") == "john_doe"
    assert _normalise_subject("Alice") == "alice"
    assert _normalise_subject("Multi   Space   Name") == "multi_space_name"


def test_temporal_bin():
    import time
    ts = int(time.mktime((2025, 3, 15, 0, 0, 0, 0, 0, 0)) * 1e9)
    result = _temporal_bin(ts)
    assert result == "2025-Q1"
    assert _temporal_bin(None) is None


def test_find_same_identity_claim(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    c1 = insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_preference",
        value="prefers dark mode",
        confidence=0.9,
        source_turn_id=sample_turn.turn_id,
    )
    found = find_same_identity_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
    )
    assert found is not None
    assert found.claim_id == c1.claim_id

    not_found = find_same_identity_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_preference",
        exclude_claim_ids=[c1.claim_id],
    )
    assert not_found is None


def test_insert_claim_empty_subject(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    with pytest.raises(ClaimValidationError, match="subject must be non-empty"):
        insert_claim(
            db,
            session_id=session_id,
            subject="",
            predicate="user_fact",
            value="something",
            confidence=0.5,
            source_turn_id=sample_turn.turn_id,
        )


def test_insert_claim_empty_value(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    with pytest.raises(ClaimValidationError, match="value must be non-empty"):
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="user_fact",
            value="   ",
            confidence=0.5,
            source_turn_id=sample_turn.turn_id,
        )
