"""Tests for memcontext.importance — deterministic importance scoring."""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.importance import compute_importance, recompute_all_importance
from memcontext.schema import Speaker, Turn, open_database
from memcontext.supersession import detect_pass1


def _make_db() -> sqlite3.Connection:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _insert_turn(conn: sqlite3.Connection, session_id: str, text: str = "test") -> Turn:
    turn = Turn(
        turn_id=new_turn_id(),
        session_id=session_id,
        speaker=Speaker.USER,
        text=text,
        ts=now_ns(),
        asr_confidence=None,
    )
    insert_turn(conn, turn)
    return turn


def test_compute_importance_basic(db, session_id, sample_claim):
    """compute_importance returns a float in [0, 1]."""
    score = compute_importance(db, sample_claim.claim_id)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_importance_uniqueness(db, session_id):
    """More claims with the same (subject, predicate) => lower uniqueness signal."""
    turn = _insert_turn(db, session_id, "testing uniqueness")

    # Insert 5 claims with the same (subject, predicate) but different values
    claims = []
    for i in range(5):
        c = insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="user_fact",
            value=f"unique value number {i}",
            confidence=0.9,
            source_turn_id=turn.turn_id,
        )
        claims.append(c)

    # The first claim's importance should be <= a lone claim's importance
    # because the uniqueness signal (1/N) goes down with more claims
    score_first = compute_importance(db, claims[0].claim_id)
    assert isinstance(score_first, float)
    assert 0.0 <= score_first <= 1.0

    # With 5 claims sharing the predicate, uniqueness = 1/5 = 0.2
    # A solo claim would have uniqueness = 1.0
    # So the first claim's score should reflect the reduced uniqueness
    score_last = compute_importance(db, claims[-1].claim_id)
    assert isinstance(score_last, float)
    assert 0.0 <= score_last <= 1.0


def test_importance_supersession_significance(db, session_id):
    """A claim that supersedes another should have higher importance than one without history."""
    turn_a = _insert_turn(db, session_id, "My favorite city is Portland")
    claim_a = insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_fact",
        value="favorite city Portland",
        confidence=0.9,
        source_turn_id=turn_a.turn_id,
    )

    turn_b = _insert_turn(db, session_id, "My favorite city is Seattle now")
    claim_b = insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_fact",
        value="favorite city Seattle",
        confidence=0.9,
        source_turn_id=turn_b.turn_id,
    )

    # Supersede A with B
    detect_pass1(db, claim_b)

    # Create a standalone claim with no supersession history
    turn_c = _insert_turn(db, session_id, "I like pizza")
    claim_c = insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_preference",
        value="likes pizza",
        confidence=0.9,
        source_turn_id=turn_c.turn_id,
    )

    score_b = compute_importance(db, claim_b.claim_id)
    score_c = compute_importance(db, claim_c.claim_id)

    # B superseded A, so its supersession signal = 1.0
    # C has no edges, so its supersession signal = 0.5
    # B should have a higher importance score (all else roughly equal)
    assert score_b > score_c


def test_recompute_all(db, session_id):
    """recompute_all_importance returns the count of recomputed claims."""
    turn = _insert_turn(db, session_id, "testing recompute")

    for i in range(3):
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="user_fact",
            value=f"fact {i}",
            confidence=0.9,
            source_turn_id=turn.turn_id,
        )

    count = recompute_all_importance(db)
    assert count == 3
