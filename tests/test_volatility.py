"""Tests for memcontext.volatility — deterministic volatility classification."""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.schema import Speaker, Turn, open_database
from memcontext.supersession import detect_pass1
from memcontext.volatility import VolatilityInfo, classify_predicate


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


def test_classify_stable(db, session_id):
    """A single claim with no supersession history is classified as 'stable'."""
    turn = _insert_turn(db, session_id, "My name is Alice")
    insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_fact",
        value="name is Alice",
        confidence=0.9,
        source_turn_id=turn.turn_id,
    )

    result = classify_predicate(db, "user", "user_fact")
    assert isinstance(result, VolatilityInfo)
    assert result.classification == "stable"


def test_classify_evolving(db, session_id):
    """One supersession event makes the predicate 'evolving'."""
    turn_a = _insert_turn(db, session_id, "My favorite city to live in is Portland")
    claim_a = insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_fact",
        value="favorite city Portland",
        confidence=0.9,
        source_turn_id=turn_a.turn_id,
    )

    turn_b = _insert_turn(db, session_id, "My favorite city to live in is Seattle now")
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
    edge = detect_pass1(db, claim_b)
    assert edge is not None, "Expected supersession to fire"

    result = classify_predicate(db, "user", "user_fact")
    assert result.classification == "evolving"
    assert result.change_count >= 1


def test_classify_volatile(db, session_id):
    """Three or more supersession events make the predicate 'volatile'."""
    # Each value shares enough content words with the predecessor for
    # Jaccard >= 0.3 so detect_pass1 fires.
    locations = [
        "favorite city Portland",
        "favorite city Seattle",
        "favorite city Denver",
        "favorite city Austin",
    ]

    prev_claim = None
    for i, loc in enumerate(locations):
        turn = _insert_turn(db, session_id, f"I moved to {loc}")
        claim = insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="user_fact",
            value=loc,
            confidence=0.9,
            source_turn_id=turn.turn_id,
        )
        if prev_claim is not None:
            detect_pass1(db, claim)
        prev_claim = claim

    result = classify_predicate(db, "user", "user_fact")
    assert result.classification == "volatile"
    assert result.change_count >= 3


def test_volatility_info_fields(db, session_id):
    """VolatilityInfo has change_count, avg_lifespan_days, current_streak_days populated."""
    turn = _insert_turn(db, session_id, "My hobby is chess")
    insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_preference",
        value="hobby is chess",
        confidence=0.9,
        source_turn_id=turn.turn_id,
    )

    result = classify_predicate(db, "user", "user_preference")
    assert isinstance(result.change_count, int)
    assert isinstance(result.avg_lifespan_days, float)
    assert isinstance(result.current_streak_days, float)
    # A stable claim should have zero changes and non-negative streak
    assert result.change_count == 0
    assert result.avg_lifespan_days >= 0.0
    assert result.current_streak_days >= 0.0
