"""Tests for memcontext.profiles — smart profile building and persistence."""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.profiles import (
    SmartProfile,
    build_smart_profile,
    format_profile,
    load_profile,
    store_profile,
)
from memcontext.schema import Speaker, Turn, open_database


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


def test_build_empty_profile(db, session_id):
    """No claims for a subject results in a profile with 0 lines."""
    profile = build_smart_profile(db, "nonexistent_subject")
    assert isinstance(profile, SmartProfile)
    assert len(profile.lines) == 0
    assert profile.total_facts == 0


def test_build_profile_with_claims(db, session_id):
    """Creating 3 claims produces a profile with lines > 0."""
    turn = _insert_turn(db, session_id, "Profile test data")

    claims_data = [
        ("user_fact", "name is Sarah Chen"),
        ("user_fact", "lives in San Francisco"),
        ("user_preference", "prefers dark mode"),
    ]

    for pred, val in claims_data:
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate=pred,
            value=val,
            confidence=0.9,
            source_turn_id=turn.turn_id,
        )

    profile = build_smart_profile(db, "user")
    assert isinstance(profile, SmartProfile)
    assert len(profile.lines) > 0
    assert profile.total_facts == 3
    assert profile.subject == "user"


def test_store_and_load_profile(db, session_id):
    """Profile can be stored and loaded back with the same data."""
    turn = _insert_turn(db, session_id, "Store load test")

    insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_fact",
        value="works at Google",
        confidence=0.9,
        source_turn_id=turn.turn_id,
    )
    insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_preference",
        value="prefers TypeScript",
        confidence=0.85,
        source_turn_id=turn.turn_id,
    )

    profile = build_smart_profile(db, "user")
    store_profile(db, profile)

    loaded = load_profile(db, "user")
    assert loaded is not None
    assert loaded.subject == profile.subject
    assert loaded.total_facts == profile.total_facts
    assert len(loaded.lines) == len(profile.lines)


def test_format_profile(db, session_id):
    """format_profile returns a string containing the subject name and [PROFILE] header."""
    turn = _insert_turn(db, session_id, "Format test")

    insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_fact",
        value="enjoys hiking",
        confidence=0.9,
        source_turn_id=turn.turn_id,
    )

    profile = build_smart_profile(db, "user")
    formatted = format_profile(profile)

    assert isinstance(formatted, str)
    assert "user" in formatted
    assert "[PROFILE]" in formatted
