"""Tests for memcontext.life_events — deterministic life event detection."""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.life_events import LifeEvent, detect_life_events, store_life_events
from memcontext.schema import Speaker, Turn, open_database


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


def test_no_life_events(db, session_id):
    """A single claim should not trigger any life event (needs min_predicates distinct predicates)."""
    turn = _insert_turn(db, session_id, "I like coffee")
    insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_preference",
        value="likes coffee",
        confidence=0.9,
        source_turn_id=turn.turn_id,
    )

    events = detect_life_events(db, "user")
    assert events == []


def test_detect_life_event(db, session_id):
    """Claims with 4+ different predicates within a tight time window trigger a life event."""
    # Use a single turn to ensure all claims share a tight timestamp
    turn = _insert_turn(db, session_id, "Big changes happening")

    predicates_and_values = [
        ("user_fact", "moved to New York"),
        ("user_event", "started new job at Google"),
        ("user_relationship", "married to Alice"),
        ("user_goal", "learn to cook Italian food"),
    ]

    for pred, val in predicates_and_values:
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate=pred,
            value=val,
            confidence=0.9,
            source_turn_id=turn.turn_id,
        )

    # Default min_predicates=3, so 4 distinct predicates should trigger
    events = detect_life_events(db, "user", min_predicates=3)
    assert len(events) >= 1
    event = events[0]
    assert isinstance(event, LifeEvent)
    assert event.subject == "user"
    assert len(event.claim_ids) >= 4
    assert len(event.predicates_affected) >= 3
    assert 0.0 < event.significance <= 1.0


def test_store_and_load(db, session_id):
    """Detected life events can be stored and read back from the life_events table."""
    turn = _insert_turn(db, session_id, "Life changes")

    predicates_and_values = [
        ("user_fact", "relocated to Boston"),
        ("user_event", "graduated from MIT"),
        ("user_relationship", "best friend is Bob"),
        ("user_goal", "run a marathon"),
    ]

    for pred, val in predicates_and_values:
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate=pred,
            value=val,
            confidence=0.9,
            source_turn_id=turn.turn_id,
        )

    events = detect_life_events(db, "user", min_predicates=3)
    assert len(events) >= 1

    count = store_life_events(db, events)
    assert count == len(events)

    # Verify the events are in the life_events table
    rows = db.execute(
        "SELECT * FROM life_events WHERE subject = ?", ("user",)
    ).fetchall()
    assert len(rows) >= 1
    assert rows[0]["subject"] == "user"
    assert rows[0]["significance"] > 0.0
