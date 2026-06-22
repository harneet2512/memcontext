from __future__ import annotations

import sqlite3

from memcontext.event_bus import CLAIM_CREATED, TURN_ADDED, EventBus
from memcontext.on_new_turn import ExtractedClaim, on_new_turn
from memcontext.schema import Speaker, Turn


def test_on_new_turn_happy_path(db: sqlite3.Connection, session_id: str, extractor_fn):
    result = on_new_turn(
        db,
        session_id=session_id,
        speaker=Speaker.USER,
        text="I prefer dark mode and use Python daily",
        extractor=extractor_fn,
    )
    assert result.admitted is True
    assert result.turn is not None
    assert len(result.created_claims) == 1
    assert result.created_claims[0].predicate == "user_preference"


def test_on_new_turn_rejected(db: sqlite3.Connection, session_id: str, extractor_fn):
    result = on_new_turn(
        db,
        session_id=session_id,
        speaker=Speaker.USER,
        text="uh um ok",
        extractor=extractor_fn,
    )
    assert result.admitted is False
    assert result.turn is None
    assert len(result.created_claims) == 0


def test_on_new_turn_empty_text_rejected(
    db: sqlite3.Connection, session_id: str, extractor_fn,
):
    result = on_new_turn(
        db,
        session_id=session_id,
        speaker=Speaker.USER,
        text="",
        extractor=extractor_fn,
    )
    assert result.admitted is False


def test_on_new_turn_with_pass1_supersession(
    db: sqlite3.Connection, session_id: str,
):
    def extract_two_kids(turn: Turn) -> list[ExtractedClaim]:
        return [
            ExtractedClaim(
                subject="user",
                predicate="user_fact",
                value="has two kids",
                confidence=0.9,
            ),
        ]

    def extract_three_kids(turn: Turn) -> list[ExtractedClaim]:
        return [
            ExtractedClaim(
                subject="user",
                predicate="user_fact",
                value="has three kids",
                confidence=0.9,
            ),
        ]

    r1 = on_new_turn(
        db,
        session_id=session_id,
        speaker=Speaker.USER,
        text="I have two kids at home",
        extractor=extract_two_kids,
    )
    assert len(r1.created_claims) == 1

    r2 = on_new_turn(
        db,
        session_id=session_id,
        speaker=Speaker.USER,
        text="Actually I have three kids at home",
        extractor=extract_three_kids,
    )
    assert len(r2.created_claims) == 1
    assert len(r2.supersession_edges) == 1
    assert r2.supersession_edges[0].old_claim_id == r1.created_claims[0].claim_id


def test_on_new_turn_out_of_vocab_predicate_coerced_to_user_fact(
    db: sqlite3.Connection, session_id: str,
):
    """An out-of-vocab predicate is no longer dropped OR value-stripped — it is
    coerced to the generic ``user_fact`` family, KEEPING subject+value as a
    structured, resolvable/served claim. The fact is never lost AND stays usable."""
    def extract_bad(turn: Turn) -> list[ExtractedClaim]:
        return [
            ExtractedClaim(
                subject="user",
                predicate="invalid_predicate_xyz",
                value="something",
                confidence=0.5,
            ),
        ]

    result = on_new_turn(
        db,
        session_id=session_id,
        speaker=Speaker.USER,
        text="This has an invalid predicate extraction",
        extractor=extract_bad,
    )
    assert result.admitted is True
    # Created as a structured fact (predicate coerced), not dropped, not value-stripped.
    assert len(result.created_claims) == 1
    assert len(result.dropped_claims) == 0
    fact = result.created_claims[0]
    assert fact.subject == "user" and fact.predicate == "user_fact" and fact.value == "something"


def test_on_new_turn_events_published(
    db: sqlite3.Connection, session_id: str, extractor_fn,
):
    bus = EventBus()
    events: list[tuple[str, dict]] = []

    def recorder(topic: str, payload: dict):
        events.append((topic, payload))

    bus.subscribe(TURN_ADDED, lambda payload: recorder(TURN_ADDED, payload))
    bus.subscribe(CLAIM_CREATED, lambda payload: recorder(CLAIM_CREATED, payload))

    on_new_turn(
        db,
        session_id=session_id,
        speaker=Speaker.USER,
        text="I prefer dark mode and use Python daily",
        extractor=extractor_fn,
        bus=bus,
    )

    topics = [t for t, _ in events]
    assert TURN_ADDED in topics
    assert CLAIM_CREATED in topics
