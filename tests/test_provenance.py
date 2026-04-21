from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.provenance import (
    claim_ids_for_turn,
    insert_output_sentence,
    sentence_ids_for_claim,
    span_for_claim,
)
from memcontext.schema import Claim, OutputSection, Speaker, Turn


def test_claim_ids_for_turn(db: sqlite3.Connection, session_id: str, sample_turn: Turn):
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="fact1", confidence=0.9, source_turn_id=sample_turn.turn_id,
    )
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="pref1", confidence=0.8, source_turn_id=sample_turn.turn_id,
    )

    ids = claim_ids_for_turn(db, sample_turn.turn_id)
    assert c1.claim_id in ids
    assert c2.claim_id in ids


def test_span_for_claim(db: sqlite3.Connection, session_id: str):
    turn = Turn(
        turn_id=new_turn_id(),
        session_id=session_id,
        speaker=Speaker.USER,
        text="I live in Toronto and work at Acme",
        ts=now_ns(),
        asr_confidence=None,
    )
    insert_turn(db, turn)

    claim = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="lives in Toronto", confidence=0.9, source_turn_id=turn.turn_id,
        char_start=0, char_end=19,
    )

    span = span_for_claim(db, claim.claim_id)
    assert span is not None
    assert span.char_start == 0
    assert span.char_end == 19
    assert span.turn_id == turn.turn_id


def test_span_for_claim_none_spans(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    claim = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="fact1", confidence=0.9, source_turn_id=sample_turn.turn_id,
    )
    span = span_for_claim(db, claim.claim_id)
    assert span is not None
    assert span.char_start is None
    assert span.char_end is None


def test_sentence_ids_for_claim(db: sqlite3.Connection, session_id: str, sample_turn: Turn):
    claim = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="fact1", confidence=0.9, source_turn_id=sample_turn.turn_id,
    )

    sent = insert_output_sentence(
        db,
        session_id=session_id,
        section=OutputSection.SUMMARY,
        ordinal=0,
        text="User reports fact1.",
        source_claim_ids=[claim.claim_id],
    )

    ids = sentence_ids_for_claim(db, claim.claim_id)
    assert sent.sentence_id in ids
