from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.schema import Claim, ClaimStatus, EdgeType, Speaker, Turn
from memcontext.supersession import detect_pass1


def _make_turn(db: sqlite3.Connection, session_id: str, speaker: Speaker, text: str) -> Turn:
    turn = Turn(
        turn_id=new_turn_id(),
        session_id=session_id,
        speaker=speaker,
        text=text,
        ts=now_ns(),
        asr_confidence=None,
    )
    insert_turn(db, turn)
    return turn


def test_pass1_same_subject_predicate_different_value(
    db: sqlite3.Connection, session_id: str,
):
    t1 = _make_turn(db, session_id, Speaker.USER, "I have two kids")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="has two kids", confidence=0.9, source_turn_id=t1.turn_id,
    )

    t2 = _make_turn(db, session_id, Speaker.USER, "Actually I have three kids")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="has three kids", confidence=0.9, source_turn_id=t2.turn_id,
    )

    edge = detect_pass1(db, c2)
    assert edge is not None
    assert edge.old_claim_id == c1.claim_id
    assert edge.new_claim_id == c2.claim_id


def test_pass1_same_value_no_supersession(
    db: sqlite3.Connection, session_id: str,
):
    t1 = _make_turn(db, session_id, Speaker.USER, "I like coffee")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="likes coffee", confidence=0.9, source_turn_id=t1.turn_id,
    )
    t2 = _make_turn(db, session_id, Speaker.USER, "I like coffee too")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="likes coffee", confidence=0.9, source_turn_id=t2.turn_id,
    )
    edge = detect_pass1(db, c2)
    assert edge is None


def test_pass1_low_jaccard_no_supersession(
    db: sqlite3.Connection, session_id: str,
):
    t1 = _make_turn(db, session_id, Speaker.USER, "I like coffee")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="lives in Toronto Canada", confidence=0.9, source_turn_id=t1.turn_id,
    )
    t2 = _make_turn(db, session_id, Speaker.USER, "My favorite color is blue")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="favorite color is blue", confidence=0.9, source_turn_id=t2.turn_id,
    )
    edge = detect_pass1(db, c2)
    assert edge is None


def test_pass1_edge_type_user_correction(
    db: sqlite3.Connection, session_id: str,
):
    t1 = _make_turn(db, session_id, Speaker.USER, "I have two kids")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="has two kids", confidence=0.9, source_turn_id=t1.turn_id,
    )
    t2 = _make_turn(db, session_id, Speaker.USER, "Actually I have three kids")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="has three kids", confidence=0.9, source_turn_id=t2.turn_id,
    )
    edge = detect_pass1(db, c2)
    assert edge is not None
    assert edge.edge_type == EdgeType.USER_CORRECTION


def test_pass1_edge_type_assistant_confirm(
    db: sqlite3.Connection, session_id: str,
):
    t1 = _make_turn(db, session_id, Speaker.USER, "I have two kids")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="has two kids", confidence=0.9, source_turn_id=t1.turn_id,
    )
    t2 = _make_turn(db, session_id, Speaker.ASSISTANT, "You mentioned having three kids")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="has three kids", confidence=0.9, source_turn_id=t2.turn_id,
    )
    edge = detect_pass1(db, c2)
    assert edge is not None
    assert edge.edge_type == EdgeType.ASSISTANT_CONFIRM


def test_pass1_same_turn_guard(
    db: sqlite3.Connection, session_id: str,
):
    t1 = _make_turn(db, session_id, Speaker.USER, "I have two kids and three dogs")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="has two kids", confidence=0.9, source_turn_id=t1.turn_id,
    )
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="has three dogs", confidence=0.9, source_turn_id=t1.turn_id,
    )
    edge = detect_pass1(db, c2)
    assert edge is None


def test_pass1_marks_old_claim_superseded(
    db: sqlite3.Connection, session_id: str,
):
    from memcontext.claims import get_claim

    t1 = _make_turn(db, session_id, Speaker.USER, "I take medication daily")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="takes medication daily morning", confidence=0.9, source_turn_id=t1.turn_id,
    )
    t2 = _make_turn(db, session_id, Speaker.USER, "I take medication at night now")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="takes medication daily night", confidence=0.9, source_turn_id=t2.turn_id,
    )
    detect_pass1(db, c2)

    old = get_claim(db, c1.claim_id)
    assert old is not None
    assert old.status == ClaimStatus.SUPERSEDED
