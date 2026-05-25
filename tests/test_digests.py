"""Tests for memcontext.digests — session digest building and persistence."""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.digests import (
    SessionDigest,
    build_session_digest,
    load_digest,
    store_digest,
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


def test_build_empty_digest(db, session_id):
    """No claims for a session produces an empty digest."""
    digest = build_session_digest(db, session_id)
    assert isinstance(digest, SessionDigest)
    assert digest.total_claims == 0
    assert digest.key_facts == []
    assert digest.updates == []


def test_build_digest_with_claims(db, session_id):
    """Creating 5 claims produces a digest with key_facts having <= 3 items."""
    turn = _insert_turn(db, session_id, "Digest test data")

    for i in range(5):
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="user_fact",
            value=f"fact number {i}",
            confidence=0.9,
            source_turn_id=turn.turn_id,
        )

    digest = build_session_digest(db, session_id)
    assert isinstance(digest, SessionDigest)
    assert digest.total_claims == 5
    assert len(digest.key_facts) <= 3
    assert digest.session_id == session_id


def test_store_and_load_digest(db, session_id):
    """Digest can be stored and loaded back (round-trip)."""
    turn = _insert_turn(db, session_id, "Round trip test")

    for i in range(4):
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="user_fact",
            value=f"round trip fact {i}",
            confidence=0.9,
            source_turn_id=turn.turn_id,
        )

    digest = build_session_digest(db, session_id)
    store_digest(db, digest)

    loaded = load_digest(db, session_id)
    assert loaded is not None
    assert loaded.session_id == digest.session_id
    assert loaded.total_claims == digest.total_claims
    assert len(loaded.key_facts) == len(digest.key_facts)
