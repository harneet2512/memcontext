"""Tests for memcontext.chains — supersession chain building and formatting."""
from __future__ import annotations

import sqlite3

from memcontext.chains import ChainStep, build_chain, format_chain
from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.schema import Speaker, Turn, open_database
from memcontext.supersession import detect_pass1


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


def test_build_chain_no_history(db, session_id):
    """A claim with no supersession history has a chain of length 1 (itself)."""
    turn = _insert_turn(db, session_id, "I like Python")
    claim = insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_preference",
        value="likes Python",
        confidence=0.9,
        source_turn_id=turn.turn_id,
    )

    chain = build_chain(db, claim.claim_id)
    assert len(chain) == 1
    assert isinstance(chain[0], ChainStep)
    assert chain[0].value == "likes Python"
    assert chain[0].edge_type == "active"
    assert chain[0].claim_id == claim.claim_id


def test_build_chain_with_supersession(db, session_id):
    """A chain with one supersession has 2 steps: old then new."""
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

    edge = detect_pass1(db, claim_b)
    assert edge is not None, "Expected supersession to fire"

    chain = build_chain(db, claim_b.claim_id)
    assert len(chain) == 2

    # First step is the old (superseded) claim
    assert chain[0].value == "favorite city Portland"
    assert chain[0].edge_type != "active"

    # Second step is the current (active) claim
    assert chain[1].value == "favorite city Seattle"
    assert chain[1].edge_type == "active"
    assert chain[1].claim_id == claim_b.claim_id


def test_format_chain(db, session_id):
    """format_chain produces readable text with dates."""
    turn_a = _insert_turn(db, session_id, "My employer company is Acme Corp")
    claim_a = insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_fact",
        value="employer company Acme Corp",
        confidence=0.9,
        source_turn_id=turn_a.turn_id,
    )

    turn_b = _insert_turn(db, session_id, "My employer company is now Google")
    claim_b = insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_fact",
        value="employer company Google",
        confidence=0.9,
        source_turn_id=turn_b.turn_id,
    )

    detect_pass1(db, claim_b)
    chain = build_chain(db, claim_b.claim_id)
    formatted = format_chain(chain)

    assert isinstance(formatted, str)
    assert len(formatted) > 0
    # Should contain date-like patterns [YYYY-MM-DD]
    assert "[" in formatted
    assert "ACTIVE" in formatted
    assert "SUPERSEDED" in formatted


def test_format_chain_empty():
    """format_chain with an empty chain returns an empty string."""
    assert format_chain([]) == ""
