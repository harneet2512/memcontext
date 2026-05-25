"""Tests for lifecycle-driven embedding indexing."""
from __future__ import annotations

import sqlite3
import struct

from memcontext.claims import (
    insert_claim,
    insert_turn,
    new_turn_id,
    now_ns,
    set_claim_status,
)
from memcontext.schema import ClaimStatus, Speaker, Turn, open_database
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


def _store_fake_embedding(conn: sqlite3.Connection, claim_id: str) -> None:
    """Store a fake embedding blob for testing purposes."""
    dim = 8
    vec = [0.1] * dim
    blob = struct.pack(f"<I{dim}f", dim, *vec)
    conn.execute(
        "INSERT OR REPLACE INTO claim_embeddings "
        "(claim_id, embedding, embedding_model_version, embedded_at_unix) "
        "VALUES (?, ?, ?, ?)",
        (claim_id, blob, "test-model-v1", 1000000),
    )


def test_superseded_claim_loses_embedding(db, session_id):
    """When a claim is superseded, its embedding row is deleted."""
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

    # Store a fake embedding for claim A
    _store_fake_embedding(db, claim_a.claim_id)

    # Verify embedding exists
    row = db.execute(
        "SELECT claim_id FROM claim_embeddings WHERE claim_id = ?",
        (claim_a.claim_id,),
    ).fetchone()
    assert row is not None, "Embedding should exist before supersession"

    # Create claim B that supersedes A
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

    # After supersession, claim A's embedding should be deleted
    # (set_claim_status with SUPERSEDED deletes from claim_embeddings)
    row = db.execute(
        "SELECT claim_id FROM claim_embeddings WHERE claim_id = ?",
        (claim_a.claim_id,),
    ).fetchone()
    assert row is None, "Embedding should be deleted after supersession"


def test_active_claim_keeps_embedding(db, session_id):
    """An active claim retains its embedding row."""
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

    _store_fake_embedding(db, claim.claim_id)

    # Verify embedding exists
    row = db.execute(
        "SELECT claim_id FROM claim_embeddings WHERE claim_id = ?",
        (claim.claim_id,),
    ).fetchone()
    assert row is not None, "Embedding should exist for active claim"

    # Claim stays active — embedding should still be there
    row2 = db.execute(
        "SELECT claim_id FROM claim_embeddings WHERE claim_id = ?",
        (claim.claim_id,),
    ).fetchone()
    assert row2 is not None, "Embedding should persist for active claim"
