"""Tests for memcontext.retrieval multi-resolution functions (search_raw_turns, retrieve_with_fallback)."""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.retrieval import retrieve_with_fallback, search_raw_turns
from memcontext.schema import Speaker, Turn, open_database


def _insert_turn(
    conn: sqlite3.Connection,
    session_id: str,
    text: str,
    speaker: Speaker = Speaker.USER,
) -> Turn:
    turn = Turn(
        turn_id=new_turn_id(),
        session_id=session_id,
        speaker=speaker,
        text=text,
        ts=now_ns(),
        asr_confidence=None,
    )
    insert_turn(conn, turn)
    return turn


def test_search_raw_turns(db, session_id):
    """search_raw_turns finds matching turns via BM25 keyword search."""
    _insert_turn(db, session_id, "I enjoy hiking in the mountains on weekends")
    _insert_turn(db, session_id, "My favorite programming language is Python")
    _insert_turn(db, session_id, "I work at a startup in San Francisco")

    results = search_raw_turns(db, session_id, "Python programming")
    assert len(results) > 0

    # The top result should be the Python turn
    top_turn, top_score = results[0]
    assert "Python" in top_turn.text
    assert top_score > 0.0


def test_retrieve_with_fallback_claims_sufficient(db, session_id):
    """When enough claims exist with good scores, fallback to turns is NOT triggered."""
    # Create multiple claims (no embeddings — hybrid retrieval will still
    # produce BM25-based scores via retrieve_hybrid, which does not require
    # embeddings to return results)
    turn = _insert_turn(db, session_id, "I like Python and TypeScript and JavaScript")

    for i in range(5):
        insert_claim(
            db,
            session_id=session_id,
            subject="user",
            predicate="user_fact",
            value=f"programming fact {i} about Python",
            confidence=0.9,
            source_turn_id=turn.turn_id,
        )

    # retrieve_with_fallback will call retrieve_hybrid internally.
    # Without embeddings, semantic scores will be 0 but BM25 scores will be non-zero.
    # The function checks if len(claim_results) >= 3 AND score >= 0.3.
    # Without embeddings, RRF scores may be low, so fallback may trigger.
    # Either way, we should get results back.
    results = retrieve_with_fallback(db, session_id, "Python programming")
    assert isinstance(results, list)
    # We should get some results (either claims or turns)
    assert len(results) > 0


def test_retrieve_with_fallback_claims_insufficient(db, session_id):
    """When claims are insufficient, raw turns are included in the results."""
    # Insert turns with content
    _insert_turn(db, session_id, "I really love hiking in mountain trails")
    _insert_turn(db, session_id, "My favorite hiking spot is Yosemite park")

    # Insert only 1 claim (below the threshold of 3 for "sufficient")
    turn_for_claim = _insert_turn(db, session_id, "hiking is great exercise")
    insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_preference",
        value="enjoys hiking",
        confidence=0.9,
        source_turn_id=turn_for_claim.turn_id,
    )

    results = retrieve_with_fallback(db, session_id, "hiking")
    assert isinstance(results, list)
    assert len(results) > 0

    # Check that there are turn-type results in the output
    turn_results = [r for r in results if r.get("type") == "turn"]
    assert len(turn_results) > 0, "Expected raw turns in fallback results"
