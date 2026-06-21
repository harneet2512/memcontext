from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim, set_claim_status
from memcontext.projections import (
    claims_grouped_by_subject_predicate,
    filtered_projection,
    rebuild_active_projection,
)
from memcontext.schema import Claim, ClaimStatus, Turn


def test_rebuild_active_projection(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="lives in Toronto", confidence=0.9, source_turn_id=sample_turn.turn_id,
    )
    insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="prefers dark mode", confidence=0.8, source_turn_id=sample_turn.turn_id,
    )

    proj = rebuild_active_projection(db, session_id)
    assert len(proj.claims) == 2


def test_filtered_projection(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="lives in Toronto", confidence=0.9, source_turn_id=sample_turn.turn_id,
    )
    insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="prefers dark mode", confidence=0.8, source_turn_id=sample_turn.turn_id,
    )

    proj = rebuild_active_projection(db, session_id)
    filtered = filtered_projection(proj, lambda c: c.predicate == "user_fact")
    assert len(filtered.claims) == 1
    assert filtered.claims[0].predicate == "user_fact"


def test_by_predicate_grouping(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="lives in Toronto", confidence=0.9, source_turn_id=sample_turn.turn_id,
    )
    insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="works at Acme", confidence=0.8, source_turn_id=sample_turn.turn_id,
    )
    insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="prefers dark mode", confidence=0.8, source_turn_id=sample_turn.turn_id,
    )

    proj = rebuild_active_projection(db, session_id)
    grouped = proj.by_predicate
    assert "user_fact" in grouped
    assert len(grouped["user_fact"]) == 2
    assert "user_preference" in grouped
    assert len(grouped["user_preference"]) == 1


def test_claims_grouped_by_subject_predicate(
    db: sqlite3.Connection, session_id: str, sample_turn: Turn,
):
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="old value", confidence=0.9, source_turn_id=sample_turn.turn_id,
    )
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_fact",
        value="new value", confidence=0.9, source_turn_id=sample_turn.turn_id,
    )
    proj = rebuild_active_projection(db, session_id)
    grouped = claims_grouped_by_subject_predicate(proj.claims)
    # FRACTURE B: key is now (subject, predicate, attribute). These values carry
    # no derivable attribute slot (no "label:" prefix, no relation verb), so the
    # attribute is "" and they still collapse newest-wins exactly as before.
    key = ("user", "user_fact", "")
    assert key in grouped
    assert grouped[key].claim_id == c2.claim_id
