"""FRACTURE B — deterministic attribute-slot identity under coarse predicates.

Covers the unit behaviour of memcontext.attribute_key and its three integration
points (Pass-1 supersession, projection collapse, enumeration slot selection)
under the coarse 'user_fact' predicate, with the model-free NullEmbedder so the
test runs in CI with zero downloads. The real-embedder evidence lives in the
runnable proof tests/proof_fractureB_identity.py.
"""
from __future__ import annotations

import sqlite3

import pytest

from memcontext.attribute_key import attribute_key, attributes_conflict
from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.projections import claims_grouped_by_subject_predicate
from memcontext.schema import Speaker, Turn, open_database
from memcontext.supersession import detect_pass1


# --------------------------------------------------------------------------- #
# attribute_key — derivation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [
        ("employer: Acme", "employer"),
        ("city: Boston", "city"),
        ("home city: Toronto", "home_city"),
        ("commute time: ~20 minutes by bike", "commute_time"),
        ("works at Acme", "work"),
        ("lives in NYC", "reside"),
        ("moved to Boston", "reside"),
        ("is allergic to peanuts", "allergic"),
        ("likes pizza", "prefer"),
        ("prefers tea", "prefer"),       # liking verbs collapse to one slot
        ("owns a car", "own"),
        # no derivable slot -> empty (the no-regression default)
        ("ClickHouse", ""),
        ("old value", ""),
        ("office located in Brooklyn", ""),
        ("", ""),
    ],
)
def test_attribute_key_derivation(value, expected):
    assert attribute_key(value) == expected


def test_attributes_conflict_abstains_when_either_empty():
    # different slots -> conflict
    assert attributes_conflict("employer: Acme", "city: Boston") is True
    assert attributes_conflict("works at Acme", "lives in NYC") is True
    # same slot -> no conflict (genuine update path stays open)
    assert attributes_conflict("employer: Acme", "employer: Globex") is False
    # either empty -> abstain (never split slot-less values)
    assert attributes_conflict("old value", "new value") is False
    assert attributes_conflict("employer: Acme", "ClickHouse") is False


# --------------------------------------------------------------------------- #
# Integration — coarse predicate, NullEmbedder (CI-safe)
# --------------------------------------------------------------------------- #

SESSION = "attr-key-session"


def _conn() -> sqlite3.Connection:
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _turn(conn) -> Turn:
    t = Turn(turn_id=new_turn_id(), session_id=SESSION, speaker=Speaker.USER,
             text="some realistic turn text about the user's life", ts=now_ns())
    insert_turn(conn, t)
    return t


def _claim(conn, value, turn):
    return insert_claim(
        conn, session_id=SESSION, subject="user", predicate="user_fact",
        value=value, confidence=0.9, source_turn_id=turn.turn_id,
    )


def test_projection_does_not_fuse_distinct_coarse_facts():
    """Six distinct 'user_fact' values must survive newest-wins collapse."""
    conn = _conn()
    vals = [
        "home city: Toronto", "employer: Acme", "favorite restaurant: Nopa",
        "commute time: 25 minutes by bike", "allergic to peanuts",
        "owns a golden retriever",
    ]
    claims = []
    for v in vals:
        claims.append(_claim(conn, v, _turn(conn)))
    grouped = claims_grouped_by_subject_predicate(claims)
    # WITHOUT the attribute these would collapse to one (user, user_fact) row.
    assert len(grouped) == len(vals)


def test_pass1_does_not_falsely_supersede_distinct_slots():
    """employer vs city under one coarse predicate must NOT supersede."""
    conn = _conn()
    t1 = _turn(conn)
    _claim(conn, "employer: Acme", t1)
    t2 = _turn(conn)
    c2 = _claim(conn, "city: Boston", t2)
    edge = detect_pass1(conn, c2)
    assert edge is None  # different slot -> no false fuse


def test_pass1_supersedes_same_slot_update():
    """employer: Acme -> employer: Globex is a same-slot update -> supersede."""
    conn = _conn()
    t1 = _turn(conn)
    c1 = _claim(conn, "employer: Acme", t1)
    t2 = _turn(conn)
    c2 = _claim(conn, "employer: Globex", t2)
    edge = detect_pass1(conn, c2)
    assert edge is not None
    assert edge.old_claim_id == c1.claim_id


def test_pass1_unchanged_for_slotless_values():
    """Values with no derivable slot keep today's additive/jaccard behaviour:
    two single-shared-token facts stay additive (no false supersession)."""
    conn = _conn()
    t1 = _turn(conn)
    _claim(conn, "likes hiking", t1)  # slot 'prefer'
    t2 = _turn(conn)
    c2 = _claim(conn, "owns a kayak", t2)  # slot 'own' -> different, no fuse
    assert detect_pass1(conn, c2) is None
