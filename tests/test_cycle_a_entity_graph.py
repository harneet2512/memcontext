"""Cycle A (entity/graph retrieval): a 2-hop query surfaces facts of a co-occurring
entity that the flat token match misses.

The discriminator: the surfaced fact contains NO query token — its only signal is
the graph neighbor boost; an unrelated entity's fact gets none.
"""
from __future__ import annotations

import sqlite3

from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.retrieval import retrieve_hybrid
from memcontext.schema import Speaker, open_database


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def test_two_hop_entity_expansion_beats_flat_token_match():
    conn = _conn()
    # Turn 1: alice and project_x co-occur in ONE turn -> graph neighbors.
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="alice leads project_x",
        extractor=PassthroughExtractor([
            {"subject": "alice", "predicate": "user_fact", "value": "lead", "confidence": 0.9},
            {"subject": "project_x", "predicate": "user_fact", "value": "active", "confidence": 0.9},
        ]),
    )
    # Turn 2: a project_x fact that does NOT mention alice.
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="project_x ships friday",
        extractor=PassthroughExtractor([
            {"subject": "project_x", "predicate": "user_fact", "value": "friday", "confidence": 0.9},
        ]),
    )
    # Turn 3: an unrelated entity (bob) -- not a neighbor of alice.
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="bob likes tea",
        extractor=PassthroughExtractor([
            {"subject": "bob", "predicate": "user_fact", "value": "tea", "confidence": 0.9},
        ]),
    )
    friday = conn.execute("SELECT claim_id FROM claims WHERE value='friday'").fetchone()["claim_id"]
    tea = conn.execute("SELECT claim_id FROM claims WHERE value='tea'").fetchone()["claim_id"]

    explain: dict[str, dict[str, float]] = {}
    hits = retrieve_hybrid(conn, session_id="s1", query="alice", top_k=20, explain=explain)
    ids = [c.claim_id for c, _ in hits]

    # Neither 'friday' nor 'tea' contains the token "alice"; the ONLY reason friday
    # scores on the entity channel is the 2-hop graph link alice -> project_x.
    assert explain[friday]["entity"] > explain[tea]["entity"], "graph neighbor gets the boost"
    assert ids.index(friday) < ids.index(tea), "co-occurring-entity fact outranks the unrelated one"
