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
    # Entity names are Capitalized on purpose: the regex entity extractor
    # (memcontext.entities.extract_entities) only recognises proper-noun-cased
    # tokens, so the co-occurrence graph is populated only when the NL fact text
    # carries them. Lowercase subjects extract NO entities, leave claim_entities
    # empty, and the 2-hop boost never fires — which previously made this test a
    # FALSE PASS that survived only on strict index-order tiebreaks in the rank
    # fusion (both candidates scored 0.0 on the entity channel). Tie-aware fusion
    # removed that index noise and exposed the dead boost; this fixture now drives
    # the graph for real so the assertion validates the actual 2-hop expansion.
    #
    # Turn 1: Alice and Project co-occur in ONE turn -> graph neighbors.
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="Alice leads Project",
        extractor=PassthroughExtractor([
            {"subject": "Alice", "predicate": "user_fact",
             "value": "Alice leads Project", "confidence": 0.9},
            {"subject": "Project", "predicate": "user_fact",
             "value": "Project is active", "confidence": 0.9},
        ]),
    )
    # Turn 2: a Project fact that does NOT mention Alice.
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="Project ships Friday",
        extractor=PassthroughExtractor([
            {"subject": "Project", "predicate": "user_fact",
             "value": "Project ships Friday", "confidence": 0.9},
        ]),
    )
    # Turn 3: an unrelated entity (Bob) -- not a neighbor of Alice.
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="Bob likes tea",
        extractor=PassthroughExtractor([
            {"subject": "Bob", "predicate": "user_fact",
             "value": "Bob likes tea", "confidence": 0.9},
        ]),
    )
    friday = conn.execute(
        "SELECT claim_id FROM claims WHERE value LIKE '%Friday%'"
    ).fetchone()["claim_id"]
    tea = conn.execute(
        "SELECT claim_id FROM claims WHERE value LIKE '%tea%'"
    ).fetchone()["claim_id"]

    explain: dict[str, dict[str, float]] = {}
    hits = retrieve_hybrid(conn, session_id="s1", query="Alice", top_k=20, explain=explain)
    ids = [c.claim_id for c, _ in hits]

    # The Friday fact does not name Alice; the ONLY reason it scores on the entity
    # channel is the 2-hop graph link Alice -> Project -> (Project's Friday fact).
    assert explain[friday]["entity"] > explain[tea]["entity"], "graph neighbor gets the boost"
    assert ids.index(friday) < ids.index(tea), "co-occurring-entity fact outranks the unrelated one"
