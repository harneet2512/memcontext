"""brain() — the deterministic world-state projection.

A read-only view over the active-claims projection. Groups the current
world-state by subject, attaches a provenance handle (source turn + character
span) to every fact, and reports a deterministic *gaps* list per subject:
vocabulary predicates with no active claim.

Zero LLM, zero network — it reads the projection and the source turns only.
This is the structured payload that distinguishes MemContext from a summary
blob or a top-k vector dump: a single current value per (subject, predicate)
with status, confidence, and a verifiable source span.
"""
from __future__ import annotations

import sqlite3

from memcontext.claims import get_turn
from memcontext.predicate_packs import active_pack
from memcontext.projections import rebuild_active_projection
from memcontext.schema import Claim, Turn


def _quote(turn: Turn | None, claim: Claim) -> str | None:
    """Exact substring of the source turn that produced *claim*, or None."""
    if turn is None or claim.char_start is None or claim.char_end is None:
        return None
    return turn.text[claim.char_start : claim.char_end]


def _fact(turn: Turn | None, claim: Claim) -> dict:
    return {
        "subject": claim.subject,
        "predicate": claim.predicate,
        "value": claim.value,
        "status": claim.status.value,
        "confidence": claim.confidence,
        "provenance": {
            "source_turn_id": claim.source_turn_id,
            "char_start": claim.char_start,
            "char_end": claim.char_end,
            "quote": _quote(turn, claim),
        },
    }


def brain(conn: sqlite3.Connection, *, session_id: str) -> dict:
    """Return the current world-state grouped by subject, plus a gaps report.

    Shape::

        {
          "session_id": ...,
          "pack": "developer",
          "vocabulary": [<predicate>, ...],            # sorted
          "subjects": {
            "<subject>": {
              "facts": [ {subject, predicate, value, status,
                          confidence, provenance:{source_turn_id,
                          char_start, char_end, quote}}, ... ],
              "gaps":  [<predicate with no active claim>, ...]  # sorted
            }, ...
          }
        }

    Deterministic and LLM-free: facts come straight from
    ``rebuild_active_projection``; gaps are ``vocabulary - filled`` per subject.
    """
    pack = active_pack()
    vocabulary = sorted(pack.predicate_families)

    projection = rebuild_active_projection(conn, session_id)

    # Group active claims by subject, preserving a stable order.
    by_subject: dict[str, list[Claim]] = {}
    for claim in projection.claims:
        by_subject.setdefault(claim.subject, []).append(claim)

    # Small turn cache so we resolve each source turn at most once.
    turn_cache: dict[str, Turn | None] = {}

    def _turn(turn_id: str) -> Turn | None:
        if turn_id not in turn_cache:
            turn_cache[turn_id] = get_turn(conn, turn_id)
        return turn_cache[turn_id]

    subjects: dict[str, dict] = {}
    for subject in sorted(by_subject):
        claims = sorted(by_subject[subject], key=lambda c: (c.predicate, c.created_ts))
        facts = [_fact(_turn(c.source_turn_id), c) for c in claims]
        filled = {c.predicate for c in claims}
        gaps = sorted(p for p in pack.predicate_families if p not in filled)
        subjects[subject] = {"facts": facts, "gaps": gaps}

    return {
        "session_id": session_id,
        "pack": pack.pack_id,
        "vocabulary": vocabulary,
        "subjects": subjects,
    }
