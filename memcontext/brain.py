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
from collections.abc import Sequence

from memcontext.claims import get_turn, row_to_claim
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


# --------------------------------------------------------- cross-session ---


def _recency_key(c: Claim) -> tuple[int, int]:
    """Sort key for picking the CURRENT claim for a slot across sessions.

    Ordered by the fact's effective onset (``valid_from_ts``, falling back to
    ``created_ts``) then by ``created_ts`` as a tiebreak. The MAX key is the
    most-recent assertion for that slot — the value the tenant currently holds.
    """
    onset = c.valid_from_ts if c.valid_from_ts is not None else c.created_ts
    return (onset, c.created_ts)


def brain_across(
    conn: sqlite3.Connection,
    *,
    session_ids: "str | Sequence[str]",
) -> dict:
    """Cross-session resolved world-state — one current value per slot per subject.

    Fracture-A fix: ``brain`` only ever projects a SINGLE session, so the
    cross-session serve path (``session_id=None`` — a tenant's whole history,
    where the product keeps one session per ingested document/conversation) got
    no resolved layer, only a top-k dump. This projects the resolved world-state
    over a SET of sessions.

    The hard part is conflict resolution: supersession runs per-session, so when
    the same ``(subject, predicate)`` slot was asserted in two different sessions
    (e.g. the user said "I live in Boston" early, then "I moved to Seattle"
    later, in separate conversations), BOTH rows remain ``active`` in their own
    session. A naive union would surface two values for one slot. Here, for each
    slot we keep ONLY the most-recent assertion (``_recency_key``) — the stale
    earlier value is dropped from the resolved view, exactly as intra-session
    supersession would have dropped it. Older instances are still in the store
    (queryable via the raw ranked channel / history intent); they are simply not
    the resolved current truth.

    Same shape as :func:`brain`, with ``session_id`` reported as the id list and a
    ``sessions`` count. Deterministic, zero-LLM: reads active claims only.
    """
    if isinstance(session_ids, str):
        sids = [session_ids]
    else:
        seen: set[str] = set()
        sids = []
        for s in session_ids:
            if s and s not in seen:
                seen.add(s)
                sids.append(s)

    pack = active_pack()
    vocabulary = sorted(pack.predicate_families)

    if not sids:
        return {
            "session_id": [],
            "sessions": 0,
            "pack": pack.pack_id,
            "vocabulary": vocabulary,
            "subjects": {},
        }

    sid_ph = ",".join("?" for _ in sids)
    rows = conn.execute(
        f"SELECT * FROM claims WHERE session_id IN ({sid_ph})"
        " AND status IN ('active','confirmed','audited')"
        " ORDER BY created_ts ASC",
        sids,
    ).fetchall()
    claims = [row_to_claim(r) for r in rows]

    # Resolve to one claim per (subject, predicate) slot — most-recent wins. NL-only
    # facts (empty predicate) carry no slot identity, so they are NOT collapsed:
    # each is kept (keyed by its own claim_id) the way a top-k channel would, while
    # structured slots resolve to the single current value.
    resolved: dict[tuple[str, str], Claim] = {}
    for c in claims:
        if not c.subject or not c.predicate:
            resolved[("", c.claim_id)] = c  # NL-only: never fuse, never overwrite
            continue
        key = (c.subject, c.predicate)
        prev = resolved.get(key)
        if prev is None or _recency_key(c) >= _recency_key(prev):
            resolved[key] = c

    by_subject: dict[str, list[Claim]] = {}
    for c in resolved.values():
        by_subject.setdefault(c.subject, []).append(c)

    turn_cache: dict[str, Turn | None] = {}

    def _turn(turn_id: str) -> Turn | None:
        if turn_id not in turn_cache:
            turn_cache[turn_id] = get_turn(conn, turn_id)
        return turn_cache[turn_id]

    subjects: dict[str, dict] = {}
    for subject in sorted(by_subject):
        s_claims = sorted(by_subject[subject], key=lambda c: (c.predicate, c.created_ts))
        facts = [_fact(_turn(c.source_turn_id), c) for c in s_claims]
        filled = {c.predicate for c in s_claims if c.predicate}
        gaps = sorted(p for p in pack.predicate_families if p not in filled)
        subjects[subject] = {"facts": facts, "gaps": gaps}

    return {
        "session_id": sids,
        "sessions": len(sids),
        "pack": pack.pack_id,
        "vocabulary": vocabulary,
        "subjects": subjects,
    }
