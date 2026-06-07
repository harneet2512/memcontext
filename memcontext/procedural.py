"""Procedural memory (EXPERIMENTAL) — detect recurring ordered action sequences
across sessions, where an action is a claim's predicate (EventTuple.action) and a
session's order is its claims by time.

OFF by default (``MEMCONTEXT_EXPERIMENTAL_PROCEDURAL=1`` to enable). This is the
plan's most speculative capability: a conversational fact store has states more
than workflows, so a "procedure" here is a recurring predicate sequence. It stays
flag-gated until it beats a synthetic procedural harness. Deterministic, zero-LLM.
"""
from __future__ import annotations

import os
import sqlite3
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

EXPERIMENTAL_FLAG = "MEMCONTEXT_EXPERIMENTAL_PROCEDURAL"


def procedural_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_FLAG, "") == "1"


@dataclass(slots=True)
class Procedure:
    """A recurring ordered action sequence with provenance to its source claims."""

    steps: tuple[str, ...]
    trigger: str
    recurrence: int
    sessions: list[str]
    source_claim_ids: list[str]


def _is_subsequence(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    if len(short) >= len(long):
        return False
    return any(long[i:i + len(short)] == short for i in range(len(long) - len(short) + 1))


def detect_procedures(
    conn: sqlite3.Connection,
    *,
    min_sessions: int = 2,
    min_steps: int = 3,
    max_steps: int = 5,
) -> list[Procedure]:
    """Recurring ordered predicate n-grams across >= ``min_sessions`` sessions.
    Pure function (callable regardless of the flag, for tests / the eval harness).
    """
    rows = conn.execute(
        "SELECT session_id, claim_id, predicate FROM claims"
        " WHERE status IN ('active','confirmed','audited') AND predicate IS NOT NULL"
        " ORDER BY session_id, created_ts ASC, claim_id ASC"
    ).fetchall()

    by_session: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
    for r in rows:
        by_session.setdefault(r["session_id"], []).append((r["predicate"], r["claim_id"]))

    gram_sessions: dict[tuple[str, ...], set[str]] = defaultdict(set)
    gram_claims: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for sid, seq in by_session.items():
        actions = [a for a, _ in seq]
        cids = [c for _, c in seq]
        for k in range(min_steps, min(max_steps, len(actions)) + 1):
            for i in range(len(actions) - k + 1):
                gram = tuple(actions[i:i + k])
                gram_sessions[gram].add(sid)
                gram_claims[gram].extend(cids[i:i + k])

    procs: list[Procedure] = []
    for gram, sids in gram_sessions.items():
        if len(sids) >= min_sessions:
            procs.append(Procedure(
                steps=gram,
                trigger=gram[0],
                recurrence=len(sids),
                sessions=sorted(sids),
                source_claim_ids=list(dict.fromkeys(gram_claims[gram])),
            ))

    # Prefer longer, more-recurrent procedures; drop a procedure subsumed (as a
    # contiguous sub-sequence) by a longer one of at least equal recurrence.
    procs.sort(key=lambda p: (-len(p.steps), -p.recurrence))
    kept: list[Procedure] = []
    for p in procs:
        if any(_is_subsequence(p.steps, q.steps) and p.recurrence <= q.recurrence for q in kept):
            continue
        kept.append(p)
    kept.sort(key=lambda p: (-p.recurrence, -len(p.steps)))
    return kept
