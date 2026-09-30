"""Memory read model — preserved evidence plus the structured state derived from it.

A Memory is one stored episode (the ``turns`` row): its verbatim content, where
it came from, and how much that source is trusted. Claims are derived from a
Memory (``claims.source_turn_id``) and keep owning state transitions; the
Memory only *reports* what became of them, so claim supersession semantics are
never applied to evidence.

Provenance chain: Claim --source_turn_id--> Memory --source_type/metadata--> Source.

No table of its own: a Memory is one-per-turn (HAR-95 slice 1), so a separate
store would mirror ``turns`` 1:1. Revisit when a Memory can outgrow a turn.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from memcontext.source_trust import QUARANTINE_THRESHOLD, trust_for_source

MemoryState = Literal["current", "historical", "mixed", "no_claims"]

CURRENT_STATUSES = frozenset({"active", "confirmed", "audited"})
HISTORICAL_STATUSES = frozenset({"superseded", "dismissed"})


@dataclass(frozen=True, slots=True)
class DerivedClaim:
    """A claim as seen from the Memory it was derived from."""

    claim_id: str
    fact: str
    status: str
    trust: float  # the claim's stored source trust (claim_metadata), not recomputed

    @property
    def quarantined(self) -> bool:
        return self.trust < QUARANTINE_THRESHOLD


@dataclass(frozen=True, slots=True)
class Memory:
    memory_id: str  # == turn_id
    session_id: str
    content: str
    speaker: str
    source_type: str
    source_metadata: str | None
    ts: int
    trust: float
    claims: tuple[DerivedClaim, ...]

    @property
    def quarantined(self) -> bool:
        return self.trust < QUARANTINE_THRESHOLD

    @property
    def state(self) -> MemoryState:
        """Whether the state derived from this evidence is still current.

        Draft claims are not yet state, so they count toward neither side.
        """
        statuses = {c.status for c in self.claims}
        current = bool(statuses & CURRENT_STATUSES)
        historical = bool(statuses & HISTORICAL_STATUSES)
        if current and historical:
            return "mixed"
        if current:
            return "current"
        if historical:
            return "historical"
        return "no_claims"


def _claim_fact(row: sqlite3.Row) -> str:
    if row["text"]:
        return row["text"]
    return " ".join(p for p in (row["subject"], row["predicate"], row["value"]) if p)


def get_memories(conn: sqlite3.Connection, memory_ids: Iterable[str]) -> dict[str, Memory]:
    """Batch-load Memories by id; unknown ids are omitted."""
    ids = list(dict.fromkeys(memory_ids))
    if not ids:
        return {}
    ph = ",".join("?" for _ in ids)
    turn_rows = conn.execute(
        f"SELECT turn_id, session_id, text, speaker, source_type, source_metadata, ts"
        f" FROM turns WHERE turn_id IN ({ph})", ids,
    ).fetchall()
    claims_by_turn: dict[str, list[DerivedClaim]] = {}
    for r in conn.execute(
        f"SELECT c.claim_id, c.source_turn_id, c.text, c.subject, c.predicate, c.value,"
        f" c.status, COALESCE(m.source_trust, 0.5) AS trust"
        # scope: a claim belongs to its turn's session, so a stray row can never pull
        # another session's claim under this Memory (defense in depth for callers
        # that did not come through the namespace-gated query path)
        f" FROM claims c JOIN turns t ON t.turn_id = c.source_turn_id"
        f" AND t.session_id = c.session_id"
        f" LEFT JOIN claim_metadata m ON m.claim_id = c.claim_id"
        f" WHERE c.source_turn_id IN ({ph}) ORDER BY c.created_ts, c.claim_id", ids,
    ).fetchall():
        claims_by_turn.setdefault(r["source_turn_id"], []).append(DerivedClaim(
            claim_id=r["claim_id"], fact=_claim_fact(r), status=r["status"], trust=float(r["trust"]),
        ))
    return {
        t["turn_id"]: Memory(
            memory_id=t["turn_id"],
            session_id=t["session_id"],
            content=t["text"],
            speaker=t["speaker"],
            source_type=t["source_type"],
            source_metadata=t["source_metadata"],
            ts=t["ts"],
            trust=trust_for_source(t["source_type"], t["speaker"]),
            claims=tuple(claims_by_turn.get(t["turn_id"], ())),
        )
        for t in turn_rows
    }


def get_memory(conn: sqlite3.Connection, memory_id: str) -> Memory | None:
    """One Memory by id (== turn_id), or None if no such turn exists."""
    return get_memories(conn, [memory_id]).get(memory_id)


def memory_for_claim(conn: sqlite3.Connection, claim_id: str) -> Memory | None:
    """The Memory a claim was derived from, or None for an unknown claim."""
    row = conn.execute(
        "SELECT source_turn_id FROM claims WHERE claim_id = ?", (claim_id,)
    ).fetchone()
    return get_memory(conn, row[0]) if row else None
