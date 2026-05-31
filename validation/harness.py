"""Block B ingestion + projection capture for the validation experiment.

Block B drives MemContext exactly as a well-behaved host AI should: plain facts
go through ``memory_store``; a correction goes through the typed ``memory_correct``
so the prior value is superseded and the projection holds the current truth.

Nothing here calls an external API — the only AI is the host model that later
answers the probes.
"""
from __future__ import annotations

import sqlite3

from validation.task import SESSION_ID, Turn


def ingest_block_b(conn: sqlite3.Connection, turns: list[Turn], upto: int | None = None) -> None:
    """Ingest the first ``upto`` turns into MemContext (all turns if None)."""
    from memcontext.claims import find_same_identity_claim
    from memcontext.mcp_tools import handle_memory_correct, handle_memory_store

    for turn in turns[: upto if upto is not None else len(turns)]:
        if turn.correction:
            head = find_same_identity_claim(
                conn,
                session_id=SESSION_ID,
                subject=turn.correction["subject"],
                predicate=turn.correction["predicate"],
            )
            if head is not None:
                handle_memory_correct(
                    conn,
                    claim_id=head.claim_id,
                    action="correct",
                    new_value=turn.correction["new_value"],
                )
                continue
            # No prior claim to supersede → fall through and store as a new fact.
        handle_memory_store(
            conn,
            text=turn.text,
            speaker=turn.speaker,
            session_id=SESSION_ID,
            claims=[dict(c) for c in turn.claims] or None,
        )


def projection_state(conn: sqlite3.Connection, subject: str, predicate: str) -> dict:
    """Deterministic projection for a (subject, predicate) slot: current value +
    provenance. This is exactly what an attached MemContext serves the host.
    """
    from memcontext.claims import find_same_identity_claim, get_turn

    head = find_same_identity_claim(
        conn, session_id=SESSION_ID, subject=subject, predicate=predicate
    )
    if head is None:
        return {"current_value": None, "status": None, "source_turn_id": None, "provenance": None}
    turn = get_turn(conn, head.source_turn_id)
    return {
        "current_value": head.value,
        "status": head.status.value,
        "source_turn_id": head.source_turn_id,
        "provenance": turn.text if turn is not None else None,
    }


def raw_transcript(turns: list[Turn], upto: int) -> str:
    """The plain conversation so far — the only context Block A (native memory) has."""
    return "\n".join(
        f"[S{t.session}] {t.speaker}: {t.text}" for t in turns[:upto]
    )
