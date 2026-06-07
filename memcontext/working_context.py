"""Working context — assemble the task-relevant slice of memory for the current
session within a token budget, derived from the working state (the recent turns),
instead of dumping all active memory.

Query-free: the recent turns ARE the retrieval cue, so an agent can ask for "the
working memory for this session, within N tokens" without formulating a query.
Deterministic, zero-LLM. Beats "return all active memory" on precision + tokens:
it scopes to what the session is currently about and stops at the budget.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from memcontext.retrieval import MemoryHit, retrieve_memory


def _toks(text: str) -> int:
    return max(1, len(text or "") // 4)


@dataclass(slots=True)
class CurrentContext:
    """The working memory assembled for a session at a point in time."""

    session_id: str
    recent_turn_ids: list[str]
    salient_entities: list[str]
    facts: list[tuple[MemoryHit, float]]
    token_budget: int
    tokens_used: int
    total_active: int
    included: int
    excluded_for_budget: int


def build_working_context(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    token_budget: int = 2000,
    recent_turns: int = 5,
    candidate_k: int = 50,
) -> CurrentContext:
    """Assemble task-relevant memory for ``session_id`` within ``token_budget``,
    cued by the last ``recent_turns`` turns. Superseded/demoted facts are already
    excluded by the unified retrieval path. Deterministic, zero-LLM.
    """
    from memcontext.claims import row_to_turn
    from memcontext.entities import extract_entities

    rows = conn.execute(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
        (session_id, recent_turns),
    ).fetchall()
    turns = [row_to_turn(r) for r in rows]
    recent_text = " ".join(t.text for t in turns if t.text)
    salient = sorted({
        e.text.lower() for t in turns for e in extract_entities(t.text or "")
    })

    hits = (
        retrieve_memory(conn, session_id=session_id, query=recent_text, top_k=candidate_k)
        if recent_text.strip() else []
    )

    # Greedy pack to the token budget (skip an over-budget item, keep filling with
    # what fits) so the working set is bounded — not the whole active store.
    packed: list[tuple[MemoryHit, float]] = []
    used = 0
    for hit, score in hits:
        cost = _toks(hit.text)
        if used + cost > token_budget:
            continue
        packed.append((hit, score))
        used += cost

    total_active = conn.execute(
        "SELECT COUNT(*) FROM claims"
        " WHERE session_id = ? AND status IN ('active','confirmed','audited')",
        (session_id,),
    ).fetchone()[0]

    return CurrentContext(
        session_id=session_id,
        recent_turn_ids=[t.turn_id for t in turns],
        salient_entities=salient,
        facts=packed,
        token_budget=token_budget,
        tokens_used=used,
        total_active=int(total_active),
        included=len(packed),
        excluded_for_budget=len(hits) - len(packed),
    )
