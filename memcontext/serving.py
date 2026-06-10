"""Served context — compose the resolved world-state + briefing into one answer.

The structured layer (resolved world-state via ``brain``, the profile briefing)
is built and stored, but historically reachable only through separate MCP tools.
This module composes it into the response an agent already gets from a query, so
the agent receives RESOLVED current truth + a session briefing + provenance by
default — not just a raw top-k dump it has to reconcile itself.

Built FRESH at serve time, so it always reflects the current resolved truth (this
also closes the "profile only rebuilt every 10th turn" staleness gap without
adding O(n) work to every ingest). Zero LLM, zero network.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

import structlog

from memcontext.brain import brain

log = structlog.get_logger(__name__)


def session_briefing(
    conn: sqlite3.Connection, *, subject: str = "user", max_tokens: int = 400
) -> str | None:
    """A compact, CURRENT profile briefing for a subject (fresh build).

    Returns the formatted profile text, or None if there's nothing to brief or the
    build fails — a briefing must never break the query that asked for it.
    """
    try:
        from memcontext.profiles import build_smart_profile, format_profile

        profile = build_smart_profile(conn, subject, max_tokens=max_tokens)
        text = format_profile(profile)
        return text or None
    except Exception:  # noqa: BLE001 — briefing is best-effort, never fatal
        log.warning("substrate.session_briefing_failed", subject=subject)
        return None


def resolved_entity_links(conn: sqlite3.Connection, session_id: str) -> dict:
    """entity_key -> sorted co-occurrence neighbors for the session.

    A read-only VIEW over claims exposing the connective structure that was
    previously reachable only through the entity-graph tool. Deterministic; it does
    NOT participate in ranking (CLAUDE.md keeps graph traversal out of retrieval).
    """
    try:
        from memcontext.entity_graph import EntityGraph

        graph = EntityGraph(conn, session_id)
        return {ek: sorted(graph.neighbors(ek)) for ek in graph.entities}
    except Exception:  # noqa: BLE001 — best-effort view, never fatal
        log.warning("substrate.entity_links_failed", session_id=session_id)
        return {}


@dataclass(frozen=True, slots=True)
class ContextBriefing:
    """Everything an agent needs at session start, in one object.

    - ``world_state``: resolved current truth grouped by subject, each fact with a
      source span + a typed list of what it superseded (from ``brain``).
    - ``briefing``: compact profile text for the subject (or None).
    - ``hits``: query-relevant (MemoryHit, score) results (empty if no query).
    - ``why``: claim_id -> ClaimExplanation for each served FACT (provenance).
    """

    session_id: str
    world_state: dict
    briefing: str | None
    hits: tuple[Any, ...]
    why: dict
    # entity_key -> sorted co-occurrence neighbors. A read-only VIEW over claims that
    # exposes the connective spine in the serve path; it is NOT a retrieval/ranking
    # channel (CLAUDE.md keeps graph traversal out of ranking).
    entity_links: dict


def build_context_briefing(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    query: str = "",
    subject: str = "user",
    top_k: int = 10,
    embedding_client: Any | None = None,
) -> ContextBriefing:
    """One call → resolved world-state + briefing + query hits + provenance.

    This is the connective spine of the serve path: it pulls together the four
    pieces that used to be reachable only via separate tools (brain, profile,
    retrieve_memory, provenance) so a caller gets resolved current memory in one
    shot. Deterministic, zero-LLM.
    """
    from memcontext.provenance import explain_claim
    from memcontext.retrieval import retrieve_memory

    world_state = brain(conn, session_id=session_id)
    briefing = session_briefing(conn, subject=subject)
    entity_links = resolved_entity_links(conn, session_id)

    hits: tuple[Any, ...] = ()
    why: dict = {}
    if query and query.strip():
        hits = tuple(
            retrieve_memory(
                conn, session_id=session_id, query=query, top_k=top_k,
                embedding_client=embedding_client,
            )
        )
        for hit, _score in hits:
            if hit.kind == "fact":
                ex = explain_claim(conn, hit.id)
                if ex is not None:
                    why[hit.id] = ex

    return ContextBriefing(
        session_id=session_id,
        world_state=world_state,
        briefing=briefing,
        hits=hits,
        why=why,
        entity_links=entity_links,
    )
