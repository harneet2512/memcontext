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
    conn: sqlite3.Connection, *, subject: str = "user", max_tokens: int = 400,
    namespace: str | None = None,
) -> str | None:
    """A compact, CURRENT profile briefing for a subject (fresh build).

    Returns the formatted profile text, or None if there's nothing to brief or the
    build fails — a briefing must never break the query that asked for it. When
    ``namespace`` is given the profile is scoped to that tenant (no cross-namespace leak).
    """
    try:
        from memcontext.profiles import build_smart_profile, format_profile

        profile = build_smart_profile(conn, subject, max_tokens=max_tokens, namespace=namespace)
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
    # cached session digest (top facts + supersession updates) for a quick summary, or None.
    digest: dict | None
    # claim_id -> {trust, quarantined, consolidated} for each served FACT hit. Parity
    # with the MCP query door: a library caller must get the same safety surface so it
    # never acts on low-trust / poisoned memory unknowingly.
    fact_trust: dict
    # episodic layer: assembled multi-slot event frames (purchases/trips/...) ranked by
    # query when embeddings exist, else most-recent; and detected life-event bursts.
    events: tuple[Any, ...]
    life_events: tuple[Any, ...]


def _load_digest_dict(conn: sqlite3.Connection, session_id: str) -> dict | None:
    """Cached session digest as a plain dict, or None. Best-effort."""
    try:
        from memcontext.digests import load_digest

        d = load_digest(conn, session_id)
        if d is None:
            return None
        return {"key_facts": d.key_facts, "updates": d.updates,
                "remaining_count": d.remaining_count, "total_claims": d.total_claims}
    except Exception:  # noqa: BLE001 — summary is best-effort
        return None


def _trust_map(conn: sqlite3.Connection, claim_ids: list[str]) -> dict:
    """claim_id -> {trust, quarantined, consolidated}, mirroring handle_memory_query."""
    if not claim_ids:
        return {}
    try:
        from memcontext.source_trust import QUARANTINE_THRESHOLD
    except Exception:  # noqa: BLE001
        QUARANTINE_THRESHOLD = 0.4
    ph = ",".join("?" for _ in claim_ids)
    rows = conn.execute(
        f"SELECT claim_id, COALESCE(consolidated, 0), COALESCE(source_trust, 0.5)"
        f" FROM claim_metadata WHERE claim_id IN ({ph})",
        claim_ids,
    ).fetchall()
    out: dict = {}
    for r in rows:
        trust = float(r[2])
        out[r[0]] = {"trust": round(trust, 3),
                     "quarantined": trust < QUARANTINE_THRESHOLD,
                     "consolidated": bool(r[1])}
    return out


def _frame_to_dict(f: Any) -> dict:
    return {"event_type": f.event_type, "item": f.item, "location": f.location,
            "time": f.time_expr, "amount": f.amount, "participants": list(f.participants),
            "missing_slots": list(f.missing_slots), "confidence": f.confidence,
            "supporting_claim_ids": list(f.supporting_claim_ids)}


def serve_event_frames(
    conn: sqlite3.Connection, *, session_id: str, query: str = "",
    embedding_client: Any | None = None, top_k: int = 8,
) -> list[dict]:
    """Event frames for the session — ranked by query when embeddings exist, else all.

    Gives the dead `retrieve_event_frames` a real caller; falls back to the full list
    in lexical-only mode (no frame embeddings). Best-effort.
    """
    try:
        from memcontext.event_frames import list_event_frames
        from memcontext.retrieval import retrieve_event_frames

        if query and query.strip():
            ranked = retrieve_event_frames(conn, session_id=session_id, query=query,
                                           top_k=top_k, embedding_client=embedding_client)
            if ranked:
                return [_frame_to_dict(f) for f, _ in ranked]
        return [_frame_to_dict(f) for f in list_event_frames(conn, session_id)[:top_k]]
    except Exception:  # noqa: BLE001 — episodic layer is additive, never fatal
        return []


def serve_life_events(
    conn: sqlite3.Connection, *, subject: str = "user", limit: int = 5,
    namespace: str | None = None,
) -> list[dict]:
    """Detected life-event bursts for a subject, most significant first. Best-effort.

    When ``namespace`` is given, RE-DETECT scoped to that tenant rather than reading
    the global ``life_events`` cache (which is subject-keyed, not namespace-keyed) —
    so one tenant's briefing never surfaces another tenant's life events.
    """
    try:
        if namespace is not None:
            from memcontext.life_events import detect_life_events

            evs = detect_life_events(conn, subject, namespace=namespace)
            evs.sort(key=lambda e: (-e.significance, -e.timestamp_start))
            return [{"summary": e.summary_text, "significance": e.significance,
                     "predicates": list(e.predicates_affected)} for e in evs[:limit]]
        rows = conn.execute(
            "SELECT summary_text, significance, predicates_affected FROM life_events"
            " WHERE subject = ? ORDER BY significance DESC, timestamp_start DESC LIMIT ?",
            (subject, limit),
        ).fetchall()
        return [{"summary": r[0], "significance": r[1],
                 "predicates": (r[2] or "").split(",") if r[2] else []} for r in rows]
    except Exception:  # noqa: BLE001
        return []


def build_context_briefing(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    query: str = "",
    subject: str = "user",
    top_k: int = 10,
    embedding_client: Any | None = None,
    namespace: str | None = None,
) -> ContextBriefing:
    """One call → resolved world-state + briefing + digest + hits + provenance + trust.

    This is the connective spine of the serve path: it pulls together the pieces
    that used to be reachable only via separate tools (brain, profile, digest,
    retrieve_memory, provenance, trust) so a caller gets resolved current memory —
    WITH its safety surface — in one shot. Deterministic, zero-LLM.

    Pass ``namespace`` in a multi-tenant deployment so the subject-keyed profile and
    life-events are scoped to that tenant (world-state/hits/events are already
    session-scoped). Leave it None for a single-tenant personal brain.
    """
    from memcontext.provenance import explain_claim
    from memcontext.retrieval import retrieve_memory

    world_state = brain(conn, session_id=session_id)
    briefing = session_briefing(conn, subject=subject, namespace=namespace)
    entity_links = resolved_entity_links(conn, session_id)
    digest = _load_digest_dict(conn, session_id)

    hits: tuple[Any, ...] = ()
    why: dict = {}
    fact_trust: dict = {}
    if query and query.strip():
        hits = tuple(
            retrieve_memory(
                conn, session_id=session_id, query=query, top_k=top_k,
                embedding_client=embedding_client,
            )
        )
        fact_trust = _trust_map(conn, [h.id for h, _ in hits if h.kind == "fact"])
        for hit, _score in hits:
            if hit.kind == "fact":
                ex = explain_claim(conn, hit.id)
                if ex is not None:
                    why[hit.id] = ex

    events = tuple(serve_event_frames(
        conn, session_id=session_id, query=query, embedding_client=embedding_client))
    life_events = tuple(serve_life_events(conn, subject=subject, namespace=namespace))

    return ContextBriefing(
        session_id=session_id,
        world_state=world_state,
        briefing=briefing,
        hits=hits,
        why=why,
        entity_links=entity_links,
        digest=digest,
        fact_trust=fact_trust,
        events=events,
        life_events=life_events,
    )
