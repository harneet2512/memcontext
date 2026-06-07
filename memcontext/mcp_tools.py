"""MCP tool handler functions — pure business logic, no MCP protocol dependency.

Each function takes a sqlite3 Connection and keyword arguments, returns a dict.
These are usable from CLI, tests, or the MCP server without importing mcp.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from typing import TYPE_CHECKING

from memcontext.brain import brain
from memcontext.claims import (
    find_same_identity_claim,
    get_claim,
    get_superseded_by,
    get_turn,
    insert_fact,
    list_active_claims,
    row_to_claim,
    set_claim_status,
)
from memcontext.extractors import PassthroughExtractor, auto_extractor
from memcontext.on_new_turn import on_new_turn
from memcontext.provenance import span_for_claim
from memcontext.schema import ClaimStatus, EdgeType, Speaker
from memcontext.supersession import write_supersession_edge

if TYPE_CHECKING:
    from memcontext.extraction_queue import ExtractionQueue
    from memcontext.on_new_turn import ExtractorFn


def handle_memory_store(
    conn: sqlite3.Connection,
    *,
    text: str,
    speaker: str = "user",
    session_id: str | None = None,
    claims: list[dict] | None = None,
    entities: list[dict] | None = None,
    extractor: ExtractorFn | None = None,
    queue: ExtractionQueue | None = None,
) -> dict:
    sid = session_id or f"session_{uuid.uuid4().hex[:8]}"
    sp = Speaker.USER if speaker == "user" else Speaker.ASSISTANT

    # Pre-structured claims extract inline (Passthrough is never deferred). With
    # no claims, use the server-injected persistent extractor + queue when
    # provided, so a deferrable (LLM) extractor runs async off the write path.
    if claims:
        ext: ExtractorFn = PassthroughExtractor(claims)
        q: ExtractionQueue | None = None
    else:
        ext = extractor or auto_extractor()
        q = queue

    from memcontext.retrieval import episode_embedder, semantic_supersession
    result = on_new_turn(
        conn, session_id=sid, speaker=sp, text=text, extractor=ext,
        queue=q, embedder=episode_embedder(), semantic=semantic_supersession(),
    )

    if entities and result.created_claims:
        rows = [
            (claim.claim_id, ent["text"].lower(), ent.get("type", "proper_noun"))
            for ent in entities
            if ent.get("text")
            for claim in result.created_claims
        ]
        if rows:
            conn.executemany(
                "INSERT OR IGNORE INTO claim_entities (claim_id, entity_text, entity_type)"
                " VALUES (?, ?, ?)",
                rows,
            )

    return {
        "turn_id": result.turn.turn_id if result.turn else None,
        "session_id": sid,
        "admitted": result.admitted,
        "claims_created": len(result.created_claims),
        "claim_ids": [c.claim_id for c in result.created_claims],
        "supersessions": len(result.supersession_edges),
    }


def handle_memory_query(
    conn: sqlite3.Connection,
    *,
    query: str,
    session_id: str | None = None,
    top_k: int = 10,
    debug: bool = False,
) -> dict:
    from memcontext.claims import get_claim, get_turn
    from memcontext.retrieval import (
        bump_access,
        classify_query_depth,
        classify_query_predicates,
        detect_history_intent,
        retrieve_memory,
        retrieve_memory_across,
    )

    explain: dict[str, dict[str, float]] | None = {} if debug else None
    # Temporal truth: a query about the PAST ("what was X before") surfaces
    # superseded facts; otherwise only current (active) facts are served.
    history = detect_history_intent(query)
    _, query_type = classify_query_predicates(query)
    if top_k == 10:
        _, top_k = classify_query_depth(query)

    # Unified two-tier retrieval (facts + episodes, source-tagged, RRF-fused).
    if session_id:
        hits = retrieve_memory(
            conn, session_id=session_id, query=query, top_k=top_k, explain=explain,
            include_superseded=history,
        )
        total = len(list_active_claims(conn, session_id))
    else:
        # Every session that has episodes — episodes exist even when a session's
        # facts are absent/pending (the Tier-1 floor), so scope by turns, not claims.
        rows = conn.execute("SELECT DISTINCT session_id FROM turns").fetchall()
        sids = [r["session_id"] if isinstance(r, sqlite3.Row) else r[0] for r in rows]
        if not sids:
            return {"claims": [], "episodes": [], "total": 0}
        hits = retrieve_memory_across(
            conn, session_ids=sids, query=query, top_k=top_k, explain=explain,
            include_superseded=history,
        )
        total = conn.execute(
            "SELECT COUNT(*) FROM claims"
            " WHERE status IN ('active','confirmed','audited')"
        ).fetchone()[0]

    max_score = hits[0][1] if hits and hits[0][1] > 0 else 1.0

    # Split the unified ranking back into source-tagged channels, preserving the
    # fused score so a consumer can re-merge by score if it wants one stream.
    claims_out: list[dict] = []
    episodes_out: list[dict] = []
    for hit, s in hits:
        norm = round(s / max_score, 4) if s > 0 else 0.0
        if hit.kind == "fact":
            c = get_claim(conn, hit.id)
            if c is None:
                continue
            claims_out.append({
                "claim_id": c.claim_id,
                "subject": c.subject,
                "predicate": c.predicate,
                "value": c.value,
                "confidence": c.confidence,
                "status": c.status.value,
                "score": norm,
            })
        else:
            t = get_turn(conn, hit.id)
            if t is None:
                continue
            episodes_out.append({
                "turn_id": t.turn_id,
                "text": t.text,
                "source_type": t.source_type.value,
                "score": norm,
            })

    # Usage reinforcement: the fact claims we served are now "accessed".
    bump_access(conn, [c["claim_id"] for c in claims_out])

    # Surface the consolidation marker: a fact graduated from a cross-session
    # recurring pattern is flagged durable, so the agent sees it isn't one-off.
    if claims_out:
        _cids = [c["claim_id"] for c in claims_out]
        _ph = ",".join("?" for _ in _cids)
        _consolidated = {
            r[0] for r in conn.execute(
                f"SELECT claim_id FROM claim_metadata"
                f" WHERE consolidated = 1 AND claim_id IN ({_ph})", _cids,
            ).fetchall()
        }
        for c in claims_out:
            c["consolidated"] = c["claim_id"] in _consolidated

    # Token accounting (zero-LLM, ~chars/4) for what we serve, by source type.
    def _toks(text: str) -> int:
        return max(1, len(text or "") // 4)
    fact_tokens = sum(_toks(c.get("value") or "") for c in claims_out)
    episode_tokens = sum(_toks(e.get("text") or "") for e in episodes_out)
    token_report = {
        "fact_tokens": fact_tokens,
        "episode_tokens": episode_tokens,
        "total_tokens": fact_tokens + episode_tokens,
        "served_items": len(claims_out) + len(episodes_out),
    }

    _READER_HINTS = {
        "assistant_recall": "Answer based on what the assistant previously said, recommended, or did.",
        "preference": "State the user's preference directly. If preferences changed, use the most recent.",
        "temporal": "Pay attention to dates and time ordering in the facts.",
        "knowledge_update": "Facts may have changed over time. Answer based on the most recent active version.",
        "fact_recall": "Answer directly from the retrieved facts.",
    }

    result: dict = {
        "claims": claims_out,
        "episodes": episodes_out,
        "total": total,
        "query_type": query_type,
        "reader_hint": _READER_HINTS.get(query_type, _READER_HINTS["fact_recall"]),
        "token_report": token_report,
    }
    if debug and explain is not None:
        served = [c["claim_id"] for c in claims_out]
        result["ranking"] = {cid: explain[cid] for cid in served if cid in explain}
    return result


def handle_memory_working_context(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    token_budget: int = 2000,
) -> dict:
    """Working context: the task-relevant memory for a session within a token
    budget, cued by recent turns (query-free) rather than all active memory."""
    from memcontext.working_context import build_working_context

    ctx = build_working_context(conn, session_id, token_budget=token_budget)
    return {
        "session_id": ctx.session_id,
        "recent_turn_ids": ctx.recent_turn_ids,
        "salient_entities": ctx.salient_entities,
        "facts": [
            {"kind": h.kind, "id": h.id, "text": h.text, "score": round(s, 4)}
            for h, s in ctx.facts
        ],
        "token_budget": ctx.token_budget,
        "tokens_used": ctx.tokens_used,
        "total_active": ctx.total_active,
        "included": ctx.included,
        "excluded_for_budget": ctx.excluded_for_budget,
    }


def handle_memory_procedures(
    conn: sqlite3.Connection,
    *,
    min_sessions: int = 2,
) -> dict:
    """Recurring procedures across sessions (EXPERIMENTAL; off unless the flag is
    set). Surfaces detected ordered action sequences + their provenance."""
    from memcontext.procedural import (
        EXPERIMENTAL_FLAG,
        detect_procedures,
        procedural_enabled,
    )

    if not procedural_enabled():
        return {"enabled": False, "procedures": [],
                "note": f"experimental; set {EXPERIMENTAL_FLAG}=1 to enable"}
    procs = detect_procedures(conn, min_sessions=min_sessions)
    return {
        "enabled": True,
        "procedures": [
            {"trigger": p.trigger, "steps": list(p.steps), "recurrence": p.recurrence,
             "sessions": p.sessions, "source_claim_ids": p.source_claim_ids}
            for p in procs
        ],
    }


def handle_memory_output_provenance(
    conn: sqlite3.Connection,
    *,
    session_id: str | None = None,
    record: list[dict] | None = None,
    claim_id: str | None = None,
    turn_id: str | None = None,
    sentence_id: str | None = None,
) -> dict:
    """Output-sentence provenance (audit-first): record which generated sentences
    cite which claims, and trace the bidirectional claim <-> sentence <-> turn
    links. Wires the previously-unreachable provenance functions.
    """
    from memcontext.provenance import (
        OutputSection,
        claim_ids_for_turn,
        insert_output_sentence,
        sentence_ids_for_claim,
        turn_id_for_sentence,
    )

    out: dict = {}
    if record and session_id:
        ids: list[str] = []
        for i, s in enumerate(record):
            row = insert_output_sentence(
                conn, session_id=session_id,
                section=OutputSection(s.get("section", "summary")),
                ordinal=int(s.get("ordinal", i)),
                text=s.get("text", ""),
                source_claim_ids=list(s.get("source_claim_ids", [])),
            )
            ids.append(row.sentence_id)
        out["recorded"] = ids
    if claim_id:
        out["cited_in"] = sentence_ids_for_claim(conn, claim_id)
    if turn_id:
        out["claims_from_turn"] = claim_ids_for_turn(conn, turn_id)
    if sentence_id:
        out["turn_of_sentence"] = turn_id_for_sentence(conn, sentence_id)
    return out


def handle_memory_forget(
    conn: sqlite3.Connection,
    *,
    claim_id: str | None = None,
    subject: str | None = None,
    session_id: str | None = None,
    predicate: str | None = None,
    reason: str = "user_request",
) -> dict:
    """Right-to-be-forgotten: hard-delete the target memory and cascade along the
    provenance + supersession graph (no residual content), audited to `decisions`.
    Specify exactly one of claim_id / subject / session_id / predicate."""
    from memcontext.forgetting import forget

    return forget(conn, claim_id=claim_id, subject=subject,
                  session_id=session_id, predicate=predicate, reason=reason)


def handle_memory_profile(
    conn: sqlite3.Connection,
    *,
    subject: str = "user",
    max_tokens: int = 500,
) -> dict:
    try:
        from memcontext.profiles import build_smart_profile, format_profile, load_profile, store_profile

        cached = load_profile(conn, subject)
        if cached:
            return {
                "subject": subject,
                "profile_text": format_profile(cached),
                "total_facts": cached.total_facts,
                "total_sessions": cached.total_sessions,
                "total_updates": cached.total_updates,
                "cached": True,
            }

        profile = build_smart_profile(conn, subject, max_tokens=max_tokens)
        store_profile(conn, profile)
        return {
            "subject": subject,
            "profile_text": format_profile(profile),
            "total_facts": profile.total_facts,
            "total_sessions": profile.total_sessions,
            "total_updates": profile.total_updates,
            "cached": False,
        }
    except Exception as exc:
        return {"subject": subject, "error": str(exc)}


def handle_memory_stats(conn: sqlite3.Connection) -> dict:
    active = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE status IN ('active','confirmed','audited')"
    ).fetchone()[0]
    superseded = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE status = 'superseded'"
    ).fetchone()[0]
    turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    profiles = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    digests = conn.execute("SELECT COUNT(*) FROM session_digests").fetchone()[0]
    events = conn.execute("SELECT COUNT(*) FROM life_events").fetchone()[0]

    return {
        "active_claims": active,
        "superseded_claims": superseded,
        "turns": turns,
        "profiles": profiles,
        "session_digests": digests,
        "life_events": events,
        "retrieval_surface": active,
        "provenance_depth": active + superseded,
    }


def handle_memory_digest(conn: sqlite3.Connection, *, session_id: str) -> dict:
    """Build, persist, and serve the session digest — the deterministic summary
    layer: top key facts by importance + supersession updates + remaining count.

    Persisting here is what finally populates the ``session_digests`` table in
    production (the builder had no caller before this tool existed).
    """
    from memcontext.digests import build_session_digest, format_digest, store_digest

    digest = build_session_digest(conn, session_id)
    try:
        store_digest(conn, digest)
    except Exception:  # caching is best-effort; never fail the read on it
        log.warning("mcp.digest_store_failed", session_id=session_id)
    return {
        "session_id": digest.session_id,
        "key_facts": digest.key_facts,
        "updates": digest.updates,
        "remaining_count": digest.remaining_count,
        "total_claims": digest.total_claims,
        "text": format_digest(digest),
    }


def handle_memory_life_events(
    conn: sqlite3.Connection,
    *,
    subject: str = "user",
    window_hours: int = 24,
    min_predicates: int = 3,
) -> dict:
    """Detect, persist, and serve life events — bursts of diverse predicate
    changes for a subject inside a time window. Deterministic, zero-LLM.

    Persisting here finally writes the ``life_events`` table (the detector had
    no caller before this tool). Note: clustering keys on *structured*
    predicates, so NL-only facts (out-of-vocab, predicate=None) don't form
    life events — only in-vocab structured facts do.
    """
    from memcontext.life_events import detect_life_events, store_life_events

    events = detect_life_events(
        conn, subject, window_hours=window_hours, min_predicates=min_predicates
    )
    try:
        store_life_events(conn, events)
    except Exception:  # best-effort cache; never fail the read on it
        log.warning("mcp.life_events_store_failed", subject=subject)
    return {
        "subject": subject,
        "count": len(events),
        "events": [
            {
                "event_id": e.event_id,
                "timestamp_start": e.timestamp_start,
                "timestamp_end": e.timestamp_end,
                "predicates_affected": list(e.predicates_affected),
                "claim_ids": list(e.claim_ids),
                "summary_text": e.summary_text,
                "significance": e.significance,
            }
            for e in events
        ],
    }


def handle_memory_events(conn: sqlite3.Connection, *, session_id: str) -> dict:
    """Assemble, persist, and serve event frames for a session — co-referent
    claims grouped into multi-slot event records (who/what/where/when/amount).
    Deterministic, zero-LLM. ``assemble_event_frames`` self-persists, so this
    finally populates the ``event_frames`` table (it had no caller before).
    """
    from memcontext.event_frames import assemble_event_frames

    frames = assemble_event_frames(conn, session_id)
    return {
        "session_id": session_id,
        "count": len(frames),
        "events": [
            {
                "event_id": f.event_id,
                "event_type": f.event_type,
                "participants": list(f.participants),
                "item": f.item,
                "location": f.location,
                "time_expr": f.time_expr,
                "amount": f.amount,
                "supporting_claim_ids": list(f.supporting_claim_ids),
                "confidence": f.confidence,
                "missing_slots": list(f.missing_slots),
            }
            for f in frames
        ],
    }


def handle_memory_volatility(
    conn: sqlite3.Connection, *, subject: str = "user", predicate: str,
) -> dict:
    """Classify how volatile a (subject, predicate) slot is from its
    supersession history: stable / evolving / volatile. Deterministic, zero-LLM.
    (Operates on structured predicates; NL-only facts have no predicate to track.)
    """
    from memcontext.volatility import classify_predicate

    v = classify_predicate(conn, subject, predicate)
    return {
        "subject": subject,
        "predicate": predicate,
        "classification": v.classification,
        "change_count": v.change_count,
        "avg_lifespan_days": v.avg_lifespan_days,
        "current_streak_days": v.current_streak_days,
    }


def handle_memory_tuples(conn: sqlite3.Connection, *, session_id: str) -> dict:
    """Project a session's active facts into event tuples
    (subject, action, object, validity window). Pure read projection, zero-LLM.
    """
    from memcontext.event_tuples import claims_to_events

    tuples = claims_to_events(list_active_claims(conn, session_id))
    return {
        "session_id": session_id,
        "count": len(tuples),
        "tuples": [
            {
                "subject": t.subject,
                "action": t.action,
                "obj": t.obj,
                "valid_from_ts": t.valid_from_ts,
                "valid_until_ts": t.valid_until_ts,
                "claim_id": t.claim_id,
            }
            for t in tuples
        ],
    }


def handle_memory_entity_graph(
    conn: sqlite3.Connection, *, session_id: str, entity: str, max_hops: int = 1,
) -> dict:
    """Return an entity's co-occurrence neighbors within a session's claim graph
    (entities mentioned together in the same turn). Deterministic, zero-LLM.
    """
    from memcontext.entity_graph import EntityGraph

    graph = EntityGraph(conn, session_id)
    return {
        "session_id": session_id,
        "entity": entity,
        "max_hops": max_hops,
        "neighbors": sorted(graph.neighbors(entity, max_hops=max_hops)),
    }


def handle_brain(
    conn: sqlite3.Connection,
    *,
    session_id: str = "default",
) -> dict:
    """Deterministic world-state projection grouped by subject (no LLM).

    Returns the current value, status, confidence, and provenance handle for
    every active fact, plus a per-subject gaps report (vocabulary predicates
    with no active claim). Reads from the projection only.
    """
    return brain(conn, session_id=session_id)


def handle_memory_payload(
    conn: sqlite3.Connection,
    *,
    question: str,
    mode: str,
    session_id: str = "default",
) -> dict:
    """Return the memory payload for a question in one of three modes.

    ``mode`` is ``summary`` (raw transcript blob), ``vector`` (top-k statements
    by similarity, local embedder), or ``memcontext`` (structured projection).
    Holds the reader constant and varies only the payload — the demo's
    apples-to-apples comparison.
    """
    from memcontext.payloads import (
        memcontext_payload,
        summary_payload,
        vector_payload,
    )

    if mode == "summary":
        return summary_payload(conn, session_id, question)
    if mode == "vector":
        return vector_payload(conn, session_id, question)
    if mode == "memcontext":
        return memcontext_payload(conn, session_id, question)
    return {"error": f"Unknown mode: {mode!r}. Use summary|vector|memcontext."}


def handle_memory_trace(
    conn: sqlite3.Connection,
    *,
    claim_id: str | None = None,
    session_id: str = "default",
    subject: str | None = None,
    predicate: str | None = None,
) -> dict:
    """Trace a claim's source and supersession lineage.

    Resolve the head claim by ``claim_id``, or by ``(session_id, subject,
    predicate)`` (the newest active claim for that slot). Returns the rich
    ``lineage`` — newest-first, each step carrying value, status, typed edge,
    source turn, and span quote — alongside the legacy head fields.
    """
    if claim_id is None:
        if not (subject and predicate):
            return {"error": "Provide claim_id, or both subject and predicate."}
        head = find_same_identity_claim(
            conn, session_id=session_id, subject=subject, predicate=predicate
        )
        if head is None:
            return {
                "error": f"No active claim for {subject}/{predicate} in {session_id}",
                "subject": subject,
                "predicate": predicate,
                "lineage": [],
            }
        claim_id = head.claim_id

    claim = get_claim(conn, claim_id)
    if claim is None:
        return {"error": f"Claim {claim_id} not found"}

    # Rich lineage (reuse build_chain): oldest-first → present newest-first.
    from memcontext.chains import build_chain

    lineage = []
    for step in reversed(build_chain(conn, claim_id)):
        step_claim = get_claim(conn, step.claim_id)
        step_turn = get_turn(conn, step.source_turn_id)
        cs = step_claim.char_start if step_claim else None
        ce = step_claim.char_end if step_claim else None
        quote = (
            step_turn.text[cs:ce]
            if step_turn and cs is not None and ce is not None
            else None
        )
        lineage.append({
            "claim_id": step.claim_id,
            "value": step.value,
            "status": step_claim.status.value if step_claim else "unknown",
            "edge_type": step.edge_type,
            "confidence": step_claim.confidence if step_claim else None,
            "source_turn_id": step.source_turn_id,
            "speaker": step_turn.speaker.value if step_turn else None,
            "text": step_turn.text if step_turn else None,
            "char_start": cs,
            "char_end": ce,
            "quote": quote,
        })

    source_turn = get_turn(conn, claim.source_turn_id)
    span = span_for_claim(conn, claim_id)

    # Legacy forward walk (kept for backward compatibility with the claim_id tool).
    chain = []
    current_id = claim_id
    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        next_id = get_superseded_by(conn, current_id)
        if next_id:
            chain.append({"from": current_id, "to": next_id})
        current_id = next_id

    return {
        "subject": claim.subject,
        "predicate": claim.predicate,
        "claim": {
            "claim_id": claim.claim_id,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "value": claim.value,
            "confidence": claim.confidence,
            "status": claim.status.value,
        },
        "source_turn": {
            "turn_id": source_turn.turn_id,
            "speaker": source_turn.speaker.value,
            "text": source_turn.text,
        } if source_turn else None,
        "char_span": {
            "start": span.char_start,
            "end": span.char_end,
        } if span and span.char_start is not None else None,
        "lineage": lineage,
        "supersession_chain": chain,
    }


def handle_memory_correct(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
    action: str,
    new_value: str | None = None,
) -> dict:
    claim = get_claim(conn, claim_id)
    if claim is None:
        return {"error": f"Claim {claim_id} not found"}

    if action == "dismiss":
        set_claim_status(conn, claim_id, ClaimStatus.DISMISSED)
        return {
            "action": "dismissed",
            "claim_id": claim_id,
            "status": "dismissed",
        }

    if action == "correct":
        if not new_value:
            return {"error": "new_value is required for correction"}

        # Correct in kind: a structured claim keeps its triple (new value); an
        # NL-only fact is corrected as NL text (it has no triple to carry).
        if claim.predicate:
            new_claim = insert_fact(
                conn,
                session_id=claim.session_id,
                source_turn_id=claim.source_turn_id,
                confidence=1.0,
                subject=claim.subject,
                predicate=claim.predicate,
                value=new_value,
            )
        else:
            new_claim = insert_fact(
                conn,
                session_id=claim.session_id,
                source_turn_id=claim.source_turn_id,
                confidence=1.0,
                text=new_value,
            )
        edge = write_supersession_edge(
            conn,
            old_claim_id=claim_id,
            new_claim_id=new_claim.claim_id,
            edge_type=EdgeType.USER_CORRECTION,
            identity_score=None,
        )
        set_claim_status(conn, claim_id, ClaimStatus.SUPERSEDED)

        return {
            "action": "corrected",
            "old_claim_id": claim_id,
            "new_claim_id": new_claim.claim_id,
            "edge_id": edge.edge_id,
            "new_value": new_value,
        }

    return {"error": f"Unknown action: {action}"}


def handle_memory_observe(
    conn: sqlite3.Connection,
    *,
    url: str,
    title: str = "",
    accessibility_tree: dict | None = None,
    session_id: str | None = None,
) -> dict:
    """Store browser observation claims from a page snapshot."""
    from datetime import datetime, timezone

    from memcontext.observe.browser import PageSnapshot, observe_page

    sid = session_id or f"observe_{uuid.uuid4().hex[:8]}"
    snapshot = PageSnapshot(
        url=url,
        title=title,
        timestamp=datetime.now(timezone.utc).isoformat(),
        accessibility_tree=accessibility_tree or {},
    )
    result = observe_page(conn, snapshot=snapshot, session_id=sid)
    return {
        "session_id": sid,
        "turn_id": result.turn_id,
        "claims_stored": len(result.claims),
        "claims": [
            {"subject": c.get("subject", ""), "predicate": c.get("predicate", ""), "value": c.get("value", "")}
            for c in result.claims
        ],
        "snapshot_id": snapshot.snapshot_id,
    }


def _capture_page(
    url: str,
    *,
    login_email: str | None = None,
    login_password: str | None = None,
    login_url: str | None = None,
    connect_browser: bool = False,
) -> tuple[str, dict, str]:
    """Capture a page's a11y tree + DOM hash. Returns (title, tree, hash).

    Three modes:
    - connect_browser=True: attach to the user's running Chrome
      (started with --remote-debugging-port=9222). Reads the page the
      user can see — inherits all auth sessions, cookies, SSO, 2FA.
    - login_email/password: launch headless, fill login form, then read.
    - neither: launch headless, read the page as-is.
    """
    import hashlib

    from patchright.sync_api import sync_playwright

    with sync_playwright() as p:
        if connect_browser:
            source = p.chromium.connect_over_cdp("http://localhost:9222")
            cookies = source.contexts[0].cookies()
            source.close()

            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            context.add_cookies(cookies)
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            owns_browser = True
        else:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            owns_browser = True

            if login_email and login_password:
                target = login_url or url
                page.goto(target, wait_until="networkidle", timeout=30000)
                email_field = page.locator(
                    "input[type='email'], input[name='email'], "
                    "input[autocomplete='email'], input[autocomplete='username']"
                ).first
                password_field = page.locator("input[type='password']").first
                email_field.fill(login_email)
                password_field.fill(login_password)
                page.locator(
                    "button[type='submit'], button:has-text('Sign in'), "
                    "button:has-text('Log in'), button:has-text('Login')"
                ).first.click()
                page.wait_for_load_state("networkidle", timeout=15000)
                if login_url and login_url != url:
                    page.goto(url, wait_until="networkidle", timeout=30000)
            else:
                page.goto(url, wait_until="networkidle", timeout=30000)

        title = page.title()

        cdp = page.context.new_cdp_session(page)
        ax_result = cdp.send("Accessibility.getFullAXTree")
        a11y = _cdp_to_tree(ax_result.get("nodes", []))

        content = page.content()
        dom_hash = hashlib.sha256(content.encode()).hexdigest()
        if owns_browser:
            browser.close()
        return title, a11y, dom_hash


def _cdp_to_tree(nodes: list[dict]) -> dict:
    """Convert CDP flat node list to nested {role, name, value, children} tree."""
    if not nodes:
        return {}

    by_id: dict[str, dict] = {}
    for node in nodes:
        if node.get("ignored"):
            by_id[node["nodeId"]] = {"_skip": True, "_child_ids": node.get("childIds", [])}
            continue
        role = node.get("role", {}).get("value", "")
        name = node.get("name", {}).get("value", "")
        value_obj = node.get("value", {})
        value = value_obj.get("value", "") if isinstance(value_obj, dict) else ""
        by_id[node["nodeId"]] = {
            "role": role,
            "name": name,
            "value": str(value) if value else "",
            "children": [],
            "_child_ids": node.get("childIds", []),
        }

    def _build(nid: str) -> dict | None:
        entry = by_id.get(nid)
        if not entry:
            return None
        child_ids = entry.pop("_child_ids", [])
        if entry.get("_skip"):
            # Ignored node: promote its children
            kids = []
            for cid in child_ids:
                kid = _build(cid)
                if kid:
                    kids.append(kid)
            return {"role": "none", "name": "", "children": kids} if kids else None
        for cid in child_ids:
            kid = _build(cid)
            if kid:
                if kid.get("role") == "none" and kid.get("children"):
                    entry["children"].extend(kid["children"])
                else:
                    entry["children"].append(kid)
        return entry

    root = _build(nodes[0]["nodeId"])
    return root or {}


def handle_memory_observe_url(
    conn: sqlite3.Connection,
    *,
    url: str,
    session_id: str | None = None,
    login_email: str | None = None,
    login_password: str | None = None,
    login_url: str | None = None,
    connect_browser: bool = False,
    allow_password_login: bool = False,
) -> dict:
    """Observe a live URL, capture a11y tree, extract and store claims.

    Auth modes (pick one), preferred first:
    - connect_browser=True — attach to the user's running Chrome on port 9222.
      Inherits all sessions: SSO, 2FA, OAuth, saved passwords. No credentials
      needed. PREFERRED — never handles raw passwords.
    - login_email/password — launch headless, fill login form, then read.
      Disabled unless allow_password_login=True (raw passwords are a hazard:
      they transit as plain args and get typed into the page).
    - neither — launch headless, read the page unauthenticated.

    If the URL was previously observed in the same session, supersession
    fires automatically for changed values.
    """
    from datetime import datetime, timezone

    from memcontext.observe.browser import PageSnapshot, observe_page
    from memcontext.observe.extractors import _url_to_subject

    sid = session_id or "observe_default"

    if login_password and not allow_password_login:
        return {
            "error": (
                "password login is disabled by default. Prefer connect_browser=true "
                "(attach to your running Chrome — inherits SSO/2FA, no credentials). "
                "To pass a raw password anyway, set allow_password_login=true."
            ),
            "url": url,
        }

    title, a11y_tree, dom_hash = _capture_page(
        url,
        login_email=login_email,
        login_password=login_password,
        login_url=login_url,
        connect_browser=connect_browser,
    )

    url_subject = _url_to_subject(url)
    prev_count = conn.execute(
        "SELECT COUNT(*) FROM claims"
        " WHERE session_id = ? AND subject = ?"
        " AND status IN ('active','confirmed','audited')",
        (sid, url_subject),
    ).fetchone()[0]
    is_revisit = prev_count > 0

    snapshot = PageSnapshot(
        url=url,
        title=title,
        timestamp=datetime.now(timezone.utc).isoformat(),
        accessibility_tree=a11y_tree,
        dom_hash=dom_hash,
    )
    result = observe_page(conn, snapshot=snapshot, session_id=sid)

    supersessions = conn.execute(
        "SELECT e.old_claim_id, e.new_claim_id, e.edge_type,"
        "       c_old.value AS old_value, c_new.value AS new_value"
        " FROM supersession_edges e"
        " JOIN claims c_old ON e.old_claim_id = c_old.claim_id"
        " JOIN claims c_new ON e.new_claim_id = c_new.claim_id"
        " WHERE c_new.source_turn_id = ?",
        (result.turn_id,),
    ).fetchall() if result.turn_id else []

    resp = {
        "session_id": sid,
        "url": url,
        "title": title,
        "dom_hash": dom_hash[:12],
        "a11y_nodes": _count_a11y_nodes(a11y_tree),
        "claims_stored": len(result.claims),
        "claims": [
            {"subject": c.get("subject", ""), "predicate": c.get("predicate", ""), "value": c.get("value", "")}
            for c in result.claims
        ],
        "snapshot_id": snapshot.snapshot_id,
        "is_revisit": is_revisit,
    }

    if supersessions:
        resp["changes_detected"] = [
            {
                "old_value": row["old_value"],
                "new_value": row["new_value"],
                "edge_type": row["edge_type"],
            }
            for row in supersessions
        ]
        resp["supersessions"] = len(supersessions)

    return resp


def _count_a11y_nodes(tree: dict) -> int:
    """Count nodes in an accessibility tree."""
    if not isinstance(tree, dict):
        return 0
    count = 1
    for child in tree.get("children", []):
        count += _count_a11y_nodes(child)
    return count
