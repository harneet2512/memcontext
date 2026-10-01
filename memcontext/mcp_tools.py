"""MCP tool handler functions — pure business logic, no MCP protocol dependency.

Each function takes a sqlite3 Connection and keyword arguments, returns a dict.
These are usable from CLI, tests, or the MCP server without importing mcp.
"""
from __future__ import annotations

import sqlite3
import uuid
from typing import TYPE_CHECKING

import structlog

from memcontext.brain import brain
from memcontext.claims import (
    get_claim,
    get_superseded_by,
    get_turn,
    insert_fact,
    insert_turn,
    list_active_claims,
    new_turn_id,
    now_ns,
    set_claim_status,
)
from memcontext.extractors import PassthroughExtractor, auto_extractor
from memcontext.on_new_turn import on_new_turn
from memcontext.provenance import span_for_claim
from memcontext.schema import Claim, ClaimStatus, EdgeType, ExtractionStatus, Speaker, Turn
from memcontext.supersession import write_supersession_edge

if TYPE_CHECKING:
    from memcontext.extraction_queue import ExtractionQueue
    from memcontext.on_new_turn import ExtractorFn

log = structlog.get_logger(__name__)


DEFAULT_SESSION_ID = "default"


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
    namespace: str = "default",
) -> dict:
    # Same default as memory_query / memory_trace / brain, so a fact stored without
    # a session is found by the calls that also omit it.
    sid = session_id or DEFAULT_SESSION_ID
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
        namespace=namespace,
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

    # EXPERIMENTAL anomaly detection (flag-gated; no-op without an embedder)
    from memcontext.anomaly import check_write
    check_write(conn, sid, text, episode_embedder())

    out = {
        "turn_id": result.turn.turn_id if result.turn else None,
        "session_id": sid,
        "admitted": result.admitted,
        "claims_created": len(result.created_claims),
        "claim_ids": [c.claim_id for c in result.created_claims],
        "supersessions": len(result.supersession_edges),
    }
    warnings = _demoted_predicate_warnings(claims) if claims and result.admitted else []
    matches = _subject_drift(conn, result, namespace)
    if matches:
        out["similar_subjects"] = [m for _new, m in matches]
        warnings += [
            f"decision subject {new!r} is new, but a similar live decision exists under "
            f"{m['subject']!r} (= {m['value']!r}). If this updates it, store it again with "
            f"the same subject {m['subject']!r} so it supersedes the old value; otherwise ignore."
            for new, m in matches
        ]
    if warnings:
        out["warnings"] = warnings
    return out


def _subject_drift(conn: sqlite3.Connection, result, namespace: str) -> list[tuple[str, dict]]:
    """(new subject, similar live decision) pairs for claims that opened a NEW slot.

    A single-valued claim that superseded or restated a value reused its subject, so
    it is fine. One that landed in an empty slot may be a changed decision recorded
    under a drifted subject, which supersession cannot link: return the likely
    matches so the caller can re-store under the existing subject. Never merges.
    """
    from memcontext.conflicts import is_single_valued, similar_subjects

    linked = {e.new_claim_id for e in result.supersession_edges}
    pairs: list[tuple[str, dict]] = []
    for c in result.created_claims:
        if not c.subject or not is_single_valued(c.predicate) or c.claim_id in linked:
            continue
        same_slot = conn.execute(
            "SELECT 1 FROM claims c JOIN turns t ON t.turn_id = c.source_turn_id"
            " WHERE c.subject = ? AND c.predicate = ? AND c.claim_id != ? AND t.namespace = ?"
            " AND c.status IN ('active','confirmed','audited') LIMIT 1",
            (c.subject, c.predicate, c.claim_id, namespace),
        ).fetchone()
        if same_slot is not None:
            continue  # a restatement of the same subject
        pairs += [(c.subject, m) for m in similar_subjects(
            conn, subject=c.subject, predicate=c.predicate, namespace=namespace)]
    return pairs


def _demoted_predicate_warnings(claims: list[dict]) -> list[str]:
    """Tell the caller when an out-of-vocab predicate demoted a claim to NL-only.

    The fact is still stored (as text), but it has no structured slot, so it
    cannot supersede or be superseded by a later value.
    """
    from memcontext.claims import predicate_in_vocab
    from memcontext.predicate_packs import active_pack

    predicates = {c.get("predicate") for c in claims}
    unknown = sorted(p for p in predicates if isinstance(p, str) and p and not predicate_in_vocab(p))
    if not unknown:
        return []
    allowed = ", ".join(sorted(active_pack().predicate_families))
    return [
        f"predicate {p!r} is not in the active pack; stored as text only, so it will not "
        f"supersede earlier values. Use one of: {allowed}"
        for p in unknown
    ]


def _session_in_namespace(conn: sqlite3.Connection, session_id: str, namespace: str) -> bool:
    """True if the session has any episode in the given namespace (tenant scope)."""
    return conn.execute(
        "SELECT 1 FROM turns WHERE session_id = ? AND namespace = ? LIMIT 1",
        (session_id, namespace),
    ).fetchone() is not None


def _record_serve_events(
    conn: sqlite3.Connection,
    *,
    request_session_id: str,
    claim_ids: list[str],
    query: str,
) -> list[str]:
    """Append served claim IDs to the answer-time verification ledger."""
    if not claim_ids:
        return []
    from memcontext.claims import now_ns

    rows = [
        (f"se_{uuid.uuid4().hex[:12]}", request_session_id, cid, query, now_ns())
        for cid in claim_ids
    ]
    conn.executemany(
        "INSERT INTO serve_events"
        " (event_id, request_session_id, claim_id, query, served_ts)"
        " VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    return [r[0] for r in rows]


def handle_memory_query(
    conn: sqlite3.Connection,
    *,
    query: str,
    session_id: str | None = None,
    top_k: int = 10,
    debug: bool = False,
    namespace: str | None = None,
    include_resolved: bool = True,
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
        # Namespace isolation: a caller bound to a namespace cannot read a session
        # owned by a different tenant.
        if namespace is not None and not _session_in_namespace(conn, session_id, namespace):
            return {"claims": [], "episodes": [], "total": 0, "denied": "namespace"}
        hits = retrieve_memory(
            conn, session_id=session_id, query=query, top_k=top_k, explain=explain,
            include_superseded=history, namespace=namespace,
        )
        active = list_active_claims(conn, session_id)
        if namespace is not None:  # a colliding session id must not reveal another tenant
            mine = {r[0] for r in conn.execute(
                "SELECT turn_id FROM turns WHERE session_id = ? AND namespace = ?",
                (session_id, namespace),
            ).fetchall()}
            active = [c for c in active if c.source_turn_id in mine]
        total = len(active)
    else:
        # Every session that has episodes — episodes exist even when a session's
        # facts are absent/pending (the Tier-1 floor), so scope by turns, not claims.
        # Namespace isolation: the cross-session sweep is bounded to the caller's
        # namespace, never "all sessions in the DB".
        if namespace is not None:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM turns WHERE namespace = ?", (namespace,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT DISTINCT session_id FROM turns").fetchall()
        sids = [r["session_id"] if isinstance(r, sqlite3.Row) else r[0] for r in rows]
        if not sids:
            return {"claims": [], "episodes": [], "total": 0}
        hits = retrieve_memory_across(
            conn, session_ids=sids, query=query, top_k=top_k, explain=explain,
            include_superseded=history, namespace=namespace,
        )
        if namespace is not None:  # the store size of other tenants is not the caller's
            total = conn.execute(
                "SELECT COUNT(*) FROM claims c JOIN turns t ON t.turn_id = c.source_turn_id"
                " WHERE c.status IN ('active','confirmed','audited') AND t.namespace = ?",
                (namespace,),
            ).fetchone()[0]
        else:
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
            from memcontext.admission import detect_durability
            claims_out.append({
                "claim_id": c.claim_id,
                "subject": c.subject,
                "predicate": c.predicate,
                "value": c.value,
                "fact": c.text,
                "confidence": c.confidence,
                "status": c.status.value,
                "score": norm,
                "source_turn_id": c.source_turn_id,
                # L3: durable instruction / standing preference / ephemeral chatter,
                # so the agent can weight standing guidance over a passing remark.
                "durability": detect_durability(c.value),
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

    # Usage reinforcement: the fact claims that RANKED are now "accessed" (claims only
    # shown as context under their evidence below are not relevance signals).
    ranked_claim_ids = [c["claim_id"] for c in claims_out]
    bump_access(conn, ranked_claim_ids)

    # Consolidation marker + source-trust spotlight: each served fact carries its
    # trust and a 'quarantined' flag (low-trust origin -- citable, not authoritative),
    # so the agent never silently acts on untrusted/poisoned memory.
    if claims_out:
        from memcontext.source_trust import QUARANTINE_THRESHOLD

        _cids = [c["claim_id"] for c in claims_out]
        _ph = ",".join("?" for _ in _cids)
        _meta = {
            r[0]: (bool(r[1]), float(r[2])) for r in conn.execute(
                f"SELECT claim_id, consolidated, COALESCE(source_trust, 0.5)"
                f" FROM claim_metadata WHERE claim_id IN ({_ph})", _cids,
            ).fetchall()
        }
        for c in claims_out:
            cons, trust = _meta.get(c["claim_id"], (False, 0.5))
            c["consolidated"] = cons
            c["trust"] = round(trust, 3)
            c["quarantined"] = trust < QUARANTINE_THRESHOLD

    # Evidence <-> state link, additive: top-level claims stay exactly as ranked; each
    # served episode names its claims (claim_ids) and lists its outdated ones inline.
    from memcontext.serving import annotate_served_evidence, served_slots

    episodes_out = annotate_served_evidence(conn, claims_out, episodes_out)
    nested_claims = [c for e in episodes_out for c in e.get("claims", ())]

    # Verification ledger: every claim the caller was shown, top-level or nested.
    serve_event_ids = _record_serve_events(
        conn,
        request_session_id=session_id or "__cross_session__",
        claim_ids=[c["claim_id"] for c in claims_out] + [c["claim_id"] for c in nested_claims],
        query=query,
    )

    # Token accounting (zero-LLM, ~chars/4) for what we serve, by source type.
    def _toks(text: str) -> int:
        return max(1, len(text or "") // 4)
    fact_tokens = sum(_toks(c.get("fact") or c.get("value") or "") for c in claims_out)
    fact_tokens += sum(_toks(c.get("fact") or c.get("value") or "") for c in nested_claims)
    episode_tokens = sum(_toks(e.get("text") or "") for e in episodes_out)
    token_report = {
        "fact_tokens": fact_tokens,
        "episode_tokens": episode_tokens,
        "total_tokens": fact_tokens + episode_tokens,
        "served_items": len(claims_out) + len(episodes_out),
        # retrieval slots: an episode and its claims served with it count as one
        "served_slots": served_slots(claims_out, episodes_out),
        # distinct evidence objects behind the served items (claims + episodes)
        "served_memories": len({c["source_turn_id"] for c in claims_out}
                               | {e["turn_id"] for e in episodes_out}),
        "nested_claims": len(nested_claims),
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
        "serve_event_ids": serve_event_ids,
    }
    # Resolved view + briefing on the MAIN query path (not tool-only): alongside the
    # raw ranked hits, the agent gets the current world-state — one value per slot
    # with provenance + typed supersession lineage — and a compact session briefing.
    # Built fresh so it is always current; best-effort so it never breaks a query.
    # The resolved view below is keyed by session id alone. When a tenant-scoped
    # caller's session id is also used in another namespace, it would mix tenants:
    # withhold it (fail closed) and say so.
    shared_session = bool(session_id) and namespace is not None and conn.execute(
        "SELECT 1 FROM turns WHERE session_id = ? AND namespace != ? LIMIT 1",
        (session_id, namespace),
    ).fetchone() is not None
    if include_resolved and session_id and shared_session:
        result["resolved_view_withheld"] = "session id shared across namespaces"
    if include_resolved and session_id and not shared_session:
        try:
            from memcontext.brain import brain
            from memcontext.serving import (
                resolved_entity_links,
                serve_event_frames,
                serve_life_events,
                session_briefing,
            )

            result["world_state"] = brain(conn, session_id=session_id)
            # namespace-scope the subject-keyed profile + life-events so a tenant's
            # query never aggregates another tenant's facts (world_state/events are
            # already session-scoped). namespace is None for single-tenant brains.
            briefing = session_briefing(conn, namespace=namespace)
            if briefing:
                result["briefing"] = briefing
            links = resolved_entity_links(conn, session_id)
            if links:
                result["entity_links"] = links
            events = serve_event_frames(conn, session_id=session_id, query=query)
            if events:
                result["events"] = events
            life = serve_life_events(conn, namespace=namespace)
            if life:
                result["life_events"] = life
            contradictions = handle_memory_contradictions(conn, session_id=session_id)
            if contradictions["count"]:
                result["contradictions"] = contradictions
        except Exception:  # noqa: BLE001 — resolved view is additive, never fatal
            pass

    if debug and explain is not None:
        result["ranking"] = {cid: explain[cid] for cid in ranked_claim_ids if cid in explain}
    return result


def handle_memory_verify(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    claim_ids: list[str],
) -> dict:
    """Verify that cited claim IDs were served to this session's query door.

    This is intentionally narrow: it checks the durable serve ledger, not whether
    a claim exists or whether the answer was semantically correct.
    """
    unique_ids = list(dict.fromkeys(claim_ids or []))
    if not unique_ids:
        return {
            "session_id": session_id,
            "verified": False,
            "served": [],
            "missing": [],
            "error": "claim_ids required",
        }
    placeholders = ",".join("?" for _ in unique_ids)
    rows = conn.execute(
        "SELECT claim_id, MAX(served_ts) AS served_ts FROM serve_events"
        f" WHERE request_session_id = ? AND claim_id IN ({placeholders})"
        " GROUP BY claim_id",
        [session_id, *unique_ids],
    ).fetchall()
    served = {r["claim_id"]: r["served_ts"] for r in rows}
    missing = [cid for cid in unique_ids if cid not in served]
    return {
        "session_id": session_id,
        "verified": not missing,
        "served": [{"claim_id": cid, "served_ts": served[cid]} for cid in unique_ids if cid in served],
        "missing": missing,
    }


def handle_memory_contradictions(
    conn: sqlite3.Connection,
    *,
    session_id: str | None = None,
) -> dict:
    """Surface unresolved contradictions where both endpoints remain active."""
    active_statuses = ("active", "confirmed", "audited")
    params: list[str] = [EdgeType.CONTRADICTS.value, *active_statuses, *active_statuses]
    session_filter = ""
    if session_id is not None:
        session_filter = " AND old.session_id = ? AND new.session_id = ?"
        params.extend([session_id, session_id])
    rows = conn.execute(
        "SELECT e.edge_id, e.identity_score, e.created_ts,"
        " old.claim_id AS old_claim_id, old.subject AS old_subject,"
        " old.predicate AS old_predicate, old.value AS old_value,"
        " old.status AS old_status, old.source_turn_id AS old_turn_id,"
        " new.claim_id AS new_claim_id, new.subject AS new_subject,"
        " new.predicate AS new_predicate, new.value AS new_value,"
        " new.status AS new_status, new.source_turn_id AS new_turn_id"
        " FROM supersession_edges e"
        " JOIN claims old ON old.claim_id = e.old_claim_id"
        " JOIN claims new ON new.claim_id = e.new_claim_id"
        " WHERE e.edge_type = ?"
        " AND old.status IN (?, ?, ?)"
        " AND new.status IN (?, ?, ?)"
        f"{session_filter}"
        " ORDER BY e.created_ts DESC",
        params,
    ).fetchall()
    contradictions = [
        {
            "edge_id": r["edge_id"],
            "edge_type": EdgeType.CONTRADICTS.value,
            "created_ts": r["created_ts"],
            "unresolved": True,
            "old": {
                "claim_id": r["old_claim_id"],
                "subject": r["old_subject"],
                "predicate": r["old_predicate"],
                "value": r["old_value"],
                "status": r["old_status"],
                "source_turn_id": r["old_turn_id"],
            },
            "new": {
                "claim_id": r["new_claim_id"],
                "subject": r["new_subject"],
                "predicate": r["new_predicate"],
                "value": r["new_value"],
                "status": r["new_status"],
                "source_turn_id": r["new_turn_id"],
            },
        }
        for r in rows
    ]
    return {
        "session_id": session_id,
        "count": len(contradictions),
        "contradictions": contradictions,
    }


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


def handle_memory_trust_status(conn: sqlite3.Connection) -> dict:
    """Trust observability: source-trust distribution, contradiction rate, forgetting
    + drift audit, tenant distribution, and a staleness proxy. Measures whether the
    trust/governance layer is working, not just recall."""
    from memcontext.trust_report import trust_status

    return trust_status(conn)


def handle_memory_profile(
    conn: sqlite3.Connection,
    *,
    subject: str = "user",
    max_tokens: int = 500,
) -> dict:
    try:
        from memcontext.profiles import (
            build_smart_profile,
            format_profile,
            load_profile,
            store_profile,
        )

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
    predicate)``: the slot's current value, i.e. the most-trusted live value (the
    newest on a tie), traced from its original assertion (``restatements`` counts
    later copies). Returns the rich ``lineage`` — newest-first, each step carrying
    value, status, typed edge, source turn, and span quote — alongside the legacy
    head fields, plus ``conflicts``: other CURRENT values competing with the head
    (same single-valued slot, a contradicts edge, or the same decision under another
    subject), newest first, each with its trust and quarantine flag.
    """
    from memcontext.conflicts import slot_conflicts, trusted_slot_head

    restatements = 0
    if claim_id is None:
        if not (subject and predicate):
            return {"error": "Provide claim_id, or both subject and predicate."}
        claim_id, restatements = trusted_slot_head(
            conn, subject=subject, predicate=predicate, session_id=session_id,
        )
        if claim_id is None:
            return {
                "error": f"No active claim for {subject}/{predicate} in {session_id}",
                "subject": subject,
                "predicate": predicate,
                "lineage": [],
            }

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
            "fact": step_claim.text if step_claim else None,
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
            "fact": claim.text,
            "confidence": claim.confidence,
            "status": claim.status.value,
        },
        # The Memory this claim was derived from, and where that evidence came from.
        "source_turn": {
            "turn_id": source_turn.turn_id,
            "speaker": source_turn.speaker.value,
            "text": source_turn.text,
            "source_type": source_turn.source_type.value,
        } if source_turn else None,
        "char_span": {
            "start": span.char_start,
            "end": span.char_end,
        } if span and span.char_start is not None else None,
        "lineage": lineage,
        "supersession_chain": chain,
        "conflicts": slot_conflicts(conn, claim_id),
        **({"restatements": restatements} if restatements else {}),
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

        # "Correct this fact" means correct the slot's CURRENT value: a superseded
        # target is redirected to its active head, never forked into a 2nd value.
        target = _resolve_correction_target(conn, claim)
        if isinstance(target, str):
            return {"error": target}

        turn = _insert_correction_turn(conn, target, new_value)
        span_start = turn.text.rfind(new_value)
        span_end = span_start + len(new_value)
        # Correct in kind: a structured claim keeps its triple (new value); an
        # NL-only fact is corrected as NL text (it has no triple to carry).
        if target.predicate:
            new_claim = insert_fact(
                conn,
                session_id=target.session_id,
                source_turn_id=turn.turn_id,
                confidence=1.0,
                subject=target.subject,
                predicate=target.predicate,
                value=new_value,
                char_start=span_start,
                char_end=span_end,
            )
        else:
            new_claim = insert_fact(
                conn,
                session_id=target.session_id,
                source_turn_id=turn.turn_id,
                confidence=1.0,
                text=new_value,
                char_start=span_start,
                char_end=span_end,
            )
        edge = write_supersession_edge(
            conn,
            old_claim_id=target.claim_id,
            new_claim_id=new_claim.claim_id,
            edge_type=EdgeType.USER_CORRECTION,
            identity_score=None,
        )
        set_claim_status(conn, target.claim_id, ClaimStatus.SUPERSEDED)
        # Close the old value's validity window exactly as Pass 1 does.
        conn.execute(
            "UPDATE claims SET valid_until_ts = ?"
            " WHERE claim_id = ?"
            " AND (valid_from_ts IS NULL OR valid_from_ts < ?)",
            (edge.created_ts, target.claim_id, edge.created_ts),
        )

        return {
            "action": "corrected",
            "requested_claim_id": claim_id,
            "corrected_claim_id": target.claim_id,
            "old_claim_id": target.claim_id,
            "new_claim_id": new_claim.claim_id,
            "source_turn_id": turn.turn_id,
            "edge_id": edge.edge_id,
            "new_value": new_value,
        }

    return {"error": f"Unknown action: {action}"}


_CORRECTABLE_STATUSES = frozenset(
    {ClaimStatus.ACTIVE, ClaimStatus.CONFIRMED, ClaimStatus.AUDITED}
)


def _resolve_correction_target(conn: sqlite3.Connection, claim: Claim) -> Claim | str:
    """The claim a correction of ``claim`` must supersede, or an error message.

    A superseded claim is followed forward along its supersession chain to the
    slot's current (active) head. A dismissed / draft claim — or a chain whose
    head is no longer active — has no current value to correct.
    """
    current = claim
    visited: set[str] = set()
    while current.status is ClaimStatus.SUPERSEDED:
        visited.add(current.claim_id)
        next_id = get_superseded_by(conn, current.claim_id)
        nxt = get_claim(conn, next_id) if next_id and next_id not in visited else None
        if nxt is None:
            return (
                f"Claim {claim.claim_id} is superseded and its supersession chain"
                " has no current claim to correct"
            )
        current = nxt
    if current.status not in _CORRECTABLE_STATUSES:
        if current.claim_id == claim.claim_id:
            return f"Claim {claim.claim_id} is {current.status.value}; it cannot be corrected"
        return (
            f"Claim {claim.claim_id} was superseded by {current.claim_id}, which is"
            f" {current.status.value}; there is no current claim to correct"
        )
    return current


def _insert_correction_turn(conn: sqlite3.Connection, target: Claim, new_value: str) -> Turn:
    """Record the correction as its own user episode (the new claim's provenance),
    in the same session and namespace as the corrected claim's source turn."""
    ns_row = conn.execute(
        "SELECT namespace FROM turns WHERE turn_id = ?", (target.source_turn_id,)
    ).fetchone()
    namespace = ns_row[0] if ns_row is not None and ns_row[0] else "default"
    text = (
        f"Correction: {target.subject} {target.predicate} is {new_value}"
        if target.predicate
        else f"Correction: {new_value}"
    )
    turn = Turn(
        turn_id=new_turn_id(),
        session_id=target.session_id,
        speaker=Speaker.USER,
        text=text,
        ts=now_ns(),
        extraction_status=ExtractionStatus.STRUCTURED,
    )
    insert_turn(conn, turn, namespace=namespace)
    return turn


def handle_tool_discover(
    conn: sqlite3.Connection,
    *,
    query: str,
    session_ids: list[str] | None = None,
    top_k: int = 10,
    use_memory: bool = False,
) -> dict:
    """Activation layer: return the curated top-K tools for a query.

    Query-only by default; ``use_memory=True`` additionally conditions on the
    user's memory via the Session-1 public surface (``retrieve_memory_across``).
    Uses the substrate's default embedder (respects MEMCONTEXT_EMBED_EPISODES).
    """
    from memcontext.retrieval import episode_embedder
    from memcontext.tool_activation import discover_tools

    results = discover_tools(
        conn,
        query=query,
        session_ids=session_ids or [],
        top_k=top_k,
        use_memory=use_memory,
        embedder=episode_embedder(),
    )
    return {
        "query": query,
        "used_memory": any(r.used_memory for r in results),
        "count": len(results),
        "tools": [
            {"tool_id": r.tool_id, "name": r.name, "score": round(r.score, 6)}
            for r in results
        ],
    }
