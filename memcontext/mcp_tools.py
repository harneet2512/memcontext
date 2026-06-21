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
    namespace: str = "default",
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

    return {
        "turn_id": result.turn.turn_id if result.turn else None,
        "session_id": sid,
        "admitted": result.admitted,
        "claims_created": len(result.created_claims),
        "claim_ids": [c.claim_id for c in result.created_claims],
        "supersessions": len(result.supersession_edges),
    }


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


def _slot_key(subject: str, predicate: str, value: str) -> tuple[str, str, str]:
    """Resolved-slot identity for a served claim — one value per slot.

    Keys on (subject, predicate, attribute-or-normalised-value). For a coarse
    catch-all predicate the value distinguishes genuinely different facts, so
    different values are NOT fused; only a single-valued ATTRIBUTE slot (read off
    the value phrasing, e.g. "lives in X" / "moved to Y" -> `reside`) collapses
    competing values to one resolved truth — mirroring the supersession layer's
    own slot model.

    C1 SLOT UNIFY (Fracture B): keys on the CANONICAL slot taxonomy
    ``memcontext.attribute_key.attribute_key`` — the exact generalized slot
    function supersession/projection/enumeration key on — so serve-dedup and
    supersession fuse on the SAME slot identity, not two divergent ones. The
    canonical fn returns ``""`` for "no derivable slot"; that empty / no-opinion
    case falls back to the value, so distinct facts never fuse (no regression).
    Deterministic, zero-LLM, no predicate/benchmark list.
    """
    from memcontext.attribute_key import attribute_key

    attr = attribute_key(value or "")
    discriminator = attr if attr else (value or "").strip().lower()
    return ((subject or "").strip().lower(), (predicate or "").strip().lower(), discriminator)


def _dedup_claims_out(claims_out: list[dict]) -> list[dict]:
    """Collapse the served claims channel to one resolved value per slot.

    Within a slot the survivor is the BEST-ranked claim; ties break toward an
    active (non-superseded) status, then the most recent. Order-stable: the kept
    survivors are emitted in their original ranked order. ADDITIVE: when no two
    claims share a slot (the single-fact / distinct-fact case) the list is
    returned byte-identical.
    """
    if len(claims_out) < 2:
        return claims_out
    _ACTIVE = {"active", "confirmed", "audited"}

    def _better(a: dict, b: dict) -> bool:
        # True if `a` should replace incumbent `b` as the slot survivor.
        if a["score"] != b["score"]:
            return a["score"] > b["score"]
        a_active = a.get("status") in _ACTIVE
        b_active = b.get("status") in _ACTIVE
        if a_active != b_active:
            return a_active
        return a.get("created_ts", 0) > b.get("created_ts", 0)

    survivor: dict[tuple[str, str, str], dict] = {}
    for c in claims_out:
        key = _slot_key(c.get("subject", ""), c.get("predicate", ""), c.get("value", ""))
        cur = survivor.get(key)
        if cur is None or _better(c, cur):
            survivor[key] = c
    if len(survivor) == len(claims_out):
        return claims_out  # nothing collapsed — byte-identical
    keep = set(id(v) for v in survivor.values())
    return [c for c in claims_out if id(c) in keep]


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
    depth_kind, depth_top_k = classify_query_depth(query)
    if top_k == 10:
        top_k = depth_top_k

    # Session-id list backing the cross-session sweep (session_id=None). Captured
    # so the resolved layer can project a cross-session world-state over exactly
    # the tenant's sessions, not just produce a top-k dump (Fracture A).
    cross_session_sids: list[str] = []

    # Unified two-tier retrieval (facts + episodes, source-tagged, RRF-fused).
    if session_id:
        # Namespace isolation: a caller bound to a namespace cannot read a session
        # owned by a different tenant.
        if namespace is not None and not _session_in_namespace(conn, session_id, namespace):
            return {"claims": [], "episodes": [], "total": 0, "denied": "namespace"}
        hits = retrieve_memory(
            conn, session_id=session_id, query=query, top_k=top_k, explain=explain,
            include_superseded=history,
        )
        total = len(list_active_claims(conn, session_id))
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
        cross_session_sids = sids
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
            from memcontext.admission import detect_durability
            claims_out.append({
                "claim_id": c.claim_id,
                "subject": c.subject,
                "predicate": c.predicate,
                "value": c.value,
                "confidence": c.confidence,
                "status": c.status.value,
                "score": norm,
                # created_ts is a slot-dedup tie-breaker (most-recent survivor);
                # internal, stripped before the payload is returned.
                "created_ts": c.created_ts,
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

    # SLOT-DEDUP (resolved-truth channel): the served claims become one value per
    # resolved slot (subject, predicate, attribute) — duplicate near-mentions and
    # competing single-valued-attribute values collapse to the best-ranked survivor,
    # so the agent reads resolved current truth, not a top-k pile of duplicates.
    # ADDITIVE: distinct facts (no shared slot) pass through byte-identical.
    claims_out = _dedup_claims_out(claims_out)

    # PROVENANCE LINEAGE on the served channel: each surviving claim carries its
    # "why" — the source span/quote it came from and the typed correction chain it
    # retired — so a served fact is self-justifying without a second tool call.
    # Best-effort; never breaks a query.
    from memcontext.provenance import explain_claim

    for c in claims_out:
        try:
            ex = explain_claim(conn, c["claim_id"])
        except Exception:  # noqa: BLE001 — provenance is additive, never fatal
            ex = None
        if ex is not None:
            quote = None
            if (
                ex.source_text is not None
                and ex.char_start is not None
                and ex.char_end is not None
            ):
                quote = ex.source_text[ex.char_start:ex.char_end]
            c["provenance"] = {
                "source_turn_id": ex.source_turn_id,
                "source_speaker": ex.source_speaker,
                "source_ts": ex.source_ts,
                "char_start": ex.char_start,
                "char_end": ex.char_end,
                "quote": quote,
                # (old_value, edge_type) for each retired predecessor, oldest first.
                "superseded": [
                    {"old_value": ov, "edge_type": et} for ov, et in ex.superseded
                ],
            }

    # Strip the internal dedup tie-breaker so the payload shape is unchanged.
    for c in claims_out:
        c.pop("created_ts", None)

    # Usage reinforcement: the fact claims we served are now "accessed".
    served_claim_ids = [c["claim_id"] for c in claims_out]
    bump_access(conn, served_claim_ids)
    serve_event_ids = _record_serve_events(
        conn,
        request_session_id=session_id or "__cross_session__",
        claim_ids=served_claim_ids,
        query=query,
    )

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

    # AGGREGATION reader hint (ADDITIVE): a generic counting/listing query
    # (depth_kind == "aggregation": "how many", "count", "list all", "every" —
    # a general intent classifier, NOT a benchmark/predicate list) over-counts
    # raw near-duplicate mentions. When an enumeration distinct-count is attached
    # below, the hint tells the reader to answer from that DISTINCT count, not the
    # number of raw retrieved rows. Overrides the per-type hint only on this intent.
    if depth_kind == "aggregation":
        reader_hint = (
            "This is a counting/listing question. Count DISTINCT occurrences, not"
            " raw mentions: multiple phrasings of the same instance are ONE. If an"
            " 'enumeration.distinct_count' is provided, use it as the count."
        )
    else:
        reader_hint = _READER_HINTS.get(query_type, _READER_HINTS["fact_recall"])

    result: dict = {
        "claims": claims_out,
        "episodes": episodes_out,
        "total": total,
        "query_type": query_type,
        "reader_hint": reader_hint,
        "token_report": token_report,
        "serve_event_ids": serve_event_ids,
    }

    # COUNTING deepening (ADDITIVE): for a generic aggregation-intent query
    # ("how many", "count", "list all" — classify_query_depth=="aggregation",
    # NOT a benchmark/predicate keyword list) the reader can over-count raw
    # near-duplicate mentions. Attach a deterministic DISTINCT count of the
    # dominant retrieved slot so the reader serves deduplicated instances.
    #
    # Strictly additive: only fires on aggregation intent AND when a real
    # embedder is configured (NullEmbedder/no-model paths skip it — the cluster
    # dedup is embedding-based) AND when scoped to a session. Never mutates
    # claims/episodes/total, so non-aggregation queries are byte-identical.
    if depth_kind == "aggregation" and session_id and claims_out:
        from memcontext.retrieval import episode_embedder, semantic_enabled

        if semantic_enabled():
            try:
                from memcontext.enumeration import enumerate_retrieved

                enum = enumerate_retrieved(
                    conn,
                    session_id=session_id,
                    retrieved_claims=claims_out,
                    embedder=episode_embedder(),
                )
                if enum is not None:
                    result["enumeration"] = {
                        "distinct_count": enum.distinct_count,
                        "t_dup": round(enum.t_dup, 4),
                        "representatives": [
                            {
                                "representative": cl.representative,
                                "member_claim_ids": list(cl.member_claim_ids),
                                "event_ts_set": list(cl.event_ts_set),
                            }
                            for cl in enum.clusters
                        ],
                    }
            except Exception:  # noqa: BLE001 — enumeration is additive, never fatal
                pass
    # Resolved view + briefing on the MAIN query path (not tool-only): alongside the
    # raw ranked hits, the agent gets the current world-state — one value per slot
    # with provenance + typed supersession lineage — and a compact session briefing.
    # Built fresh so it is always current; best-effort so it never breaks a query.
    if include_resolved and session_id:
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

    # FRACTURE A FIX (ADDITIVE): the CROSS-SESSION sweep (session_id=None) is the
    # real multi-session / whole-tenant-history path, and historically it returned
    # ONLY ranked claims + raw episodes — a top-k dump, the exact thing the product
    # claims to beat. Give it the same resolved layer the single-session path has:
    # a cross-session world_state (one CURRENT value per slot across the tenant's
    # sessions, stale superseded values absent) plus, on aggregation intent, a
    # cross-session distinct count. Best-effort so it never breaks a query.
    if include_resolved and session_id is None and cross_session_sids:
        try:
            from memcontext.brain import brain_across

            result["world_state"] = brain_across(conn, session_ids=cross_session_sids)
            # Namespace-scoped briefing / life-events for the tenant (subject-keyed,
            # so they must be namespace-scoped to avoid cross-tenant aggregation).
            from memcontext.serving import serve_life_events, session_briefing

            briefing = session_briefing(conn, namespace=namespace)
            if briefing:
                result["briefing"] = briefing
            life = serve_life_events(conn, namespace=namespace)
            if life:
                result["life_events"] = life
        except Exception:  # noqa: BLE001 — resolved view is additive, never fatal
            pass

        # Cross-session distinct count for aggregation intent ("how many ... in
        # total"): reuse enumeration's sequence-capable session scope so the count
        # spans the tenant's sessions (the cross-session caller enumeration was
        # built for but never had). Embedding-based, so real-embedder only.
        if depth_kind == "aggregation" and claims_out:
            from memcontext.retrieval import episode_embedder, semantic_enabled

            if semantic_enabled():
                try:
                    from memcontext.enumeration import enumerate_retrieved

                    enum = enumerate_retrieved(
                        conn,
                        session_id=cross_session_sids,
                        retrieved_claims=claims_out,
                        embedder=episode_embedder(),
                    )
                    if enum is not None:
                        result["enumeration"] = {
                            "distinct_count": enum.distinct_count,
                            "t_dup": round(enum.t_dup, 4),
                            "representatives": [
                                {
                                    "representative": cl.representative,
                                    "member_claim_ids": list(cl.member_claim_ids),
                                    "event_ts_set": list(cl.event_ts_set),
                                }
                                for cl in enum.clusters
                            ],
                        }
                except Exception:  # noqa: BLE001 — enumeration is additive, never fatal
                    pass

    if debug and explain is not None:
        served = [c["claim_id"] for c in claims_out]
        result["ranking"] = {cid: explain[cid] for cid in served if cid in explain}
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


def handle_brain_across(
    conn: sqlite3.Connection,
    *,
    session_ids: list[str] | None = None,
    namespace: str | None = None,
) -> dict:
    """Cross-session resolved world-state — one current value per slot per subject,
    spanning a SET of sessions (a tenant's whole history). No LLM.

    Pass an explicit ``session_ids`` list, or a ``namespace`` to resolve over every
    session that has episodes in that tenant. When neither is given, resolves over
    every session in the store. Stale superseded values (per-slot, most-recent-wins)
    are absent; older instances remain queryable via the raw/history channel.
    """
    from memcontext.brain import brain_across

    if session_ids is None:
        if namespace is not None:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM turns WHERE namespace = ?", (namespace,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT DISTINCT session_id FROM turns").fetchall()
        session_ids = [
            r["session_id"] if isinstance(r, sqlite3.Row) else r[0] for r in rows
        ]
    return brain_across(conn, session_ids=session_ids)


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
