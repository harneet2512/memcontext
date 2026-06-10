"""Orchestrator — wire admission → extract → supersede → project → publish.

One synchronous function that takes a raw turn (text + speaker + metadata)
and an extractor callable, runs the whole substrate pipeline, and emits
events on the bus.

The extractor callable is injected (not imported) so domain-specific
extractors can be plugged in without the substrate depending on any LLM stack.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from memcontext import admission
from memcontext.claims import (
    ClaimValidationError,
    get_turn,
    insert_claim,
    insert_turn,
    new_turn_id,
    now_ns,
)
from memcontext.event_bus import (
    CLAIM_CREATED,
    CLAIM_SUPERSEDED,
    PROJECTION_UPDATED,
    TURN_ADDED,
    EventBus,
)
from memcontext.projections import rebuild_active_projection
from memcontext.schema import (
    Claim,
    ExtractionStatus,
    Speaker,
    SupersessionEdge,
    Turn,
)
from memcontext.supersession import detect_pass1
from memcontext.supersession_semantic import SemanticSupersession

if TYPE_CHECKING:
    from memcontext.extraction_queue import ExtractionQueue
    from memcontext.retrieval import EmbeddingClient

log = structlog.get_logger(__name__)


# ---------------------------------------------------------- extractor API ---


@dataclass(frozen=True, slots=True)
class ExtractedClaim:
    """Input-facing claim shape returned by the extractor.

    Mirrors Claim minus server-assigned fields (claim_id, status,
    created_ts, session_id, source_turn_id).
    """

    subject: str
    predicate: str
    value: str
    confidence: float
    value_normalised: str | None = None
    char_start: int | None = None
    char_end: int | None = None


ExtractorFn = Callable[[Turn], list[ExtractedClaim]]


# ------------------------------------------------------------- result type ---


@dataclass(frozen=True, slots=True)
class TurnResult:
    """What on_new_turn actually did.

    turn is None iff admission rejected the text.
    dropped_claims lists raw extractor outputs that failed validation.
    """

    turn: Turn | None
    admitted: bool
    admission_reason: str
    created_claims: tuple[Claim, ...]
    supersession_edges: tuple[SupersessionEdge, ...]
    dropped_claims: tuple[tuple[ExtractedClaim, str], ...]


# ----------------------------------------------------------- orchestrator ---


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """What `run_extraction` produced for one episode."""

    created_claims: tuple[Claim, ...]
    supersession_edges: tuple[SupersessionEdge, ...]
    dropped_claims: tuple[tuple[ExtractedClaim, str], ...]


def _set_extraction_status(
    conn: sqlite3.Connection, episode_id: str, status: ExtractionStatus
) -> None:
    conn.execute(
        "UPDATE turns SET extraction_status = ? WHERE turn_id = ?",
        (status.value, episode_id),
    )


def run_extraction(
    conn: sqlite3.Connection,
    *,
    episode_id: str,
    session_id: str,
    extractor: ExtractorFn,
    semantic: SemanticSupersession | None = None,
    bus: EventBus | None = None,
    done_status: ExtractionStatus = ExtractionStatus.DONE,
) -> ExtractionResult:
    """Extract facts from a stored episode and supersede/project (Tier-2 tail).

    This is the extract -> insert-facts -> Pass-1 -> Pass-2 -> rebuild-projection
    sequence, factored out of `on_new_turn` so it can run either inline (sync
    extractors) or later on a queue worker (async LLM extraction). It reloads the
    episode by id, so the caller need not hold the Turn. Marks the episode's
    `extraction_status` (`done_status` if facts were produced, else SKIPPED).
    """
    bus = bus or EventBus()
    turn = get_turn(conn, episode_id)
    if turn is None:
        log.warning("substrate.extraction_episode_missing", episode_id=episode_id)
        return ExtractionResult((), (), ())

    if hasattr(extractor, "set_context"):
        prior_rows = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? AND ts < ? ORDER BY ts DESC LIMIT 8",
            (session_id, turn.ts),
        ).fetchall()
        prior_turns = [
            Turn(
                turn_id=r["turn_id"], session_id=r["session_id"],
                speaker=Speaker(r["speaker"]), text=r["text"],
                ts=r["ts"], asr_confidence=r["asr_confidence"],
            )
            for r in reversed(prior_rows)
        ]
        extractor.set_context(prior_turns)

    extracted = extractor(turn)

    created: list[Claim] = []
    edges: list[SupersessionEdge] = []
    dropped: list[tuple[ExtractedClaim, str]] = []

    for ec in extracted:
        try:
            claim = insert_claim(
                conn,
                session_id=session_id,
                subject=ec.subject,
                predicate=ec.predicate,
                value=ec.value,
                confidence=ec.confidence,
                source_turn_id=turn.turn_id,
                value_normalised=ec.value_normalised,
                char_start=ec.char_start,
                char_end=ec.char_end,
            )
        except ClaimValidationError as exc:
            log.warning(
                "substrate.claim_dropped",
                session_id=session_id,
                turn_id=turn.turn_id,
                subject=ec.subject,
                predicate=ec.predicate,
                reason=str(exc),
            )
            dropped.append((ec, str(exc)))
            continue

        created.append(claim)
        bus.publish(
            CLAIM_CREATED,
            {
                "claim_id": claim.claim_id,
                "session_id": session_id,
                "predicate": claim.predicate,
                "status": claim.status.value,
            },
        )

        edge1 = detect_pass1(conn, claim)
        if edge1 is not None:
            edges.append(edge1)
            bus.publish(
                CLAIM_SUPERSEDED,
                {
                    "old_claim_id": edge1.old_claim_id,
                    "new_claim_id": edge1.new_claim_id,
                    "edge_type": edge1.edge_type.value,
                    "identity_score": edge1.identity_score,
                },
            )

        if semantic is not None and edge1 is None:
            edge2 = semantic.detect(conn, claim, new_turn_text=turn.text)
            if edge2 is not None:
                edges.append(edge2)
                bus.publish(
                    CLAIM_SUPERSEDED,
                    {
                        "old_claim_id": edge2.old_claim_id,
                        "new_claim_id": edge2.new_claim_id,
                        "edge_type": edge2.edge_type.value,
                        "identity_score": edge2.identity_score,
                    },
                )

    proj = rebuild_active_projection(conn, session_id)
    bus.publish(
        PROJECTION_UPDATED,
        {"session_id": session_id, "active_count": len(proj.claims)},
    )

    try:
        # Importance for EVERY new claim at ingest, not only superseded ones, so the
        # importance ranking signal has real values instead of a flat 0.5 default in
        # the common never-superseded case. Deterministic + zero-LLM; new_claim_ids
        # are already in `created`, so only the retired old claims need a recompute.
        from memcontext.importance import compute_importance
        for _claim in created:
            compute_importance(conn, _claim.claim_id)
        for edge in edges:
            compute_importance(conn, edge.old_claim_id)

        turn_count = conn.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        if turn_count % 10 == 0 and turn_count > 0:
            from memcontext.profiles import build_smart_profile, store_profile
            profile = build_smart_profile(conn, "user")
            store_profile(conn, profile)
            # Session digest alongside the profile (same per-session cadence): top facts
            # by importance + supersession updates, cached so the serve path
            # (build_context_briefing) can return a session summary without rebuilding it
            # per query. Previously digests were only ever built via the memory_digest tool.
            from memcontext.digests import build_session_digest, store_digest
            store_digest(conn, build_session_digest(conn, session_id))

            # Episodic layer: assemble multi-slot event frames (purchases, trips,
            # appointments, named artifacts...) and detect life-event bursts. Both are
            # deterministic and were previously ONLY built via their MCP tools, so the
            # episodic memory layer was dormant in any live deployment. assemble is
            # idempotent (clears the session first); life_events are cleared by subject
            # before re-detect (random ids). Frame embeddings backfill only when a real
            # embedder is configured — lexical-only mode still assembles + serves frames.
            from memcontext.event_frames import assemble_event_frames
            from memcontext.life_events import detect_life_events, store_life_events
            assemble_event_frames(conn, session_id)
            conn.execute("DELETE FROM life_events WHERE subject = ?", ("user",))
            store_life_events(conn, detect_life_events(conn, "user"))
            from memcontext.retrieval import backfill_event_frame_embeddings, episode_embedder
            _emb = episode_embedder()
            if _emb is not None:
                backfill_event_frame_embeddings(conn, session_id, client=_emb)

        # Cross-session consolidation — graduate facts that recur across >= 3 distinct
        # sessions into durable 'consolidated' facts. Previously this ran ONLY via the
        # manual `memcontext consolidate` CLI, so it never happened in a live deployment;
        # wiring it here closes that "never auto-runs" gap. Triggered on a GLOBAL turn
        # cadence (not per-session): the recurrence it detects spans many — often short —
        # sessions, so a per-session counter would rarely fire. Deterministic, zero-LLM,
        # and self-guarded (contested slots are skipped), so it is safe to run inline.
        global_turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        if global_turns % 25 == 0 and global_turns > 0:
            from memcontext.consolidate import consolidate_facts
            consolidate_facts(conn)
            # Importance DECAY over time: recency + stability are time-dependent signals,
            # but importance was only evaluated at INSERT — so a fact's stored importance
            # never re-decayed and long-stable facts never accrued stability. (Live
            # recency still ranks via retrieval's temporal channel; this fixes the STORED
            # signal that digests/profiles read.) Recompute the active set on this coarse
            # cadence so decay actually runs. O(active claims) per tick — fine for a
            # personal brain; batch by age if the corpus ever gets very large.
            from memcontext.importance import recompute_all_importance
            recompute_all_importance(conn)
    except Exception:  # noqa: BLE001
        pass

    _set_extraction_status(
        conn, episode_id, done_status if created else ExtractionStatus.SKIPPED
    )
    return ExtractionResult(tuple(created), tuple(edges), tuple(dropped))


def on_new_turn(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    speaker: Speaker,
    text: str,
    extractor: ExtractorFn,
    bus: EventBus | None = None,
    semantic: SemanticSupersession | None = None,
    asr_confidence: float | None = None,
    turn_id: str | None = None,
    queue: ExtractionQueue | None = None,
    embedder: EmbeddingClient | None = None,
    namespace: str = "default",
) -> TurnResult:
    """Ingest one turn (episode) end-to-end.

    Tier-1 (always, synchronous, zero-LLM): admit -> persist episode -> embed
    episode (if an embedder is given). Tier-2 (facts):
    - If the extractor is deferrable (an LLMExtractor) AND a `queue` is provided,
      enqueue the episode and return immediately with no facts (status `pending`);
      the facts are produced later by `queue.drain()` / the worker.
    - Otherwise extraction runs inline via `run_extraction` (current behaviour),
      and the returned TurnResult carries the created facts.
    """
    bus = bus or EventBus()

    adm = admission.admit(text)
    if not adm.admitted:
        log.info(
            "substrate.turn_rejected",
            session_id=session_id,
            reason=adm.reason,
            text_len=len(text),
        )
        return TurnResult(
            turn=None,
            admitted=False,
            admission_reason=adm.reason,
            created_claims=(),
            supersession_edges=(),
            dropped_claims=(),
        )

    turn = Turn(
        turn_id=turn_id or new_turn_id(),
        session_id=session_id,
        speaker=speaker,
        text=text,
        ts=now_ns(),
        asr_confidence=asr_confidence,
    )
    insert_turn(conn, turn, namespace=namespace)
    bus.publish(TURN_ADDED, {"turn_id": turn.turn_id, "session_id": session_id})

    # Tier-1 floor: embed the episode synchronously (local model, never an LLM).
    # Never block ingest on an embedding failure.
    if embedder is not None:
        try:
            from memcontext.retrieval import embed_and_store_episode
            embed_and_store_episode(conn, turn, client=embedder)
        except Exception:  # noqa: BLE001
            log.warning("substrate.episode_embed_failed", turn_id=turn.turn_id)

    deferrable = bool(getattr(extractor, "is_deferrable", False))
    if deferrable and queue is not None:
        _set_extraction_status(conn, turn.turn_id, ExtractionStatus.PENDING)
        queue.enqueue(turn.turn_id, session_id)
        return TurnResult(
            turn=turn,
            admitted=True,
            admission_reason=adm.reason,
            created_claims=(),
            supersession_edges=(),
            dropped_claims=(),
        )

    result = run_extraction(
        conn,
        episode_id=turn.turn_id,
        session_id=session_id,
        extractor=extractor,
        semantic=semantic,
        bus=bus,
        done_status=ExtractionStatus.STRUCTURED,
    )
    return TurnResult(
        turn=turn,
        admitted=True,
        admission_reason=adm.reason,
        created_claims=result.created_claims,
        supersession_edges=result.supersession_edges,
        dropped_claims=result.dropped_claims,
    )
