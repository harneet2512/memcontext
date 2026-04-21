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

import structlog

from memcontext import admission
from memcontext.claims import (
    ClaimValidationError,
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
from memcontext.schema import Claim, Speaker, SupersessionEdge, Turn
from memcontext.supersession import detect_pass1
from memcontext.supersession_semantic import SemanticSupersession

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
) -> TurnResult:
    """Ingest one turn end-to-end.

    Pipeline:
      1. Admission filter (noise regex).
      2. Persist Turn.
      3. Run injected extractor.
      4. For each extracted claim:
           a. Validate + insert.
           b. Pass-1 deterministic supersession.
           c. Pass-2 semantic supersession (if provided).
      5. Rebuild active projection.
      6. Return TurnResult.
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
    insert_turn(conn, turn)
    bus.publish(TURN_ADDED, {"turn_id": turn.turn_id, "session_id": session_id})

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

    return TurnResult(
        turn=turn,
        admitted=True,
        admission_reason=adm.reason,
        created_claims=tuple(created),
        supersession_edges=tuple(edges),
        dropped_claims=tuple(dropped),
    )
