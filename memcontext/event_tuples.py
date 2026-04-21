"""Event-tuple projection over Claims.

A claim (subject, predicate, value) with temporal-validity window
(valid_from_ts, valid_until_ts) projects onto an event tuple —
(subject, action, object, valid_from, valid_until). This is a pure
read-side projection — it never writes back to the substrate.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from memcontext.schema import Claim


@dataclass(frozen=True, slots=True)
class EventTuple:
    """(subject, action, object, valid_from, valid_until) projection of a Claim."""

    subject: str
    action: str
    obj: str
    valid_from_ts: int | None
    valid_until_ts: int | None
    claim_id: str


def claim_to_event(claim: Claim) -> EventTuple:
    return EventTuple(
        subject=claim.subject,
        action=claim.predicate,
        obj=claim.value,
        valid_from_ts=claim.valid_from_ts,
        valid_until_ts=claim.valid_until_ts,
        claim_id=claim.claim_id,
    )


def claims_to_events(claims: Iterable[Claim]) -> list[EventTuple]:
    """Order-preserving projection from Claims to EventTuples."""
    return [claim_to_event(c) for c in claims]


__all__ = [
    "EventTuple",
    "claim_to_event",
    "claims_to_events",
]
