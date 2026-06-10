"""Simple synchronous in-memory pub/sub — a public INTEGRATION SEAM.

`on_new_turn` publishes lifecycle events (claim created/superseded, projection
updated, ...). MemContext ships NO internal subscriber by design — this is an
extension point for the *host* (a UI, an async refinement worker, an audit log) to
`subscribe()` and react to memory changes without polling. It is intentionally
"unconsumed" in the package itself; that is not dead code, it is the hook.

Event names and payload shapes:
- turn.added                     {turn_id, session_id}
- claim.created                  {claim_id, session_id, predicate, status}
- claim.superseded               {old_claim_id, new_claim_id, edge_type,
                                  identity_score?}
- claim.status_changed           {claim_id, status}
- projection.updated             {session_id, active_count}
- output_sentence.added          {sentence_id, session_id, section,
                                  source_claim_ids}

Synchronous delivery. Subscriber exceptions are logged but do not crash
the publisher.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

log = structlog.get_logger(__name__)


TURN_ADDED = "turn.added"
CLAIM_CREATED = "claim.created"
CLAIM_SUPERSEDED = "claim.superseded"
CLAIM_STATUS_CHANGED = "claim.status_changed"
PROJECTION_UPDATED = "projection.updated"
OUTPUT_SENTENCE_ADDED = "output_sentence.added"


Payload = dict[str, Any]
Callback = Callable[[Payload], None]


class EventBus:
    """Tiny synchronous pub/sub."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Callback]] = {}

    def subscribe(self, event: str, callback: Callback) -> None:
        self._subs.setdefault(event, []).append(callback)
        log.debug("substrate.event_subscribe", event_name=event)

    def unsubscribe(self, event: str, callback: Callback) -> None:
        if event in self._subs and callback in self._subs[event]:
            self._subs[event].remove(callback)

    def publish(self, event: str, payload: Payload) -> None:
        """Deliver `payload` to every subscriber synchronously."""
        subs = self._subs.get(event, ())
        for cb in subs:
            try:
                cb(payload)
            except Exception:
                log.exception(
                    "substrate.event_subscriber_failed",
                    event_name=event,
                    callback=getattr(cb, "__qualname__", repr(cb)),
                )

    def subscriber_count(self, event: str) -> int:
        return len(self._subs.get(event, ()))
