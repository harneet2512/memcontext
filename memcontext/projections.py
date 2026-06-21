"""Projections — materialised views over active claims.

Two projection kinds:

1. Active-claims projection — the session's current active/confirmed claim
   set, one row per (subject, predicate).

2. Filtered projection — filter of the active-claims projection by an
   arbitrary predicate matcher. Domain-specific code (e.g. a differential
   engine) provides the matcher function.

Pure Python, no LLM, no randomness.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import structlog

from memcontext.claims import list_active_claims
from memcontext.schema import Claim, Turn

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ActiveProjection:
    """The authoritative active claim set for a session at a point in time.

    `episodes` is populated ONLY in the degraded case — when the session has no
    active facts (extraction disabled or still pending) — with the most-recent
    episodes, so the projection is never empty while episodes exist. When facts
    are present, `episodes` is empty and `claims` is authoritative.
    """

    session_id: str
    claims: tuple[Claim, ...]
    episodes: tuple[Turn, ...] = ()

    @property
    def is_episode_backed(self) -> bool:
        """True when this projection is degraded to episodes (no active facts)."""
        return not self.claims and bool(self.episodes)

    @property
    def by_predicate(self) -> dict[str, tuple[Claim, ...]]:
        """Claims grouped by predicate (stable order preserved)."""
        groups: dict[str, list[Claim]] = {}
        for c in self.claims:
            groups.setdefault(c.predicate, []).append(c)
        return {k: tuple(v) for k, v in groups.items()}

    def filtered(self, predicate_matcher: Callable[[Claim], bool]) -> FilteredProjection:
        filtered = tuple(c for c in self.claims if predicate_matcher(c))
        return FilteredProjection(session_id=self.session_id, claims=filtered)


@dataclass(frozen=True, slots=True)
class FilteredProjection:
    """Active claims filtered by domain-specific criteria."""

    session_id: str
    claims: tuple[Claim, ...]


# ---------------------------------------------------------- rebuild helpers ---


def rebuild_active_projection(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    episode_fallback_k: int = 10,
) -> ActiveProjection:
    """Rebuild the active-claims projection for a session.

    Draws from active facts. When a session has NO active facts (Tier-2 empty —
    extraction disabled or still pending), it degrades to the `episode_fallback_k`
    most-recent episodes so the projection is never empty while episodes exist
    (the Tier-1 graceful-degradation floor).
    """
    claims = tuple(list_active_claims(conn, session_id))
    episodes: tuple[Turn, ...] = ()
    if not claims:
        from memcontext.claims import row_to_turn
        rows = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
            (session_id, episode_fallback_k),
        ).fetchall()
        episodes = tuple(row_to_turn(r) for r in rows)
    log.debug(
        "substrate.active_projection_rebuilt",
        session_id=session_id,
        active_count=len(claims),
        episode_fallback=len(episodes),
    )
    return ActiveProjection(
        session_id=session_id, claims=claims, episodes=episodes
    )


def filtered_projection(
    active: ActiveProjection,
    predicate_matcher: Callable[[Claim], bool],
) -> FilteredProjection:
    """Project active claims to a domain-specific subset."""
    return active.filtered(predicate_matcher)


def claims_grouped_by_subject_predicate(
    claims: Iterable[Claim],
) -> dict[tuple[str, str], Claim]:
    """Keep only the newest active claim per identity slot.

    Safety net for downstream consumers — if two active claims land for
    the same identity, return the most recent.

    FRACTURE B: the identity key is ``(subject, predicate, attribute)``, where
    ``attribute`` is a deterministic slot token read off the VALUE
    (attribute_key.py). Under a COARSE predicate ('user_fact'), keying on only
    ``(subject, predicate)`` would collapse EVERY personal fact (residence,
    employer, hobby …) into one row — newest-wins silently deletes the rest.
    The attribute splits genuinely-different slots apart while still collapsing
    true restatements of the same slot. The attribute defaults to ``""`` when no
    slot is derivable, so fine-grained predicates key exactly as before (no
    regression): the tuple becomes ``(subject, predicate, "")``.
    """
    from memcontext.attribute_key import attribute_key

    out: dict[tuple[str, str, str], Claim] = {}
    for c in claims:
        key = (c.subject, c.predicate, attribute_key(c.value))
        prev = out.get(key)
        if prev is None or c.created_ts > prev.created_ts:
            out[key] = c
    return out
