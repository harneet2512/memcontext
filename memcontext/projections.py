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
from memcontext.schema import Claim

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ActiveProjection:
    """The authoritative active claim set for a session at a point in time."""

    session_id: str
    claims: tuple[Claim, ...]

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
    conn: sqlite3.Connection, session_id: str
) -> ActiveProjection:
    """Rebuild the active-claims projection for a session."""
    claims = tuple(list_active_claims(conn, session_id))
    log.debug(
        "substrate.active_projection_rebuilt",
        session_id=session_id,
        active_count=len(claims),
    )
    return ActiveProjection(session_id=session_id, claims=claims)


def filtered_projection(
    active: ActiveProjection,
    predicate_matcher: Callable[[Claim], bool],
) -> FilteredProjection:
    """Project active claims to a domain-specific subset."""
    return active.filtered(predicate_matcher)


def claims_grouped_by_subject_predicate(
    claims: Iterable[Claim],
) -> dict[tuple[str, str], Claim]:
    """Keep only the newest active claim per (subject, predicate).

    Safety net for downstream consumers — if two active claims land for
    the same identity, return the most recent.
    """
    out: dict[tuple[str, str], Claim] = {}
    for c in claims:
        key = (c.subject, c.predicate)
        prev = out.get(key)
        if prev is None or c.created_ts > prev.created_ts:
            out[key] = c
    return out
