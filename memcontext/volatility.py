"""Volatility classification — deterministic, zero LLM."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

_NS_PER_DAY: float = 86400 * 1e9


@dataclass(frozen=True, slots=True)
class VolatilityInfo:
    classification: str  # "stable" | "evolving" | "volatile"
    change_count: int
    avg_lifespan_days: float
    current_streak_days: float


def classify_predicate(
    conn: sqlite3.Connection, subject: str, predicate: str
) -> VolatilityInfo:
    """Classify the volatility of a (subject, predicate) pair.

    Examines the supersession history to determine how frequently the
    predicate value changes for this subject.

    * **stable** — 0 supersession events
    * **evolving** — 1-2 supersession events
    * **volatile** — 3+ supersession events

    All computation is deterministic; no LLM calls.
    """
    # Count supersession events where both the old and new claim share
    # the given (subject, predicate).
    change_rows = conn.execute(
        "SELECT e.edge_id, c_old.valid_from_ts, c_old.valid_until_ts"
        " FROM supersession_edges e"
        " JOIN claims c_old ON e.old_claim_id = c_old.claim_id"
        " JOIN claims c_new ON e.new_claim_id = c_new.claim_id"
        " WHERE c_old.subject = ? AND c_old.predicate = ?"
        " AND c_new.subject = ? AND c_new.predicate = ?",
        (subject, predicate, subject, predicate),
    ).fetchall()

    change_count = len(change_rows)

    # Classification.
    if change_count == 0:
        classification = "stable"
    elif change_count <= 2:
        classification = "evolving"
    else:
        classification = "volatile"

    # Average lifespan of superseded claims (those that have valid_until_ts).
    lifespan_total = 0.0
    lifespan_n = 0
    for row in change_rows:
        vfrom = row["valid_from_ts"]
        vuntil = row["valid_until_ts"]
        if vfrom is not None and vuntil is not None and vuntil > vfrom:
            lifespan_total += (vuntil - vfrom) / _NS_PER_DAY
            lifespan_n += 1
    avg_lifespan_days = lifespan_total / lifespan_n if lifespan_n > 0 else 0.0

    # Current streak: how long the current active claim has been alive.
    now_ns = time.time_ns()
    active_row = conn.execute(
        "SELECT created_ts FROM claims"
        " WHERE subject = ? AND predicate = ?"
        " AND status IN ('active', 'confirmed', 'audited')"
        " ORDER BY created_ts DESC LIMIT 1",
        (subject, predicate),
    ).fetchone()
    if active_row is not None:
        current_streak_days = (now_ns - active_row["created_ts"]) / _NS_PER_DAY
    else:
        current_streak_days = 0.0

    result = VolatilityInfo(
        classification=classification,
        change_count=change_count,
        avg_lifespan_days=round(avg_lifespan_days, 4),
        current_streak_days=round(current_streak_days, 4),
    )
    log.debug(
        "volatility.classified",
        subject=subject,
        predicate=predicate,
        classification=classification,
        change_count=change_count,
    )
    return result
