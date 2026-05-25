"""Importance scoring — deterministic, zero LLM."""
from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

# --------------------------------------------------------- signal weights ---

_W_UNIQUENESS: float = 0.25
_W_SUPERSESSION: float = 0.20
_W_CONFIDENCE: float = 0.15
_W_RECENCY: float = 0.15
_W_STABILITY: float = 0.15
_W_CROSS_SESSION: float = 0.10

_NS_PER_DAY: float = 86400 * 1e9
_RECENCY_HALF_LIFE_DAYS: float = 30.0
_STABILITY_PLATEAU_DAYS: float = 180.0
_CROSS_SESSION_CAP: int = 5


# ------------------------------------------------------- signal helpers ---


def _signal_uniqueness(conn: sqlite3.Connection, subject: str, predicate: str) -> float:
    """1/N where N = count of active claims with the same (subject, predicate)."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM claims"
        " WHERE subject = ? AND predicate = ?"
        " AND status IN ('active', 'confirmed', 'audited')",
        (subject, predicate),
    ).fetchone()
    n = row["n"] if row else 1
    return 1.0 / max(n, 1)


def _signal_supersession(conn: sqlite3.Connection, claim_id: str) -> float:
    """1.0 if this claim superseded another, 0.0 if it was superseded, else 0.5."""
    # Check if this claim is on the *new* side of any edge (it replaced something).
    row_new = conn.execute(
        "SELECT 1 FROM supersession_edges WHERE new_claim_id = ? LIMIT 1",
        (claim_id,),
    ).fetchone()
    if row_new is not None:
        return 1.0

    # Check if this claim is on the *old* side (it was superseded).
    row_old = conn.execute(
        "SELECT 1 FROM supersession_edges WHERE old_claim_id = ? LIMIT 1",
        (claim_id,),
    ).fetchone()
    if row_old is not None:
        return 0.0

    # No edges at all.
    return 0.5


def _signal_confidence(conn: sqlite3.Connection, claim_id: str) -> float:
    """Pass-through from the claim's confidence column."""
    row = conn.execute(
        "SELECT confidence FROM claims WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    return float(row["confidence"]) if row else 0.5


def _signal_recency(created_ts: int, now_ns: int) -> float:
    """Exponential decay: exp(-days/30), clamped to [0, 1]."""
    days = (now_ns - created_ts) / _NS_PER_DAY
    score = math.exp(-days / _RECENCY_HALF_LIFE_DAYS)
    return max(0.0, min(1.0, score))


def _signal_stability(created_ts: int, now_ns: int) -> float:
    """Linear ramp: min(1.0, days_active / 180)."""
    days_active = (now_ns - created_ts) / _NS_PER_DAY
    return min(1.0, days_active / _STABILITY_PLATEAU_DAYS)


def _signal_cross_session(conn: sqlite3.Connection, claim_id: str) -> float:
    """Distinct session count sharing the same entity_key, normalized by 5."""
    row = conn.execute(
        "SELECT entity_key FROM claim_metadata WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        return 0.0

    entity_key = row["entity_key"]
    count_row = conn.execute(
        "SELECT COUNT(DISTINCT c.session_id) AS cnt"
        " FROM claim_metadata cm"
        " JOIN claims c ON cm.claim_id = c.claim_id"
        " WHERE cm.entity_key = ?"
        " AND c.status IN ('active', 'confirmed', 'audited')",
        (entity_key,),
    ).fetchone()
    cnt = count_row["cnt"] if count_row else 0
    return min(1.0, cnt / _CROSS_SESSION_CAP)


# --------------------------------------------------------- public API ---


def compute_importance(conn: sqlite3.Connection, claim_id: str) -> float:
    """Compute and persist the importance score for a single claim.

    Six deterministic signals are combined via weighted sum, normalized to
    [0, 1].  The result is written back to
    ``claim_metadata.importance_score``.

    Returns the computed score.
    """
    # Fetch the claim row for subject/predicate/created_ts.
    row = conn.execute(
        "SELECT subject, predicate, created_ts FROM claims WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        log.warning("importance.claim_not_found", claim_id=claim_id)
        return 0.0

    subject: str = row["subject"]
    predicate: str = row["predicate"]
    created_ts: int = row["created_ts"]
    now_ns = time.time_ns()

    score = (
        _W_UNIQUENESS * _signal_uniqueness(conn, subject, predicate)
        + _W_SUPERSESSION * _signal_supersession(conn, claim_id)
        + _W_CONFIDENCE * _signal_confidence(conn, claim_id)
        + _W_RECENCY * _signal_recency(created_ts, now_ns)
        + _W_STABILITY * _signal_stability(created_ts, now_ns)
        + _W_CROSS_SESSION * _signal_cross_session(conn, claim_id)
    )
    # Clamp to [0, 1] for safety (weights already sum to 1.0).
    score = max(0.0, min(1.0, score))

    conn.execute(
        "UPDATE claim_metadata SET importance_score = ? WHERE claim_id = ?",
        (score, claim_id),
    )
    log.debug(
        "importance.computed",
        claim_id=claim_id,
        score=round(score, 4),
    )
    return score


def recompute_all_importance(conn: sqlite3.Connection) -> int:
    """Recompute importance scores for every active claim.

    Returns the number of claims updated.
    """
    rows = conn.execute(
        "SELECT claim_id FROM claims"
        " WHERE status IN ('active', 'confirmed', 'audited')"
    ).fetchall()

    count = 0
    for row in rows:
        compute_importance(conn, row["claim_id"])
        count += 1

    log.info("importance.recomputed_all", count=count)
    return count
