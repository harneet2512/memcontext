"""Utility-weighted retention — demote low-utility, old claims out of active
retrieval so the active set stays small and high-value (token efficiency).

Deterministic, zero-LLM. Demotion is a reversible flag on ``claim_metadata``
(``demoted``); the claim and its full provenance are never deleted, so a demoted
fact remains traceable and can be reinstated.
"""
from __future__ import annotations

import sqlite3
import time

import structlog

log = structlog.get_logger()

_NS_PER_DAY = 86_400 * 1_000_000_000


def compute_utility(conn: sqlite3.Connection, claim_id: str) -> float:
    """Deterministic utility in [0, 1]: how worth-keeping a claim is.

    Combines the signals MemContext already tracks — importance, usage
    (access_count), recency, and confidence — into one score. Zero-LLM.
    """
    row = conn.execute(
        "SELECT c.confidence, c.created_ts,"
        " COALESCE(m.importance_score, 0.5) AS importance,"
        " COALESCE(m.access_count, 0) AS access_count,"
        " m.last_accessed_ts AS last_accessed_ts"
        " FROM claims c LEFT JOIN claim_metadata m ON c.claim_id = m.claim_id"
        " WHERE c.claim_id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        return 0.5
    importance = float(row["importance"])
    confidence = float(row["confidence"] or 0.0)
    access = float(row["access_count"])
    usage = access / (access + 3.0)  # saturating: 0 at 0, -> 1 as access grows
    # Recency from the most recent of creation / last access — a recently-served
    # fact is "fresher" than its creation date alone (connects last_accessed_ts).
    last_seen = max(int(row["created_ts"]), int(row["last_accessed_ts"] or 0))
    age_days = max(0.0, (time.time_ns() - last_seen) / _NS_PER_DAY)
    recency = 1.0 / (1.0 + age_days / 90.0)  # decays over ~90 days
    return round(
        0.4 * importance + 0.3 * usage + 0.2 * recency + 0.1 * confidence, 4
    )


def demote_low_utility(
    conn: sqlite3.Connection,
    *,
    threshold: float = 0.35,
    min_age_days: float = 30.0,
) -> int:
    """Demote active claims with utility < ``threshold`` AND older than
    ``min_age_days`` out of active retrieval (``claim_metadata.demoted = 1``).

    Reversible; never deletes. Returns the number demoted. Zero-LLM.
    """
    cutoff_ts = time.time_ns() - int(min_age_days * _NS_PER_DAY)
    rows = conn.execute(
        "SELECT claim_id FROM claims"
        " WHERE status IN ('active','confirmed','audited') AND created_ts < ?",
        (cutoff_ts,),
    ).fetchall()
    demoted = 0
    for r in rows:
        cid = r[0]
        if compute_utility(conn, cid) < threshold:
            conn.execute(
                "UPDATE claim_metadata SET demoted = 1 WHERE claim_id = ?", (cid,)
            )
            demoted += 1
    log.info("substrate.demoted_low_utility", count=demoted, threshold=threshold)
    return demoted
