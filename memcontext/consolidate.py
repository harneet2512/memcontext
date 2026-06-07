"""Episodic -> semantic consolidation — graduate a fact that recurs across many
sessions into a single durable, high-importance 'consolidated' fact.

Deterministic, zero-LLM. The graduated (canonical) claim records provenance to
ALL its source claims; the redundant duplicates are demoted out of active
retrieval (not deleted). A consolidated fact can still be superseded or demoted
later. Reuses the importance (durability) + demoted (retention) machinery.
"""
from __future__ import annotations

import json
import sqlite3

import structlog

log = structlog.get_logger()


def consolidate_facts(conn: sqlite3.Connection, *, min_sessions: int = 3) -> int:
    """Graduate cross-session-recurring facts. Returns the number consolidated.

    A group of active claims sharing the same (subject, predicate, value) across
    >= ``min_sessions`` DISTINCT sessions graduates — UNLESS the slot is contested
    (the same (subject, predicate) has another active value), which signals a
    volatile/unsettled fact, not a stable semantic one. The earliest member becomes
    the durable consolidated fact (importance boosted, source provenance recorded);
    the remaining duplicates are demoted out of active retrieval.
    """
    groups = conn.execute(
        "SELECT subject, predicate, value,"
        " COUNT(DISTINCT session_id) AS n_sessions"
        " FROM claims"
        " WHERE status IN ('active','confirmed','audited')"
        "   AND subject IS NOT NULL AND predicate IS NOT NULL AND value IS NOT NULL"
        " GROUP BY subject, predicate, value"
        " HAVING n_sessions >= ?",
        (min_sessions,),
    ).fetchall()

    consolidated = 0
    for g in groups:
        subj, pred, val = g["subject"], g["predicate"], g["value"]

        # Contradiction / volatility guard: a different active value for the same
        # (subject, predicate) slot means it's contested -> not a stable fact.
        distinct_values = conn.execute(
            "SELECT COUNT(DISTINCT value) AS n FROM claims"
            " WHERE subject=? AND predicate=? AND value IS NOT NULL"
            "   AND status IN ('active','confirmed','audited')",
            (subj, pred),
        ).fetchone()["n"]
        if distinct_values > 1:
            continue

        members = conn.execute(
            "SELECT claim_id FROM claims"
            " WHERE subject=? AND predicate=? AND value=?"
            "   AND status IN ('active','confirmed','audited')"
            " ORDER BY created_ts ASC, claim_id ASC",
            (subj, pred, val),
        ).fetchall()
        ids = [m["claim_id"] for m in members]
        if len(ids) < 2:
            continue
        canonical, dups = ids[0], ids[1:]

        # Canonical becomes the durable consolidated fact: boosted importance,
        # consolidated flag, full source provenance; never demoted.
        conn.execute(
            "UPDATE claim_metadata"
            " SET consolidated = 1, consolidated_sources = ?,"
            "     importance_score = MAX(COALESCE(importance_score, 0.5), 0.9),"
            "     demoted = 0"
            " WHERE claim_id = ?",
            (json.dumps(ids), canonical),
        )
        # Redundant duplicates leave active retrieval (provenance preserved).
        placeholders = ",".join("?" for _ in dups)
        conn.execute(
            f"UPDATE claim_metadata SET demoted = 1 WHERE claim_id IN ({placeholders})",
            tuple(dups),
        )
        consolidated += 1
        log.info(
            "substrate.consolidated", subject=subj, predicate=pred,
            sessions=g["n_sessions"], sources=len(ids),
        )
    return consolidated
