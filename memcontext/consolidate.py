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

    A group of active claims of one namespace sharing the same (subject, predicate,
    value) across >= ``min_sessions`` DISTINCT sessions graduates — UNLESS the slot is contested
    (the same (subject, predicate) has another active value), which signals a
    volatile/unsettled fact, not a stable semantic one. The earliest member becomes
    the durable consolidated fact (importance boosted, source provenance recorded);
    the remaining duplicates are demoted out of active retrieval.
    """
    # Every query is confined to one namespace (tenant): pooling tenants let one
    # tenant's copy become canonical and demote another tenant's facts.
    live = "c.status IN ('active','confirmed','audited')"
    scoped = "FROM claims c JOIN turns t ON t.turn_id = c.source_turn_id"
    groups = conn.execute(
        f"SELECT t.namespace AS namespace, c.subject AS subject, c.predicate AS predicate,"
        f" c.value AS value, COUNT(DISTINCT c.session_id) AS n_sessions {scoped}"
        f" WHERE {live}"
        "   AND c.subject IS NOT NULL AND c.predicate IS NOT NULL AND c.value IS NOT NULL"
        " GROUP BY t.namespace, c.subject, c.predicate, c.value"
        " HAVING n_sessions >= ?",
        (min_sessions,),
    ).fetchall()

    consolidated = 0
    for g in groups:
        ns, subj, pred, val = g["namespace"], g["subject"], g["predicate"], g["value"]

        # Contradiction / volatility guard: a different active value for the same
        # (subject, predicate) slot in this tenant means it's contested.
        distinct_values = conn.execute(
            f"SELECT COUNT(DISTINCT c.value) AS n {scoped}"
            f" WHERE t.namespace=? AND c.subject=? AND c.predicate=? AND c.value IS NOT NULL"
            f"   AND {live}",
            (ns, subj, pred),
        ).fetchone()["n"]
        if distinct_values > 1:
            continue

        members = conn.execute(
            f"SELECT c.claim_id {scoped}"
            f" WHERE t.namespace=? AND c.subject=? AND c.predicate=? AND c.value=? AND {live}"
            " ORDER BY c.created_ts ASC, c.claim_id ASC",
            (ns, subj, pred, val),
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
            "substrate.consolidated", namespace=ns, subject=subj, predicate=pred,
            sessions=g["n_sessions"], sources=len(ids),
        )
    return consolidated
