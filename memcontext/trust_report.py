"""Trust observability — measure whether the trust/governance layer is working,
not just whether recall is (GOVERNANCE_AUDIT G was ABSENT).

Surfaces, deterministically and zero-LLM, the signals that the rest of the layer
produces: source-trust distribution + quarantine (P3/P4), contradiction-surfacing
rate (the typed edges), forgetting + drift audit (P2/P4 via the decisions log),
tenant distribution (P5), and a cold-memory (staleness) proxy.
"""
from __future__ import annotations

import json
import sqlite3


def trust_status(conn: sqlite3.Connection) -> dict:
    """Compute the trust/governance posture metrics from the live substrate."""

    def scalar(sql: str, *args) -> int:
        row = conn.execute(sql, args).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    active = scalar("SELECT COUNT(*) FROM claims WHERE status IN ('active','confirmed')")
    superseded = scalar("SELECT COUNT(*) FROM claims WHERE status = 'superseded'")
    total_claims = active + superseded

    trusted = scalar("SELECT COUNT(*) FROM claim_metadata WHERE source_trust >= 0.7")
    neutral = scalar("SELECT COUNT(*) FROM claim_metadata WHERE source_trust >= 0.4 AND source_trust < 0.7")
    quarantined = scalar("SELECT COUNT(*) FROM claim_metadata WHERE source_trust < 0.4")
    trust_total = trusted + neutral + quarantined

    total_edges = scalar("SELECT COUNT(*) FROM supersession_edges")
    contradictions = scalar(
        "SELECT COUNT(*) FROM supersession_edges WHERE edge_type IN ('contradicts','dismissed_by_user')"
    )

    forget_actions = scalar("SELECT COUNT(*) FROM decisions WHERE kind = 'forget'")
    drift_blocked = scalar("SELECT COUNT(*) FROM decisions WHERE kind = 'drift_blocked'")
    claims_erased = 0
    for (snap,) in conn.execute(
        "SELECT claim_state_snapshot FROM decisions WHERE kind = 'forget'"
    ).fetchall():
        try:
            claims_erased += len(json.loads(snap))
        except (TypeError, json.JSONDecodeError):
            pass

    namespaces = {
        r[0]: int(r[1]) for r in conn.execute(
            "SELECT namespace, COUNT(DISTINCT session_id) FROM turns GROUP BY namespace"
        ).fetchall()
    }

    # Staleness proxy: active claims never read back (cold) -> may be stale/unused.
    cold = scalar(
        "SELECT COUNT(*) FROM claims c"
        " LEFT JOIN claim_metadata m ON c.claim_id = m.claim_id"
        " WHERE c.status IN ('active','confirmed') AND COALESCE(m.access_count, 0) = 0"
    )

    def frac(n: int, d: int) -> float:
        return round(n / d, 3) if d else 0.0

    return {
        "active_claims": active,
        "superseded_claims": superseded,
        "supersession_rate": frac(superseded, total_claims),
        "source_trust": {"trusted": trusted, "neutral": neutral, "quarantined": quarantined},
        "quarantined_fraction": frac(quarantined, trust_total),
        "contradiction_rate": frac(contradictions, total_edges),
        "forgetting": {
            "forget_actions": forget_actions,
            "claims_erased": claims_erased,
            "drift_blocked": drift_blocked,
        },
        "namespaces": namespaces,
        "tenant_count": len(namespaces),
        "cold_fraction": frac(cold, active),
    }
