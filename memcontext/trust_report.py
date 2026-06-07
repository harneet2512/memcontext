"""Trust observability — measure whether the trust/governance layer is working,
not just whether recall is (GOVERNANCE_AUDIT G was ABSENT).

Surfaces, deterministically and zero-LLM, the signals that the rest of the layer
produces: source-trust distribution + quarantine (P3/P4), contradiction-surfacing
rate (the typed edges), forgetting + drift audit (P2/P4 via the decisions log),
tenant distribution (P5), and a cold-memory (staleness) proxy.
"""
from __future__ import annotations

import contextlib
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

    from memcontext.source_trust import QUARANTINE_THRESHOLD

    trusted = scalar("SELECT COUNT(*) FROM claim_metadata WHERE source_trust >= 0.7")
    neutral = scalar(
        "SELECT COUNT(*) FROM claim_metadata WHERE source_trust >= ? AND source_trust < 0.7",
        QUARANTINE_THRESHOLD,
    )
    quarantined = scalar(
        "SELECT COUNT(*) FROM claim_metadata WHERE source_trust < ?", QUARANTINE_THRESHOLD
    )
    trust_total = trusted + neutral + quarantined

    total_edges = scalar("SELECT COUNT(*) FROM supersession_edges")
    contradictions = scalar(
        "SELECT COUNT(*) FROM supersession_edges"
        " WHERE edge_type IN ('contradicts','dismissed_by_user')"
    )

    forget_actions = scalar("SELECT COUNT(*) FROM decisions WHERE kind = 'forget'")
    drift_blocked = scalar("SELECT COUNT(*) FROM decisions WHERE kind = 'drift_blocked'")
    claims_erased = 0
    for (snap,) in conn.execute(
        "SELECT claim_state_snapshot FROM decisions WHERE kind = 'forget'"
    ).fetchall():
        with contextlib.suppress(TypeError, json.JSONDecodeError):
            claims_erased += len(json.loads(snap))

    namespaces = {
        r[0]: int(r[1]) for r in conn.execute(
            "SELECT namespace, COUNT(DISTINCT session_id) FROM turns GROUP BY namespace"
        ).fetchall()
    }

    # Staleness: active claims older than their slot's volatility window. A volatile
    # slot (changes often) goes stale faster than a stable one -- so "old" is relative
    # to how fast that (subject, predicate) actually changes, not an absolute age.
    import time

    from memcontext.profiles import _volatility_label

    supersession_counts: dict[tuple[str, str], int] = {}
    for s_subj, s_pred, s_n in conn.execute(
        "SELECT c.subject, c.predicate, COUNT(*)"
        " FROM supersession_edges e JOIN claims c ON e.new_claim_id = c.claim_id"
        " WHERE c.subject IS NOT NULL AND c.predicate IS NOT NULL"
        " GROUP BY c.subject, c.predicate"
    ).fetchall():
        supersession_counts[(s_subj, s_pred)] = int(s_n)
    _DAY_NS = 86_400 * 1_000_000_000
    _windows = {"stable": 365 * _DAY_NS, "evolving": 90 * _DAY_NS, "volatile": 14 * _DAY_NS}
    _now = time.time_ns()
    stale = 0
    for a_subj, a_pred, a_ts in conn.execute(
        "SELECT subject, predicate, COALESCE(event_ts, valid_from_ts, created_ts)"
        " FROM claims WHERE status IN ('active','confirmed')"
    ).fetchall():
        tier = _volatility_label(supersession_counts.get((a_subj, a_pred), 0))
        if a_ts is not None and (_now - int(a_ts)) > _windows[tier]:
            stale += 1

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
        "staleness": {"stale": stale, "stale_fraction": frac(stale, active)},
    }
