"""Life event detection — deterministic, zero LLM."""
from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

log = structlog.get_logger(__name__)

_NS_PER_HOUR: int = 3600 * 10**9


@dataclass(frozen=True, slots=True)
class LifeEvent:
    event_id: str
    subject: str
    timestamp_start: int  # nanoseconds
    timestamp_end: int
    claim_ids: tuple[str, ...]
    predicates_affected: tuple[str, ...]
    summary_text: str
    significance: float  # 0-1


def _format_ns_date(ts_ns: int) -> str:
    """Format a nanosecond timestamp as YYYY-MM-DD."""
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=UTC)
    return dt.strftime("%Y-%m-%d")


def detect_life_events(
    conn: sqlite3.Connection,
    subject: str,
    *,
    window_hours: int = 24,
    min_predicates: int = 3,
    namespace: str | None = None,
) -> list[LifeEvent]:
    """Detect life events for a subject by clustering temporally close claims.

    A *life event* is a burst of diverse predicate changes within a sliding
    window.  When ``min_predicates`` or more distinct predicates appear
    inside a ``window_hours``-wide window, the cluster is surfaced as a
    ``LifeEvent``.

    When ``namespace`` is given, only claims whose source turn is in that namespace
    are clustered — so life-event detection never mixes tenants (claims are not
    namespace-keyed; isolation is applied here).

    All computation is deterministic; no LLM calls.
    """
    ns_clause = ""
    params: tuple = (subject,)
    if namespace is not None:
        ns_clause = " AND source_turn_id IN (SELECT turn_id FROM turns WHERE namespace = ?)"
        params = (subject, namespace)
    rows = conn.execute(
        "SELECT claim_id, predicate, value, created_ts FROM claims"
        " WHERE subject = ?"
        " AND status IN ('active', 'confirmed', 'audited')"
        f"{ns_clause}"
        " ORDER BY created_ts ASC",
        params,
    ).fetchall()

    if not rows:
        return []

    # Total distinct predicate families for this subject (denominator for
    # significance) — same namespace scope.
    total_families_row = conn.execute(
        "SELECT COUNT(DISTINCT predicate) AS n FROM claims"
        " WHERE subject = ?"
        " AND status IN ('active', 'confirmed', 'audited')"
        f"{ns_clause}",
        params,
    ).fetchone()
    total_families = total_families_row["n"] if total_families_row else 1
    total_families = max(total_families, 1)  # avoid division by zero

    window_ns = window_hours * _NS_PER_HOUR
    events: list[LifeEvent] = []
    used_claim_ids: set[str] = set()

    # Sliding-window: for each claim, look ahead within window_ns and
    # collect the group.  Greedily consume claims so they are not
    # double-counted.
    i = 0
    while i < len(rows):
        if rows[i]["claim_id"] in used_claim_ids:
            i += 1
            continue

        anchor_ts: int = rows[i]["created_ts"]
        window_end_ts = anchor_ts + window_ns

        # Collect all claims within the window that have not been used.
        group: list[sqlite3.Row] = []
        j = i
        while j < len(rows) and rows[j]["created_ts"] <= window_end_ts:
            if rows[j]["claim_id"] not in used_claim_ids:
                group.append(rows[j])
            j += 1

        # Count distinct predicates in this group.
        predicates_in_group: dict[str, str] = {}
        for row in group:
            pred = row["predicate"]
            if pred not in predicates_in_group:
                predicates_in_group[pred] = row["value"]

        if len(predicates_in_group) >= min_predicates:
            claim_ids = tuple(r["claim_id"] for r in group)
            predicates_affected = tuple(sorted(predicates_in_group.keys()))
            ts_start = group[0]["created_ts"]
            ts_end = group[-1]["created_ts"]

            # Significance: ratio of affected predicates to total families.
            significance = min(1.0, len(predicates_in_group) / total_families)

            # Deterministic summary.
            date_str = _format_ns_date(ts_start)
            changes = ", ".join(
                f"{pred} -> {predicates_in_group[pred]}"
                for pred in predicates_affected
            )
            summary_text = f"Changes on {date_str}: {changes}"

            event_id = f"le_{uuid.uuid4().hex[:12]}"
            events.append(
                LifeEvent(
                    event_id=event_id,
                    subject=subject,
                    timestamp_start=ts_start,
                    timestamp_end=ts_end,
                    claim_ids=claim_ids,
                    predicates_affected=predicates_affected,
                    summary_text=summary_text,
                    significance=round(significance, 4),
                )
            )

            # Mark all claims in this group as used.
            for r in group:
                used_claim_ids.add(r["claim_id"])
            i = j  # advance past the window
        else:
            i += 1

    log.debug(
        "life_events.detected",
        subject=subject,
        count=len(events),
    )
    return events


def store_life_events(conn: sqlite3.Connection, events: list[LifeEvent]) -> int:
    """Persist life events into the ``life_events`` table — idempotent per subject.

    Event ids are freshly minted each detection, so INSERT OR REPLACE alone would
    pile up duplicates on re-detection. We clear each subject's prior rows first, so
    re-detecting a subject replaces (not appends) its events regardless of the caller.

    Returns the number of events stored.
    """
    for subject in {ev.subject for ev in events}:
        conn.execute("DELETE FROM life_events WHERE subject = ?", (subject,))

    count = 0
    for ev in events:
        conn.execute(
            "INSERT OR REPLACE INTO life_events"
            " (event_id, subject, timestamp_start, timestamp_end,"
            "  claim_ids, predicates_affected, summary_text, significance)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ev.event_id,
                ev.subject,
                ev.timestamp_start,
                ev.timestamp_end,
                ",".join(ev.claim_ids),
                ",".join(ev.predicates_affected),
                ev.summary_text,
                ev.significance,
            ),
        )
        count += 1

    log.info("life_events.stored", count=count)
    return count
