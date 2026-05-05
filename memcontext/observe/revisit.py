"""Re-visit flow -- detect changes between page visits.

Compares two PageSnapshots of the same URL, identifies changed claims,
and triggers supersession for outdated ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChangeReport:
    """Report of changes detected between two visits to the same URL."""

    url: str
    added_claims: list[dict] = field(default_factory=list)
    removed_claims: list[dict] = field(default_factory=list)
    changed_claims: list[tuple[dict, dict]] = field(
        default_factory=list
    )  # (old, new)
    unchanged_count: int = 0


def diff_snapshots(
    old_claims: list[dict],
    new_claims: list[dict],
    url: str,
) -> ChangeReport:
    """Compare claims from two visits to the same page.

    Matching is by (subject, predicate) identity. If a claim has the same
    subject and predicate but different value, it's a change. New subjects
    are additions. Missing subjects are removals.
    """
    report = ChangeReport(url=url)

    old_by_key: dict[tuple[str, str], dict] = {}
    for c in old_claims:
        key = (c.get("subject", ""), c.get("predicate", ""))
        old_by_key[key] = c

    new_by_key: dict[tuple[str, str], dict] = {}
    for c in new_claims:
        key = (c.get("subject", ""), c.get("predicate", ""))
        new_by_key[key] = c

    all_keys = set(old_by_key.keys()) | set(new_by_key.keys())

    for key in all_keys:
        old_claim = old_by_key.get(key)
        new_claim = new_by_key.get(key)

        if old_claim and not new_claim:
            report.removed_claims.append(old_claim)
        elif new_claim and not old_claim:
            report.added_claims.append(new_claim)
        elif old_claim and new_claim:
            if old_claim.get("value") != new_claim.get("value"):
                report.changed_claims.append((old_claim, new_claim))
            else:
                report.unchanged_count += 1

    return report


def apply_changes(
    conn,
    *,
    change_report: ChangeReport,
    session_id: str,
) -> dict:
    """Apply detected changes to the memory store.

    - Added claims: store as new claims
    - Changed claims: store new version, supersession handles the rest
    - Removed claims: optionally dismiss (not automatic -- may reappear)

    Returns stats dict.
    """
    from memcontext.extractors import PassthroughExtractor
    from memcontext.on_new_turn import on_new_turn
    from memcontext.schema import Speaker

    stats: dict = {"added": 0, "changed": 0, "supersessions": 0, "errors": []}

    # Store added claims
    if change_report.added_claims:
        text = f"[Re-visit: {change_report.url}] New content detected"
        pt = PassthroughExtractor(change_report.added_claims)
        result = on_new_turn(
            conn,
            session_id=session_id,
            speaker=Speaker.ASSISTANT,
            text=text,
            extractor=pt,
        )
        stats["added"] = len(result.created_claims)

    # Store changed claims (supersession fires automatically via on_new_turn)
    if change_report.changed_claims:
        new_claims = [new for _, new in change_report.changed_claims]
        text = f"[Re-visit: {change_report.url}] Content changed"
        pt = PassthroughExtractor(new_claims)
        result = on_new_turn(
            conn,
            session_id=session_id,
            speaker=Speaker.ASSISTANT,
            text=text,
            extractor=pt,
        )
        stats["changed"] = len(result.created_claims)
        stats["supersessions"] = len(result.supersession_edges)

    return stats
