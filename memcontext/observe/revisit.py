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


def _claim_diff_key(claim: dict) -> str:
    """Stable identity key for diffing observation claims.

    Uses obs_key (set by AccessibilityTreeExtractor) when present,
    which encodes the a11y role + label. Falls back to
    subject:predicate:value_hash for claims without obs_key.
    """
    obs_key = claim.get("obs_key", "")
    if obs_key:
        subj = claim.get("subject", "")
        return f"{subj}|{obs_key}"
    import hashlib
    subj = claim.get("subject", "")
    pred = claim.get("predicate", "")
    val = claim.get("value", "")
    val_hash = hashlib.sha256(val.encode()).hexdigest()[:12]
    return f"{subj}|{pred}|{val_hash}"


def diff_snapshots(
    old_claims: list[dict],
    new_claims: list[dict],
    url: str,
) -> ChangeReport:
    """Compare claims from two visits to the same page.

    Matching uses obs_key (role+label identity) when available,
    falling back to value hash. This correctly distinguishes multiple
    claims sharing the same (subject, predicate) — e.g., multiple
    headings or links from the same page.
    """
    report = ChangeReport(url=url)

    old_by_key: dict[str, dict] = {}
    for c in old_claims:
        old_by_key[_claim_diff_key(c)] = c

    new_by_key: dict[str, dict] = {}
    for c in new_claims:
        new_by_key[_claim_diff_key(c)] = c

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
