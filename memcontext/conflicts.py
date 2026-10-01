"""Same-kind decision conflicts: two CURRENT values for one kind of decision.

Supersession keeps one current value per (subject, predicate) slot, but two values
can still both be current:
- in one slot, when the source-trust guard refuses to let a low-trust value retire a
  trusted one (both stay active behind a ``contradicts`` edge), and
- across slots, when the same decision is recorded again under a new subject
  ("ci-toolkit/ci" then "ci-toolkit/ci-provider"); structural supersession keys on the
  exact subject and cannot link them.

A consumer shown both values as unrelated facts cannot tell which is current (the
Claude Code recall eval caught Claude writing a stale CI provider this way). This
module finds them so they can be flagged, newest first. Deterministic, zero-LLM.

"Same kind" means the same SINGLE-VALUED predicate (one current value per slot by
declaration, e.g. ``decision_made``) and either the same subject or the same project
prefix (text before the last '/') with one topic's words contained in the other's.
Multi-valued predicates (``blocker``, ``todo``) never conflict: several current
values are normal there.
"""
from __future__ import annotations

import re
import sqlite3

from memcontext.predicate_packs import active_pack

LIVE_STATUSES = ("active", "confirmed", "audited")


def _prefix_and_topic(subject: str) -> tuple[str, frozenset[str]]:
    prefix, _, topic = subject.strip().lower().rpartition("/")
    return prefix, frozenset(re.findall(r"[a-z0-9]+", topic))


def same_decision_kind(a: str | None, b: str | None) -> bool:
    """Whether two subjects name the same kind of decision (see module docstring)."""
    if not a or not b:
        return False
    if a.strip().lower() == b.strip().lower():
        return True
    pa, ta = _prefix_and_topic(a)
    pb, tb = _prefix_and_topic(b)
    return pa == pb and bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


def is_single_valued(predicate: str | None) -> bool:
    return bool(predicate) and predicate in active_pack().single_valued


def live_same_kind(
    conn: sqlite3.Connection, *, subject: str | None, predicate: str | None,
    namespace: str | None,
) -> list[sqlite3.Row]:
    """Every live claim of the same kind of decision, newest first (including any
    claim with exactly this subject). ``namespace=None`` means unrestricted (the
    single-tenant / shared-token case); otherwise only that tenant's claims.

    Rows carry the claim columns plus ``trust`` (stored source trust, 0.5 if unset).
    """
    if not subject or not is_single_valued(predicate):
        return []
    placeholders = ",".join("?" for _ in LIVE_STATUSES)
    sql = (
        "SELECT c.*, COALESCE(m.source_trust, 0.5) AS trust FROM claims c"
        " JOIN turns t ON t.turn_id = c.source_turn_id"
        " LEFT JOIN claim_metadata m ON m.claim_id = c.claim_id"
        f" WHERE c.predicate = ? AND c.status IN ({placeholders})"
    )
    args: list = [predicate, *LIVE_STATUSES]
    if namespace is not None:
        sql += " AND t.namespace = ?"
        args.append(namespace)
    rows = conn.execute(sql + " ORDER BY c.created_ts DESC, c.claim_id", args).fetchall()
    return [r for r in rows if same_decision_kind(subject, r["subject"])]
