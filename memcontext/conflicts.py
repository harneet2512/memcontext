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


# ── memory_trace by slot ──────────────────────────────────────────────────────

_LIVE_IN = ",".join("?" for _ in LIVE_STATUSES)
_WITH_TRUST = (
    "SELECT c.*, t.namespace AS namespace, COALESCE(m.source_trust, 0.5) AS trust"
    " FROM claims c JOIN turns t ON t.turn_id = c.source_turn_id"
    " LEFT JOIN claim_metadata m ON m.claim_id = c.claim_id"
)


def _norm_value(value: str | None) -> str:
    return " ".join((value or "").lower().split())


def _live_slot_rows(
    conn: sqlite3.Connection, *, subject: str, predicate: str, session_id: str | None,
) -> list[sqlite3.Row]:
    """Live claims of one (subject, predicate) slot, with namespace and trust.

    A slot is namespace-wide (supersession's scope): the caller's session only
    selects WHICH namespace, then every live claim of the slot in it counts, since
    the current value or its copies may live in other sessions. A session with no
    claim in the slot falls back to any session (the stdio MCP case).
    """
    from memcontext.claims import _normalise_subject

    base = f"{_WITH_TRUST} WHERE c.subject = ? AND c.predicate = ? AND c.status IN ({_LIVE_IN})"
    args = [_normalise_subject(subject), predicate, *LIVE_STATUSES]
    in_session = []
    if session_id is not None:
        in_session = conn.execute(base + " AND c.session_id = ?", [*args, session_id]).fetchall()
    if not in_session:
        return conn.execute(base, args).fetchall()
    ns = max(in_session, key=lambda r: (r["trust"], r["created_ts"]))["namespace"]
    return conn.execute(base + " AND t.namespace = ?", [*args, ns]).fetchall()


def trusted_slot_head(
    conn: sqlite3.Connection, *, subject: str, predicate: str, session_id: str | None,
) -> tuple[str | None, int]:
    """The claim representing a slot's current value, and its number of restatements.

    The current value is the one with the highest source trust, the newest on a tie,
    so a newer low-trust value the trust guard refused to let supersede never becomes
    "current". Its EARLIEST live copy is returned: the original assertion, which
    carries the supersession lineage; later copies are restatements.
    """
    rows = _live_slot_rows(conn, subject=subject, predicate=predicate, session_id=session_id)
    if not rows:
        return None, 0
    best = max(rows, key=lambda r: (r["trust"], r["created_ts"]))
    copies = [r for r in rows if r["namespace"] == best["namespace"]
              and _norm_value(r["value"]) == _norm_value(best["value"])]
    head = min(copies, key=lambda r: (r["created_ts"], r["claim_id"]))
    return head["claim_id"], len(copies) - 1


def _contradicted(conn: sqlite3.Connection, a: str, b: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM supersession_edges WHERE edge_type = 'contradicts' AND"
        " ((old_claim_id = ? AND new_claim_id = ?) OR (old_claim_id = ? AND new_claim_id = ?))",
        (a, b, b, a),
    ).fetchone() is not None


def slot_conflicts(conn: sqlite3.Connection, claim_id: str) -> list[dict]:
    """Other CURRENT values competing with a claim, newest first, one per value.

    Competing means: in the same slot when the predicate is single-valued or a
    ``contradicts`` edge links the two, or the same kind of decision under another
    subject (``live_same_kind``). Confined to the claim's own namespace. A value
    equal to the claim's is agreement (a restatement), not a conflict.
    """
    from memcontext.source_trust import QUARANTINE_THRESHOLD

    head = conn.execute(f"{_WITH_TRUST} WHERE c.claim_id = ?", (claim_id,)).fetchone()
    if head is None or head["subject"] is None or head["status"] not in LIVE_STATUSES:
        return []
    ns = head["namespace"]
    single = is_single_valued(head["predicate"])
    same_slot = conn.execute(
        f"{_WITH_TRUST} WHERE t.namespace = ? AND c.subject = ? AND c.predicate = ?"
        f" AND c.status IN ({_LIVE_IN})",
        (ns, head["subject"], head["predicate"], *LIVE_STATUSES),
    ).fetchall()
    candidates = [r for r in same_slot if single or _contradicted(conn, claim_id, r["claim_id"])]
    candidates += live_same_kind(conn, subject=head["subject"], predicate=head["predicate"],
                                 namespace=ns)
    out: list[dict] = []
    seen_values = {_norm_value(head["value"])}
    for r in sorted(candidates, key=lambda r: (-r["created_ts"], r["claim_id"])):
        value = _norm_value(r["value"])
        if r["claim_id"] == claim_id or value in seen_values:
            continue
        seen_values.add(value)
        trust = float(r["trust"])
        out.append({
            "claim_id": r["claim_id"], "subject": r["subject"], "value": r["value"],
            "fact": r["text"], "status": r["status"], "trust": round(trust, 3),
            "quarantined": trust < QUARANTINE_THRESHOLD, "source_turn_id": r["source_turn_id"],
            "newer_than_head": r["created_ts"] > head["created_ts"],
        })
    return out


# ── capture-time subject drift ────────────────────────────────────────────────

# Words that name the container or kind of thing rather than the decision's topic:
# two subjects sharing only these ("billing_service_language" / "go_service_logging")
# are not the same decision.
_CONTAINER_WORDS = frozenset({
    "a", "an", "the", "of", "for", "and", "to", "in", "on", "our", "we",
    "project", "app", "application", "service", "services", "system", "systems",
    "config", "configuration", "setup", "choice", "decision", "decisions",
    "tool", "tools", "library", "libraries", "framework", "version",
})
SIMILAR_SUBJECTS_LIMIT = 3


def _topic_words(subject: str) -> frozenset[str]:
    _, _, topic = subject.strip().lower().rpartition("/")
    return frozenset(re.findall(r"[a-z0-9]+", topic)) - _CONTAINER_WORDS


_MIN_TOPIC_OVERLAP = 0.3  # same bar Pass-1 uses for value overlap
_COMMON_WORD_MIN_SUBJECTS = 3
_COMMON_WORD_SHARE = 0.5


def _corpus_common_words(subjects: list[str]) -> frozenset[str]:
    """Project prefixes: a word that LEADS a multi-word topic in at least half of the
    distinct subjects (min 3), e.g. "har95" in har95_context_grouping, har95_storage_model.

    Only the leading position counts, and only when more words follow it, so a topic
    word that many subjects are genuinely about ("authentication") is never dropped.
    """
    distinct = {s for s in subjects if s}
    if len(distinct) < _COMMON_WORD_MIN_SUBJECTS:
        return frozenset()
    counts: dict[str, int] = {}
    for s in distinct:
        _, _, topic = s.strip().lower().rpartition("/")
        words = re.findall(r"[a-z0-9]+", topic)
        if len(words) >= 2:
            counts[words[0]] = counts.get(words[0], 0) + 1
    floor = max(_COMMON_WORD_MIN_SUBJECTS, _COMMON_WORD_SHARE * len(distinct))
    return frozenset(w for w, n in counts.items() if n >= floor)


def similar_subjects(
    conn: sqlite3.Connection, *, subject: str, predicate: str, namespace: str | None,
    limit: int = SIMILAR_SUBJECTS_LIMIT,
) -> list[dict]:
    """Live decisions under OTHER subjects that look like the same decision.

    For a single-valued predicate only. Two subjects resemble each other when their
    topics (the part after the last '/', any project prefix ignored) share a word
    that is not a container word. Ranked by topic-word Jaccard, then newest; one
    entry (the newest live value) per subject. Used to WARN at capture time; nothing
    is merged or superseded.
    """
    mine = _topic_words(subject)
    if not mine or not is_single_valued(predicate):
        return []
    sql = f"{_WITH_TRUST} WHERE c.predicate = ? AND c.subject != ? AND c.status IN ({_LIVE_IN})"
    args: list = [predicate, subject, *LIVE_STATUSES]
    if namespace is not None:
        sql += " AND t.namespace = ?"
        args.append(namespace)
    rows = conn.execute(sql, args).fetchall()
    # A word that most live subjects share (a project prefix like "har95_") names
    # the project, not the decision: it is never evidence that two are the same.
    common = _corpus_common_words([r["subject"] or "" for r in rows])
    mine = mine - common
    best: dict[str, tuple[float, int, sqlite3.Row]] = {}
    for r in rows:
        theirs = _topic_words(r["subject"] or "") - common
        shared = mine & theirs
        if not shared:
            continue
        overlap = len(shared) / len(mine | theirs)
        if overlap < _MIN_TOPIC_OVERLAP:  # one common word among many is not the same topic
            continue
        score = (overlap, r["created_ts"], r)
        if r["subject"] not in best or score[1] > best[r["subject"]][1]:
            best[r["subject"]] = score
    ranked = sorted(best.values(), key=lambda x: (-x[0], -x[1]))[:limit]
    return [{"subject": r["subject"], "predicate": r["predicate"], "value": r["value"],
             "claim_id": r["claim_id"]} for _j, _ts, r in ranked]
