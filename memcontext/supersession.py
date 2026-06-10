"""Supersession Pass 1 — deterministic structural matcher.

Match: same (session_id, normalised_subject, predicate), different value.

Discriminate edge_type from speaker + context:
- assistant speaker on new claim → ASSISTANT_CONFIRM
- user speaker on new claim and old claim was also user → USER_CORRECTION
- user speaker on new claim but old claim was assistant → CONTRADICTS
- new value is a strict token-subset of old value → REFINES

No LLM, no randomness. Guard: supersession never fires within the same
turn — claims emitted together from one utterance are additive.
"""
from __future__ import annotations

import sqlite3
import uuid

import structlog

from memcontext.claims import now_ns, set_claim_status
from memcontext.schema import Claim, ClaimStatus, EdgeType, Speaker, SupersessionEdge

log = structlog.get_logger(__name__)


def _new_edge_id() -> str:
    return f"ed_{uuid.uuid4().hex[:12]}"


def _tokens(value: str) -> set[str]:
    import re
    return {t for t in re.split(r"[\s,;/]+", value.lower().strip()) if t}


def _classify_edge(
    *,
    old_claim: Claim,
    new_claim: Claim,
    new_turn_speaker: Speaker,
    old_turn_speaker: Speaker,
) -> EdgeType:
    """Return the typed edge kind for a Pass-1 supersession.

    Order: REFINES → ASSISTANT_CONFIRM → USER_CORRECTION → CONTRADICTS.
    """
    old_tokens = _tokens(old_claim.value)
    new_tokens = _tokens(new_claim.value)
    if old_tokens and new_tokens and new_tokens < old_tokens:
        return EdgeType.REFINES
    if new_turn_speaker is Speaker.ASSISTANT:
        return EdgeType.ASSISTANT_CONFIRM
    if (
        new_turn_speaker is Speaker.USER
        and old_turn_speaker is Speaker.USER
    ):
        return EdgeType.USER_CORRECTION
    return EdgeType.CONTRADICTS


def _get_speaker(conn: sqlite3.Connection, turn_id: str) -> Speaker:
    row = conn.execute("SELECT speaker FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
    if row is None:
        raise ValueError(f"turn {turn_id!r} not found when classifying supersession")
    return Speaker(row["speaker"])


def _claim_trust(conn: sqlite3.Connection, claim_id: str) -> float:
    """Source-trust weight of a claim (0.5 neutral if unset)."""
    row = conn.execute(
        "SELECT COALESCE(source_trust, 0.5) FROM claim_metadata WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    return float(row[0]) if row else 0.5


def _record_drift_blocked(
    conn: sqlite3.Connection, new_claim: Claim, old_claim: Claim, edge_type: EdgeType
) -> None:
    """Audit a blocked low-trust override (a belief-drift attempt) to `decisions`,
    so it is countable for trust observability (P6)."""
    import json
    import time
    import uuid

    try:
        conn.execute(
            "INSERT INTO decisions (decision_id, session_id, kind, target_type,"
            " target_id, claim_state_snapshot, ts)"
            " VALUES (?, ?, 'drift_blocked', 'claim', ?, ?, ?)",
            (f"dec_{uuid.uuid4().hex[:12]}", new_claim.session_id, old_claim.claim_id,
             json.dumps({"attempted_claim_id": new_claim.claim_id,
                         "new_value": new_claim.value, "old_value": old_claim.value,
                         "edge_type": edge_type.value}),
             time.time_ns()),
        )
    except Exception:  # noqa: BLE001
        pass
    log.info("substrate.supersession_blocked_low_trust",
             new_claim_id=new_claim.claim_id, old_claim_id=old_claim.claim_id,
             edge_type=edge_type.value)


# Single-valued life attributes: a new value for the SAME attribute supersedes the
# prior one even when surface phrasing differs ("lives in NYC" -> "moved to Boston").
# This connects claims by the attribute slot they describe, not by exact tokens —
# the head of the entity/attribute-identity problem. Deterministic, zero-LLM,
# high-precision (conservative trigger phrases to avoid false supersession).
#
# NB: this is value-level, NOT predicate-level. The personal_assistant pack emits
# coarse predicates ("user_fact"), so `single_valued` cannot distinguish residence
# from employer from hobby — putting "user_fact" in single_valued would make every
# fact clobber every other fact. The attribute slot is read off the VALUE phrasing.
_SINGLE_VALUED_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "residence": (
        "lives in", "live in", "living in", "moved to", "relocated to",
        "relocating to", "resides in", "reside in",
    ),
    "employer": (
        "works at", "work at", "working at", "employed at", "employed by",
    ),
}


def _attribute_of(value: str) -> str | None:
    """Map a claim value to a single-valued attribute slot, or None.

    Matches trigger phrases on WORD-TOKEN boundaries (a contiguous token
    subsequence), NOT raw substrings — so "works at" does not match inside
    "frameworks at" and "lives in" does not match inside "olives in".
    Deterministic, zero-LLM.
    """
    import re as _re

    vtoks = _re.findall(r"[a-z0-9]+", value.lower())
    for attr, triggers in _SINGLE_VALUED_ATTRIBUTES.items():
        for trigger in triggers:
            ptoks = trigger.split()
            n = len(ptoks)
            if any(vtoks[i : i + n] == ptoks for i in range(len(vtoks) - n + 1)):
                return attr
    return None


def detect_pass1(
    conn: sqlite3.Connection,
    new_claim: Claim,
) -> SupersessionEdge | None:
    """Deterministic Pass-1 supersession for a freshly-inserted claim.

    Returns the created edge (and marks the old claim superseded) or None if
    no prior matching active/confirmed claim exists.

    Pass-1 is purely structural (keys on subject+predicate). NL-only facts have
    no structured triple, so they cannot be matched here — they fall through to
    the Pass-2 semantic path (see `supersession_semantic`).
    """
    if not new_claim.subject or not new_claim.predicate:
        return None
    rows = conn.execute(
        "SELECT * FROM claims WHERE session_id = ? AND subject = ? AND predicate = ?"
        " AND status IN ('active','confirmed') AND claim_id != ?"
        " AND source_turn_id != ?"
        " ORDER BY created_ts DESC",
        (
            new_claim.session_id,
            new_claim.subject,
            new_claim.predicate,
            new_claim.claim_id,
            new_claim.source_turn_id,
        ),
    ).fetchall()
    if not rows:
        return None

    from memcontext.claims import row_to_claim
    from memcontext.predicate_packs import active_pack

    new_value_norm = new_claim.value.strip().lower()
    best_match: Claim | None = None

    if new_claim.predicate in active_pack().single_valued:
        # Cardinality supersession: a single-valued (subject, predicate) slot holds
        # ONE current value, so a new value supersedes the newest prior active claim
        # regardless of token overlap (e.g. Postgres -> DynamoDB). Deterministic.
        for row in rows:
            candidate = row_to_claim(row)
            if candidate.value.strip().lower() != new_value_norm:
                best_match = candidate
                break

    if best_match is None:
        # Attribute-cardinality: the new value names a single-valued life attribute
        # (residence, employer). A prior active claim describing the SAME attribute
        # with a different value is superseded — even across surface phrasing
        # ("lives in NYC" -> "moved to Boston"). This resolves the slot to one
        # current truth where the coarse predicate alone cannot. Deterministic.
        new_attr = _attribute_of(new_claim.value)
        if new_attr is not None:
            for row in rows:
                candidate = row_to_claim(row)
                if candidate.value.strip().lower() == new_value_norm:
                    continue
                if _attribute_of(candidate.value) == new_attr:
                    best_match = candidate
                    break

    if best_match is None and new_claim.predicate not in active_pack().single_valued:
        # Multi-valued / undeclared: distinguish a value UPDATE (supersede) from an
        # ADDITIVE fact (keep both) on a shared (subject, predicate).
        import re as _re
        _noise = {"the","a","an","is","was","to","for","and","or","of","in","on","at",
                  "it","my","i","me","we","up","so","no","not","but","with","has","had",
                  "be","do","did","will","been","just","very","really","also","about",
                  "some","from","that","this","more","than","each","during"}
        _quantity = {"zero","one","two","three","four","five","six","seven","eight",
                     "nine","ten","eleven","twelve","couple","few","several","many",
                     "single","both","dozen","hundred","thousand"}

        def _content(v: str) -> set[str]:
            return set(_re.findall(r"[a-z0-9]+", v.lower())) - _noise

        def _nonnum(toks: set[str]) -> set[str]:
            return {t for t in toks if not (t.isdigit() or t in _quantity)}

        new_content = _content(new_claim.value)
        new_nn = _nonnum(new_content)
        best_jaccard: float = 0.0
        for row in rows:
            candidate = row_to_claim(row)
            if candidate.value.strip().lower() == new_value_norm:
                continue
            old_content = _content(candidate.value)
            if not (old_content and new_content):
                continue
            # (1) Quantity correction: the non-numeric content is identical and only a
            # number/quantifier changed ("has two kids" -> "has three kids"). That IS a
            # replacement even though only the head noun is shared — the count updated.
            old_nn = _nonnum(old_content)
            if new_nn and new_nn == old_nn and new_content != old_content:
                best_match = candidate
                break
            # (2) General overlap: require >= 2 shared CONTENT tokens. A single shared
            # token is usually just the relation verb — "likes" pizza vs sushi,
            # "allergic to" peanuts vs shellfish — which are ADDITIVE facts, not
            # replacements. Over-supersession silently deletes valid memory, so default
            # to keeping both unless the overlap is substantial.
            shared = old_content & new_content
            jaccard = len(shared) / len(old_content | new_content)
            if len(shared) >= 2 and jaccard >= 0.3 and jaccard > best_jaccard:
                best_jaccard = jaccard
                best_match = candidate

    if best_match is None:
        return None

    old_claim = best_match

    new_speaker = _get_speaker(conn, new_claim.source_turn_id)
    old_speaker = _get_speaker(conn, old_claim.source_turn_id)
    edge_type = _classify_edge(
        old_claim=old_claim,
        new_claim=new_claim,
        new_turn_speaker=new_speaker,
        old_turn_speaker=old_speaker,
    )

    # Source-trust guard (Phase 3): a markedly lower-trust source must NOT REPLACE
    # or REFUTE a higher-trust fact (e.g. a browsed-page value overriding what the
    # user stated). Confirmations / refinements are unaffected.
    if edge_type in (EdgeType.USER_CORRECTION, EdgeType.CONTRADICTS) and (
        _claim_trust(conn, new_claim.claim_id) + 0.2 < _claim_trust(conn, old_claim.claim_id)
    ):
        _record_drift_blocked(conn, new_claim, old_claim, edge_type)
        return None

    edge = write_supersession_edge(
        conn,
        old_claim_id=old_claim.claim_id,
        new_claim_id=new_claim.claim_id,
        edge_type=edge_type,
        identity_score=None,
    )
    set_claim_status(conn, old_claim.claim_id, ClaimStatus.SUPERSEDED)
    conn.execute(
        "UPDATE claims SET valid_until_ts = ?"
        " WHERE claim_id = ?"
        " AND (valid_from_ts IS NULL OR valid_from_ts < ?)",
        (edge.created_ts, old_claim.claim_id, edge.created_ts),
    )
    log.info(
        "substrate.supersession_pass1",
        session_id=new_claim.session_id,
        old_claim_id=old_claim.claim_id,
        new_claim_id=new_claim.claim_id,
        edge_type=edge_type.value,
    )
    return edge


def write_supersession_edge(
    conn: sqlite3.Connection,
    *,
    old_claim_id: str,
    new_claim_id: str,
    edge_type: EdgeType,
    identity_score: float | None,
) -> SupersessionEdge:
    """Insert a typed supersession edge (no status side-effect here)."""
    edge_id = _new_edge_id()
    ts = now_ns()
    conn.execute(
        "INSERT INTO supersession_edges"
        " (edge_id, old_claim_id, new_claim_id, edge_type, identity_score, created_ts)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (edge_id, old_claim_id, new_claim_id, edge_type.value, identity_score, ts),
    )
    return SupersessionEdge(
        edge_id=edge_id,
        old_claim_id=old_claim_id,
        new_claim_id=new_claim_id,
        edge_type=edge_type,
        identity_score=identity_score,
        created_ts=ts,
    )


def record_user_dismissal(
    conn: sqlite3.Connection,
    *,
    dismissed_claim_id: str,
    replacement_claim_id: str | None,
) -> SupersessionEdge | None:
    """Record a user action dismissing a claim.

    If `replacement_claim_id` is None, only the status is set to DISMISSED.
    Otherwise an edge with edge_type=DISMISSED_BY_USER is written.
    """
    set_claim_status(conn, dismissed_claim_id, ClaimStatus.DISMISSED)
    if replacement_claim_id is None:
        return None
    return write_supersession_edge(
        conn,
        old_claim_id=dismissed_claim_id,
        new_claim_id=replacement_claim_id,
        edge_type=EdgeType.DISMISSED_BY_USER,
        identity_score=None,
    )
