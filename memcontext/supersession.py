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


def detect_pass1(
    conn: sqlite3.Connection,
    new_claim: Claim,
) -> SupersessionEdge | None:
    """Deterministic Pass-1 supersession for a freshly-inserted claim.

    Returns the created edge (and marks the old claim superseded) or None if
    no prior matching active/confirmed claim exists.
    """
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
    import re as _re
    _noise = {"the","a","an","is","was","to","for","and","or","of","in","on","at",
              "it","my","i","me","we","up","so","no","not","but","with","has","had",
              "be","do","did","will","been","just","very","really","also","about",
              "some","from","that","this","more","than","each","during"}

    new_content = set(_re.findall(r"[a-z0-9]+", new_claim.value.lower())) - _noise

    best_match: Claim | None = None
    best_jaccard: float = 0.0

    for row in rows:
        candidate = row_to_claim(row)
        if candidate.value.strip().lower() == new_claim.value.strip().lower():
            continue
        old_content = set(_re.findall(r"[a-z0-9]+", candidate.value.lower())) - _noise
        if old_content and new_content:
            jaccard = len(old_content & new_content) / len(old_content | new_content)
            if jaccard >= 0.3 and jaccard > best_jaccard:
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
