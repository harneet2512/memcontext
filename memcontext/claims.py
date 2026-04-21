"""Claim CRUD + active-state queries.

Pure storage layer — no LLMs, no embeddings. Validation:
- predicate must be in the active pack's predicate families
- 0 <= confidence <= 1
- source_turn_id must reference an existing turn
- Non-empty subject and value

Invalid claims raise `ClaimValidationError`.
"""
from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from time import time_ns

import structlog

from memcontext.schema import (
    Claim,
    ClaimStatus,
    EdgeType,
    Speaker,
    SupersessionEdge,
    Turn,
)

log = structlog.get_logger(__name__)


class ClaimValidationError(ValueError):
    """Raised when a claim fails write-time validation."""


# ---------------------------------------------------------------- helpers ---


_last_ts: list[int] = [0]


def now_ns() -> int:
    """Strictly-increasing nanosecond timestamp.

    Wall-clock `time_ns()` unless doing so would not strictly increase since
    the last call, in which case we bump by 1 ns.
    """
    wall = time_ns()
    next_ts = wall if wall > _last_ts[0] else _last_ts[0] + 1
    _last_ts[0] = next_ts
    return next_ts


def new_claim_id() -> str:
    return f"cl_{uuid.uuid4().hex[:12]}"


def new_turn_id() -> str:
    return f"tu_{uuid.uuid4().hex[:12]}"


def row_to_claim(row: sqlite3.Row) -> Claim:
    """Map a sqlite3.Row from `claims` into the typed `Claim` dataclass."""
    keys = set(row.keys())
    return Claim(
        claim_id=row["claim_id"],
        session_id=row["session_id"],
        subject=row["subject"],
        predicate=row["predicate"],
        value=row["value"],
        value_normalised=row["value_normalised"],
        confidence=row["confidence"],
        source_turn_id=row["source_turn_id"],
        status=ClaimStatus(row["status"]),
        created_ts=row["created_ts"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        valid_from_ts=row["valid_from_ts"] if "valid_from_ts" in keys else None,
        valid_until_ts=row["valid_until_ts"] if "valid_until_ts" in keys else None,
    )


def _temporal_bin(valid_from_ts: int | None) -> str | None:
    """Coarse YYYY-Qn bin for `claim_metadata.temporal_bin`."""
    if valid_from_ts is None:
        return None
    seconds = valid_from_ts / 1e9
    dt = datetime.fromtimestamp(seconds, tz=UTC)
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year:04d}-Q{quarter}"


def _normalise_subject(subject: str) -> str:
    """Lowercase, strip whitespace, collapse internal runs to a single underscore."""
    import re

    cleaned = subject.strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned


# --------------------------------------------------------------- turns CRUD ---


def insert_turn(conn: sqlite3.Connection, turn: Turn) -> None:
    """Insert a turn. Duplicate IDs raise `sqlite3.IntegrityError`."""
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, speaker, text, ts, asr_confidence)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            turn.turn_id,
            turn.session_id,
            turn.speaker.value,
            turn.text,
            turn.ts,
            turn.asr_confidence,
        ),
    )
    log.debug(
        "substrate.turn_inserted",
        session_id=turn.session_id,
        turn_id=turn.turn_id,
        speaker=turn.speaker.value,
    )


def get_turn(conn: sqlite3.Connection, turn_id: str) -> Turn | None:
    row = conn.execute(
        "SELECT turn_id, session_id, speaker, text, ts, asr_confidence"
        " FROM turns WHERE turn_id = ?",
        (turn_id,),
    ).fetchone()
    if row is None:
        return None
    return Turn(
        turn_id=row["turn_id"],
        session_id=row["session_id"],
        speaker=Speaker(row["speaker"]),
        text=row["text"],
        ts=row["ts"],
        asr_confidence=row["asr_confidence"],
    )


# -------------------------------------------------------------- claims CRUD ---


def validate_claim(
    *,
    subject: str,
    predicate: str,
    value: str,
    confidence: float,
    source_turn_id: str,
    char_start: int | None,
    char_end: int | None,
    turn_text_len: int | None = None,
    allowed_predicates: frozenset[str] | None = None,
) -> None:
    """Write-time validation. Raises `ClaimValidationError` on any violation.

    If `allowed_predicates` is None, attempts to load from the active pack.
    """
    if not subject or not subject.strip():
        raise ClaimValidationError("subject must be non-empty")

    if allowed_predicates is not None:
        allowed = allowed_predicates
    else:
        try:
            from memcontext.predicate_packs import active_pack
            allowed = active_pack().predicate_families
        except Exception:  # noqa: BLE001
            allowed = None

    if allowed is not None and predicate not in allowed:
        raise ClaimValidationError(
            f"predicate {predicate!r} not in allowed set; "
            f"allowed = {sorted(allowed)}"
        )
    if not value or not value.strip():
        raise ClaimValidationError("value must be non-empty")
    if not (0.0 <= confidence <= 1.0):
        raise ClaimValidationError(f"confidence {confidence} outside [0, 1]")
    if not source_turn_id:
        raise ClaimValidationError("source_turn_id must be non-empty")

    if char_start is not None or char_end is not None:
        if char_start is None or char_end is None:
            raise ClaimValidationError(
                "char_start and char_end must either both be set or both be None"
            )
        if char_start < 0 or char_end < char_start:
            raise ClaimValidationError(
                f"invalid span ({char_start}, {char_end}); require 0 <= start <= end"
            )
        if turn_text_len is not None and char_end > turn_text_len:
            raise ClaimValidationError(
                f"char_end {char_end} exceeds source turn length {turn_text_len}"
            )


def insert_claim(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: str,
    predicate: str,
    value: str,
    confidence: float,
    source_turn_id: str,
    value_normalised: str | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    claim_id: str | None = None,
    status: ClaimStatus = ClaimStatus.ACTIVE,
    valid_from: int | None = None,
    valid_until: int | None = None,
    allowed_predicates: frozenset[str] | None = None,
) -> Claim:
    """Insert a new claim after validation.

    Checks that `source_turn_id` refers to an existing turn.
    Returns the persisted `Claim` with its server-assigned id + timestamp.
    """
    turn = get_turn(conn, source_turn_id)
    if turn is None:
        raise ClaimValidationError(
            f"source_turn_id {source_turn_id!r} does not reference any turn"
        )
    validate_claim(
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=confidence,
        source_turn_id=source_turn_id,
        char_start=char_start,
        char_end=char_end,
        turn_text_len=len(turn.text),
        allowed_predicates=allowed_predicates,
    )

    cid = claim_id or new_claim_id()
    ts = now_ns()
    valid_from_ts = ts if valid_from is None else valid_from
    valid_until_ts = valid_until
    if valid_until_ts is not None and valid_until_ts <= valid_from_ts:
        raise ClaimValidationError(
            f"valid_until ({valid_until_ts}) must be > valid_from ({valid_from_ts})"
        )
    norm_subject = _normalise_subject(subject)
    conn.execute(
        "INSERT INTO claims (claim_id, session_id, subject, predicate, value,"
        " value_normalised, confidence, source_turn_id, status, created_ts,"
        " char_start, char_end, valid_from_ts, valid_until_ts)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            cid,
            session_id,
            norm_subject,
            predicate,
            value,
            value_normalised,
            confidence,
            source_turn_id,
            status.value,
            ts,
            char_start,
            char_end,
            valid_from_ts,
            valid_until_ts,
        ),
    )
    conn.execute(
        "INSERT INTO claim_metadata (claim_id, entity_key, predicate_family,"
        " temporal_bin) VALUES (?, ?, ?, ?)",
        (cid, norm_subject, predicate, _temporal_bin(valid_from_ts)),
    )
    log.info(
        "substrate.claim_inserted",
        session_id=session_id,
        claim_id=cid,
        subject=norm_subject,
        predicate=predicate,
        confidence=confidence,
    )
    return Claim(
        claim_id=cid,
        session_id=session_id,
        subject=norm_subject,
        predicate=predicate,
        value=value,
        value_normalised=value_normalised,
        confidence=confidence,
        source_turn_id=source_turn_id,
        status=status,
        created_ts=ts,
        char_start=char_start,
        char_end=char_end,
        valid_from_ts=valid_from_ts,
        valid_until_ts=valid_until_ts,
    )


def get_claim(conn: sqlite3.Connection, claim_id: str) -> Claim | None:
    row = conn.execute(
        "SELECT * FROM claims WHERE claim_id = ?", (claim_id,)
    ).fetchone()
    return row_to_claim(row) if row is not None else None


def set_claim_status(
    conn: sqlite3.Connection, claim_id: str, status: ClaimStatus
) -> None:
    cur = conn.execute(
        "UPDATE claims SET status = ? WHERE claim_id = ?", (status.value, claim_id)
    )
    if cur.rowcount == 0:
        raise ClaimValidationError(f"claim {claim_id!r} not found")
    log.info(
        "substrate.claim_status_set",
        claim_id=claim_id,
        status=status.value,
    )


def list_active_claims(
    conn: sqlite3.Connection, session_id: str
) -> list[Claim]:
    """All claims in the session with status in {active, confirmed, audited}."""
    rows = conn.execute(
        "SELECT * FROM claims WHERE session_id = ?"
        " AND status IN ('active','confirmed','audited')"
        " ORDER BY created_ts ASC",
        (session_id,),
    ).fetchall()
    return [row_to_claim(r) for r in rows]


def list_claims_with_lifecycle(
    conn: sqlite3.Connection, session_id: str, mode: str = "current_truth",
) -> list[Claim]:
    if mode == "current_truth":
        return list_active_claims(conn, session_id)
    rows = conn.execute(
        "SELECT * FROM claims WHERE session_id = ?"
        " AND status IN ('active','confirmed','audited','superseded')"
        " ORDER BY created_ts ASC", (session_id,),
    ).fetchall()
    return [row_to_claim(r) for r in rows]


def list_supersession_pairs(
    conn: sqlite3.Connection, session_id: str,
) -> list[tuple[Claim, Claim, SupersessionEdge]]:
    rows = conn.execute(
        "SELECT e.edge_id, e.old_claim_id, e.new_claim_id,"
        "       e.edge_type, e.identity_score, e.created_ts"
        " FROM supersession_edges e"
        " JOIN claims c_old ON e.old_claim_id = c_old.claim_id"
        " JOIN claims c_new ON e.new_claim_id = c_new.claim_id"
        " WHERE c_old.session_id = ? AND c_new.session_id = ?"
        " ORDER BY e.created_ts DESC", (session_id, session_id),
    ).fetchall()
    pairs: list[tuple[Claim, Claim, SupersessionEdge]] = []
    for r in rows:
        old_row = conn.execute("SELECT * FROM claims WHERE claim_id = ?", (r["old_claim_id"],)).fetchone()
        new_row = conn.execute("SELECT * FROM claims WHERE claim_id = ?", (r["new_claim_id"],)).fetchone()
        if old_row is None or new_row is None:
            continue
        edge = SupersessionEdge(
            edge_id=r["edge_id"], old_claim_id=r["old_claim_id"],
            new_claim_id=r["new_claim_id"], edge_type=EdgeType(r["edge_type"]),
            identity_score=r["identity_score"], created_ts=r["created_ts"],
        )
        pairs.append((row_to_claim(old_row), row_to_claim(new_row), edge))
    return pairs


def list_claims_for_turn(conn: sqlite3.Connection, turn_id: str) -> list[Claim]:
    rows = conn.execute(
        "SELECT * FROM claims WHERE source_turn_id = ? ORDER BY created_ts ASC",
        (turn_id,),
    ).fetchall()
    return [row_to_claim(r) for r in rows]


def get_superseded_by(
    conn: sqlite3.Connection, claim_id: str
) -> str | None:
    """Return the `new_claim_id` that supersedes `claim_id`, or None."""
    row = conn.execute(
        "SELECT new_claim_id FROM supersession_edges"
        " WHERE old_claim_id = ?"
        " ORDER BY created_ts DESC, edge_id DESC LIMIT 1",
        (claim_id,),
    ).fetchone()
    return row["new_claim_id"] if row is not None else None


def find_same_identity_claim(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: str,
    predicate: str,
    exclude_claim_ids: Iterable[str] = (),
) -> Claim | None:
    """Return the newest ACTIVE/CONFIRMED claim with matching (subject, predicate)."""
    excluded = tuple(exclude_claim_ids)
    placeholders = ",".join("?" * len(excluded)) if excluded else "''"
    norm = _normalise_subject(subject)
    sql = (
        "SELECT * FROM claims WHERE session_id = ? AND subject = ? AND predicate = ?"
        " AND status IN ('active','confirmed')"
        f" AND claim_id NOT IN ({placeholders})"
        " ORDER BY created_ts DESC LIMIT 1"
    )
    row = conn.execute(sql, (session_id, norm, predicate, *excluded)).fetchone()
    return row_to_claim(row) if row is not None else None
