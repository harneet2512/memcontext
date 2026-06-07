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
    ExtractionStatus,
    SourceType,
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
    """Map a sqlite3.Row from `claims` into the typed `Claim` dataclass.

    The structured triple is optional at the storage layer (NULL for NL-only
    facts). It is surfaced as the empty string ``""`` rather than ``None`` so the
    dataclass stays non-Optional and every existing consumer keeps working; an
    empty ``subject``/``predicate``/``value`` is the "no structured field"
    sentinel (test with truthiness). The NL ``text`` is always present from v4 on.
    """
    keys = set(row.keys())
    return Claim(
        claim_id=row["claim_id"],
        session_id=row["session_id"],
        subject=row["subject"] or "",
        predicate=row["predicate"] or "",
        value=row["value"] or "",
        value_normalised=row["value_normalised"],
        confidence=row["confidence"],
        source_turn_id=row["source_turn_id"],
        status=ClaimStatus(row["status"]),
        created_ts=row["created_ts"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        valid_from_ts=row["valid_from_ts"] if "valid_from_ts" in keys else None,
        valid_until_ts=row["valid_until_ts"] if "valid_until_ts" in keys else None,
        event_ts=row["event_ts"] if "event_ts" in keys else None,
        text=row["text"] if "text" in keys else None,
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
    """Insert a turn (episode). Duplicate IDs raise `sqlite3.IntegrityError`."""
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, speaker, text, ts, asr_confidence,"
        " source_type, source_metadata, extraction_status)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            turn.turn_id,
            turn.session_id,
            turn.speaker.value,
            turn.text,
            turn.ts,
            turn.asr_confidence,
            turn.source_type.value,
            turn.source_metadata,
            turn.extraction_status.value,
        ),
    )
    log.debug(
        "substrate.turn_inserted",
        session_id=turn.session_id,
        turn_id=turn.turn_id,
        speaker=turn.speaker.value,
        source_type=turn.source_type.value,
    )


def row_to_turn(row: sqlite3.Row) -> Turn:
    """Map a `turns` row into a `Turn`, tolerating pre-v3 rows missing columns."""
    keys = set(row.keys())
    return Turn(
        turn_id=row["turn_id"],
        session_id=row["session_id"],
        speaker=Speaker(row["speaker"]),
        text=row["text"],
        ts=row["ts"],
        asr_confidence=row["asr_confidence"],
        source_type=(
            SourceType(row["source_type"])
            if "source_type" in keys and row["source_type"] is not None
            else SourceType.CONVERSATION
        ),
        source_metadata=row["source_metadata"] if "source_metadata" in keys else None,
        extraction_status=(
            ExtractionStatus(row["extraction_status"])
            if "extraction_status" in keys and row["extraction_status"] is not None
            else ExtractionStatus.PENDING
        ),
    )


def get_turn(conn: sqlite3.Connection, turn_id: str) -> Turn | None:
    row = conn.execute(
        "SELECT * FROM turns WHERE turn_id = ?",
        (turn_id,),
    ).fetchone()
    if row is None:
        return None
    return row_to_turn(row)


# -------------------------------------------------------------- claims CRUD ---


def predicate_in_vocab(
    predicate: str, allowed_predicates: frozenset[str] | None = None
) -> bool:
    """Whether `predicate` belongs to the active (or given) predicate pack.

    Governs only whether the optional STRUCTURED triple is attached — NOT whether
    a fact is stored. An out-of-vocab predicate demotes the fact to NL-only; it is
    never dropped. If the pack cannot be loaded the predicate is treated as in-vocab
    (we cannot prove otherwise, so we keep the structure).
    """
    if allowed_predicates is not None:
        allowed: frozenset[str] | None = allowed_predicates
    else:
        try:
            from memcontext.predicate_packs import active_pack
            allowed = active_pack().predicate_families
        except Exception:  # noqa: BLE001
            allowed = None
    return allowed is None or predicate in allowed


def validate_claim(
    *,
    text: str | None,
    subject: str | None,
    predicate: str | None,
    value: str | None,
    confidence: float,
    source_turn_id: str,
    char_start: int | None,
    char_end: int | None,
    turn_text_len: int | None = None,
) -> None:
    """Write-time validation. Raises `ClaimValidationError` on any violation.

    A fact is either STRUCTURED (subject+predicate+value all present, all
    non-empty) or NL-only (text present, no triple). Predicate vocabulary is NOT
    gated here — see `predicate_in_vocab`; an out-of-vocab predicate demotes to
    NL-only, it never fails validation.
    """
    structured = bool(subject or predicate or value)
    if structured:
        if not (subject and subject.strip()):
            raise ClaimValidationError("structured fact: subject must be non-empty")
        if not (predicate and predicate.strip()):
            raise ClaimValidationError("structured fact: predicate must be non-empty")
        if not (value and value.strip()):
            raise ClaimValidationError("structured fact: value must be non-empty")
    elif not (text and text.strip()):
        raise ClaimValidationError("NL fact: text must be non-empty")

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


def insert_fact(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    source_turn_id: str,
    confidence: float,
    text: str | None = None,
    subject: str | None = None,
    predicate: str | None = None,
    value: str | None = None,
    value_normalised: str | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    claim_id: str | None = None,
    status: ClaimStatus = ClaimStatus.ACTIVE,
    valid_from: int | None = None,
    valid_until: int | None = None,
    event_ts: int | None = None,
    allowed_predicates: frozenset[str] | None = None,
) -> Claim:
    """Insert a fact (NL-first). The persisted unit of Tier-2 memory.

    A fact ALWAYS stores NL ``text`` and links back to its source episode
    (``source_turn_id``). The structured triple ``(subject, predicate, value)`` is
    an OPTIONAL precision layer, attached only when all three are supplied AND the
    predicate is in the active pack. An out-of-vocab predicate DEMOTES the fact to
    NL-only (triple stored as NULL) — it is never dropped. ``text`` is synthesised
    from the triple when not supplied, so structured facts retrieve exactly as
    before. Returns the persisted ``Claim`` (NL-only facts carry ``""`` triple).
    """
    turn = get_turn(conn, source_turn_id)
    if turn is None:
        raise ClaimValidationError(
            f"source_turn_id {source_turn_id!r} does not reference any turn"
        )

    # Treat the empty-string triple sentinel (from a Claim round-trip) as absent,
    # so re-inserting a previously NL-only fact stays NL-only.
    subject = subject or None
    predicate = predicate or None
    value = value or None

    structured = bool(subject and predicate and value)
    # Demote an out-of-vocab structured triple to NL-only — never drop the fact.
    if structured and not predicate_in_vocab(predicate or "", allowed_predicates):
        if not text:
            text = f"{_normalise_subject(subject or '')} {predicate} {value}"
        log.info(
            "substrate.predicate_demoted_to_nl",
            session_id=session_id,
            predicate=predicate,
        )
        subject = predicate = value = None
        structured = False

    norm_subject = _normalise_subject(subject) if structured and subject else None
    # NL form. Structured facts synthesise it to exactly match the legacy
    # `claim_retrieval_text` triple string, so retrieval/embeddings are unchanged.
    if not text and structured:
        text = f"{norm_subject} {predicate} {value}"

    validate_claim(
        text=text,
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=confidence,
        source_turn_id=source_turn_id,
        char_start=char_start,
        char_end=char_end,
        turn_text_len=len(turn.text),
    )

    cid = claim_id or new_claim_id()
    ts = now_ns()
    valid_from_ts = ts if valid_from is None else valid_from
    valid_until_ts = valid_until
    if valid_until_ts is not None and valid_until_ts <= valid_from_ts:
        raise ClaimValidationError(
            f"valid_until ({valid_until_ts}) must be > valid_from ({valid_from_ts})"
        )

    conn.execute(
        "INSERT INTO claims (claim_id, session_id, text, subject, predicate, value,"
        " value_normalised, confidence, source_turn_id, status, created_ts,"
        " char_start, char_end, valid_from_ts, valid_until_ts, event_ts)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            cid,
            session_id,
            text,
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
            event_ts,
        ),
    )
    # Lightweight entities (no LLM) from the NL text — extracted first so an
    # NL-only fact can anchor its metadata entity_key on its most salient entity.
    ents: list = []
    try:
        from memcontext.entities import extract_entities
        ents = list(extract_entities(text or ""))
    except Exception:  # noqa: BLE001
        ents = []

    # claim_metadata for EVERY fact (NL-only included). Previously structured-only,
    # which left NL facts invisible to the entity/temporal/importance channels and
    # without an importance row. NL facts now anchor entity_key on their top entity
    # and mark predicate_family='nl'; the importance row is created by importance.py.
    if structured and norm_subject and predicate:
        meta_entity, meta_family = norm_subject, predicate
    else:
        meta_entity = _normalise_subject(ents[0].text) if ents else ""
        meta_family = "nl"
    # Source trust (Phase 3): intrinsic to the claim, derived from its source
    # episode's origin (user vs tool vs browser vs assistant) so retrieval and
    # supersession can weigh how much to trust it.
    from memcontext.source_trust import trust_for_source
    _trow = conn.execute(
        "SELECT source_type, speaker FROM turns WHERE turn_id = ?", (source_turn_id,)
    ).fetchone()
    _src_trust = trust_for_source(_trow[0], _trow[1]) if _trow else 0.5
    conn.execute(
        "INSERT INTO claim_metadata (claim_id, entity_key, predicate_family,"
        " temporal_bin, source_trust) VALUES (?, ?, ?, ?, ?)",
        (cid, meta_entity, meta_family, _temporal_bin(valid_from_ts), _src_trust),
    )

    for ent in ents:
        conn.execute(
            "INSERT OR IGNORE INTO claim_entities (claim_id, entity_text, entity_type)"
            " VALUES (?, ?, ?)",
            (cid, ent.text.lower(), ent.entity_type),
        )

    log.info(
        "substrate.fact_inserted",
        session_id=session_id,
        claim_id=cid,
        structured=structured,
        predicate=predicate,
        confidence=confidence,
    )
    return Claim(
        claim_id=cid,
        session_id=session_id,
        subject=norm_subject or "",
        predicate=predicate or "",
        value=value or "",
        value_normalised=value_normalised,
        confidence=confidence,
        source_turn_id=source_turn_id,
        status=status,
        created_ts=ts,
        char_start=char_start,
        char_end=char_end,
        valid_from_ts=valid_from_ts,
        valid_until_ts=valid_until_ts,
        event_ts=event_ts,
        text=text,
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
    event_ts: int | None = None,
    allowed_predicates: frozenset[str] | None = None,
) -> Claim:
    """Insert a STRUCTURED fact. Thin wrapper over `insert_fact`.

    Back-compat entry point for callers that already have a triple. An
    out-of-vocab predicate no longer raises — it demotes to an NL-only fact (the
    returned `Claim` then carries an empty triple). Genuinely malformed input
    (empty value, bad confidence/span, unknown turn) still raises
    `ClaimValidationError`.
    """
    return insert_fact(
        conn,
        session_id=session_id,
        source_turn_id=source_turn_id,
        confidence=confidence,
        subject=subject,
        predicate=predicate,
        value=value,
        value_normalised=value_normalised,
        char_start=char_start,
        char_end=char_end,
        claim_id=claim_id,
        status=status,
        valid_from=valid_from,
        valid_until=valid_until,
        event_ts=event_ts,
        allowed_predicates=allowed_predicates,
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
    if status == ClaimStatus.SUPERSEDED:
        conn.execute("DELETE FROM claim_embeddings WHERE claim_id = ?", (claim_id,))
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


def get_supersession_chain(
    conn: sqlite3.Connection, claim_id: str
) -> list[tuple["Claim", str]]:
    """Walk backwards from a claim through supersession edges.

    Returns [(predecessor_claim, edge_type), ...] in chronological order
    (oldest first). The input claim itself is NOT included.
    """
    chain: list[tuple[Claim, str]] = []
    current = claim_id
    visited: set[str] = set()
    while current and current not in visited:
        visited.add(current)
        row = conn.execute(
            "SELECT old_claim_id, edge_type FROM supersession_edges"
            " WHERE new_claim_id = ?"
            " ORDER BY created_ts DESC LIMIT 1",
            (current,),
        ).fetchone()
        if row is None:
            break
        old_id = row["old_claim_id"]
        old_claim = get_claim(conn, old_id)
        if old_claim is not None:
            chain.append((old_claim, row["edge_type"]))
        current = old_id
    chain.reverse()
    return chain


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
