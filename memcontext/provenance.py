"""Forward/back-link provenance utilities.

Supports the provenance demo: output sentence → claim → turn.
All functions are deterministic SELECTs — no LLM, no side effects.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, cast

import structlog

from memcontext.schema import OutputSection, OutputSentence

log = structlog.get_logger(__name__)


# ------------------------------------------------- forward / back-link API ---


def claim_ids_for_turn(conn: sqlite3.Connection, turn_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT claim_id FROM claims WHERE source_turn_id = ? ORDER BY created_ts ASC",
        (turn_id,),
    ).fetchall()
    return [r["claim_id"] for r in rows]


def sentence_ids_for_claim(
    conn: sqlite3.Connection, claim_id: str
) -> list[str]:
    """Sentences whose `source_claim_ids` JSON array contains `claim_id`."""
    pattern = f'%"{claim_id}"%'
    rows = conn.execute(
        "SELECT sentence_id FROM output_sentences WHERE source_claim_ids LIKE ?"
        " ORDER BY ordinal ASC",
        (pattern,),
    ).fetchall()
    return [r["sentence_id"] for r in rows]


def turn_id_for_sentence(
    conn: sqlite3.Connection, sentence_id: str
) -> str | None:
    """Resolve the source turn for a sentence by walking back via claims."""
    row = conn.execute(
        "SELECT source_claim_ids FROM output_sentences WHERE sentence_id = ?",
        (sentence_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        decoded = json.loads(row["source_claim_ids"])
    except json.JSONDecodeError as exc:
        log.error(
            "substrate.malformed_source_claim_ids",
            sentence_id=sentence_id,
            error=str(exc),
        )
        raise
    if not isinstance(decoded, list) or not decoded:
        return None
    decoded_list = cast(list[Any], decoded)
    first = str(decoded_list[0])
    claim_row = conn.execute(
        "SELECT source_turn_id FROM claims WHERE claim_id = ?", (first,)
    ).fetchone()
    return claim_row["source_turn_id"] if claim_row is not None else None


# ------------------------------------------------------------------ spans ---


@dataclass(frozen=True, slots=True)
class ClaimSpan:
    """Exact substring of a source turn that produced a claim."""

    claim_id: str
    turn_id: str
    char_start: int | None
    char_end: int | None


def span_for_claim(conn: sqlite3.Connection, claim_id: str) -> ClaimSpan | None:
    row = conn.execute(
        "SELECT claim_id, source_turn_id, char_start, char_end FROM claims"
        " WHERE claim_id = ?",
        (claim_id,),
    ).fetchone()
    if row is None:
        return None
    return ClaimSpan(
        claim_id=row["claim_id"],
        turn_id=row["source_turn_id"],
        char_start=row["char_start"],
        char_end=row["char_end"],
    )


# -------------------------------------------------- output sentence insertion ---


class OutputSentenceValidationError(ValueError):
    """Raised when an output sentence is missing provenance."""


def insert_output_sentence(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    section: OutputSection,
    ordinal: int,
    text: str,
    source_claim_ids: list[str],
    sentence_id: str | None = None,
) -> OutputSentence:
    """Insert a validated output sentence.

    A sentence without source_claim_ids is rejected. Each claim_id must
    resolve to an existing row in claims.
    """
    if not source_claim_ids:
        raise OutputSentenceValidationError(
            "source_claim_ids must be non-empty"
        )
    placeholders = ",".join("?" * len(source_claim_ids))
    rows = conn.execute(
        f"SELECT claim_id FROM claims WHERE claim_id IN ({placeholders})",
        tuple(source_claim_ids),
    ).fetchall()
    found = {r["claim_id"] for r in rows}
    missing = [c for c in source_claim_ids if c not in found]
    if missing:
        raise OutputSentenceValidationError(
            f"source_claim_ids reference unknown claims: {missing}"
        )

    sid = sentence_id or f"os_{uuid.uuid4().hex[:12]}"
    payload = json.dumps(list(source_claim_ids))
    conn.execute(
        "INSERT INTO output_sentences (sentence_id, session_id, section, ordinal, text,"
        " source_claim_ids) VALUES (?, ?, ?, ?, ?, ?)",
        (sid, session_id, section.value, ordinal, text, payload),
    )
    log.info(
        "substrate.output_sentence_inserted",
        session_id=session_id,
        sentence_id=sid,
        section=section.value,
        n_source_claims=len(source_claim_ids),
    )
    return OutputSentence(
        sentence_id=sid,
        session_id=session_id,
        section=section,
        ordinal=ordinal,
        text=text,
        source_claim_ids=tuple(source_claim_ids),
    )
