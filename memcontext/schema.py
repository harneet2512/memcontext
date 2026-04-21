"""SQLite schema + typed data model for the memory substrate.

Defines the core data model — claims (subject-predicate-value triples),
turns (conversation utterances), supersession edges (how facts evolve),
and output sentences (generated text backed by claims).

Tables:
- turns — conversation turns
- claims — atomic facts extracted from turns, with temporal validity
- supersession_edges — typed edges linking old → new claims
- decisions — audit trail for user actions
- output_sentences — generated text with provenance back to claims
- claim_embeddings — sidecar for retrieval vectors
- claim_metadata — sidecar for multi-signal retrieval fusion
- event_frames / event_frame_claims / event_frame_embeddings — compositional events
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import structlog

log = structlog.get_logger(__name__)


# ------------------------------------------------------------------ enums ---


class Speaker(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ClaimStatus(StrEnum):
    """Lifecycle state of a claim.

    DRAFT and AUDITED extend the lifecycle for audit-and-revise workflows.
    DRAFT claims are not considered active for projection. AUDITED claims
    have passed audit and are treated as active.
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    DRAFT = "draft"
    AUDITED = "audited"


class EdgeType(StrEnum):
    """Typed supersession edge vocabulary.

    - USER_CORRECTION — user restates with a different value (Pass 1).
    - ASSISTANT_CONFIRM — assistant confirms a prior user claim (Pass 1).
    - SEMANTIC_REPLACE — same identity under cosine threshold (Pass 2).
    - REFINES — new value is a narrower subset of the old one (Pass 1).
    - CONTRADICTS — later speaker refutes an earlier value (Pass 1).
    - RULED_OUT — tied to a hypothesis becoming dead (verifier).
    - DISMISSED_BY_USER — explicit user dismissal action.
    """

    USER_CORRECTION = "user_correction"
    ASSISTANT_CONFIRM = "assistant_confirm"
    SEMANTIC_REPLACE = "semantic_replace"
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    RULED_OUT = "ruled_out"
    DISMISSED_BY_USER = "dismissed_by_user"


class OutputSection(StrEnum):
    """Section labels for generated output sentences."""

    SUMMARY = "summary"
    DETAIL = "detail"
    ANALYSIS = "analysis"
    ACTION = "action"


# ------------------------------------------------------------ dataclasses ---


@dataclass(frozen=True, slots=True)
class Turn:
    """One conversation turn."""

    turn_id: str
    session_id: str
    speaker: Speaker
    text: str
    ts: int
    asr_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Claim:
    """One claim — an atomic fact extracted from a turn.

    `char_start` / `char_end` index into the source turn's `text` and bound
    the substring that produced this claim. Both `None` means the extractor
    could not localise the span (falls back to full-turn highlight).

    `valid_from_ts` / `valid_until_ts` carry the temporal-validity window.
    Default: `valid_from_ts == created_ts`, `valid_until_ts == None` (unbounded).
    Pass-1 supersession sets `valid_until_ts` on the superseded claim.
    """

    claim_id: str
    session_id: str
    subject: str
    predicate: str
    value: str
    value_normalised: str | None
    confidence: float
    source_turn_id: str
    status: ClaimStatus
    created_ts: int
    char_start: int | None = None
    char_end: int | None = None
    valid_from_ts: int | None = None
    valid_until_ts: int | None = None


@dataclass(frozen=True, slots=True)
class SupersessionEdge:
    """Lifecycle edge between two claims.

    `identity_score` is the Pass-2 cosine similarity when `edge_type ==
    SEMANTIC_REPLACE`; `None` for the deterministic Pass-1 kinds.
    """

    edge_id: str
    old_claim_id: str
    new_claim_id: str
    edge_type: EdgeType
    identity_score: float | None
    created_ts: int


@dataclass(frozen=True, slots=True)
class OutputSentence:
    """One generated output sentence backed by source claims."""

    sentence_id: str
    session_id: str
    section: OutputSection
    ordinal: int
    text: str
    source_claim_ids: tuple[str, ...]


# ----------------------------------------------------------------- schema ---


_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id         TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    speaker         TEXT NOT NULL CHECK (speaker IN ('user','assistant','system')),
    text            TEXT NOT NULL,
    ts              INTEGER NOT NULL,
    asr_confidence  REAL
);
CREATE INDEX IF NOT EXISTS idx_turns_session_ts ON turns(session_id, ts);

CREATE TABLE IF NOT EXISTS claims (
    claim_id          TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    subject           TEXT NOT NULL,
    predicate         TEXT NOT NULL,
    value             TEXT NOT NULL,
    value_normalised  TEXT,
    confidence        REAL NOT NULL,
    source_turn_id    TEXT NOT NULL REFERENCES turns(turn_id) ON DELETE CASCADE,
    status            TEXT NOT NULL CHECK (
                          status IN (
                              'active','superseded','confirmed','dismissed',
                              'draft','audited'
                          )
                      ),
    created_ts        INTEGER NOT NULL,
    char_start        INTEGER,
    char_end          INTEGER,
    valid_from_ts     INTEGER,
    valid_until_ts    INTEGER,
    CHECK (
        valid_until_ts IS NULL
        OR valid_from_ts IS NULL
        OR valid_until_ts > valid_from_ts
    )
);
CREATE INDEX IF NOT EXISTS idx_claims_active
    ON claims(session_id, subject, predicate, status);
CREATE INDEX IF NOT EXISTS idx_claims_temporal
    ON claims(valid_from_ts, valid_until_ts);

CREATE TABLE IF NOT EXISTS supersession_edges (
    edge_id         TEXT PRIMARY KEY,
    old_claim_id    TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    new_claim_id    TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    edge_type       TEXT NOT NULL CHECK (edge_type IN (
                        'user_correction','assistant_confirm','semantic_replace',
                        'refines','contradicts','ruled_out','dismissed_by_user'
                    )),
    identity_score  REAL,
    created_ts      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_supersession_old ON supersession_edges(old_claim_id);
CREATE INDEX IF NOT EXISTS idx_supersession_new ON supersession_edges(new_claim_id);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id           TEXT PRIMARY KEY,
    session_id            TEXT NOT NULL,
    kind                  TEXT NOT NULL,
    target_type           TEXT NOT NULL,
    target_id             TEXT NOT NULL,
    claim_state_snapshot  TEXT NOT NULL,
    ts                    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS output_sentences (
    sentence_id       TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    section           TEXT NOT NULL CHECK (section IN ('summary','detail','analysis','action')),
    ordinal           INTEGER NOT NULL,
    text              TEXT NOT NULL,
    source_claim_ids  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_output_session_section
    ON output_sentences(session_id, section, ordinal);

CREATE TABLE IF NOT EXISTS claim_embeddings (
    claim_id                TEXT PRIMARY KEY REFERENCES claims(claim_id) ON DELETE CASCADE,
    embedding               BLOB NOT NULL,
    embedding_model_version TEXT NOT NULL,
    embedded_at_unix        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claim_embeddings_model
    ON claim_embeddings(embedding_model_version);

CREATE TABLE IF NOT EXISTS claim_metadata (
    claim_id           TEXT PRIMARY KEY REFERENCES claims(claim_id) ON DELETE CASCADE,
    entity_key         TEXT NOT NULL,
    predicate_family   TEXT NOT NULL,
    temporal_bin       TEXT
);
CREATE INDEX IF NOT EXISTS idx_claim_metadata_entity ON claim_metadata(entity_key);
CREATE INDEX IF NOT EXISTS idx_claim_metadata_temporal ON claim_metadata(temporal_bin);

CREATE TABLE IF NOT EXISTS event_frames (
    event_id             TEXT PRIMARY KEY,
    event_type           TEXT NOT NULL,
    participants         TEXT NOT NULL,
    item                 TEXT,
    location             TEXT,
    time_expr            TEXT,
    amount               TEXT,
    supporting_claim_ids TEXT NOT NULL,
    source_turn_ids      TEXT NOT NULL,
    session_id           TEXT NOT NULL,
    confidence           REAL NOT NULL,
    missing_slots        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_frames_session ON event_frames(session_id);

CREATE TABLE IF NOT EXISTS event_frame_claims (
    event_id  TEXT NOT NULL REFERENCES event_frames(event_id) ON DELETE CASCADE,
    claim_id  TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    slot_role TEXT NOT NULL,
    PRIMARY KEY (event_id, claim_id)
);

CREATE TABLE IF NOT EXISTS event_frame_embeddings (
    event_id                TEXT PRIMARY KEY REFERENCES event_frames(event_id) ON DELETE CASCADE,
    embedding               BLOB NOT NULL,
    embedding_model_version TEXT NOT NULL,
    embedded_at_unix        INTEGER NOT NULL
);
"""


def open_database(path: str | Path) -> sqlite3.Connection:
    """Open (or create) a SQLite DB with PRAGMAs and schema.

    Idempotent — safe to call against an existing DB. `path` may be
    `":memory:"` for unit tests.
    """
    is_memory = str(path) == ":memory:"
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    if not is_memory:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA cache_size=-20000")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript(_SCHEMA_SQL)
    log.debug("substrate.db_opened", path=str(path))
    return conn
