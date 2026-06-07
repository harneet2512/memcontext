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

import contextlib
import os
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


class SourceType(StrEnum):
    """Origin of an episode (Tier-1 retrievable unit).

    Every turn / tool-call result / browser observation is stored as an
    episode tagged with one of these. The default is a conversation turn.
    """

    CONVERSATION = "conversation"
    TOOL_CALL = "tool_call"
    BROWSER = "browser"


class ExtractionStatus(StrEnum):
    """Tier-2 fact-extraction state for an episode.

    - PENDING — async LLM extraction queued, not yet run.
    - DONE — async extraction ran and wrote facts.
    - SKIPPED — admission-rejected or extractor returned nothing.
    - STRUCTURED — synchronous (Passthrough/Simple) extraction already
      produced facts; no async pass needed.
    """

    PENDING = "pending"
    DONE = "done"
    SKIPPED = "skipped"
    STRUCTURED = "structured"


# ------------------------------------------------------------ dataclasses ---


@dataclass(frozen=True, slots=True)
class Turn:
    """One episode — a conversation turn, tool-call result, or browser observation.

    `source_type` distinguishes the origin; `source_metadata` carries
    JSON-encoded provenance (url/title/tool_name/...). `extraction_status`
    tracks the Tier-2 async fact-extraction lifecycle for this episode.
    Defaults keep legacy call sites (conversation turns) unchanged.
    """

    turn_id: str
    session_id: str
    speaker: Speaker
    text: str
    ts: int
    asr_confidence: float | None = None
    source_type: SourceType = SourceType.CONVERSATION
    source_metadata: str | None = None
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING


@dataclass(frozen=True, slots=True)
class Claim:
    """One claim — an atomic fact extracted from a turn.

    `char_start` / `char_end` index into the source turn's `text` and bound
    the substring that produced this claim. Both `None` means the extractor
    could not localise the span (falls back to full-turn highlight).

    `valid_from_ts` / `valid_until_ts` carry the temporal-validity window.
    Default: `valid_from_ts == created_ts`, `valid_until_ts == None` (unbounded).
    Pass-1 supersession sets `valid_until_ts` on the superseded claim.

    `text` is the NL form of the fact (Tier-2 NL-first). It is the always-present
    representation; `(subject, predicate, value)` is the optional structured
    precision layer attached only on high-confidence extraction. `text` is
    populated from the v4 migration onward; pre-v4 rows leave it `None` and fall
    back to the synthesized triple (see `claim_retrieval_text`).
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
    event_ts: int | None = None
    text: str | None = None


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
    turn_id           TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    speaker           TEXT NOT NULL CHECK (speaker IN ('user','assistant','system')),
    text              TEXT NOT NULL,
    ts                INTEGER NOT NULL,
    asr_confidence    REAL,
    source_type       TEXT NOT NULL DEFAULT 'conversation'
                        CHECK (source_type IN ('conversation','tool_call','browser')),
    source_metadata   TEXT,
    extraction_status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_turns_session_ts ON turns(session_id, ts);

CREATE TABLE IF NOT EXISTS claims (
    claim_id          TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    text              TEXT,
    subject           TEXT,
    predicate         TEXT,
    value             TEXT,
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
    event_ts          INTEGER,
    CHECK (
        valid_until_ts IS NULL
        OR valid_from_ts IS NULL
        OR valid_until_ts > valid_from_ts
    ),
    -- A fact is NL-first: `text` is always present. The structured triple
    -- (subject, predicate, value) is an optional all-or-nothing precision layer.
    CHECK (
        (subject IS NULL AND predicate IS NULL AND value IS NULL)
        OR (subject IS NOT NULL AND predicate IS NOT NULL AND value IS NOT NULL)
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

CREATE TABLE IF NOT EXISTS turn_embeddings (
    turn_id                 TEXT PRIMARY KEY REFERENCES turns(turn_id) ON DELETE CASCADE,
    embedding               BLOB NOT NULL,
    embedding_model_version TEXT NOT NULL,
    embedded_at_unix        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turn_embeddings_model
    ON turn_embeddings(embedding_model_version);

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

CREATE TABLE IF NOT EXISTS claim_entities (
    claim_id    TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('person','organization','location','proper_noun')),
    PRIMARY KEY (claim_id, entity_text)
);
CREATE INDEX IF NOT EXISTS idx_claim_entities_text ON claim_entities(entity_text);

CREATE TABLE IF NOT EXISTS profiles (
    subject         TEXT PRIMARY KEY,
    profile_text    TEXT NOT NULL,
    profile_data    TEXT NOT NULL,
    claim_count     INTEGER NOT NULL,
    session_count   INTEGER NOT NULL,
    built_at_ts     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS session_digests (
    session_id      TEXT PRIMARY KEY,
    digest_text     TEXT NOT NULL,
    digest_data     TEXT NOT NULL,
    claim_count     INTEGER NOT NULL,
    built_at_ts     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS life_events (
    event_id            TEXT PRIMARY KEY,
    subject             TEXT NOT NULL,
    timestamp_start     INTEGER NOT NULL,
    timestamp_end       INTEGER NOT NULL,
    claim_ids           TEXT NOT NULL,
    predicates_affected TEXT NOT NULL,
    summary_text        TEXT NOT NULL,
    significance        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_life_events_subject ON life_events(subject);
"""


# Bump when adding a migration step below. Existing databases upgrade forward
# on open; fresh databases get every step applied once.
SCHEMA_VERSION = 5


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply ordered forward migrations keyed on ``PRAGMA user_version``.

    Steps v1–v3 only ADD columns/indexes. Step v4 is the one destructive
    exception: SQLite cannot drop a ``NOT NULL`` constraint in place, so making
    the structured triple optional requires a full ``claims`` table-rebuild
    (the sanctioned CREATE-new / copy / DROP / RENAME procedure). To add a
    migration: append an ``if current < N`` block and bump ``SCHEMA_VERSION``.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current >= SCHEMA_VERSION:
        return

    if current < 1:
        # v1: importance_score on claim_metadata (previously an ad-hoc ALTER).
        try:
            conn.execute(
                "ALTER TABLE claim_metadata ADD COLUMN importance_score REAL DEFAULT 0.5"
            )
        except sqlite3.OperationalError:
            pass  # column already present on pre-versioning databases

    if current < 2:
        # v2: covering index for list_active_claims (session_id + status filter,
        # created_ts order) — the hottest read, run by every retrieve_hybrid.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claims_active_recency"
            " ON claims(session_id, status, created_ts)"
        )

    if current < 3:
        # v3 (Tier-1 episodes): promote turns to first-class retrievable episodes.
        # Additive — ALTER ADD COLUMN with constant defaults + a new sidecar.
        # The source_type CHECK lives only in _SCHEMA_SQL (fresh DBs); migrated
        # DBs enforce the domain in the Python write layer to avoid a turns
        # table-rebuild. turn_embeddings mirrors claim_embeddings so the existing
        # vector encode/decode + model-version logic is reused verbatim.
        for col_sql in (
            "ALTER TABLE turns ADD COLUMN source_type TEXT NOT NULL"
            " DEFAULT 'conversation'",
            "ALTER TABLE turns ADD COLUMN source_metadata TEXT",
            "ALTER TABLE turns ADD COLUMN extraction_status TEXT NOT NULL"
            " DEFAULT 'pending'",
        ):
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(col_sql)  # column already present on partial upgrades
        conn.execute(
            "CREATE TABLE IF NOT EXISTS turn_embeddings ("
            " turn_id TEXT PRIMARY KEY REFERENCES turns(turn_id) ON DELETE CASCADE,"
            " embedding BLOB NOT NULL,"
            " embedding_model_version TEXT NOT NULL,"
            " embedded_at_unix INTEGER NOT NULL)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_turn_embeddings_model"
            " ON turn_embeddings(embedding_model_version)"
        )

    if current < 4:
        # v4 (Tier-2 NL-first facts): make the structured triple optional so an
        # out-of-vocab / NL-only fact can be stored as `text` alone. SQLite cannot
        # drop NOT NULL in place, so rebuild `claims`. foreign_keys MUST be OFF for
        # the DROP so it does NOT cascade-delete the sidecars (claim_embeddings,
        # claim_metadata, claim_entities, event_frame_claims, supersession_edges) —
        # their claim_id values are preserved identically across the RENAME. `text`
        # is backfilled from the legacy triple (every legacy row has a full triple).
        # Guarded so fresh/already-migrated DBs (nullable subject) skip the rebuild.
        info = conn.execute("PRAGMA table_info(claims)").fetchall()
        subject_not_null = any(r[1] == "subject" and r[3] for r in info)
        has_text_col = any(r[1] == "text" for r in info)
        if subject_not_null or not has_text_col:
            conn.execute("PRAGMA foreign_keys=OFF")
            try:
                conn.execute("BEGIN")
                conn.execute("DROP TABLE IF EXISTS claims_new")
                conn.execute(
                    "CREATE TABLE claims_new ("
                    " claim_id TEXT PRIMARY KEY,"
                    " session_id TEXT NOT NULL,"
                    " text TEXT,"
                    " subject TEXT,"
                    " predicate TEXT,"
                    " value TEXT,"
                    " value_normalised TEXT,"
                    " confidence REAL NOT NULL,"
                    " source_turn_id TEXT NOT NULL"
                    " REFERENCES turns(turn_id) ON DELETE CASCADE,"
                    " status TEXT NOT NULL CHECK (status IN"
                    " ('active','superseded','confirmed','dismissed','draft','audited')),"
                    " created_ts INTEGER NOT NULL,"
                    " char_start INTEGER,"
                    " char_end INTEGER,"
                    " valid_from_ts INTEGER,"
                    " valid_until_ts INTEGER,"
                    " event_ts INTEGER,"
                    " CHECK (valid_until_ts IS NULL OR valid_from_ts IS NULL"
                    " OR valid_until_ts > valid_from_ts),"
                    " CHECK ((subject IS NULL AND predicate IS NULL AND value IS NULL)"
                    " OR (subject IS NOT NULL AND predicate IS NOT NULL"
                    " AND value IS NOT NULL)))"
                )
                conn.execute(
                    "INSERT INTO claims_new"
                    " (claim_id, session_id, text, subject, predicate, value,"
                    "  value_normalised, confidence, source_turn_id, status,"
                    "  created_ts, char_start, char_end, valid_from_ts,"
                    "  valid_until_ts, event_ts)"
                    " SELECT claim_id, session_id,"
                    "  subject || ' ' || predicate || ' ' || value,"
                    "  subject, predicate, value, value_normalised, confidence,"
                    "  source_turn_id, status, created_ts, char_start, char_end,"
                    "  valid_from_ts, valid_until_ts, event_ts"
                    " FROM claims"
                )
                conn.execute("DROP TABLE claims")
                conn.execute("ALTER TABLE claims_new RENAME TO claims")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_claims_active"
                    " ON claims(session_id, subject, predicate, status)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_claims_temporal"
                    " ON claims(valid_from_ts, valid_until_ts)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_claims_active_recency"
                    " ON claims(session_id, status, created_ts)"
                )
                conn.execute("COMMIT")
            finally:
                conn.execute("PRAGMA foreign_keys=ON")

    if current < 5:
        # v5: usage/access signals on claim_metadata — retrieval reinforcement
        # (incremented when a claim is served; a ranking signal + decay input).
        for _col in (
            "ALTER TABLE claim_metadata ADD COLUMN access_count INTEGER DEFAULT 0",
            "ALTER TABLE claim_metadata ADD COLUMN last_accessed_ts INTEGER",
        ):
            try:
                conn.execute(_col)
            except sqlite3.OperationalError:
                pass  # column already present

    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    log.debug("substrate.db_migrated", from_version=current, to_version=SCHEMA_VERSION)


class DatabaseUnavailableError(RuntimeError):
    """The database could not be opened — locked by another process, or corrupt.

    Carries a friendly, path-scoped message (the path is the caller's own) and
    no traceback detail, so CLI/tool layers can surface it cleanly.
    """


def open_database(path: str | Path) -> sqlite3.Connection:
    """Open (or create) a SQLite DB with PRAGMAs and schema.

    Idempotent — safe to call against an existing DB. `path` may be
    `":memory:"` for unit tests. Raises `DatabaseUnavailableError` (never a raw
    sqlite traceback) if the file is locked by another process or corrupt.
    """
    is_memory = str(path) == ":memory:"
    try:
        conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        if not is_memory:
            # Restrict the DB file to the owner (best-effort; limited on Windows).
            try:
                os.chmod(str(path), 0o600)
            except OSError:
                pass
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA cache_size=-20000")
        conn.execute("PRAGMA foreign_keys=ON")

        conn.executescript(_SCHEMA_SQL)
        _migrate(conn)
    except sqlite3.OperationalError as exc:  # subclass of DatabaseError — catch first
        detail = str(exc).lower()
        if "lock" in detail or "busy" in detail:
            raise DatabaseUnavailableError(
                f"database is locked by another MemContext process: {path}. "
                "Close the other client (or stop `memcontext serve`) and retry."
            ) from None
        raise DatabaseUnavailableError(
            f"could not open database {path} (it may be read-only or in use)."
        ) from None
    except sqlite3.DatabaseError:  # malformed / corrupt file
        raise DatabaseUnavailableError(
            f"database file is unreadable or corrupt: {path}. "
            "Restore from a backup, or remove the file to start fresh."
        ) from None

    log.debug("substrate.db_opened", path=str(path))
    return conn
