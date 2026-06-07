"""Schema migration v2 -> v4: turns become episodes, claims become NL-first facts.

The v4 step rebuilds `claims` (SQLite cannot drop NOT NULL in place) to make the
structured triple optional and add the NL `text` column. The rebuild must:
  - preserve every claim row with its claim_id and triple,
  - backfill `text` from the legacy triple (no data loss),
  - leave subject/predicate/value NULLABLE going forward,
  - NOT cascade-delete the sidecars (foreign_keys OFF during the DROP),
  - restore foreign_keys ON afterwards.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from memcontext.schema import open_database

# A minimal v2-shaped database: legacy NOT NULL claims, no `text` column, plus
# the five sidecars whose rows must survive the rebuild.
_V2_SCHEMA = """
CREATE TABLE turns (
    turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, speaker TEXT NOT NULL,
    text TEXT NOT NULL, ts INTEGER NOT NULL, asr_confidence REAL
);
CREATE TABLE claims (
    claim_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
    subject TEXT NOT NULL, predicate TEXT NOT NULL, value TEXT NOT NULL,
    value_normalised TEXT, confidence REAL NOT NULL,
    source_turn_id TEXT NOT NULL REFERENCES turns(turn_id) ON DELETE CASCADE,
    status TEXT NOT NULL, created_ts INTEGER NOT NULL,
    char_start INTEGER, char_end INTEGER,
    valid_from_ts INTEGER, valid_until_ts INTEGER, event_ts INTEGER
);
CREATE TABLE supersession_edges (
    edge_id TEXT PRIMARY KEY,
    old_claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    new_claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL, identity_score REAL, created_ts INTEGER NOT NULL
);
CREATE TABLE claim_embeddings (
    claim_id TEXT PRIMARY KEY REFERENCES claims(claim_id) ON DELETE CASCADE,
    embedding BLOB NOT NULL, embedding_model_version TEXT NOT NULL,
    embedded_at_unix INTEGER NOT NULL
);
CREATE TABLE claim_metadata (
    claim_id TEXT PRIMARY KEY REFERENCES claims(claim_id) ON DELETE CASCADE,
    entity_key TEXT NOT NULL, predicate_family TEXT NOT NULL, temporal_bin TEXT
);
CREATE TABLE claim_entities (
    claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    entity_text TEXT NOT NULL, entity_type TEXT NOT NULL,
    PRIMARY KEY (claim_id, entity_text)
);
"""


def _build_v2_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(_V2_SCHEMA)
    conn.execute(
        "INSERT INTO turns VALUES ('t1','s','user','I live in Paris',100,NULL)"
    )
    # c1 (superseded) -> c2 (active): a real supersession chain.
    conn.execute(
        "INSERT INTO claims VALUES"
        " ('c1','s','user','user_location','Paris',NULL,0.9,'t1','superseded',"
        " 100,NULL,NULL,100,200,NULL)"
    )
    conn.execute(
        "INSERT INTO claims VALUES"
        " ('c2','s','user','user_location','Berlin',NULL,0.9,'t1','active',"
        " 200,NULL,NULL,200,NULL,NULL)"
    )
    conn.execute(
        "INSERT INTO supersession_edges VALUES"
        " ('e1','c1','c2','user_correction',NULL,200)"
    )
    conn.execute("INSERT INTO claim_embeddings VALUES ('c1', X'00', 'm', 0)")
    conn.execute(
        "INSERT INTO claim_metadata VALUES ('c1','user','user_location','2026-Q1')"
    )
    conn.execute("INSERT INTO claim_entities VALUES ('c1','paris','location')")
    conn.execute("PRAGMA user_version=2")
    conn.commit()
    conn.close()


def test_v2_to_v4_upgrade_preserves_data_and_sidecars(tmp_path: Path):
    path = str(tmp_path / "v2.db")
    _build_v2_db(path)

    conn = open_database(path)
    try:
        # Reached v4.
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5

        # Structured triple is now NULLABLE; the text column exists.
        info = {r[1]: r[3] for r in conn.execute("PRAGMA table_info(claims)")}
        assert info["subject"] == 0, "subject must be nullable after v4"
        assert info["predicate"] == 0
        assert info["value"] == 0
        assert "text" in info

        # Every claim survived, with triple intact and text backfilled.
        rows = {
            r["claim_id"]: r
            for r in conn.execute(
                "SELECT claim_id, text, subject, predicate, value, status FROM claims"
            )
        }
        assert set(rows) == {"c1", "c2"}, "no claim rows lost"
        assert rows["c1"]["text"] == "user user_location Paris"
        assert rows["c2"]["text"] == "user user_location Berlin"
        assert rows["c1"]["subject"] == "user" and rows["c1"]["value"] == "Paris"
        assert rows["c1"]["status"] == "superseded"
        assert rows["c2"]["status"] == "active"

        # Sidecars survived the DROP (foreign_keys was OFF during the rebuild).
        def count(table: str) -> int:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        assert count("supersession_edges") == 1, "supersession edge lost"
        assert count("claim_embeddings") == 1, "embedding lost"
        assert count("claim_metadata") == 1, "metadata lost"
        assert count("claim_entities") == 1, "entity lost"

        # The edge still references the surviving claims.
        edge = conn.execute(
            "SELECT old_claim_id, new_claim_id FROM supersession_edges"
        ).fetchone()
        assert (edge["old_claim_id"], edge["new_claim_id"]) == ("c1", "c2")

        # foreign_keys enforcement is restored.
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_v2_to_v4_allows_nl_only_fact_after_upgrade(tmp_path: Path):
    """After v4 the rebuilt table accepts an NL-only fact (NULL triple, text set)."""
    path = str(tmp_path / "v2b.db")
    _build_v2_db(path)
    conn = open_database(path)
    try:
        conn.execute(
            "INSERT INTO claims"
            " (claim_id, session_id, text, subject, predicate, value, confidence,"
            "  source_turn_id, status, created_ts)"
            " VALUES ('c3','s','an unstructured note', NULL, NULL, NULL, 0.7,"
            "  't1','active', 300)"
        )
        row = conn.execute(
            "SELECT text, subject FROM claims WHERE claim_id='c3'"
        ).fetchone()
        assert row["text"] == "an unstructured note"
        assert row["subject"] is None

        # The all-or-nothing CHECK rejects a partial triple.
        try:
            conn.execute(
                "INSERT INTO claims"
                " (claim_id, session_id, text, subject, predicate, value, confidence,"
                "  source_turn_id, status, created_ts)"
                " VALUES ('c4','s','bad', 'user', NULL, NULL, 0.7, 't1','active', 400)"
            )
            raised = False
        except sqlite3.IntegrityError:
            raised = True
        assert raised, "partial triple must violate the all-or-nothing CHECK"
    finally:
        conn.close()


def test_fresh_db_is_born_at_v4_and_skips_rebuild(tmp_path: Path):
    """A fresh DB is created directly at the v4 shape (nullable triple + text)."""
    path = str(tmp_path / "fresh.db")
    conn = open_database(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
        info = {r[1]: r[3] for r in conn.execute("PRAGMA table_info(claims)")}
        assert info["subject"] == 0
        assert "text" in info
    finally:
        conn.close()
