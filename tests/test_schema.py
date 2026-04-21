from __future__ import annotations

import sqlite3

import pytest

from memcontext.schema import (
    ClaimStatus,
    EdgeType,
    OutputSection,
    Speaker,
    open_database,
)


def test_open_database_memory():
    conn = open_database(":memory:")
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    expected = {
        "turns",
        "claims",
        "supersession_edges",
        "decisions",
        "output_sentences",
        "claim_embeddings",
        "claim_metadata",
        "event_frames",
        "event_frame_claims",
        "event_frame_embeddings",
    }
    assert expected <= tables


def test_open_database_idempotent(tmp_path):
    path = str(tmp_path / "test.db")
    conn1 = open_database(path)
    conn1.close()
    conn2 = open_database(path)
    tables = {
        r[0]
        for r in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "claims" in tables
    conn2.close()


def test_foreign_keys_enabled(db: sqlite3.Connection):
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO claims (claim_id, session_id, subject, predicate, value,"
            " confidence, source_turn_id, status, created_ts)"
            " VALUES ('c1','s1','subj','pred','val',0.5,'nonexistent_turn','active',1)"
        )


def test_enum_values():
    assert ClaimStatus.ACTIVE.value == "active"
    assert ClaimStatus.SUPERSEDED.value == "superseded"
    assert EdgeType.USER_CORRECTION.value == "user_correction"
    assert EdgeType.SEMANTIC_REPLACE.value == "semantic_replace"
    assert Speaker.USER.value == "user"
    assert Speaker.ASSISTANT.value == "assistant"
    assert OutputSection.SUMMARY.value == "summary"

    assert ClaimStatus("active") == ClaimStatus.ACTIVE
    assert EdgeType("contradicts") == EdgeType.CONTRADICTS
    assert Speaker("user") == Speaker.USER
