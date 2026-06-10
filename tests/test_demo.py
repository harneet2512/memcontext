"""Smoke test for the "one corrected fact, three memories" demo.

Deterministic and model-free: in-memory SQLite, no embedder (the vector payload
is intentionally not exercised here — it needs a model). Asserts the end state
the demo depends on: DynamoDB is the active database value with a verifiable
span, Postgres is superseded by a typed user_correction edge, and the memcontext
payload exposes a current value with provenance that the summary payload cannot.
"""
from __future__ import annotations

import sqlite3

import pytest

# demo/ is an untracked sample package (not part of the shipped product); skip
# this scenario smoke test cleanly when it isn't importable in a fresh checkout.
pytest.importorskip("demo.scenario")
from demo.scenario import seed_demo
from memcontext.brain import brain
from memcontext.mcp_tools import handle_memory_trace


@pytest.fixture()
def seeded(db: sqlite3.Connection):
    manifest = seed_demo(db, pack="developer")
    return db, manifest


def test_brain_reports_current_value_with_provenance(seeded):
    conn, manifest = seeded
    ws = brain(conn, session_id=manifest["session_id"])

    main_db = ws["subjects"]["main_database"]
    assert len(main_db["facts"]) == 1, "only the current value should be active"
    fact = main_db["facts"][0]
    assert fact["value"] == "DynamoDB"
    assert fact["status"] == "active"
    assert fact["predicate"] == "decision_made"

    prov = fact["provenance"]
    assert prov["char_start"] is not None and prov["char_end"] is not None
    assert prov["quote"] == "DynamoDB"

    # gaps = vocabulary predicates with no active claim for this subject
    assert "decision_made" not in main_db["gaps"]
    assert "todo" in main_db["gaps"]


def test_postgres_superseded_via_typed_correction(seeded):
    conn, manifest = seeded
    trace = handle_memory_trace(
        conn,
        session_id=manifest["session_id"],
        subject="main_database",
        predicate="decision_made",
    )

    lineage = trace["lineage"]
    # newest-first: active DynamoDB on top, superseded Postgres beneath
    assert lineage[0]["value"] == "DynamoDB"
    assert lineage[0]["status"] == "active"

    postgres = next(row for row in lineage if row["value"] == "Postgres")
    assert postgres["status"] == "superseded"
    assert postgres["edge_type"] == "user_correction"
    assert postgres["quote"] == "Postgres"


