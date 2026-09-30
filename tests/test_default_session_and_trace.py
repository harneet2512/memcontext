"""Store, query, trace and brain must agree on where a fact lives.

Regression: memory_store without a session_id invented a random `session_xxxx`,
while memory_trace / brain default to "default" — so in real Claude Code runs
(Claude rarely passes session_id) trace-by-slot always answered "No active claim"
even though the claim and its supersession edge were correct, and Claude told the
user the history "isn't recorded". Supersession is namespace-wide, so tracing a
slot must not depend on guessing the session either.
"""
from __future__ import annotations

from memcontext.mcp_tools import handle_brain, handle_memory_store, handle_memory_trace
from memcontext.schema import open_database

DB_CLAIM = {"subject": "orders service", "predicate": "user_fact", "value": "uses PostgreSQL for orders"}
NEW_CLAIM = {"subject": "orders service", "predicate": "user_fact", "value": "uses SQLite for orders"}


def test_store_without_session_uses_the_default_session():
    conn = open_database(":memory:")
    r = handle_memory_store(conn, text="The orders service uses PostgreSQL for orders.", claims=[DB_CLAIM])
    assert r["session_id"] == "default"
    facts = [f for s in handle_brain(conn)["subjects"].values() for f in s["facts"]]
    assert any("PostgreSQL" in f["fact"] for f in facts)


def test_trace_by_slot_finds_the_fact_in_any_session():
    conn = open_database(":memory:")
    handle_memory_store(conn, text="The orders service uses PostgreSQL for orders.",
                        session_id="claude-a", claims=[DB_CLAIM])
    handle_memory_store(conn, text="Change of plan: the orders service uses SQLite for orders.",
                        session_id="claude-b", claims=[NEW_CLAIM])

    trace = handle_memory_trace(conn, subject="orders service", predicate="user_fact")
    assert "error" not in trace
    values = [step["value"] for step in trace["lineage"]]
    assert values[0] == "uses SQLite for orders"
    assert "uses PostgreSQL for orders" in values
