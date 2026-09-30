"""Facts stored with an out-of-vocabulary predicate must never be served blank.

Regression: `insert_claim` deliberately demotes an unknown predicate to an NL-only
fact (triple NULL, content in `text`), but memory_query / memory_trace / brain
served only the empty `value`, and memory_store never said it had demoted the
claim. Claude Code picked `predicate="uses_database"` in a real run and got back
`"value": ""` for its own decision.
"""
from __future__ import annotations

import pytest

from memcontext import mcp_server
from memcontext.mcp_tools import (
    handle_brain,
    handle_memory_query,
    handle_memory_store,
    handle_memory_trace,
)
from memcontext.predicate_packs import active_pack
from memcontext.schema import open_database
from memcontext.trace_view import render_trace_table

OFF_VOCAB = {"subject": "orders service", "predicate": "uses_database", "value": "PostgreSQL 16"}


@pytest.fixture
def conn():
    return open_database(":memory:")


def _store(conn, claim):
    return handle_memory_store(conn, text="The orders service uses PostgreSQL 16 for persistence.",
                               session_id="s1", claims=[claim])


def test_store_warns_when_a_predicate_is_demoted(conn):
    r = _store(conn, OFF_VOCAB)
    assert r["claims_created"] == 1
    assert any("uses_database" in w for w in r["warnings"])


def test_store_has_no_warning_for_in_vocab_predicates(conn):
    r = _store(conn, {"subject": "orders service", "predicate": "user_fact",
                      "value": "uses PostgreSQL 16"})
    assert "warnings" not in r


def test_query_serves_the_fact_text_for_nl_only_claims(conn):
    _store(conn, OFF_VOCAB)
    claims = handle_memory_query(conn, query="orders service database", session_id="s1")["claims"]
    assert claims and all(c["fact"] for c in claims)
    assert any("PostgreSQL 16" in c["fact"] for c in claims)


def test_trace_and_brain_serve_the_fact_text(conn):
    cid = _store(conn, OFF_VOCAB)["claim_ids"][0]
    trace = handle_memory_trace(conn, claim_id=cid)
    assert "PostgreSQL 16" in trace["claim"]["fact"]
    assert "PostgreSQL 16" in trace["lineage"][0]["fact"]
    assert "PostgreSQL 16" in render_trace_table(trace)

    facts = [f for s in handle_brain(conn, session_id="s1")["subjects"].values() for f in s["facts"]]
    assert any("PostgreSQL 16" in f["fact"] for f in facts)


def test_memory_store_tool_lists_the_active_predicates():
    desc = mcp_server.memory_store_description()
    for predicate in active_pack().predicate_families:
        assert predicate in desc
