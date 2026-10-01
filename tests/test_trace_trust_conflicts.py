"""memory_trace by slot: the most-trusted current value wins, and conflicts are shown.

Regression (HAR-95 GAP-5): the slot head was the newest active claim. When the
source-trust guard refuses a low-trust override, both values stay active, so a
newer web page became the "current" value of the slot, with a lineage of one and
nothing saying the trusted value was still live.
"""
from __future__ import annotations

import pytest

from memcontext.claims import insert_turn, new_turn_id, now_ns
from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import handle_memory_store, handle_memory_trace
from memcontext.on_new_turn import run_extraction
from memcontext.predicate_packs import active_pack
from memcontext.schema import SourceType, Speaker, Turn, open_database
from memcontext.trace_view import render_trace_table


@pytest.fixture
def conn(monkeypatch):
    monkeypatch.setenv("ACTIVE_PACK", "general,developer")  # decision_made is single-valued
    active_pack.cache_clear()
    return open_database(":memory:")


def _store(conn, value, *, subject="billing-api/database", session="s1", namespace="default"):
    return handle_memory_store(conn, text=f"Decision: {subject} is {value}.", session_id=session,
                               namespace=namespace,
                               claims=[{"subject": subject, "predicate": "decision_made",
                                        "value": value}])


def _web(conn, value, *, subject="billing-api/database", session="s1", namespace="default"):
    t = Turn(turn_id=new_turn_id(), session_id=session, speaker=Speaker.USER, ts=now_ns(),
             text=f"Forum: {subject} switched to {value}.", source_type=SourceType.BROWSER)
    insert_turn(conn, t, namespace=namespace)
    run_extraction(conn, episode_id=t.turn_id, session_id=session, extractor=PassthroughExtractor(
        [{"subject": subject, "predicate": "decision_made", "value": value}]))


def _trace(conn, subject="billing-api/database", session="s1"):
    return handle_memory_trace(conn, subject=subject, predicate="decision_made", session_id=session)


def test_trusted_value_beats_a_newer_untrusted_one(conn):
    _store(conn, "MySQL")
    _store(conn, "PostgreSQL")       # user correction: supersedes MySQL
    _web(conn, "MongoDB")            # blocked by the trust guard: stays active
    t = _trace(conn)
    assert t["claim"]["value"] == "PostgreSQL"
    assert [s["value"] for s in t["lineage"]] == ["PostgreSQL", "MySQL"]
    [c] = t["conflicts"]
    assert c["value"] == "MongoDB" and c["quarantined"] is True and c["newer_than_head"] is True


def test_equal_trust_falls_back_to_the_newest_value(conn):
    _store(conn, "PostgreSQL")
    _store(conn, "CockroachDB", subject="billing-api/database-engine")  # same kind, new subject
    t = _trace(conn)
    assert t["claim"]["value"] == "PostgreSQL"  # the traced slot's own value
    assert [c["value"] for c in t["conflicts"]] == ["CockroachDB"]
    assert t["conflicts"][0]["subject"] == "billing-api/database-engine"


def test_a_restated_value_traces_to_its_original_assertion(conn):
    _store(conn, "MySQL")
    _store(conn, "PostgreSQL")
    _store(conn, "PostgreSQL", session="s2")  # restatement, no supersession edge of its own
    t = _trace(conn)
    assert [s["value"] for s in t["lineage"]] == ["PostgreSQL", "MySQL"]
    assert t["restatements"] == 1
    assert t["conflicts"] == []


def test_no_conflicts_when_the_slot_has_one_current_value(conn):
    _store(conn, "MySQL")
    _store(conn, "PostgreSQL")
    t = _trace(conn)
    assert t["conflicts"] == [] and "restatements" not in t


def test_conflicts_never_cross_namespaces(conn):
    _store(conn, "PostgreSQL", namespace="tenantA")
    _store(conn, "SQLite", namespace="tenantB")  # same session id "s1", other tenant
    t = _trace(conn)
    assert t["claim"]["value"] in {"PostgreSQL", "SQLite"}
    assert t["conflicts"] == []


def test_trace_by_claim_id_shows_the_trusted_rival_of_an_untrusted_claim(conn):
    _store(conn, "PostgreSQL")
    _web(conn, "MongoDB")
    web_id = conn.execute("SELECT claim_id FROM claims WHERE value = 'MongoDB'").fetchone()[0]
    t = handle_memory_trace(conn, claim_id=web_id)
    assert t["claim"]["value"] == "MongoDB"  # an explicit claim id is traced as asked
    assert [(c["value"], c["quarantined"]) for c in t["conflicts"]] == [("PostgreSQL", False)]


def test_rendered_trace_shows_the_conflict(conn):
    _store(conn, "PostgreSQL")
    _web(conn, "MongoDB")
    text = render_trace_table(_trace(conn))
    assert "ACTIVE      PostgreSQL" in text
    assert "CONFLICT    1 other current value(s)" in text
    assert "MongoDB" in text and "untrusted source" in text
