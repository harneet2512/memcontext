"""memory_store warns when a new decision subject resembles an existing live one.

Regression (HAR-95, Claude Code recall eval, dev split): Claude re-recorded a changed
decision under a new subject ("project/authentication" -> "auth-system/authentication-
method"). Supersession keys on the exact subject, so both values stayed current and
the hook injected both. memory_store now returns the likely matches so the agent can
re-store under the existing subject. It only warns: it never merges or supersedes.
"""
from __future__ import annotations

import pytest

from memcontext.conflicts import similar_subjects
from memcontext.mcp_tools import handle_memory_store
from memcontext.predicate_packs import active_pack
from memcontext.schema import open_database


@pytest.fixture
def conn(monkeypatch):
    monkeypatch.setenv("ACTIVE_PACK", "general,developer")  # decision_made is single-valued
    active_pack.cache_clear()
    return open_database(":memory:")


def _decide(conn, subject, value, *, predicate="decision_made", namespace="default", session="s1"):
    return handle_memory_store(conn, text=f"Decision: {subject} is {value}.", session_id=session,
                               namespace=namespace,
                               claims=[{"subject": subject, "predicate": predicate, "value": value}])


def test_new_subject_resembling_a_live_decision_is_flagged(conn):
    _decide(conn, "project/authentication", "JWT")
    r = _decide(conn, "auth-system/authentication-method", "server-side session cookies",
                session="s2")
    assert [m["subject"] for m in r["similar_subjects"]] == ["project/authentication"]
    assert r["similar_subjects"][0]["value"] == "JWT"
    assert any("project/authentication" in w and "same subject" in w for w in r["warnings"])


def test_the_warning_never_merges_or_supersedes(conn):
    _decide(conn, "project/authentication", "JWT")
    r = _decide(conn, "auth-system/authentication-method", "session cookies", session="s2")
    assert r["supersessions"] == 0
    active = conn.execute("SELECT value FROM claims WHERE status = 'active' ORDER BY created_ts")
    assert [row[0] for row in active] == ["JWT", "session cookies"]


def test_reusing_the_subject_supersedes_and_does_not_warn(conn):
    _decide(conn, "project/authentication", "JWT")
    r = _decide(conn, "project/authentication", "session cookies", session="s2")
    assert r["supersessions"] == 1
    assert "similar_subjects" not in r and "warnings" not in r


def test_first_decision_does_not_warn(conn):
    r = _decide(conn, "project/authentication", "JWT")
    assert "similar_subjects" not in r


def test_unrelated_decisions_do_not_warn(conn):
    _decide(conn, "billing-api/database", "PostgreSQL")
    r = _decide(conn, "billing-api/cache", "Redis")
    assert "similar_subjects" not in r


def test_container_words_alone_do_not_make_subjects_similar(conn):
    _decide(conn, "billing_service_language", "Go 1.22")
    r = _decide(conn, "go_service_logging_library", "zap")
    assert "similar_subjects" not in r


def test_multi_valued_predicates_do_not_warn(conn):
    _decide(conn, "release/blockers", "flaky tests", predicate="blocker")
    r = _decide(conn, "release/blocker", "missing signing key", predicate="blocker")
    assert "similar_subjects" not in r


def test_matches_stay_in_the_namespace_and_are_live(conn):
    _decide(conn, "project/authentication", "JWT", namespace="tenantB")
    _decide(conn, "web/authentication", "OAuth")
    _decide(conn, "web/authentication", "SAML")  # supersedes OAuth: not live any more
    r = _decide(conn, "auth-system/authentication-method", "passkeys", session="s2")
    assert [(m["subject"], m["value"]) for m in r["similar_subjects"]] == [
        ("web/authentication", "SAML"),
    ]


def test_similar_subjects_ranks_closest_first_and_caps(conn):
    for subj in ("api/auth-token-ttl", "project/authentication", "web/authentication-provider",
                 "mobile/authentication", "admin/authentication"):
        _decide(conn, subj, "x")
    got = similar_subjects(conn, subject="auth-system/authentication-method",
                           predicate="decision_made", namespace="default")
    # topic-word Jaccard vs {authentication, method}: 1/2 for the bare "authentication"
    # subjects (newest first), 1/3 for "authentication-provider"; "auth-token-ttl"
    # shares no word. Capped at 3.
    assert [m["subject"] for m in got] == [
        "admin/authentication", "mobile/authentication", "project/authentication",
    ]


def test_a_shared_project_prefix_is_not_evidence_of_the_same_decision(conn):
    # Live demo: every HAR-95 subject starts "har95_", so a new GAP-9 decision was
    # flagged as resembling three unrelated decisions.
    for topic, value in (("memory_granularity", "one memory per turn"),
                         ("context_grouping", "grouping is additive"),
                         ("derived_caches_tenancy", "no migration"),
                         ("storage_model", "extend the turns table")):
        _decide(conn, f"har95_{topic}", value)
    r = _decide(conn, "har95_gap-9_tie_ranking", "ties share a rank", session="s2")
    assert "similar_subjects" not in r


def test_real_drift_under_a_shared_prefix_is_still_flagged(conn):
    for topic, value in (("memory_granularity", "one memory per turn"),
                         ("context_grouping", "grouping is additive"),
                         ("auth_method", "JWT")):
        _decide(conn, f"har95_{topic}", value)
    r = _decide(conn, "har95_authentication_method", "session cookies", session="s2")
    assert [m["subject"] for m in r["similar_subjects"]] == ["har95_auth_method"]


def test_one_shared_word_among_many_is_not_drift(conn):
    # Live rehearsal: a new memory_query provenance decision was flagged as resembling
    # "memory granularity" and "memory storage model" through the word "memory" alone.
    for topic in ("memory_granularity", "memory_storage_model", "context_grouping"):
        _decide(conn, f"har95_{topic}", "x")
    r = _decide(conn, "har95_memory_query_provenance_fields", "source_turn_id by default",
                session="s2")
    assert "similar_subjects" not in r
