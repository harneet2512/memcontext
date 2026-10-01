"""Prompt hook: two current values for the same kind of decision are flagged, newest first.

Regression: the Claude Code recall eval saw Claude write a stale CI provider. The
decision had been re-recorded under a new subject, so supersession never linked the
two values; both stayed current and the hook injected them as two unrelated facts
with nothing saying which was newer.

"Same kind of decision" is deterministic: the same single-valued predicate (one
current value per slot, e.g. decision_made) and either the same subject or the same
project prefix with one topic's words contained in the other's. Multi-valued
predicates (blocker, todo) never conflict: several current values are normal there.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from memcontext import http_server
from memcontext.authz import register_principal
from memcontext.claims import insert_turn, new_turn_id, now_ns
from memcontext.conflicts import live_same_kind, same_decision_kind
from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import handle_memory_store
from memcontext.on_new_turn import run_extraction
from memcontext.predicate_packs import active_pack
from memcontext.schema import SourceType, Speaker, Turn


@pytest.fixture
def conn(monkeypatch):
    monkeypatch.setenv("ACTIVE_PACK", "general,developer")  # the live Claude Code pack
    active_pack.cache_clear()
    monkeypatch.setattr(http_server, "_hook_extractor", lambda _turn: [], raising=False)
    http_server.init_db(":memory:")
    c = http_server._conn  # type: ignore[attr-defined]
    register_principal(c, token="tokA", principal="A", namespace="tenantA", can_write=True)
    register_principal(c, token="tokB", principal="B", namespace="tenantB", can_write=True)
    c.commit()
    return c


@pytest.fixture
def client(conn):
    return TestClient(http_server.app)


def _store(conn, subject, value, *, predicate="decision_made", namespace="tenantA", session="s1"):
    return handle_memory_store(
        conn, text=f"Decision: {subject} is {value}.", session_id=session, namespace=namespace,
        claims=[{"subject": subject, "predicate": predicate, "value": value}],
    )


def _injected(client, prompt, tok="tokA") -> str:
    r = client.post("/api/hooks/user_prompt_submit", headers={"authorization": f"Bearer {tok}"},
                    json={"prompt": prompt, "session_id": "fresh"})
    assert r.status_code == 200
    return (r.json().get("hookSpecificOutput") or {}).get("additionalContext", "")


# ------------------------------------------------------------- same kind ---


@pytest.mark.parametrize(("a", "b", "same"), [
    ("ci-toolkit/ci", "ci-toolkit/ci", True),
    ("ci-toolkit/ci", "ci-toolkit/ci-provider", True),
    ("ci-toolkit/ci provider", "ci-toolkit/ci", True),
    ("billing-api/database", "billing-api/cache", False),
    ("billing-api/database", "orders-api/database", False),  # different project
    ("ci-toolkit/ci", "", False),
])
def test_same_decision_kind(a, b, same):
    assert same_decision_kind(a, b) is same


# ----------------------------------------------------------------- hook ---


def test_decision_recorded_under_a_new_subject_is_flagged_newest_first(conn, client):
    _store(conn, "ci-toolkit/ci", "GitHub Actions")
    new = _store(conn, "ci-toolkit/ci-provider", "GitLab CI")
    assert new["supersessions"] == 0  # different subject: supersession cannot link them
    ctx = _injected(client, "Which CI system do we use for ci-toolkit?")
    assert "CONFLICT" in ctx
    assert ctx.index("GitLab CI") < ctx.index("GitHub Actions")  # newest first
    assert "newest" in ctx.split("GitLab CI", 1)[1].split("\n", 1)[0]


def test_conflict_partner_is_shown_even_when_retrieval_only_surfaces_the_stale_value(conn, client):
    """The prompt matches only the old wording; the newer value must still appear."""
    _store(conn, "ci-toolkit/ci", "GitHub Actions workflows")
    _store(conn, "ci-toolkit/ci-provider", "GitLab CI")
    ctx = _injected(client, "Update the GitHub Actions workflows for ci-toolkit")
    assert "CONFLICT" in ctx and "GitLab CI" in ctx
    assert ctx.index("GitLab CI") < ctx.index("GitHub Actions")


def test_two_live_values_in_one_slot_are_flagged(conn, client):
    """The trust guard keeps both values live when a web page contradicts the user."""
    _store(conn, "ci-toolkit/ci", "GitHub Actions")
    t = Turn(turn_id=new_turn_id(), session_id="web", speaker=Speaker.USER, ts=now_ns(),
             text="Blog: ci-toolkit moved to CircleCI.", source_type=SourceType.BROWSER)
    insert_turn(conn, t, namespace="tenantA")
    run_extraction(conn, episode_id=t.turn_id, session_id="web", extractor=PassthroughExtractor(
        [{"subject": "ci-toolkit/ci", "predicate": "decision_made", "value": "CircleCI"}]))
    ctx = _injected(client, "Which CI system does ci-toolkit use?")
    assert "CONFLICT" in ctx
    circle = next(line for line in ctx.splitlines() if "CircleCI" in line)
    assert "untrusted" in circle  # newest, but from a low-trust source
    assert ctx.index("CircleCI") < ctx.index("GitHub Actions")


def test_a_superseded_value_is_not_a_conflict(conn, client):
    _store(conn, "ci-toolkit/ci", "GitHub Actions")
    assert _store(conn, "ci-toolkit/ci", "GitLab CI")["supersessions"] == 1
    ctx = _injected(client, "Which CI system do we use for ci-toolkit?")
    assert "GitLab CI" in ctx and "GitHub Actions" not in ctx and "CONFLICT" not in ctx


def test_multi_valued_predicates_never_conflict(conn, client):
    _store(conn, "ci-toolkit/release", "flaky integration tests", predicate="blocker")
    _store(conn, "ci-toolkit/release", "missing signing key", predicate="blocker")
    ctx = _injected(client, "What is blocking the ci-toolkit release?")
    assert "flaky integration tests" in ctx and "missing signing key" in ctx
    assert "CONFLICT" not in ctx


def test_unrelated_decisions_are_not_flagged(conn, client):
    _store(conn, "billing-api/database", "PostgreSQL")
    _store(conn, "billing-api/cache", "Redis")
    ctx = _injected(client, "What database and cache does billing-api use?")
    assert "PostgreSQL" in ctx and "Redis" in ctx and "CONFLICT" not in ctx


def test_conflict_lookup_stays_inside_the_namespace(conn, client):
    _store(conn, "ci-toolkit/ci", "GitHub Actions", namespace="tenantA")
    _store(conn, "ci-toolkit/ci-provider", "GitLab CI", namespace="tenantB")
    ctx = _injected(client, "Which CI system do we use for ci-toolkit?", tok="tokA")
    assert "GitHub Actions" in ctx
    assert "GitLab CI" not in ctx and "CONFLICT" not in ctx


def test_live_same_kind_returns_newest_first(conn):
    _store(conn, "ci-toolkit/ci", "GitHub Actions")
    _store(conn, "ci-toolkit/ci-provider", "GitLab CI")
    rows = live_same_kind(conn, subject="ci-toolkit/ci", predicate="decision_made",
                          namespace="tenantA")
    assert [r["value"] for r in rows] == ["GitLab CI", "GitHub Actions"]
