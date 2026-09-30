"""Claude Code hooks: what memory they inject and what they write.

Regressions covered:
- A: UserPromptSubmit stored the prompt but never injected memory, so a fresh
  session never saw prior decisions.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from memcontext import http_server
from memcontext.authz import register_principal
from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.mcp_tools import handle_memory_store
from memcontext.schema import Speaker, Turn


@pytest.fixture
def conn(monkeypatch):
    # Episode-only prompt capture: these tests are about injection, not extraction.
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


def _h(tok: str) -> dict:
    return {"authorization": f"Bearer {tok}"}


def _store(conn, subject: str, predicate: str, value: str, *, session: str = "old",
           namespace: str = "tenantA") -> dict:
    return handle_memory_store(conn, text=f"{subject} {predicate} {value}", session_id=session,
                               namespace=namespace,
                               claims=[{"subject": subject, "predicate": predicate, "value": value}])


def _prompt(client, prompt: str, tok: str = "tokA", session: str = "fresh"):
    return client.post("/api/hooks/user_prompt_submit", headers=_h(tok),
                       json={"prompt": prompt, "session_id": session})


def _injected(resp) -> str:
    out = resp.json().get("hookSpecificOutput") or {}
    return out.get("additionalContext", "")


# ── A: UserPromptSubmit injects current memory ──────────────────────────────

def test_prompt_submit_injects_current_decision_from_a_prior_session(conn, client):
    _store(conn, "orders service", "user_fact", "primary database is MySQL 5.7")
    newer = _store(conn, "orders service", "user_fact", "primary database is PostgreSQL")
    assert newer["supersessions"] == 1
    r = _prompt(client, "Which database does the orders service use?")
    assert r.status_code == 200
    out = r.json()["hookSpecificOutput"]
    assert out["hookEventName"] == "UserPromptSubmit"
    ctx = out["additionalContext"]
    assert ctx.startswith("[MemContext] Current project memory relevant to this prompt:\n- ")
    assert "PostgreSQL" in ctx
    assert "MySQL" not in ctx  # superseded
    assert "orders_service / user_fact:" in ctx


def test_prompt_submit_serves_nl_only_facts_and_skips_actions_and_episodes(conn, client):
    _store(conn, "orders service", "decision_made", "use SQLite")  # out-of-vocab -> NL-only
    _store(conn, "orders_service.py", "action", "Write on D:/proj/orders_service.py")
    handle_memory_store(conn, text="Earlier chat: orders service might use Redis.",
                        session_id="chat", namespace="tenantA", extractor=lambda _t: [])
    ctx = _injected(_prompt(client, "Let's refactor the orders service storage layer"))
    assert "use SQLite" in ctx
    assert "Write on" not in ctx
    assert "Redis" not in ctx


def test_prompt_submit_injects_nothing_when_nothing_is_relevant(conn, client):
    _store(conn, "orders service", "user_fact", "uses PostgreSQL")
    r = _prompt(client, "Tell me a joke about penguins wearing hats")
    assert r.status_code == 200
    assert "hookSpecificOutput" not in r.json()


def test_prompt_submit_injection_is_namespace_scoped(conn, client):
    _store(conn, "orders service", "user_fact", "uses PostgreSQL", namespace="tenantA")
    ctx_b = _injected(_prompt(client, "Which database does the orders service use?", tok="tokB"))
    assert "PostgreSQL" not in ctx_b


def test_prompt_submit_still_stores_the_prompt_in_the_callers_namespace(conn, client):
    _prompt(client, "We decided the orders service will use PostgreSQL going forward.")
    rows = conn.execute("SELECT namespace, session_id FROM turns").fetchall()
    assert [(r[0], r[1]) for r in rows] == [("tenantA", "fresh")]


def test_prompt_submit_injection_is_capped(conn, client):
    for i in range(20):
        _store(conn, f"orders service part{i}", "user_fact",
               f"orders service component {i} " + "detail " * 40, session=f"s{i}")
    ctx = _injected(_prompt(client, "What do we know about the orders service?"))
    lines = ctx.split("\n")[1:]
    assert 0 < len(lines) <= 6
    assert len(ctx) <= 1500


def test_prompt_submit_latency_with_500_claims_is_under_a_second(conn, client):
    ts = now_ns()
    for i in range(500):
        tid = new_turn_id()
        insert_turn(conn, Turn(turn_id=tid, session_id=f"s{i % 25}", speaker=Speaker.USER,
                               text=f"module{i} uses library{i}", ts=ts + i),
                    namespace="tenantA")
        insert_claim(conn, session_id=f"s{i % 25}", subject=f"module{i}",
                     predicate="user_fact", value=f"uses library{i}", confidence=0.9,
                     source_turn_id=tid)
    _prompt(client, "warm up the query path for module1")
    start = time.perf_counter()
    r = _prompt(client, "Which library does module42 use?")
    elapsed = time.perf_counter() - start
    assert "library42" in _injected(r)
    assert elapsed < 1.0, f"user_prompt_submit took {elapsed:.3f}s"
