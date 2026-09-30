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


def test_relevance_ignores_predicate_names(conn, client):
    # "user_fact" is a label, not content: a prompt mentioning "user" or "fact"
    # must not pull in every user_* claim.
    _store(conn, "orders service", "user_fact", "primary database is PostgreSQL")
    r = _prompt(client, "What should the user see as a fact sheet on login?")
    assert "hookSpecificOutput" not in r.json()
    r = _pre_tool(client, "Bash", {"command": "grep-user fact-check login"})
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


# ── B: PreToolUse keywords come from the file, not its directory path ───────

def _pre_tool(client, tool_name: str, tool_input: dict, tok: str = "tokA"):
    return client.post("/api/hooks/pre_tool_use", headers=_h(tok),
                       json={"tool_name": tool_name, "tool_input": tool_input})


def test_pre_tool_use_injects_decision_not_path_noise(conn, client):
    for n in range(50):
        _store(conn, f"file{n}.py", "action", f"Write on D:/proj/src/pkg{n}/file{n}.py",
               session=f"s{n}")
    _store(conn, "orders service", "decision_made", "use SQLite", session="decisions")
    r = _pre_tool(client, "Write", {"file_path": "D:/proj/src/orders/orders_service.py",
                                    "content": "print('hi')"})
    ctx = _injected(r)
    assert "use SQLite" in ctx
    assert "Write on" not in ctx
    assert "None" not in ctx  # NL-only facts render their text, not a null triple


def test_query_keywords_drop_directory_prefixes():
    kw = http_server._extract_query_keywords(
        "Edit", {"file_path": "C:\\Users\\me\\AppData\\Local\\Temp\\claude\\billing_invoice.py"})
    assert kw is not None
    assert set(kw.split()) == {"billing", "invoice"}


def test_query_keywords_combine_basename_and_command_text():
    kw = http_server._extract_query_keywords("Bash", {"command": "pytest tests/orders -k refund"})
    assert kw is not None
    assert {"pytest", "orders", "refund"} <= set(kw.split())


# ── C: PostToolUse keeps tool actions as episodes, never as claims ──────────

def _post_tool(client, tool_name: str, tool_input: dict, tok: str = "tokA"):
    return client.post("/api/hooks/post_tool_use", headers=_h(tok),
                       json={"tool_name": tool_name, "tool_input": tool_input, "session_id": "s1"})


def test_post_tool_use_stores_episodes_not_action_claims(conn, client):
    write = {"file_path": "D:/proj/src/orders/orders_service.py", "content": "x = 1"}
    for _ in range(3):
        assert _post_tool(client, "Write", write).status_code == 200
    assert _post_tool(client, "Bash", {"command": "npm run build --prefix web"}).status_code == 200
    assert _post_tool(client, "Bash", {"command": "python manage.py migrate orders"}).status_code == 200
    actions = conn.execute("SELECT COUNT(*) FROM claims WHERE predicate = 'action'").fetchone()[0]
    assert actions == 0
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
    turns = conn.execute("SELECT namespace, text FROM turns").fetchall()
    assert len(turns) == 5
    assert {t[0] for t in turns} == {"tenantA"}
    assert any("python manage.py migrate orders" in t[1] for t in turns)


# ── D: no regex-extracted junk claims from prompts ──────────────────────────

def test_prompt_capture_with_regex_fallback_stores_episode_only(conn, client, monkeypatch):
    from structlog.testing import capture_logs

    from memcontext import extractors, mcp_tools
    monkeypatch.setattr(http_server, "_hook_extractor", None)
    monkeypatch.setattr(mcp_tools, "auto_extractor", extractors.SimpleExtractor)
    with capture_logs() as logs:
        _prompt(client, "We decided we will use MySQL 5.7 for the orders database.")
        assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 1
        _prompt(client, "The team prefers tabs over spaces in the billing service.")
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
    warned = [e for e in logs if e["log_level"] == "warning"
              and "MEMCONTEXT_EXTRACTOR_BACKEND" in str(e)]
    assert len(warned) == 1


def test_prompt_capture_uses_a_real_extractor_when_one_is_selected(conn, client, monkeypatch):
    from memcontext import mcp_tools
    from memcontext.on_new_turn import ExtractedClaim

    def llm_like(_turn):
        return [ExtractedClaim(subject="orders service", predicate="user_fact",
                               value="database is MySQL 5.7", confidence=0.9)]

    monkeypatch.setattr(http_server, "_hook_extractor", None)
    monkeypatch.setattr(mcp_tools, "auto_extractor", lambda: llm_like)
    _prompt(client, "We decided we will use MySQL 5.7 for the orders database.")
    values = [r[0] for r in conn.execute("SELECT value FROM claims")]
    assert values == ["database is MySQL 5.7"]
