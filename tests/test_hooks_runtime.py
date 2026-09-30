"""Claude Code hook endpoints: authorization, namespace isolation, and liveness.

Regressions covered:
- hooks ignored the principal's namespace and write permission: a read-only token
  could write through /api/hooks/*, and pre_tool_use injected other tenants' claims.
- hook handlers ran blocking work (extraction, embedding, SQLite) directly on the
  event loop, stalling every other request (Claude Code then cancels hooks at 5-10s).
- the extractor was re-selected on every hook call, probing Ollama each time.
"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from memcontext import http_server
from memcontext.authz import register_principal
from memcontext.mcp_tools import handle_memory_store


@pytest.fixture
def conn(monkeypatch):
    monkeypatch.setattr(http_server, "_hook_extractor", None, raising=False)
    http_server.init_db(":memory:")
    c = http_server._conn  # type: ignore[attr-defined]
    register_principal(c, token="tokA", principal="A", namespace="tenantA", can_write=True)
    register_principal(c, token="tokB", principal="B", namespace="tenantB", can_write=True)
    register_principal(c, token="tokRO", principal="RO", namespace="tenantA", can_write=False)
    c.commit()
    return c


def _h(tok: str) -> dict:
    return {"authorization": f"Bearer {tok}"}


PROMPT = "We decided the orders service will use PostgreSQL for persistence going forward."


def test_read_only_principal_cannot_write_through_hooks(conn):
    client = TestClient(http_server.app)
    before = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    r1 = client.post("/api/hooks/user_prompt_submit", json={"prompt": PROMPT, "session_id": "s1"},
                     headers=_h("tokRO"))
    r2 = client.post("/api/hooks/post_tool_use", headers=_h("tokRO"), json={
        "tool_name": "Write", "session_id": "s1",
        "tool_input": {"file_path": "/repo/orders/db_config.py", "content": "x"}})
    assert r1.status_code == 403 and r2.status_code == 403
    assert conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == before


def test_hook_writes_land_in_the_callers_namespace(conn):
    client = TestClient(http_server.app)
    r = client.post("/api/hooks/user_prompt_submit", json={"prompt": PROMPT, "session_id": "s1"},
                    headers=_h("tokA"))
    assert r.status_code == 200
    ns = {row[0] for row in conn.execute("SELECT namespace FROM turns")}
    assert ns == {"tenantA"}


def test_pre_tool_use_never_injects_another_namespaces_claims(conn):
    handle_memory_store(conn, text="Tenant A secret: the acquisition target is Contoso Ltd.",
                        session_id="sa", namespace="tenantA",
                        claims=[{"subject": "acquisition", "predicate": "user_fact",
                                 "value": "target is Contoso Ltd"}])
    client = TestClient(http_server.app)
    body = {"tool_name": "Bash", "tool_input": {"command": "python acquisition target report"}}

    as_b = client.post("/api/hooks/pre_tool_use", json=body, headers=_h("tokB")).json()
    assert "Contoso" not in str(as_b)

    as_a = client.post("/api/hooks/pre_tool_use", json=body, headers=_h("tokA")).json()
    assert "Contoso" in as_a["hookSpecificOutput"]["additionalContext"]


def test_extractor_is_selected_once_not_per_hook_call(conn, monkeypatch):
    from memcontext import extractors, mcp_tools

    calls = []

    def counting():
        calls.append(1)
        return extractors.SimpleExtractor()

    # Patch both import sites: callers may resolve it from either module.
    monkeypatch.setattr(extractors, "auto_extractor", counting)
    monkeypatch.setattr(mcp_tools, "auto_extractor", counting)
    client = TestClient(http_server.app)
    for i in range(3):
        client.post("/api/hooks/user_prompt_submit",
                    json={"prompt": f"{PROMPT} Iteration number {i}.", "session_id": "s1"},
                    headers=_h("tokA"))
    assert len(calls) <= 1


def test_slow_hook_does_not_block_other_requests(conn, monkeypatch):
    from memcontext import mcp_tools

    def slow_store(*_a, **_k):
        time.sleep(1.0)
        return {"ok": True}

    monkeypatch.setattr(mcp_tools, "handle_memory_store", slow_store)

    async def run():
        transport = httpx.ASGITransport(app=http_server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            async def hook():
                await client.post("/api/hooks/user_prompt_submit", headers=_h("tokA"),
                                  json={"prompt": PROMPT, "session_id": "s1"})
                return time.monotonic()

            async def health():
                await asyncio.sleep(0.2)  # let the slow hook start first
                await client.get("/health")
                return time.monotonic()

            hook_done, health_done = await asyncio.gather(hook(), health())
            return hook_done, health_done

    hook_done, health_done = asyncio.run(run())
    # If the hook blocked the event loop, /health could only finish after it.
    assert health_done < hook_done
