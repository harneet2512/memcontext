"""HTTP transport: every route except /health is authenticated, and read endpoints
stay inside the caller's namespace.

Regressions covered:
- the bearer middleware only guarded /api/*, so the MCP app mounted at /mcp by
  `serve-http` (bound to 0.0.0.0 by default) answered memory_query/memory_forget
  with no token at all.
- /api/memory/trace returned any tenant's claim by id; /api/memory/status counted
  every tenant.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from memcontext import http_server
from memcontext.authz import register_principal
from memcontext.cli import serve_http
from memcontext.mcp_tools import handle_memory_store


def _h(tok: str) -> dict:
    return {"authorization": f"Bearer {tok}"}


@pytest.fixture
def shared_token(monkeypatch):
    monkeypatch.setenv("MEMCONTEXT_HTTP_TOKEN", "tok-shared")
    monkeypatch.setattr(http_server, "_http_token", None)
    http_server.init_db(":memory:")
    return "tok-shared"


@pytest.fixture
def tenants(monkeypatch):
    monkeypatch.setattr(http_server, "_http_token", None)
    http_server.init_db(":memory:")
    c = http_server._conn  # type: ignore[attr-defined]
    register_principal(c, token="tokA", principal="A", namespace="tenantA", can_write=True)
    register_principal(c, token="tokB", principal="B", namespace="tenantB", can_write=True)
    c.commit()
    a = handle_memory_store(c, text="The orders service database is PostgreSQL.", session_id="sa",
                            namespace="tenantA",
                            claims=[{"subject": "orders service", "predicate": "user_fact",
                                     "value": "database is PostgreSQL"}])
    handle_memory_store(c, text="Tenant B billing runs on Stripe.", session_id="sb",
                        namespace="tenantB",
                        claims=[{"subject": "billing", "predicate": "user_fact",
                                 "value": "runs on Stripe"}])
    return a["claim_ids"][0]


@pytest.mark.parametrize("path", ["/mcp/", "/mcp", "/docs", "/openapi.json"])
def test_non_api_routes_require_the_token(shared_token, path):
    client = TestClient(http_server.app)
    assert client.get(path).status_code == 401
    assert client.get(path, headers=_h(shared_token)).status_code != 401


def test_health_stays_public(shared_token):
    assert TestClient(http_server.app).get("/health").status_code == 200


def test_mcp_rejects_namespace_scoped_principals(tenants):
    # The mounted MCP app is not namespace-bound, so a tenant token must not reach it.
    r = TestClient(http_server.app).post("/mcp/", json={}, headers=_h("tokB"))
    assert r.status_code == 403


def test_trace_hides_other_tenants_claims(tenants):
    client = TestClient(http_server.app)
    as_b = client.post("/api/memory/trace", json={"claim_id": tenants}, headers=_h("tokB"))
    assert "PostgreSQL" not in as_b.text
    as_a = client.post("/api/memory/trace", json={"claim_id": tenants}, headers=_h("tokA"))
    assert "PostgreSQL" in as_a.text


def test_status_counts_only_the_callers_namespace(tenants):
    client = TestClient(http_server.app)
    b = client.get("/api/memory/status", headers=_h("tokB")).json()
    assert b["total_claims"] == 1 and b["turns"] == 1 and b["sessions"] == 1


def test_serve_http_binds_loopback_by_default():
    host = next(p for p in serve_http.params if p.name == "host")
    assert host.default == "127.0.0.1"
    assert "--host" in CliRunner().invoke(serve_http, ["--help"]).output
