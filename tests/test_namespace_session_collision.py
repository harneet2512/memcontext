"""Tenant isolation must not depend on session ids being unique across namespaces.

Regression (pre-existing, found by HAR-95's hook-conflict namespace test): retrieval
scoped by session_id only. When two tenants used the same session id (the hooks
default to "hooks" when Claude Code sends none; MCP callers pick their own ids),
memory_query(namespace="tenantA") served tenant B's claims and episodes, on both the
cross-session and the single-session path, and the prompt hook injected them.
"""
from __future__ import annotations

import json

import pytest

from memcontext.mcp_tools import handle_memory_query, handle_memory_store
from memcontext.retrieval import retrieve_memory
from memcontext.schema import open_database
from memcontext.serving import iter_served_claims

SECRET = "Kubernetes on bare metal"


@pytest.fixture
def conn():
    c = open_database(":memory:")
    handle_memory_store(c, text="Decision: we deploy with Docker Compose.", session_id="shared",
                        namespace="tenantA",
                        claims=[{"subject": "deploy target", "predicate": "user_fact",
                                 "value": "Docker Compose"}])
    handle_memory_store(c, text=f"Decision: we deploy with {SECRET}.", session_id="shared",
                        namespace="tenantB",
                        claims=[{"subject": "deploy target", "predicate": "user_fact",
                                 "value": SECRET}])
    return c


@pytest.mark.parametrize("session_id", [None, "shared"])
def test_query_never_serves_another_tenants_memory(conn, session_id):
    out = handle_memory_query(conn, query="How do we deploy?", session_id=session_id,
                              namespace="tenantA")
    assert any("Docker Compose" in (c["fact"] or "") for c in iter_served_claims(out))
    assert SECRET not in json.dumps(out)  # claims, episodes, resolved view: nothing of B


def test_resolved_view_is_withheld_for_a_session_shared_across_tenants(conn):
    out = handle_memory_query(conn, query="How do we deploy?", session_id="shared",
                              namespace="tenantA")
    assert "world_state" not in out
    assert out["resolved_view_withheld"] == "session id shared across namespaces"


def test_retrieval_filters_by_namespace(conn):
    hits = retrieve_memory(conn, session_id="shared", query="deploy", top_k=10, namespace="tenantB")
    assert hits and all("Docker Compose" not in h.text for h, _ in hits)


def test_unscoped_single_tenant_query_is_unchanged(conn):
    """namespace=None is the single-tenant / shared-token mode: it sees everything."""
    out = handle_memory_query(conn, query="How do we deploy?", session_id="shared")
    facts = " ".join(c["fact"] or "" for c in iter_served_claims(out))
    assert "Docker Compose" in facts and SECRET in facts


def test_unshared_session_keeps_its_resolved_view():
    c = open_database(":memory:")
    handle_memory_store(c, text="Decision: we deploy with Docker Compose.", session_id="own",
                        namespace="tenantA",
                        claims=[{"subject": "deploy target", "predicate": "user_fact",
                                 "value": "Docker Compose"}])
    out = handle_memory_query(c, query="How do we deploy?", session_id="own", namespace="tenantA")
    assert "world_state" in out and "resolved_view_withheld" not in out


@pytest.mark.parametrize("session_id", [None, "shared"])
def test_total_counts_only_the_callers_namespace(conn, session_id):
    """`total` must not reveal the size of another tenant's store."""
    handle_memory_store(conn, text="Decision: logs go to Loki.", session_id="other",
                        namespace="tenantB",
                        claims=[{"subject": "log sink", "predicate": "user_fact", "value": "Loki"}])
    out = handle_memory_query(conn, query="How do we deploy?", session_id=session_id,
                              namespace="tenantA")
    assert out["total"] == 1  # tenantA's one active claim; tenantB holds two


def test_unscoped_total_is_unchanged(conn):
    assert handle_memory_query(conn, query="How do we deploy?")["total"] == 2
