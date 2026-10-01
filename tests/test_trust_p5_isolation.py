"""Trust layer Phase 5 — namespace isolation / access control.

Memory is partitioned by tenant namespace (the scope above session). A caller
bound to one namespace cannot read another tenant's sessions, and the cross-session
sweep is bounded to the caller's namespace — never "all sessions in the DB".
"""
from __future__ import annotations

import sqlite3

from memcontext.mcp_tools import handle_memory_query, handle_memory_store
from memcontext.schema import open_database


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _store(conn, subject, value, session_id, namespace=None):
    kw = {} if namespace is None else {"namespace": namespace}
    handle_memory_store(
        conn, text=f"{subject} likes {value}", session_id=session_id,
        claims=[{"subject": subject, "predicate": "user_fact", "value": value, "confidence": 0.9}],
        **kw,
    )


def test_namespace_isolates_cross_session_sweep():
    conn = _conn()
    _store(conn, "alice", "coffee", "sA", namespace="tenantA")
    _store(conn, "bob", "tea", "sB", namespace="tenantB")

    a = {c["subject"] for c in handle_memory_query(conn, query="likes", namespace="tenantA")["claims"]}
    assert "alice" in a and "bob" not in a

    b = {c["subject"] for c in handle_memory_query(conn, query="likes", namespace="tenantB")["claims"]}
    assert "bob" in b and "alice" not in b


def test_namespace_denies_foreign_session():
    conn = _conn()
    _store(conn, "bob", "tea", "sB", namespace="tenantB")

    denied = handle_memory_query(conn, query="tea", session_id="sB", namespace="tenantA")
    assert denied.get("denied") == "namespace"
    assert denied["claims"] == []

    ok = handle_memory_query(conn, query="tea", session_id="sB", namespace="tenantB")
    assert any(c["subject"] == "bob" for c in ok["claims"])


def test_default_namespace_keeps_single_tenant_unchanged():
    conn = _conn()
    _store(conn, "alice", "coffee", "s1")  # no namespace -> 'default'
    assert conn.execute("SELECT DISTINCT namespace FROM turns").fetchone()[0] == "default"
    # a query with no namespace still sees everything (backward compatible)
    assert len(handle_memory_query(conn, query="likes")["claims"]) >= 1
