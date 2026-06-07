"""Trust layer Phase 7 — per-principal access control (closes GOVERNANCE_AUDIT D).

A bearer token resolves to a principal scoped to a namespace + read/write
permission (tokens stored hashed). The HTTP transport binds the caller to its
namespace and gates mutations on the write permission.
"""
from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from memcontext import http_server
from memcontext.authz import hash_token, register_principal, resolve_principal
from memcontext.schema import open_database


# ── substrate authz ──────────────────────────────────────
def test_register_and_resolve_principal():
    conn = open_database(":memory:")
    register_principal(conn, token="secret-A", principal="svcA", namespace="tenantA", can_write=True)
    p = resolve_principal(conn, "secret-A")
    assert p is not None and p.name == "svcA" and p.namespace == "tenantA" and p.can_write is True
    assert resolve_principal(conn, "wrong-token") is None


def test_token_stored_hashed_never_plaintext():
    conn = open_database(":memory:")
    register_principal(conn, token="topsecret", principal="x", namespace="n")
    stored = conn.execute("SELECT token_hash FROM principals").fetchone()[0]
    assert stored == hash_token("topsecret")
    assert "topsecret" not in stored


# ── HTTP transport enforcement ───────────────────────────
def _http_conn() -> sqlite3.Connection:
    http_server.init_db(":memory:")
    return http_server._conn  # type: ignore[attr-defined]


def _store_body(subject, value, session_id):
    return {"text": f"{subject} likes {value}", "session_id": session_id,
            "claims": [{"subject": subject, "predicate": "user_fact", "value": value}]}


def test_http_unknown_token_is_rejected():
    conn = _http_conn()
    register_principal(conn, token="good", principal="p", namespace="tenantA")
    conn.commit()
    client = TestClient(http_server.app)
    r = client.post("/api/memory/query", json={"query": "x"},
                    headers={"authorization": "Bearer nope"})
    assert r.status_code == 401


def test_http_principal_scopes_to_its_namespace():
    conn = _http_conn()
    register_principal(conn, token="tokA", principal="A", namespace="tenantA", can_write=True)
    register_principal(conn, token="tokB", principal="B", namespace="tenantB", can_write=True)
    conn.commit()
    client = TestClient(http_server.app)

    client.post("/api/memory/store", json=_store_body("alice", "coffee", "sA"),
                headers={"authorization": "Bearer tokA"})
    client.post("/api/memory/store", json=_store_body("bob", "tea", "sB"),
                headers={"authorization": "Bearer tokB"})

    rA = client.post("/api/memory/query", json={"query": "likes"},
                     headers={"authorization": "Bearer tokA"})
    subjects = {c["subject"] for c in rA.json()["claims"]}
    assert "alice" in subjects and "bob" not in subjects  # A sees only tenantA


def test_http_read_only_principal_cannot_write():
    conn = _http_conn()
    register_principal(conn, token="ro", principal="reader", namespace="tenantA", can_write=False)
    conn.commit()
    client = TestClient(http_server.app)
    r = client.post("/api/memory/store", json=_store_body("x", "v", "s"),
                    headers={"authorization": "Bearer ro"})
    assert r.status_code == 403
