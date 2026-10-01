"""Lexical mode must never load the embedding model at query time.

Regression: query-time retrieval fell back to `_default_embedding_client()` whenever
the DB held stored vectors, ignoring MEMCONTEXT_EMBED_EPISODES=0. In the normal
deployment, serve-http (semantic) writes turn/claim vectors and the stdio MCP server
(lexical) reads the same DB, so the MCP server lazily loaded the model inside a tool
call — which deadlocks the Windows stdio transport (memory_query hung; Claude Code
had to kill the server).
"""
from __future__ import annotations

import pytest

from memcontext import retrieval
from memcontext.mcp_tools import handle_memory_query, handle_memory_store
from memcontext.schema import open_database
from memcontext.serving import iter_served_claims


class _ExplodingClient:
    """Stands in for the default client: any embed() call means a model load."""

    model_version = "bge-m3-test"

    def embed(self, texts):
        raise AssertionError("lexical mode loaded the embedding model at query time")


class _StoringClient:
    model_version = "bge-m3-test"

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def db_with_foreign_vectors(monkeypatch):
    """A DB whose turns/claims were embedded by another (semantic) process."""
    conn = open_database(":memory:")
    monkeypatch.setenv("MEMCONTEXT_EMBED_EPISODES", "1")
    monkeypatch.setattr(retrieval, "episode_embedder", lambda: _StoringClient())
    handle_memory_store(conn, text="The orders service uses PostgreSQL for persistence.",
                        session_id="s1",
                        claims=[{"subject": "orders service", "predicate": "user_fact",
                                 "value": "uses PostgreSQL for persistence"}])
    retrieval.backfill_embeddings(conn, "s1", client=_StoringClient())
    stored = conn.execute("SELECT COUNT(*) FROM turn_embeddings").fetchone()[0]
    stored += conn.execute("SELECT COUNT(*) FROM claim_embeddings").fetchone()[0]
    assert stored > 0, "fixture must hold vectors written by a semantic process"
    monkeypatch.undo()
    return conn


def test_lexical_query_ignores_stored_vectors(db_with_foreign_vectors, monkeypatch):
    monkeypatch.setenv("MEMCONTEXT_EMBED_EPISODES", "0")
    monkeypatch.setattr(retrieval, "_default_embedding_client", lambda: _ExplodingClient())

    result = handle_memory_query(db_with_foreign_vectors,
                                 query="which database does the orders service use",
                                 session_id="s1")
    assert any("PostgreSQL" in (c.get("fact") or "") for c in iter_served_claims(result))
