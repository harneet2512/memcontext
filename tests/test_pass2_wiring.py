"""Pass-2 semantic supersession is now WIRED into the production ingest path
(was constructed nowhere -> dormant). Two guards:

1. Safe gating: the helper returns None when no real embedder is configured
   (NullEmbedder cosine is always 1.0 and would supersede everything), and a
   constructed instance when an embedder is present.
2. Wiring: handle_memory_store actually invokes the helper (red before this fix).
"""
from __future__ import annotations

import sqlite3

import memcontext.retrieval as R
from memcontext.schema import open_database
from memcontext.supersession_semantic import NullEmbedder, SemanticSupersession


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def test_semantic_supersession_gated_on_real_embedder(monkeypatch):
    # embeddings off (test/CI default) -> None, so Pass-2 stays inert (safe).
    monkeypatch.setattr(R, "episode_embedder", lambda: None)
    assert R.semantic_supersession() is None
    # a real embedder present -> a constructed Pass-2 instance.
    monkeypatch.setattr(R, "episode_embedder", lambda: NullEmbedder())
    assert isinstance(R.semantic_supersession(), SemanticSupersession)


def test_handle_store_invokes_pass2(monkeypatch):
    from memcontext.mcp_tools import handle_memory_store

    calls = {"n": 0}

    def _spy():
        calls["n"] += 1
        return None  # inert: don't actually supersede in this wiring test

    monkeypatch.setattr(R, "semantic_supersession", _spy)

    conn = _conn()
    handle_memory_store(
        conn, text="I use Postgres for the project daily",
        claims=[{"subject": "user", "predicate": "user_fact",
                 "value": "postgres", "confidence": 0.9}],
    )
    assert calls["n"] >= 1, "handle_memory_store wires Pass-2 (invokes semantic_supersession)"
