"""Regression: degraded mode without embedding backends.

Root cause fixed (LIPI — Plumbing/Integration): ``episode_embedder`` keyed off
``MEMCONTEXT_EMBED_EPISODES`` only, so with the env flag on (the default) it
returned an EmbeddingClient whose local driver was unimportable. On the second
same-predicate write, Pass-2 semantic supersession invoked ``embed()`` and the
RuntimeError propagated out of ``handle_memory_store`` mid-write — the README's
"graceful degradation" claim was false. Now the embedder is capability-gated
(``backend_available``) and Pass-2 failures are caught and logged.
"""
from __future__ import annotations

import sqlite3

import pytest

import memcontext.retrieval as R
from memcontext.mcp_tools import handle_memory_store
from memcontext.schema import open_database


@pytest.fixture
def db(tmp_path):
    conn = open_database(str(tmp_path / "t.db"))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def _store_preference(conn, value: str) -> dict:
    return handle_memory_store(
        conn,
        text=f"By the way, I really like {value} on weekends",
        session_id="s",
        claims=[{"subject": "user", "predicate": "user_preference",
                 "value": value, "confidence": 0.9}],
    )


def test_episode_embedder_returns_none_without_backend(monkeypatch):
    """Env flag on + no importable backend -> no client (capability, not config)."""
    monkeypatch.delenv(R.EPISODE_EMBED_ENV, raising=False)
    monkeypatch.delenv(R.MODAL_URL_ENV, raising=False)
    monkeypatch.setattr(R, "_default_client", None)
    # Simulate a host where neither driver is installed.
    monkeypatch.setattr(
        R.importlib.util, "find_spec", lambda name: None if name != "builtins" else object()
    )
    assert R.episode_embedder() is None
    assert R.semantic_enabled() is False


def test_repeated_same_predicate_writes_do_not_crash(db, monkeypatch):
    """Two writes that reach Pass-2 must not raise when embed() fails."""
    monkeypatch.delenv(R.EPISODE_EMBED_ENV, raising=False)
    monkeypatch.setattr(R, "_default_client", None)

    class BrokenClient:
        def embed(self, texts):
            raise RuntimeError("FlagEmbedding is not installed")

        @property
        def model_version(self):
            return "broken"

        def backend_available(self):
            return False

    # Force a client into the Pass-2 path to prove the write survives even if a
    # client object exists but its backend is dead at call time.
    from memcontext.supersession_semantic import SemanticSupersession

    monkeypatch.setattr(R, "episode_embedder", lambda: BrokenClient())
    monkeypatch.setattr(
        R, "semantic_supersession", lambda: SemanticSupersession(BrokenClient())
    )

    r1 = _store_preference(db, "pizza")
    assert r1["claims_created"] == 1
    r2 = _store_preference(db, "sushi")
    assert r2["claims_created"] == 1  # second write completes despite dead embedder


def test_semantic_supersession_none_when_no_backend(monkeypatch):
    monkeypatch.delenv(R.EPISODE_EMBED_ENV, raising=False)
    monkeypatch.setattr(R, "_default_client", None)
    monkeypatch.setattr(
        R.importlib.util, "find_spec", lambda name: None if name != "builtins" else object()
    )
    assert R.semantic_supersession() is None


def test_status_reports_lexical_mode(tmp_path, monkeypatch):
    """status surfaces degraded mode instead of claiming semantic memory."""
    from click.testing import CliRunner

    from memcontext.cli import main

    monkeypatch.delenv(R.EPISODE_EMBED_ENV, raising=False)
    monkeypatch.setattr(R, "_default_client", None)
    monkeypatch.setattr(
        R.importlib.util, "find_spec", lambda name: None if name != "builtins" else object()
    )
    db_path = str(tmp_path / "s.db")
    r = CliRunner()
    assert r.invoke(main, ["init", "--db", db_path]).exit_code == 0
    result = r.invoke(main, ["status", "--db", db_path])
    assert result.exit_code == 0
    assert "OFF" in result.output
