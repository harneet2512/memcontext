"""Embeddings are a prerequisite, not a silent degrade: the serving guard warns
loudly when semantic is off and refuses to serve under MEMCONTEXT_REQUIRE_EMBEDDINGS=1.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

import memcontext.retrieval as R
from memcontext.cli import main
from memcontext.schema import open_database


def test_semantic_policy(monkeypatch):
    # ON -> enabled, no raise
    monkeypatch.setattr(R, "episode_embedder", lambda: object())
    assert R.semantic_enabled() is True
    assert R.enforce_semantic_policy() is True

    # OFF, non-strict -> degraded (False), loud warning but no raise
    monkeypatch.setattr(R, "episode_embedder", lambda: None)
    monkeypatch.delenv("MEMCONTEXT_REQUIRE_EMBEDDINGS", raising=False)
    assert R.semantic_enabled() is False
    assert R.enforce_semantic_policy() is False

    # OFF, strict -> refuses (embeddings required)
    monkeypatch.setenv("MEMCONTEXT_REQUIRE_EMBEDDINGS", "1")
    with pytest.raises(RuntimeError):
        R.enforce_semantic_policy()


def test_status_surfaces_semantic_mode(tmp_path):
    db = str(tmp_path / "m.db")
    open_database(db).close()
    r = CliRunner().invoke(main, ["status", "--db", db])
    assert r.exit_code == 0, r.output
    assert "Semantic memory:" in r.output  # never hidden (OFF in CI)
