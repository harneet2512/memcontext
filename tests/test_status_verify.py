"""`memcontext status` must not claim semantic memory works without testing it.

Regression: status printed "Semantic memory: ON" from an import check alone
(EmbeddingClient.backend_available), while every real embed call raised.
`status --verify` now performs one real embed and reports ON (verified) or
BROKEN (exit non-zero); plain `status` says ON is unverified.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

import memcontext.retrieval as R
from memcontext.cli import main
from memcontext.schema import open_database


class _OkEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _BrokenEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("model weights missing")


@pytest.fixture()
def db(tmp_path) -> str:
    path = str(tmp_path / "m.db")
    open_database(path).close()
    return path


def _status(db: str, *extra: str):
    return CliRunner().invoke(main, ["status", "--db", db, *extra])


def test_status_verify_reports_verified_dim_when_embed_works(monkeypatch, db):
    emb = _OkEmbedder()
    monkeypatch.setattr(R, "episode_embedder", lambda: emb)

    r = _status(db, "--verify")

    assert r.exit_code == 0, r.output
    assert "Semantic memory: ON (verified, 4-d," in r.output
    assert emb.calls == [["probe"]]


def test_status_verify_reports_broken_and_exits_nonzero_when_embed_raises(monkeypatch, db):
    monkeypatch.setattr(R, "episode_embedder", lambda: _BrokenEmbedder())

    r = _status(db, "--verify")

    assert r.exit_code != 0
    assert "Semantic memory: BROKEN (RuntimeError)" in r.output
    assert "lexical-only" in r.output
    assert "ON" not in r.output.split("Semantic memory:")[1].splitlines()[0]


def test_status_without_verify_says_on_is_unverified_and_never_embeds(monkeypatch, db):
    monkeypatch.setattr(R, "episode_embedder", lambda: _BrokenEmbedder())

    r = _status(db)

    assert r.exit_code == 0, r.output
    assert "Semantic memory: ON (backend installed; not verified" in r.output
    assert "memcontext status --verify" in r.output


def test_status_off_wording_unchanged(monkeypatch, db):
    monkeypatch.setattr(R, "episode_embedder", lambda: None)

    for extra in ((), ("--verify",)):
        r = _status(db, *extra)
        assert r.exit_code == 0, r.output
        assert "Semantic memory: OFF -- lexical-only (BM25)" in r.output
