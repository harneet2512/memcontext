"""The stdio MCP server must load the embedding model before its event loop starts.

Regression: the model was loaded lazily inside the first memory_store call. On
Windows the stdio transport's reader thread blocks in ReadFile(stdin), and the
lazy DLL load waited on that handle until the client sent another message — the
first memory_store hung for minutes in Claude Code.
"""
from __future__ import annotations

import asyncio

import pytest

from memcontext import mcp_server, retrieval


class _StopServer(Exception):
    pass


def test_stdio_server_prewarms_before_event_loop(tmp_path, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(mcp_server, "prewarm_embedder", lambda: order.append("prewarm"))

    def fake_run(coro):
        order.append("loop")
        coro.close()
        raise _StopServer

    monkeypatch.setattr(asyncio, "run", fake_run)
    with pytest.raises(_StopServer):
        mcp_server.run_server(db_path=str(tmp_path / "m.db"), transport="stdio")
    assert order == ["prewarm", "loop"]


class _Embedder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(texts)
        if self.fail:
            raise ValueError("boom")
        return [[0.0] for _ in texts]


def test_prewarm_embeds_once_when_semantic(monkeypatch):
    emb = _Embedder()
    monkeypatch.setattr(retrieval, "episode_embedder", lambda: emb)
    mcp_server.prewarm_embedder()
    assert emb.calls == [["warmup"]]


def test_prewarm_failure_is_reported_not_fatal(monkeypatch, capsys):
    monkeypatch.setattr(retrieval, "episode_embedder", lambda: _Embedder(fail=True))
    mcp_server.prewarm_embedder()
    assert "warmup failed (ValueError)" in capsys.readouterr().err


def test_prewarm_is_noop_in_lexical_mode(monkeypatch, capsys):
    monkeypatch.setattr(retrieval, "episode_embedder", lambda: None)
    mcp_server.prewarm_embedder()
    assert capsys.readouterr().err == ""
