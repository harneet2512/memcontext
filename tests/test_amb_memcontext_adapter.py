from __future__ import annotations

import json
import os
import sqlite3

import pytest

from evals.amb_memcontext.provider import (
    Document,
    MemContextFullProvider,
)
from evals.amb_memcontext.router_llm import (
    OPENROUTER_READER_MODEL,
    TOKENROUTER_EXTRACTOR_MODEL,
    TOKENROUTER_JUDGE_MODEL,
    _coerce_to_schema,
)
from evals.amb_memcontext.run import _configure_tokenrouter_models, _register_provider
from memcontext.extractors import SimpleExtractor


class _Schema:
    properties = {
        "answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "citations": {"type": "array"},
    }
    required = ["answer", "reasoning", "citations"]


def _provider(tmp_path):
    provider = MemContextFullProvider(
        extractor=SimpleExtractor(),
        embedder=None,
        semantic=None,
        require_full=False,
        retrieval_depth=12,
    )
    provider.prepare(tmp_path / "amb_store", unit_ids={"u1", "u2"}, reset=True)
    return provider


def test_prepare_creates_clean_db_and_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMCONTEXT_EMBED_EPISODES", "0")
    store = tmp_path / "amb_store"
    store.mkdir()
    stale = store / "memcontext.db"
    stale.write_text("stale", encoding="utf-8")

    provider = _provider(tmp_path)

    db_path = store / "memcontext.db"
    config_path = store / "memcontext_run_config.json"
    assert db_path.exists()
    assert config_path.exists()
    assert sqlite3.connect(db_path).execute("SELECT name FROM sqlite_master").fetchall()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["provider"] == "memcontext-full"
    assert config["unit_ids"] == ["u1", "u2"]
    provider.cleanup()


def test_ingest_stores_turns_claims_and_retrieves_relevant_documents(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMCONTEXT_EMBED_EPISODES", "0")
    provider = _provider(tmp_path)
    provider.ingest(
        [
            Document(
                id="session-a",
                user_id="u1",
                timestamp="2024-01-02T10:00:00Z",
                context="LongMemEval synthetic memory",
                messages=[
                    {"role": "user", "content": "I prefer green tea with breakfast."},
                    {"role": "assistant", "content": "I will remember that."},
                ],
                content="",
            )
        ]
    )

    rows = provider.conn.execute("SELECT namespace, session_id, text FROM turns").fetchall()
    claims = provider.conn.execute("SELECT predicate, value FROM claims").fetchall()
    assert [(r["namespace"], r["session_id"]) for r in rows] == [
        ("u1", "session-a"),
        ("u1", "session-a"),
    ]
    assert any("green tea" in row["value"] for row in claims)

    docs, raw = provider.retrieve(
        "What drink does the user prefer at breakfast?", k=4, user_id="u1"
    )
    assert raw is not None
    assert raw["provider"] == "memcontext-full"
    assert raw["session_ids"] == ["session-a"]
    assert any("green tea" in doc.content.lower() for doc in docs)
    assert raw["hits"]
    provider.cleanup()


def test_user_id_namespace_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMCONTEXT_EMBED_EPISODES", "0")
    provider = _provider(tmp_path)
    provider.ingest(
        [
            Document(
                id="doc-u1",
                user_id="u1",
                content="I love sushi for lunch.",
                timestamp="2024-01-02",
            ),
            Document(
                id="doc-u2",
                user_id="u2",
                content="I love pasta for dinner.",
                timestamp="2024-01-03",
            ),
        ]
    )

    docs_u2, raw_u2 = provider.retrieve("What food does the user love?", k=5, user_id="u2")
    content_u2 = "\n".join(doc.content.lower() for doc in docs_u2)
    assert raw_u2["session_ids"] == ["doc-u2"]
    assert "pasta" in content_u2
    assert "sushi" not in content_u2
    provider.cleanup()


def test_query_timestamp_filters_future_hits(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMCONTEXT_EMBED_EPISODES", "0")
    provider = _provider(tmp_path)
    provider.ingest(
        [
            Document(
                id="old",
                user_id="u1",
                content="I live in Boston.",
                timestamp="2024-01-01T00:00:00Z",
            ),
            Document(
                id="future",
                user_id="u1",
                content="I live in Seattle.",
                timestamp="2024-02-01T00:00:00Z",
            ),
        ]
    )

    docs, raw = provider.retrieve(
        "Where does the user live?",
        k=6,
        user_id="u1",
        query_timestamp="2024-01-15T00:00:00Z",
    )
    body = "\n".join(doc.content.lower() for doc in docs)
    assert raw["valid_at_ts"] is not None
    assert "boston" in body
    assert "seattle" not in body
    provider.cleanup()


def test_runner_runtime_registers_memcontext(monkeypatch):
    pytest.importorskip("memory_bench.memory")
    monkeypatch.setenv("MEMCONTEXT_EMBED_EPISODES", "0")
    _register_provider()
    from memory_bench.memory import REGISTRY

    assert REGISTRY["memcontext-full"] is MemContextFullProvider


def test_runner_defaults_to_openrouter_reader_and_tokenrouter_extractor_judge(monkeypatch):
    monkeypatch.delenv("MEMCONTEXT_EXTRACTOR_BACKEND", raising=False)
    monkeypatch.delenv("MEMCONTEXT_EXTRACTOR_ENDPOINT", raising=False)
    monkeypatch.delenv("MEMCONTEXT_EXTRACTOR_MODEL", raising=False)
    monkeypatch.delenv("MEMCONTEXT_EXTRACTOR_API_KEY", raising=False)
    monkeypatch.setenv("TOKENROUTER_AMB_EXTRACTOR_KEY", "test-extractor-key")
    monkeypatch.delenv("OMB_ANSWER_LLM", raising=False)
    monkeypatch.delenv("OMB_ANSWER_MODEL", raising=False)
    monkeypatch.delenv("OMB_JUDGE_LLM", raising=False)
    monkeypatch.delenv("OMB_JUDGE_MODEL", raising=False)

    _configure_tokenrouter_models()

    assert os.environ["MEMCONTEXT_EXTRACTOR_BACKEND"] == "openrouter"
    assert os.environ["MEMCONTEXT_EXTRACTOR_ENDPOINT"].endswith("/chat/completions")
    assert "tokenrouter.com" in os.environ["MEMCONTEXT_EXTRACTOR_ENDPOINT"]
    assert os.environ["MEMCONTEXT_EXTRACTOR_MODEL"] == TOKENROUTER_EXTRACTOR_MODEL
    assert os.environ["MEMCONTEXT_EXTRACTOR_API_KEY"] == "test-extractor-key"
    assert os.environ["MEMCONTEXT_EXTRACTOR_REASONING_EFFORT"] == "none"
    assert os.environ["OMB_ANSWER_LLM"] == "openrouter-reader"
    assert os.environ["OMB_ANSWER_MODEL"] == OPENROUTER_READER_MODEL
    assert os.environ["OMB_ANSWER_REASONING_EFFORT"] == "high"
    assert os.environ["OMB_JUDGE_LLM"] == "tokenrouter-judge"
    assert os.environ["OMB_JUDGE_MODEL"] == TOKENROUTER_JUDGE_MODEL
    assert os.environ["OMB_JUDGE_REASONING_EFFORT"] == "low"
    monkeypatch.delenv("MEMCONTEXT_EXTRACTOR_API_KEY", raising=False)


def test_router_llm_coerces_missing_required_schema_fields():
    data = _coerce_to_schema({"response": "The user prefers green tea."}, _Schema())

    assert data["answer"] == "The user prefers green tea."
    assert data["reasoning"] == ""
    assert data["citations"] == []
