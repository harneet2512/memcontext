"""Event-frame serving must honour the embedding policy and degrade to the list.

Regression: `retrieve_event_frames` embedded the query with the process-default
client before checking whether any frame had a stored embedding. With episode
embeddings off (lexical-only mode) no frame is ever embedded, so the call could
never rank anything, yet it loaded the full local model on the query path
(~400s cold on Windows) or, with no backend installed, raised; `serve_event_frames`
then swallowed the error and served no frames instead of its documented fallback.
"""
from __future__ import annotations

import json

import pytest

from memcontext import retrieval
from memcontext.retrieval import EmbeddingClient, retrieve_event_frames
from memcontext.schema import open_database
from memcontext.serving import serve_event_frames

SESSION = "s-frames"


@pytest.fixture
def conn():
    c = open_database(":memory:")
    c.execute(
        "INSERT INTO event_frames (event_id, event_type, participants, item, location,"
        " time_expr, amount, supporting_claim_ids, source_turn_ids, session_id,"
        " confidence, missing_slots) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ev_1", "purchase", json.dumps(["user"]), "laptop", None, "last week", None,
         json.dumps([]), json.dumps([]), SESSION, 0.8, json.dumps([])),
    )
    return c


@pytest.fixture
def no_model(monkeypatch):
    """Fail loudly if anything tries to embed with the default client."""
    def _boom(self, texts):
        raise AssertionError("query path loaded the embedding model")
    monkeypatch.setattr(EmbeddingClient, "embed", _boom)
    monkeypatch.setattr(retrieval, "_default_client", None)


class _CountingClient:
    model_version = "stub-v1"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


def test_lexical_only_mode_never_embeds_the_query(conn, no_model):
    # conftest sets MEMCONTEXT_EMBED_EPISODES=0 (lexical-only mode)
    assert retrieve_event_frames(conn, session_id=SESSION, query="what did I buy") == []


def test_lexical_only_mode_serves_the_frame_list(conn, no_model):
    served = serve_event_frames(conn, session_id=SESSION, query="what did I buy")
    assert [f["item"] for f in served] == ["laptop"]


def test_no_query_embedding_when_no_frame_is_embedded(conn):
    client = _CountingClient()
    assert retrieve_event_frames(conn, session_id=SESSION, query="laptop", embedding_client=client) == []
    assert client.calls == 0


def test_embedded_frames_are_still_ranked(conn):
    client = _CountingClient()
    retrieval.backfill_event_frame_embeddings(conn, SESSION, client=client)
    ranked = retrieve_event_frames(conn, session_id=SESSION, query="laptop", embedding_client=client)
    assert [f.event_id for f, _ in ranked] == ["ev_1"]
