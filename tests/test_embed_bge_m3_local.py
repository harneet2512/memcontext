"""Regression: the local BGE-M3 path must accept the numpy array FlagEmbedding returns.

`BGEM3FlagModel.encode(..., return_dense=True)` returns ``{"dense_vecs": np.ndarray}``.
A truthiness check on that array (``vectors or []``) raises
``ValueError: The truth value of an array with more than one element is ambiguous``,
which silently disabled every episode embedding and all Pass-2 supersession
(the failure is caught and logged as a warning by the ingest pipeline).

CI installs ``.[dev]`` without the embedding model, so this stubs the model class
with the library's real return shape instead of downloading weights.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from memcontext import retrieval


class _FakeBGEM3:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def encode(self, texts, **_kwargs):
        return {"dense_vecs": np.ones((len(texts), 4), dtype=np.float32)}


@pytest.fixture
def fake_flagembedding(monkeypatch):
    mod = types.ModuleType("FlagEmbedding")
    mod.BGEM3FlagModel = _FakeBGEM3  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "FlagEmbedding", mod)
    monkeypatch.delenv(retrieval.MODAL_URL_ENV, raising=False)
    return mod


def test_local_bge_m3_accepts_numpy_dense_vecs(fake_flagembedding):
    client = retrieval.EmbeddingClient()
    vecs = client._embed_bge_m3_local(["orders service uses PostgreSQL", "second text"])
    assert len(vecs) == 2
    assert vecs[0] == [1.0, 1.0, 1.0, 1.0]
    assert all(isinstance(x, float) for x in vecs[1])


def test_local_bge_m3_accepts_single_text(fake_flagembedding):
    client = retrieval.EmbeddingClient()
    assert client._embed_bge_m3_local(["only one"]) == [[1.0, 1.0, 1.0, 1.0]]
