from __future__ import annotations

from memcontext.retrieval import BGE_M3_EMBED_DIM, BGE_M3_MODEL_ID


def test_product_default_embedder_is_bge_m3():
    assert BGE_M3_MODEL_ID == "BAAI/bge-m3"
    assert BGE_M3_EMBED_DIM == 1024
