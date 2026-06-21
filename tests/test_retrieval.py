from __future__ import annotations

import math
import struct

from memcontext.retrieval import (
    _bm25_scores,
    _cosine_fallback,
    _cosine_normalised,
    _decode_vector,
    _encode_vector,
    _rrf_ranks,
    _tokenize_for_bm25,
    claim_retrieval_text,
)
from memcontext.schema import Claim, ClaimStatus


def _fake_claim(subject: str, predicate: str, value: str) -> Claim:
    return Claim(
        claim_id="cl_test",
        session_id="s1",
        subject=subject,
        predicate=predicate,
        value=value,
        value_normalised=None,
        confidence=0.9,
        source_turn_id="tu_test",
        status=ClaimStatus.ACTIVE,
        created_ts=1,
        char_start=None,
        char_end=None,
        valid_from_ts=1,
        valid_until_ts=None,
    )


def test_encode_decode_vector_roundtrip():
    vec = [0.1, 0.2, 0.3, 0.4, 0.5]
    encoded = _encode_vector(vec)
    decoded = _decode_vector(encoded)
    assert len(decoded) == len(vec)
    for a, b in zip(vec, decoded):
        assert abs(a - b) < 1e-6


def test_encode_vector_format():
    vec = [1.0, 2.0, 3.0]
    encoded = _encode_vector(vec)
    length_prefix = struct.unpack("<I", encoded[:4])[0]
    assert length_prefix == 3


def test_cosine_normalised_identical():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(_cosine_normalised(a, b) - 1.0) < 1e-6


def test_cosine_normalised_orthogonal():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert abs(_cosine_normalised(a, b)) < 1e-6


def test_cosine_fallback_identical():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(_cosine_fallback(a, b) - 1.0) < 1e-6


def test_cosine_fallback_orthogonal():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert abs(_cosine_fallback(a, b)) < 1e-6


def test_cosine_fallback_zero_vector():
    a = [0.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert _cosine_fallback(a, b) == 0.0


def test_bm25_scores_basic():
    c1 = _fake_claim("cat", "user_fact", "sat on the mat")
    c2 = _fake_claim("dog", "user_fact", "played in the park")
    c3 = _fake_claim("cat", "user_fact", "chased a mouse")

    query_tokens = _tokenize_for_bm25("cat mat")
    scores = _bm25_scores(query_tokens, [c1, c2, c3])
    assert len(scores) == 3
    assert scores[0] > scores[1]


def test_bm25_scores_empty():
    scores = _bm25_scores([], [])
    assert scores == []


def test_rrf_ranks_basic():
    scores = [0.9, 0.1, 0.5]
    ranks = _rrf_ranks(scores)
    assert ranks[0] == 1
    assert ranks[2] == 2
    assert ranks[1] == 3


def test_rrf_ranks_ties():
    # Competition ranking: equal scores share the same (lowest) rank, so a FLAT /
    # degenerate channel stays NEUTRAL in fusion instead of injecting index-ordered
    # noise that buries a needle another channel ranked #1
    # (proven: results/proof_fusion_master.py — a flat channel buried an entity-needle
    # to #4 under the old strict-index ranking; tie-aware ranking restores it to #1).
    #
    # Verified as a GENERAL PROPERTY over random inputs — NOT hardcoded expected
    # values: for every pair, equal scores => equal rank, higher score => lower rank.
    import random
    rng = random.Random(0)
    for _ in range(100):
        scores = [rng.choice([0.1, 0.4, 0.7, 1.0]) for _ in range(rng.randint(1, 25))]
        ranks = _rrf_ranks(scores)
        for i in range(len(scores)):
            for j in range(len(scores)):
                if scores[i] == scores[j]:
                    assert ranks[i] == ranks[j]   # equal score -> equal rank (flat channel neutral)
                elif scores[i] > scores[j]:
                    assert ranks[i] < ranks[j]    # higher score -> better (lower) rank
    # the degenerate case the fix targets: a wholly flat channel collapses to ONE rank
    assert set(_rrf_ranks([0.5] * 8)) == {1}


def test_claim_retrieval_text():
    claim = _fake_claim("user", "user_preference", "prefers dark mode")
    text = claim_retrieval_text(claim)
    assert "user" in text
    assert "user_preference" in text
    assert "prefers dark mode" in text


def test_tokenize_for_bm25():
    tokens = _tokenize_for_bm25("Hello World! 123 test-case")
    assert "hello" in tokens
    assert "world" in tokens
    assert "123" in tokens
    assert "test" in tokens
    assert "case" in tokens
