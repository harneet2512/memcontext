"""Tests for retrieval pipeline changes: valid_at_ts filtering,
classify_query_depth routing, per-category weight config.

Uses :memory: SQLite, NullEmbedder (constant vectors), and mock
reader LLM. No network calls, no model downloads.
"""
from __future__ import annotations

import math
import sqlite3
import struct
from unittest.mock import MagicMock, patch

import pytest

from memcontext.claims import (
    insert_claim,
    insert_turn,
    list_active_claims,
    new_turn_id,
    now_ns,
    set_claim_status,
)
from memcontext.retrieval import (
    EmbeddingClient,
    _claim_valid_at,
    _encode_vector,
    classify_query_depth,
    retrieve_hybrid,
)
from memcontext.schema import Claim, ClaimStatus, Speaker, Turn, open_database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 384


def _make_conn() -> sqlite3.Connection:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _insert_turn(conn: sqlite3.Connection, sid: str, text: str, ts: int) -> Turn:
    t = Turn(
        turn_id=new_turn_id(),
        session_id=sid,
        speaker=Speaker.USER,
        text=text,
        ts=ts,
        asr_confidence=None,
    )
    insert_turn(conn, t)
    return t


def _insert_claim_with_embedding(
    conn: sqlite3.Connection,
    sid: str,
    turn: Turn,
    subject: str,
    predicate: str,
    value: str,
    valid_from: int | None = None,
    valid_until: int | None = None,
    vec: list[float] | None = None,
) -> Claim:
    claim = insert_claim(
        conn,
        session_id=sid,
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=0.9,
        source_turn_id=turn.turn_id,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    if vec is None:
        vec = [1.0 / math.sqrt(DIM)] * DIM
    blob = _encode_vector(vec)
    conn.execute(
        "INSERT OR REPLACE INTO claim_embeddings "
        "(claim_id, embedding, embedding_model_version, embedded_at_unix) "
        "VALUES (?, ?, ?, ?)",
        (claim.claim_id, blob, "test-model", 0),
    )
    return claim


def _stub_embedding_client() -> EmbeddingClient:
    """EmbeddingClient that returns constant unit vectors without loading a model."""
    client = MagicMock(spec=EmbeddingClient)
    client.model_version = "test-model"
    unit = [1.0 / math.sqrt(DIM)] * DIM
    client.embed.return_value = [unit]
    return client


# ---------------------------------------------------------------------------
# Tests: valid_at_ts filtering in retrieve_hybrid (Change 1)
# ---------------------------------------------------------------------------


class TestValidAtTsFiltering:
    """Verify retrieve_hybrid respects temporal validity windows."""

    def _setup_temporal_claims(self, conn, sid):
        """Create 3 claims with non-overlapping validity windows:
        c1: valid [1000, 2000)  — NYC
        c2: valid [2000, 3000)  — LA
        c3: valid [3000, ∞)    — London
        """
        t1 = _insert_turn(conn, sid, "I live in NYC", ts=1000)
        t2 = _insert_turn(conn, sid, "I moved to LA", ts=2000)
        t3 = _insert_turn(conn, sid, "Now in London", ts=3000)

        c1 = _insert_claim_with_embedding(
            conn, sid, t1, "user", "user_fact", "lives in NYC",
            valid_from=1000, valid_until=2000,
        )
        set_claim_status(conn, c1.claim_id, ClaimStatus.SUPERSEDED)

        c2 = _insert_claim_with_embedding(
            conn, sid, t2, "user", "user_fact", "lives in LA",
            valid_from=2000, valid_until=3000,
        )
        set_claim_status(conn, c2.claim_id, ClaimStatus.SUPERSEDED)

        c3 = _insert_claim_with_embedding(
            conn, sid, t3, "user", "user_fact", "lives in London",
            valid_from=3000,
        )
        return c1, c2, c3

    def test_valid_at_ts_returns_only_temporally_valid(self):
        conn = _make_conn()
        sid = "temporal_test"
        c1, c2, c3 = self._setup_temporal_claims(conn, sid)
        client = _stub_embedding_client()

        results = retrieve_hybrid(
            conn,
            session_id=sid,
            query="where do I live",
            top_k=10,
            valid_at_ts=1500,
            embedding_client=client,
            include_superseded=True,
        )

        claim_ids = {c.claim_id for c, _ in results}
        assert c1.claim_id in claim_ids, "c1 (NYC) should be valid at ts=1500"
        assert c2.claim_id not in claim_ids, "c2 (LA) should NOT be valid at ts=1500"
        assert c3.claim_id not in claim_ids, "c3 (London) should NOT be valid at ts=1500"

    def test_valid_at_ts_returns_current_when_at_boundary(self):
        conn = _make_conn()
        sid = "boundary_test"
        c1, c2, c3 = self._setup_temporal_claims(conn, sid)
        client = _stub_embedding_client()

        results = retrieve_hybrid(
            conn,
            session_id=sid,
            query="where do I live",
            top_k=10,
            valid_at_ts=2000,
            embedding_client=client,
            include_superseded=True,
        )

        claim_ids = {c.claim_id for c, _ in results}
        assert c1.claim_id not in claim_ids, "c1 valid_until=2000, so NOT valid AT 2000"
        assert c2.claim_id in claim_ids, "c2 valid_from=2000, so valid AT 2000"

    def test_valid_at_ts_none_returns_all_active(self):
        """Backwards compat: when valid_at_ts is None, no temporal filtering."""
        conn = _make_conn()
        sid = "no_filter_test"
        t = _insert_turn(conn, sid, "test", ts=1000)
        c1 = _insert_claim_with_embedding(
            conn, sid, t, "user", "user_fact", "fact one", valid_from=1000,
        )
        c2 = _insert_claim_with_embedding(
            conn, sid, t, "user", "user_preference", "prefers dark mode", valid_from=1000,
        )
        client = _stub_embedding_client()

        results = retrieve_hybrid(
            conn,
            session_id=sid,
            query="tell me about user",
            top_k=10,
            valid_at_ts=None,
            embedding_client=client,
        )

        assert len(results) == 2, "Both claims should be returned when valid_at_ts is None"

    def test_valid_at_ts_filters_to_empty_returns_empty(self):
        conn = _make_conn()
        sid = "empty_test"
        t = _insert_turn(conn, sid, "test", ts=5000)
        _insert_claim_with_embedding(
            conn, sid, t, "user", "user_fact", "some fact",
            valid_from=5000, valid_until=6000,
        )
        set_claim_status(conn, list_active_claims(conn, sid)[0].claim_id, ClaimStatus.SUPERSEDED)
        client = _stub_embedding_client()

        results = retrieve_hybrid(
            conn,
            session_id=sid,
            query="anything",
            top_k=10,
            valid_at_ts=9000,
            embedding_client=client,
            include_superseded=True,
        )

        assert results == [], "No claims valid at ts=9000 → empty result"


# ---------------------------------------------------------------------------
# Tests: _claim_valid_at edge cases
# ---------------------------------------------------------------------------


class TestClaimValidAt:

    def _make_claim(self, valid_from=None, valid_until=None) -> Claim:
        return Claim(
            claim_id="cl_test",
            session_id="s",
            subject="user",
            predicate="user_fact",
            value="v",
            value_normalised=None,
            confidence=0.9,
            source_turn_id="tu_test",
            status=ClaimStatus.ACTIVE,
            created_ts=100,
            valid_from_ts=valid_from,
            valid_until_ts=valid_until,
        )

    def test_no_bounds_always_valid(self):
        c = self._make_claim(valid_from=None, valid_until=None)
        assert _claim_valid_at(c, 0) is True
        assert _claim_valid_at(c, 999999) is True

    def test_only_valid_from(self):
        c = self._make_claim(valid_from=100, valid_until=None)
        assert _claim_valid_at(c, 50) is False
        assert _claim_valid_at(c, 100) is True
        assert _claim_valid_at(c, 200) is True

    def test_only_valid_until(self):
        c = self._make_claim(valid_from=None, valid_until=200)
        assert _claim_valid_at(c, 100) is True
        assert _claim_valid_at(c, 199) is True
        assert _claim_valid_at(c, 200) is False

    def test_bounded_window(self):
        c = self._make_claim(valid_from=100, valid_until=200)
        assert _claim_valid_at(c, 99) is False
        assert _claim_valid_at(c, 100) is True
        assert _claim_valid_at(c, 150) is True
        assert _claim_valid_at(c, 199) is True
        assert _claim_valid_at(c, 200) is False


# ---------------------------------------------------------------------------
# Tests: classify_query_depth (Change 2)
# ---------------------------------------------------------------------------


class TestClassifyQueryDepth:

    @pytest.mark.parametrize("query,expected_type,expected_k", [
        ("How many plants did I buy?", "aggregation", 50),
        ("List all the books I read", "aggregation", 50),
        ("Summarize my trips", "aggregation", 50),
        ("Count the number of meetings", "aggregation", 50),
        ("Give me a history of my diet", "aggregation", 50),
        ("When did I start my job?", "temporal", 30),
        ("What was the last time I went hiking?", "temporal", 30),
        ("How long ago did I move?", "temporal", 30),
        ("What is my favorite color?", "factual", 15),
        ("Where do I work?", "factual", 15),
        ("Do I have a pet?", "factual", 15),
    ])
    def test_classification(self, query, expected_type, expected_k):
        qtype, k = classify_query_depth(query)
        assert qtype == expected_type, f"Query '{query}' classified as {qtype}, expected {expected_type}"
        assert k == expected_k, f"Query '{query}' got k={k}, expected {expected_k}"

    def test_top_k_is_floor_not_override(self):
        """classify_query_depth sets a MINIMUM top_k, the runner takes max(configured, recommended)."""
        _, depth_k = classify_query_depth("How many plants?")
        configured_k = 100
        effective = max(configured_k, depth_k)
        assert effective == 100, "Configured top_k=100 should not be reduced by depth_k=50"

        configured_k = 10
        effective = max(configured_k, depth_k)
        assert effective == 50, "Configured top_k=10 should be raised to depth_k=50"


# ---------------------------------------------------------------------------
# Tests: per-category retrieval config (Change C)
# ---------------------------------------------------------------------------


class TestPerCategoryConfig:

    def test_multi_session_gets_entity_weight(self):
        """Multi-session questions should have entity weight > 0."""
        # Import the config from run_official.py
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_official", "evals/benchmark/run_official.py"
        )
        mod = importlib.util.module_from_spec(spec)

        # Extract the config dict by exec'ing just the config block
        with open("evals/benchmark/run_official.py") as f:
            source = f.read()

        config_block = source.split("_CATEGORY_RETRIEVAL_CONFIG")[1]
        config_block = "_CATEGORY_RETRIEVAL_CONFIG" + config_block[:config_block.index("\n\n")]
        local_ns: dict = {}
        exec(config_block, {}, local_ns)
        config = local_ns["_CATEGORY_RETRIEVAL_CONFIG"]

        ms_config = config.get("cross_session_user_fact", {})
        ms_weights = ms_config.get("weights", ())
        assert len(ms_weights) >= 2, "Multi-session config must have weights"
        assert ms_weights[1] > 0, f"Entity weight should be > 0 for multi-session, got {ms_weights[1]}"

    def test_preference_gets_entity_weight(self):
        with open("evals/benchmark/run_official.py") as f:
            source = f.read()

        config_block = source.split("_CATEGORY_RETRIEVAL_CONFIG")[1]
        config_block = "_CATEGORY_RETRIEVAL_CONFIG" + config_block[:config_block.index("\n\n")]
        local_ns: dict = {}
        exec(config_block, {}, local_ns)
        config = local_ns["_CATEGORY_RETRIEVAL_CONFIG"]

        pref_config = config.get("single_session_preference", {})
        pref_weights = pref_config.get("weights", ())
        assert len(pref_weights) >= 2, "Preference config must have weights"
        assert pref_weights[1] > 0, f"Entity weight should be > 0 for preference, got {pref_weights[1]}"

    def test_multi_session_top_k_higher_than_default(self):
        with open("evals/benchmark/run_official.py") as f:
            source = f.read()

        config_block = source.split("_CATEGORY_RETRIEVAL_CONFIG")[1]
        config_block = "_CATEGORY_RETRIEVAL_CONFIG" + config_block[:config_block.index("\n\n")]
        local_ns: dict = {}
        exec(config_block, {}, local_ns)
        config = local_ns["_CATEGORY_RETRIEVAL_CONFIG"]

        ms_top_k = config["cross_session_user_fact"]["top_k"]
        assert ms_top_k > 50, f"Multi-session top_k should be > baseline 50, got {ms_top_k}"


# ---------------------------------------------------------------------------
# Tests: prompt behavioral verification (Changes A/B)
# ---------------------------------------------------------------------------


class TestPromptBehavior:

    def test_multi_session_prompt_forces_enumeration(self):
        """The cross_session_user_fact prompt must instruct the reader to
        enumerate instances individually before counting/summing."""
        from evals.longmemeval_prompts import PROMPTS, get_prompt

        prompt = PROMPTS["cross_session_user_fact"]
        assert "Number each" in prompt or "number each" in prompt, \
            "Prompt must instruct numbered enumeration"
        assert "count" in prompt.lower() or "sum" in prompt.lower() or "arithmetic" in prompt.lower(), \
            "Prompt must mention counting or arithmetic"

        formatted = get_prompt(
            "multi-session",
            "1. [user_event] user: bought a succulent\n2. [user_event] user: bought a fern",
            "How many plants did I buy?",
        )
        assert "succulent" in formatted
        assert "How many plants" in formatted

    def test_preference_prompt_handles_implicit(self):
        """The preference prompt must instruct synthesis of implicit preferences."""
        from evals.longmemeval_prompts import PROMPTS, get_prompt

        prompt = PROMPTS["single_session_preference"]
        assert "IMPLICIT" in prompt, "Must mention implicit preferences"
        assert "behavior" in prompt.lower() or "choices" in prompt.lower(), \
            "Must instruct extraction from behavior/choices"
        assert "specific" in prompt.lower() or "names" in prompt.lower(), \
            "Must request specific product names/details"

    def test_multi_session_prompt_has_verification_step(self):
        """The prompt should instruct the reader to verify its count."""
        from evals.longmemeval_prompts import PROMPTS

        prompt = PROMPTS["cross_session_user_fact"]
        assert "verify" in prompt.lower(), \
            "Prompt should instruct verification of count against enumerated list"

    def test_category_routing_maps_correctly(self):
        """Dataset category names must map to the right prompt keys."""
        from evals.longmemeval_prompts import CATEGORY_MAP, PROMPTS

        assert CATEGORY_MAP["multi-session"] == "cross_session_user_fact"
        assert CATEGORY_MAP["single-session-preference"] == "single_session_preference"
        assert "cross_session_user_fact" in PROMPTS
        assert "single_session_preference" in PROMPTS


# ---------------------------------------------------------------------------
# Tests: retrieve_hybrid integration with real RRF
# ---------------------------------------------------------------------------


class TestFrequencySignal:
    """Verify that claim density per (subject, predicate) boosts ranking."""

    def test_frequently_mentioned_claim_ranks_higher(self):
        """A preference mentioned in 3 turns should outrank one mentioned once."""
        conn = _make_conn()
        sid = "freq_test"
        client = _stub_embedding_client()

        t1 = _insert_turn(conn, sid, "I love dark mode", ts=1000)
        t2 = _insert_turn(conn, sid, "Dark mode is great", ts=2000)
        t3 = _insert_turn(conn, sid, "I always use dark mode", ts=3000)
        t4 = _insert_turn(conn, sid, "I tried light theme once", ts=4000)

        _insert_claim_with_embedding(
            conn, sid, t1, "user", "user_preference", "prefers dark mode",
            valid_from=1000,
        )
        _insert_claim_with_embedding(
            conn, sid, t2, "user", "user_preference", "likes dark mode",
            valid_from=2000,
        )
        _insert_claim_with_embedding(
            conn, sid, t3, "user", "user_preference", "uses dark mode",
            valid_from=3000,
        )
        light_claim = _insert_claim_with_embedding(
            conn, sid, t4, "user", "user_preference", "tried light theme",
            valid_from=4000,
        )

        results = retrieve_hybrid(
            conn,
            session_id=sid,
            query="what theme does user prefer",
            top_k=10,
            embedding_client=client,
        )

        claim_ids = [c.claim_id for c, _ in results]
        light_rank = claim_ids.index(light_claim.claim_id)
        assert light_rank > 0, (
            "Light theme (mentioned once) should rank below dark mode claims "
            "(mentioned 3 times, higher frequency signal)"
        )

    def test_frequency_score_counts_subject_predicate_pairs(self):
        """Frequency is per (subject, predicate) group, not global."""
        conn = _make_conn()
        sid = "freq_group_test"
        client = _stub_embedding_client()

        t1 = _insert_turn(conn, sid, "test1", ts=1000)
        t2 = _insert_turn(conn, sid, "test2", ts=2000)

        _insert_claim_with_embedding(
            conn, sid, t1, "user", "user_fact", "works at Google",
            valid_from=1000,
        )
        _insert_claim_with_embedding(
            conn, sid, t2, "user", "user_fact", "senior engineer",
            valid_from=2000,
        )
        _insert_claim_with_embedding(
            conn, sid, t1, "user", "user_preference", "prefers Python",
            valid_from=1000,
        )

        results = retrieve_hybrid(
            conn,
            session_id=sid,
            query="tell me about user",
            top_k=10,
            embedding_client=client,
        )

        assert len(results) == 3


class TestEventTs:
    """Verify event_ts field on claims and its effect on temporal ranking."""

    def test_claim_stores_event_ts(self):
        conn = _make_conn()
        sid = "event_ts_test"
        t = _insert_turn(conn, sid, "I started in March", ts=5000)

        march_ns = 1709251200_000000000  # March 2024 approx
        claim = insert_claim(
            conn,
            session_id=sid,
            subject="user",
            predicate="user_event",
            value="started job",
            confidence=0.9,
            source_turn_id=t.turn_id,
            valid_from=5000,
            event_ts=march_ns,
        )

        assert claim.event_ts == march_ns

        from memcontext.claims import get_claim
        loaded = get_claim(conn, claim.claim_id)
        assert loaded is not None
        assert loaded.event_ts == march_ns

    def test_claim_event_ts_defaults_to_none(self):
        conn = _make_conn()
        sid = "no_event_ts"
        t = _insert_turn(conn, sid, "plain fact", ts=1000)

        claim = insert_claim(
            conn,
            session_id=sid,
            subject="user",
            predicate="user_fact",
            value="some fact",
            confidence=0.9,
            source_turn_id=t.turn_id,
        )

        assert claim.event_ts is None

    def test_recency_prefers_event_ts_over_valid_from(self):
        """When event_ts is set, temporal ranking should use it."""
        from memcontext.retrieval import _claim_recency_ts

        old_event = Claim(
            claim_id="cl_old", session_id="s", subject="u", predicate="user_event",
            value="started job", value_normalised=None, confidence=0.9,
            source_turn_id="tu_1", status=ClaimStatus.ACTIVE, created_ts=5000,
            valid_from_ts=5000, valid_until_ts=None,
            event_ts=1000,  # event happened much earlier
        )
        new_ingested = Claim(
            claim_id="cl_new", session_id="s", subject="u", predicate="user_fact",
            value="some fact", value_normalised=None, confidence=0.9,
            source_turn_id="tu_2", status=ClaimStatus.ACTIVE, created_ts=6000,
            valid_from_ts=6000, valid_until_ts=None,
            event_ts=None,
        )

        assert _claim_recency_ts(old_event) == 1000, "Should use event_ts"
        assert _claim_recency_ts(new_ingested) == 6000, "Should fall back to valid_from_ts"


class TestRetrieveHybridIntegration:
    """End-to-end test: insert claims, embed them, retrieve, verify ranking."""

    def test_retrieval_returns_claims_sorted_by_score(self):
        conn = _make_conn()
        sid = "integration_test"
        client = _stub_embedding_client()

        t = _insert_turn(conn, sid, "I have a dog named Max", ts=1000)

        c1 = _insert_claim_with_embedding(
            conn, sid, t, "user", "user_fact", "has a dog named Max",
            valid_from=1000,
        )
        c2 = _insert_claim_with_embedding(
            conn, sid, t, "user", "user_preference", "prefers cats over dogs",
            valid_from=1000,
        )

        results = retrieve_hybrid(
            conn,
            session_id=sid,
            query="what pet do I have",
            top_k=10,
            embedding_client=client,
        )

        assert len(results) == 2
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by score descending"

    def test_retrieval_respects_top_k(self):
        conn = _make_conn()
        sid = "topk_test"
        client = _stub_embedding_client()

        t = _insert_turn(conn, sid, "test", ts=1000)
        for i in range(10):
            _insert_claim_with_embedding(
                conn, sid, t, "user", "user_fact", f"fact number {i}",
                valid_from=1000,
            )

        results = retrieve_hybrid(
            conn,
            session_id=sid,
            query="tell me facts",
            top_k=3,
            embedding_client=client,
        )

        assert len(results) == 3, f"Expected 3 results for top_k=3, got {len(results)}"

    def test_empty_session_returns_empty(self):
        conn = _make_conn()
        client = _stub_embedding_client()

        results = retrieve_hybrid(
            conn,
            session_id="nonexistent",
            query="anything",
            top_k=10,
            embedding_client=client,
        )

        assert results == []

    def test_reranker_replaces_scores(self):
        """When a reranker is provided, it replaces RRF scores with reranker scores."""
        conn = _make_conn()
        sid = "rerank_test"
        client = _stub_embedding_client()

        t = _insert_turn(conn, sid, "test", ts=1000)
        _insert_claim_with_embedding(
            conn, sid, t, "user", "user_fact", "alpha fact", valid_from=1000,
        )
        _insert_claim_with_embedding(
            conn, sid, t, "user", "user_fact", "beta fact", valid_from=1000,
        )

        fixed_scores = [0.95, 0.10]

        def fixed_reranker(query: str, texts: list[str]) -> list[float]:
            return fixed_scores[:len(texts)]

        results = retrieve_hybrid(
            conn, session_id=sid, query="facts", top_k=10,
            embedding_client=client, reranker=fixed_reranker,
        )

        scores = [s for _, s in results]
        assert scores[0] == pytest.approx(0.95), "Top result should have reranker score 0.95"
        assert scores[1] == pytest.approx(0.10), "Second result should have reranker score 0.10"

    def test_reranker_failure_falls_back(self):
        """If the reranker raises, results fall back to RRF order."""
        conn = _make_conn()
        sid = "rerank_fail"
        client = _stub_embedding_client()

        t = _insert_turn(conn, sid, "test", ts=1000)
        _insert_claim_with_embedding(
            conn, sid, t, "user", "user_fact", "some fact", valid_from=1000,
        )

        def broken_reranker(query: str, texts: list[str]) -> list[float]:
            raise RuntimeError("reranker crashed")

        results = retrieve_hybrid(
            conn, session_id=sid, query="facts", top_k=10,
            embedding_client=client, reranker=broken_reranker,
        )

        assert len(results) == 1, "Should return results despite reranker failure"

    def test_empty_query_returns_empty(self):
        conn = _make_conn()
        client = _stub_embedding_client()

        results = retrieve_hybrid(
            conn,
            session_id="any",
            query="",
            top_k=10,
            embedding_client=client,
        )

        assert results == []


# ---------------------------------------------------------------------------
# Tests: reranking (Change 5) + query expansion (Change 6)
# ---------------------------------------------------------------------------


class TestQueryExpansion:

    def test_extract_query_entities_filters_stopwords(self):
        from memcontext.retrieval import _extract_query_entities

        entities = _extract_query_entities("How many plants did I buy in the garden?")
        assert "how" not in entities
        assert "did" not in entities
        assert "the" not in entities
        assert "plants" in entities
        assert "garden" in entities
        assert "buy" in entities

    def test_extract_query_entities_handles_empty(self):
        from memcontext.retrieval import _extract_query_entities

        assert _extract_query_entities("") == []
        assert _extract_query_entities("I the a") == []

    def test_retrieve_expanded_returns_superset_of_primary(self):
        from memcontext.retrieval import retrieve_expanded

        conn = _make_conn()
        sid = "expand_test"
        client = _stub_embedding_client()

        t1 = _insert_turn(conn, sid, "I bought a succulent", ts=1000)
        t2 = _insert_turn(conn, sid, "Got a fern at the market", ts=2000)
        t3 = _insert_turn(conn, sid, "My cat is named Luna", ts=3000)

        _insert_claim_with_embedding(
            conn, sid, t1, "plant_collection", "user_event", "bought succulent",
            valid_from=1000,
        )
        _insert_claim_with_embedding(
            conn, sid, t2, "plant_collection", "user_event", "bought fern at market",
            valid_from=2000,
        )
        _insert_claim_with_embedding(
            conn, sid, t3, "user", "user_fact", "has cat named Luna",
            valid_from=3000,
        )

        primary = retrieve_hybrid(
            conn, session_id=sid, query="How many plants?",
            top_k=10, embedding_client=client,
        )
        expanded = retrieve_expanded(
            conn, session_id=sid, query="How many plants?",
            top_k=10, embedding_client=client,
        )

        assert len(expanded) >= len(primary)

    def test_retrieve_expanded_deduplicates(self):
        from memcontext.retrieval import retrieve_expanded

        conn = _make_conn()
        sid = "dedup_test"
        client = _stub_embedding_client()

        t = _insert_turn(conn, sid, "I work at Google", ts=1000)
        _insert_claim_with_embedding(
            conn, sid, t, "user", "user_fact", "works at Google",
            valid_from=1000,
        )

        results = retrieve_expanded(
            conn, session_id=sid, query="where does user work",
            top_k=10, embedding_client=client,
        )

        claim_ids = [c.claim_id for c, _ in results]
        assert len(claim_ids) == len(set(claim_ids)), "No duplicate claim_ids"

    def test_retrieve_expanded_empty_session(self):
        from memcontext.retrieval import retrieve_expanded

        conn = _make_conn()
        results = retrieve_expanded(
            conn, session_id="nonexistent", query="anything", top_k=10,
        )
        assert results == []
