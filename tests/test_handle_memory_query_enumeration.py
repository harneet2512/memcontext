"""Wiring test: enumeration deepening attached to the PRODUCT serve door.

`handle_memory_query` attaches a deterministic DISTINCT-count `enumeration`
block for generic aggregation-intent queries ("how many", "count", "list all"
— routed by `classify_query_depth`, NOT a benchmark keyword list) when a real
embedder is configured. This proves:

  * an aggregation query gets a CORRECT distinct_count (near-duplicate mentions
    of the same instance collapse; genuinely distinct instances stay apart);
  * a NORMAL (non-aggregation) query gets NO enumeration block AND the
    claims/episodes/total are byte-identical to the same query run with
    enumeration force-disabled — proving the change is strictly ADDITIVE and
    cannot create a new regression.

Model-free: a deterministic concept embedder is injected so `semantic_enabled()`
is True with zero model download (the same proven stub the enumeration unit
tests use). NullEmbedder/no-embedder paths skip enumeration entirely, which is
covered implicitly by the byte-identical assertion.
"""
from __future__ import annotations

import math
import sqlite3

import pytest

import memcontext.retrieval as R
from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.mcp_tools import handle_memory_query
from memcontext.schema import Speaker, Turn, open_database

SESSION = "enum-serve-session"


# --------------------------------------------------------------------------- #
# Deterministic, model-free embedder (concept buckets -> bimodal cosine).
# Same construction as tests/test_enumeration.py::ConceptEmbedder.
# --------------------------------------------------------------------------- #
class _ConceptEmbedder:
    model_version = "concept-serve-stub-v1"

    CONCEPTS = {
        "sushi": ("ate sushi", "had sushi for lunch", "grabbed some sushi"),
        "ramen": ("ate ramen", "had a bowl of ramen", "got ramen for dinner"),
        "tacos": ("ate tacos", "had tacos", "ordered tacos"),
    }

    def __init__(self) -> None:
        self._dim = len(self.CONCEPTS) + 4
        self._concept_idx = {c: i for i, c in enumerate(self.CONCEPTS)}

    def _concept_of(self, text: str) -> str | None:
        low = text.lower()
        for concept, phrasings in self.CONCEPTS.items():
            if any(p in low for p in phrasings):
                return concept
        return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self._dim
            concept = self._concept_of(t)
            if concept is not None:
                v[self._concept_idx[concept]] = 1.0
            else:
                # deterministic spread for unknown text (orthogonal-ish)
                for k in range(self._dim):
                    v[k] = ((hash_stable(t + str(k))) % 1000) / 1000.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def hash_stable(s: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big")


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


_TS = [0]


def _add(conn: sqlite3.Connection, *, subject: str, predicate: str, value: str) -> str:
    _TS[0] += 1
    t = Turn(
        turn_id=new_turn_id(),
        session_id=SESSION,
        speaker=Speaker.USER,
        text=value,
        ts=now_ns(),
        asr_confidence=None,
    )
    insert_turn(conn, t)
    claim = insert_claim(
        conn,
        session_id=SESSION,
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=0.9,
        source_turn_id=t.turn_id,
    )
    return claim.claim_id


def _seed_two_distinct_meals(conn: sqlite3.Connection) -> None:
    """Two DISTINCT meal instances, each mentioned several ways (the regression
    shape: more raw mentions than distinct things). gold distinct = 2."""
    _add(conn, subject="user", predicate="user_event", value="ate sushi")
    _add(conn, subject="user", predicate="user_event", value="had sushi for lunch")
    _add(conn, subject="user", predicate="user_event", value="grabbed some sushi")
    _add(conn, subject="user", predicate="user_event", value="ate ramen")
    _add(conn, subject="user", predicate="user_event", value="had a bowl of ramen")


def test_aggregation_query_gets_distinct_count(conn, monkeypatch):
    # Real embedder configured -> semantic_enabled() True, enumeration fires.
    monkeypatch.setattr(R, "episode_embedder", lambda: _ConceptEmbedder())
    _seed_two_distinct_meals(conn)

    result = handle_memory_query(
        conn, query="how many different meals have I eaten?", session_id=SESSION
    )

    assert result["query_type"] is not None
    assert "enumeration" in result
    enum = result["enumeration"]
    # 5 raw mentions collapse to 2 distinct instances (sushi, ramen).
    assert enum["distinct_count"] == 2
    assert len(enum["representatives"]) == 2
    # every served representative carries its supporting claim ids (instance-preserving)
    assert all(r["member_claim_ids"] for r in enum["representatives"])


def test_normal_query_has_no_enumeration_and_is_byte_identical(conn, monkeypatch):
    # Embedder available (so the only thing gating enumeration off is the query
    # NOT being aggregation intent — this is the strict ADDITIVE proof).
    monkeypatch.setattr(R, "episode_embedder", lambda: _ConceptEmbedder())
    _seed_two_distinct_meals(conn)

    normal_q = "what did I eat for lunch?"  # no aggregation keyword
    with_emb = handle_memory_query(conn, query=normal_q, session_id=SESSION)

    # No enumeration block on a non-aggregation query.
    assert "enumeration" not in with_emb

    # Byte-identical core payload to the same query run with enumeration
    # impossible to fire (no embedder -> semantic disabled). The non-aggregation
    # path must be untouched either way.
    monkeypatch.setattr(R, "episode_embedder", lambda: None)
    no_emb = handle_memory_query(conn, query=normal_q, session_id=SESSION)

    assert "enumeration" not in no_emb
    assert with_emb["claims"] == no_emb["claims"]
    assert with_emb["episodes"] == no_emb["episodes"]
    assert with_emb["total"] == no_emb["total"]


def test_aggregation_query_without_embedder_skips_enumeration(conn, monkeypatch):
    # Aggregation INTENT but no real embedder -> enumeration must be skipped
    # (the cluster dedup is embedding-based; NullEmbedder/None must not fire it).
    monkeypatch.setattr(R, "episode_embedder", lambda: None)
    _seed_two_distinct_meals(conn)

    result = handle_memory_query(
        conn, query="how many different meals have I eaten?", session_id=SESSION
    )
    assert "enumeration" not in result
