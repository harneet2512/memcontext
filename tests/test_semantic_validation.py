"""Semantic-VALUE validation: prove the engine matches by MEANING, not words.

The existing semantic tests use a bag-of-words embedder (cosine ~= word overlap ~=
BM25), so they validate the plumbing but not the differentiator. Here a concept-
aware deterministic embedder makes "Berlin"/"reside" near via a 'location' concept
with ZERO shared words -- something BM25 cannot do -- so a pass proves the semantic
core does real work. Model-free; CI-safe.
"""
from __future__ import annotations

import math
import sqlite3

from memcontext.claims import insert_fact
from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.retrieval import backfill_embeddings, retrieve_hybrid
from memcontext.schema import Speaker, open_database
from memcontext.supersession_semantic import SemanticSupersession


class ConceptEmbedder:
    """Deterministic concept-level embedder: texts sharing a CONCEPT are near
    even with no shared words (unlike bag-of-words / BM25). Model-free."""

    model_version = "concept-stub-v1"
    CONCEPTS = {
        "location": ("munich", "berlin", "paris", "city", "live", "lives", "moved",
                     "reside", "resides", "home", "located"),
        "beverage": ("coffee", "tea", "espresso", "latte", "drink", "drinks"),
        "database": ("postgres", "mysql", "sqlite", "redis", "database"),
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        dims = list(self.CONCEPTS)
        out: list[list[float]] = []
        for t in texts:
            toks = set(t.lower().replace("_", " ").split())
            v = [1.0 if toks & set(self.CONCEPTS[c]) else 0.0 for c in dims]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _ingest(conn, value):
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text=f"a fact about {value}",
        extractor=PassthroughExtractor(
            [{"subject": "user", "predicate": "user_fact", "value": value, "confidence": 0.9}]),
    )


def test_semantic_retrieval_matches_by_concept_not_words():
    conn = _conn()
    emb = ConceptEmbedder()
    _ingest(conn, "berlin")   # 'location' concept
    _ingest(conn, "coffee")   # 'beverage' concept
    backfill_embeddings(conn, "s1", client=emb)
    ids = {r["value"]: r["claim_id"] for r in conn.execute("SELECT value, claim_id FROM claims").fetchall()}
    b, c = ids["berlin"], ids["coffee"]

    # Two queries that share NO word with either fact. The semantic ranking must
    # FLIP with the query's concept -- BM25 (zero word overlap) cannot do this, so
    # the flip isolates true meaning-based matching from every lexical/tiebreak signal.
    loc: dict[str, dict[str, float]] = {}
    retrieve_hybrid(conn, session_id="s1", query="where do I reside",
                    top_k=10, embedding_client=emb, explain=loc)
    bev: dict[str, dict[str, float]] = {}
    retrieve_hybrid(conn, session_id="s1", query="what do I drink each morning",
                    top_k=10, embedding_client=emb, explain=bev)

    assert loc[b]["semantic"] > loc[c]["semantic"], "location query -> location fact (berlin)"
    assert bev[c]["semantic"] > bev[b]["semantic"], "beverage query -> beverage fact (coffee)"


def test_pass2_supersedes_by_concept_not_words():
    conn = _conn()
    emb = ConceptEmbedder()
    sem = SemanticSupersession(emb, threshold=0.8)

    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="I live in Munich these days",
        extractor=PassthroughExtractor([]),
    )
    tid = conn.execute("SELECT turn_id FROM turns LIMIT 1").fetchone()["turn_id"]
    insert_fact(conn, session_id="s1", source_turn_id=tid, confidence=0.9,
                text="I live in Munich these days")

    on_new_turn(
        conn, session_id="s2", speaker=Speaker.USER, text="I have moved to Berlin now",
        extractor=PassthroughExtractor([]),
    )
    tid2 = conn.execute("SELECT turn_id FROM turns WHERE session_id='s2' LIMIT 1").fetchone()["turn_id"]
    # NL-only fact (no predicate) -> Pass-2 NL mode
    new = insert_fact(conn, session_id="s1", source_turn_id=tid2, confidence=0.9,
                      text="I have moved to Berlin now")

    # Munich and Berlin share NO word; only the 'location' concept links them.
    edge = sem.detect(conn, new, new_turn_text="I have moved to Berlin now")
    assert edge is not None, "Pass-2 supersedes by MEANING (location), not shared words"
