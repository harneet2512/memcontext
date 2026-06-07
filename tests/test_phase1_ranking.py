"""Phase 1 (product-grade RAG): importance is a live ranking signal + observable.

RED before: retrieve_hybrid never read claim_metadata.importance_score, so a
computed signal was discarded at ranking time.
"""
from __future__ import annotations

import sqlite3

from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.retrieval import retrieve_hybrid
from memcontext.schema import Speaker, open_database


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _ingest(conn, subject, value):
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER,
        text=f"{subject} likes {value}",
        extractor=PassthroughExtractor(
            [{"subject": subject, "predicate": "user_fact",
              "value": value, "confidence": 0.9}]
        ),
    )


def test_importance_changes_ranking_and_is_observable():
    conn = _conn()
    # Two distinct-subject claims (no supersession) with identical query signals
    # for "coffee": same predicate, confidence; query matches both equally.
    _ingest(conn, "alice", "coffee")
    _ingest(conn, "bob", "coffee")
    rows = conn.execute("SELECT claim_id, subject FROM claims").fetchall()
    a = next(r["claim_id"] for r in rows if r["subject"] == "alice")
    b = next(r["claim_id"] for r in rows if r["subject"] == "bob")

    # Equalize everything that isn't importance (recency), then split importance.
    conn.execute("UPDATE claims SET created_ts=1000, valid_from_ts=1000")
    conn.execute("UPDATE claim_metadata SET importance_score=0.99 WHERE claim_id=?", (a,))
    conn.execute("UPDATE claim_metadata SET importance_score=0.01 WHERE claim_id=?", (b,))

    explain: dict[str, dict[str, float]] = {}
    hits = retrieve_hybrid(conn, session_id="s1", query="coffee", top_k=10,
                           explain=explain)
    ids = [c.claim_id for c, _ in hits]

    assert {a, b} <= set(ids), "both active claims retrieved"
    # Observability (item 6): per-signal breakdown incl importance + final.
    assert "importance" in explain[a] and "final" in explain[a]
    assert set(explain[a]) >= {"semantic", "bm25", "entity", "temporal", "importance", "final"}
    # Importance is wired and monotonic in importance_score.
    assert explain[a]["importance"] > explain[b]["importance"]
    # And it flips the order: higher importance ranks first when all else is equal.
    assert ids.index(a) < ids.index(b)


def test_explain_is_opt_in_default_off():
    """retrieve_hybrid without explain still returns the legacy shape (no crash)."""
    conn = _conn()
    _ingest(conn, "alice", "tea")
    hits = retrieve_hybrid(conn, session_id="s1", query="tea", top_k=5)
    assert hits and all(len(h) == 2 for h in hits)
