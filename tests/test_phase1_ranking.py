"""Phase 1 (product-grade RAG): importance is a live ranking signal + observable.

RED before: retrieve_hybrid never read claim_metadata.importance_score, so a
computed signal was discarded at ranking time.
"""
from __future__ import annotations

import sqlite3

from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.retrieval import retrieve_hybrid
from memcontext.schema import SCHEMA_VERSION, Speaker, open_database


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


def test_v5_usage_columns_exist():
    """Schema v5 adds access_count + last_accessed_ts to claim_metadata."""
    conn = _conn()
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(claim_metadata)").fetchall()}
    assert {"access_count", "last_accessed_ts"} <= cols
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_serving_a_fact_reinforces_usage_and_reports_tokens():
    """handle_memory_query bumps usage for served facts + returns a token report."""
    from memcontext.mcp_tools import handle_memory_query

    conn = _conn()
    _ingest(conn, "alice", "coffee")
    cid = conn.execute("SELECT claim_id FROM claims").fetchone()["claim_id"]
    before = conn.execute(
        "SELECT COALESCE(access_count, 0) FROM claim_metadata WHERE claim_id = ?", (cid,)
    ).fetchone()[0]

    res = handle_memory_query(conn, query="coffee", session_id="s1", top_k=5)

    after = conn.execute(
        "SELECT access_count FROM claim_metadata WHERE claim_id = ?", (cid,)
    ).fetchone()[0]
    assert after == before + 1, "serving a fact reinforces its usage (RED before v5)"

    tr = res["token_report"]
    assert tr["total_tokens"] == tr["fact_tokens"] + tr["episode_tokens"]
    assert tr["served_items"] == len(res["claims"]) + len(res["episodes"])


def test_query_debug_exposes_ranking_breakdown():
    """debug=True surfaces the per-claim signal breakdown through the door."""
    from memcontext.mcp_tools import handle_memory_query

    conn = _conn()
    _ingest(conn, "alice", "coffee")
    res = handle_memory_query(conn, query="coffee", session_id="s1", top_k=5, debug=True)
    assert "ranking" in res and res["claims"]
    cid = res["claims"][0]["claim_id"]
    assert {"importance", "usage", "final"} <= set(res["ranking"][cid])


def test_cli_query_is_unified_and_reindex_importance_works(tmp_path):
    """cli query now serves the unified two-tier path; reindex-importance wires
    the previously-dormant recompute_all_importance."""
    from click.testing import CliRunner

    from memcontext.cli import main

    db = str(tmp_path / "m.db")
    conn = open_database(db)
    conn.row_factory = sqlite3.Row
    _ingest(conn, "alice", "coffee")
    conn.commit()
    conn.close()

    runner = CliRunner()
    r = runner.invoke(main, ["query", "coffee", "--db", db, "--session", "s1"])
    assert r.exit_code == 0, r.output
    # unified output shape (kind=fact|episode) — was a facts-only claim shape before
    assert '"kind"' in r.output

    r = runner.invoke(main, ["reindex-importance", "--db", db])
    assert r.exit_code == 0, r.output
    assert "Recomputed importance" in r.output


def test_importance_is_computed_at_ingest_not_only_on_supersession():
    """LIPI fix: every new claim gets importance computed at ingest, so the
    importance ranking channel isn't inert (flat 0.5) for never-superseded claims."""
    conn = _conn()
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="alice strong fact",
        extractor=PassthroughExtractor(
            [{"subject": "alice", "predicate": "user_fact", "value": "x", "confidence": 0.99}]),
    )
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="bob weak fact",
        extractor=PassthroughExtractor(
            [{"subject": "bob", "predicate": "user_fact", "value": "y", "confidence": 0.20}]),
    )
    scores = [r[0] for r in conn.execute("SELECT importance_score FROM claim_metadata").fetchall()]
    assert len(scores) == 2
    # computed at ingest (no supersession here): values differ or aren't the flat default
    assert len({round(s, 4) for s in scores}) > 1 or any(abs(s - 0.5) > 1e-6 for s in scores)
