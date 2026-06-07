"""Trust layer Phase 2 — cascade-consistent, verifiable deletion (right-to-be-forgotten).

The test most RAG systems fail: after forgetting a subject, NO residual content
remains in any table (claims, sidecars, supersession edges, summaries, output
sentences, orphaned turns) — and the deletion is audited and provable.
"""
from __future__ import annotations

import json
import sqlite3

from memcontext.digests import build_session_digest, store_digest
from memcontext.extractors import PassthroughExtractor
from memcontext.forgetting import forget
from memcontext.on_new_turn import on_new_turn
from memcontext.provenance import OutputSection, insert_output_sentence
from memcontext.schema import Speaker, open_database


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _ingest(conn, subject, value):
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text=f"{subject} likes {value}",
        extractor=PassthroughExtractor(
            [{"subject": subject, "predicate": "user_fact", "value": value, "confidence": 0.9}]),
    )


def _orphans(conn, table):
    return conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE claim_id NOT IN (SELECT claim_id FROM claims)"
    ).fetchone()[0]


def _count(conn, sql, *args):
    return conn.execute(sql, args).fetchone()[0]


def test_forget_leaves_zero_residual_and_is_audited():
    conn = _conn()
    _ingest(conn, "alice", "coffee")
    _ingest(conn, "alice", "berlin")
    _ingest(conn, "bob", "tea")  # co-subject survivor

    # build a derived summary (references alice) + record an output sentence citing alice
    store_digest(conn, build_session_digest(conn, "s1"))
    acid = conn.execute("SELECT claim_id FROM claims WHERE subject='alice' LIMIT 1").fetchone()[0]
    insert_output_sentence(conn, session_id="s1", section=OutputSection.SUMMARY,
                           ordinal=0, text="Alice likes coffee.", source_claim_ids=[acid])

    res = forget(conn, subject="alice")
    assert res["forgotten"] == 2 and res["decision_id"]

    # 1. the claims themselves are gone
    assert _count(conn, "SELECT COUNT(*) FROM claims WHERE subject='alice'") == 0
    # 2. FK cascade: no orphaned sidecar rows
    for t in ("claim_metadata", "claim_entities", "claim_embeddings"):
        assert _orphans(conn, t) == 0, f"{t} cascaded"
    # 3. no supersession edge dangles to a forgotten claim
    assert _count(
        conn,
        "SELECT COUNT(*) FROM supersession_edges"
        " WHERE old_claim_id NOT IN (SELECT claim_id FROM claims)"
        " OR new_claim_id NOT IN (SELECT claim_id FROM claims)") == 0
    # 4. the stale summary + the citing output sentence are removed (no residual content)
    assert _count(conn, "SELECT COUNT(*) FROM session_digests") == 0
    assert _count(conn, "SELECT COUNT(*) FROM output_sentences") == 0
    # 5. alice's orphaned source turns are gone; bob's claim AND turn survive
    assert _count(conn, "SELECT COUNT(*) FROM claims WHERE subject='bob'") == 1
    assert _count(conn, "SELECT COUNT(*) FROM turns") == 1
    # 6. verifiable audit row with the full snapshot
    row = conn.execute(
        "SELECT kind, target_type, target_id, claim_state_snapshot FROM decisions"
    ).fetchone()
    assert (row["kind"], row["target_type"], row["target_id"]) == ("forget", "subject", "alice")
    assert len(json.loads(row["claim_state_snapshot"])) == 2


def test_forget_by_claim_id_is_surgical():
    conn = _conn()
    _ingest(conn, "user", "x")
    _ingest(conn, "user", "y")
    cid = conn.execute("SELECT claim_id FROM claims WHERE value='x'").fetchone()["claim_id"]
    res = forget(conn, claim_id=cid)
    assert res["forgotten"] == 1
    assert _count(conn, "SELECT COUNT(*) FROM claims WHERE value='x'") == 0
    assert _count(conn, "SELECT COUNT(*) FROM claims WHERE value='y'") == 1  # untouched


def test_forget_via_mcp_door():
    from memcontext.mcp_tools import handle_memory_forget
    conn = _conn()
    _ingest(conn, "alice", "coffee")
    res = handle_memory_forget(conn, subject="alice")
    assert res["forgotten"] == 1 and res["decision_id"]
    assert _count(conn, "SELECT COUNT(*) FROM claims WHERE subject='alice'") == 0


def test_forget_via_cli(tmp_path):
    from click.testing import CliRunner

    from memcontext.cli import main

    db = str(tmp_path / "m.db")
    conn = open_database(db)
    conn.row_factory = sqlite3.Row
    _ingest(conn, "alice", "coffee")
    conn.commit()
    conn.close()

    r = CliRunner().invoke(main, ["forget", "--db", db, "--subject", "alice"])
    assert r.exit_code == 0, r.output
    assert '"forgotten": 1' in r.output
    conn2 = open_database(db)
    assert conn2.execute("SELECT COUNT(*) FROM claims WHERE subject='alice'").fetchone()[0] == 0


def test_forget_redacts_shared_surviving_turn():
    """A turn shared between a forgotten subject and a survivor keeps no raw PII."""
    conn = _conn()
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER,
        text="alice likes coffee and bob likes tea",
        extractor=PassthroughExtractor([
            {"subject": "alice", "predicate": "user_fact", "value": "coffee", "confidence": 0.9},
            {"subject": "bob", "predicate": "user_fact", "value": "tea", "confidence": 0.9},
        ]),
    )
    forget(conn, subject="alice")

    row = conn.execute("SELECT text FROM turns").fetchone()
    assert row is not None  # the shared turn survives (bob's claim remains)
    text = row["text"].lower()
    assert "alice" not in text and "coffee" not in text  # forgotten PII redacted
    assert "bob" in text and "tea" in text                # survivor intact
    assert "[redacted]" in text
    assert _count(conn, "SELECT COUNT(*) FROM claims WHERE subject='bob'") == 1
