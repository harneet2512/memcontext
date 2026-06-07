"""Trust layer Phase 6 — trust observability.

Measures whether the trust/governance layer is working (not just recall): the
metrics move when the underlying trust/forget/drift/tenant state changes.
"""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim
from memcontext.extractors import PassthroughExtractor
from memcontext.forgetting import forget
from memcontext.mcp_tools import handle_memory_store, handle_memory_trust_status
from memcontext.on_new_turn import on_new_turn
from memcontext.supersession import detect_pass1
from memcontext.schema import Speaker, open_database


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _store(conn, subject, value, session_id, namespace):
    handle_memory_store(
        conn, text=f"{subject} likes {value}", session_id=session_id,
        claims=[{"subject": subject, "predicate": "user_fact", "value": value, "confidence": 0.9}],
        namespace=namespace,
    )


def test_trust_status_distribution_and_tenants():
    conn = _conn()
    _store(conn, "alice", "coffee", "s1", "tenantA")
    _store(conn, "bob", "tea", "s2", "tenantB")
    cid = conn.execute("SELECT claim_id FROM claims WHERE subject='alice'").fetchone()["claim_id"]
    conn.execute("UPDATE claim_metadata SET source_trust=0.3 WHERE claim_id=?", (cid,))  # quarantine

    st = handle_memory_trust_status(conn)
    assert st["active_claims"] == 2
    assert st["source_trust"]["quarantined"] == 1
    assert st["quarantined_fraction"] == 0.5
    assert st["tenant_count"] == 2
    assert set(st["namespaces"]) == {"tenantA", "tenantB"}


def test_trust_status_forget_and_drift_audit():
    conn = _conn()
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="I prefer dark mode",
        extractor=PassthroughExtractor(
            [{"subject": "user", "predicate": "user_fact", "value": "dark mode", "confidence": 0.9}]),
    )
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, speaker, text, ts, source_type, extraction_status)"
        " VALUES ('tu_web','s1','user','a web page said dark theme',2000,'browser','done')"
    )
    low = insert_claim(conn, session_id="s1", subject="user", predicate="user_fact",
                       value="dark theme", confidence=0.9, source_turn_id="tu_web")
    detect_pass1(conn, low)  # blocked low-trust override -> drift event

    forget(conn, subject="user")  # erase + audit

    st = handle_memory_trust_status(conn)
    assert st["forgetting"]["drift_blocked"] == 1
    assert st["forgetting"]["forget_actions"] == 1
    assert st["forgetting"]["claims_erased"] >= 1


def test_trust_status_cli(tmp_path):
    from click.testing import CliRunner

    from memcontext.cli import main

    db = str(tmp_path / "m.db")
    conn = open_database(db)
    _store(conn, "alice", "coffee", "s1", "default")
    conn.commit()
    conn.close()

    r = CliRunner().invoke(main, ["trust-status", "--db", db])
    assert r.exit_code == 0, r.output
    assert '"active_claims"' in r.output and '"source_trust"' in r.output
