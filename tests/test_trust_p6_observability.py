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
    # Low-trust web source tries to override the user's stated residence (a single-
    # valued attribute → deterministic supersession). See the note in
    # test_trust_p4_antipoisoning: the old "dark mode"/"dark theme" vehicle relied on
    # an over-loose Jaccard since tightened; the trust-guard block+audit path is intact.
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="I live in Portland",
        extractor=PassthroughExtractor(
            [{"subject": "user", "predicate": "user_fact", "value": "lives in Portland", "confidence": 0.9}]),
    )
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, speaker, text, ts, source_type, extraction_status)"
        " VALUES ('tu_web','s1','user','a web page claims the user lives in Denver',2000,'browser','done')"
    )
    low = insert_claim(conn, session_id="s1", subject="user", predicate="user_fact",
                       value="lives in Denver", confidence=0.9, source_turn_id="tu_web")
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


_DAY_NS = 86_400 * 1_000_000_000


def _age_active(conn, days):
    import time
    t = time.time_ns() - days * _DAY_NS
    conn.execute(
        "UPDATE claims SET created_ts=?, valid_from_ts=?, event_ts=NULL"
        " WHERE status IN ('active','confirmed')", (t, t))


def test_staleness_respects_stable_window():
    conn = _conn()
    _store(conn, "user", "hiking", "s1", "default")  # stable slot (no supersessions)

    _age_active(conn, 100)  # 100d < 365d stable window -> fresh
    assert handle_memory_trust_status(conn)["staleness"]["stale"] == 0

    _age_active(conn, 400)  # 400d > 365d -> stale
    st = handle_memory_trust_status(conn)
    assert st["staleness"]["stale"] == 1 and st["staleness"]["stale_fraction"] == 1.0


def test_staleness_window_is_shorter_for_volatile_slots():
    conn = _conn()
    _store(conn, "user", "current", "s1", "default")
    active = conn.execute(
        "SELECT claim_id, source_turn_id FROM claims WHERE status IN ('active','confirmed')"
    ).fetchone()
    # manufacture 3 supersessions on the (user, user_fact) slot -> volatile
    for i in range(3):
        conn.execute(
            "INSERT INTO claims (claim_id, session_id, text, subject, predicate, value,"
            " confidence, source_turn_id, status, created_ts)"
            " VALUES (?, 's1', 'x', 'user', 'user_fact', ?, 0.9, ?, 'superseded', 1)",
            (f"old{i}", f"v{i}", active["source_turn_id"]))
        conn.execute(
            "INSERT INTO supersession_edges (edge_id, old_claim_id, new_claim_id, edge_type, created_ts)"
            " VALUES (?, ?, ?, 'semantic_replace', 1)", (f"e{i}", f"old{i}", active["claim_id"]))

    _age_active(conn, 30)  # 30d > 14d volatile window -> stale (would be fresh if stable)
    assert handle_memory_trust_status(conn)["staleness"]["stale"] == 1
