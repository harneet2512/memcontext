"""memory_correct must respect the supersession chain and carry its own provenance.

Regressions for three defects in ``handle_memory_correct``:
- correcting an already-SUPERSEDED claim forked the chain (two active values);
- correcting a DISMISSED claim resurrected it;
- the corrected claim reused the OLD claim's source turn (wrong quote, wrong
  trust) and the old claim never got a ``valid_until_ts``.
"""
from __future__ import annotations

import sqlite3

import pytest

from memcontext.claims import get_claim, get_turn, insert_fact, insert_turn, new_turn_id, now_ns
from memcontext.mcp_tools import handle_memory_correct, handle_memory_store, handle_memory_trace
from memcontext.schema import ClaimStatus, SourceType, Speaker, Turn, open_database
from memcontext.source_trust import TRUSTED_USER

SUBJ = "project"
PRED = "user_fact"


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _store(conn: sqlite3.Connection, value: str, *, sid: str = "s1", namespace: str = "default") -> str:
    out = handle_memory_store(
        conn, text=f"The project {value}", session_id=sid, namespace=namespace,
        claims=[{"subject": SUBJ, "predicate": PRED, "value": value, "confidence": 0.9}],
    )
    return out["claim_ids"][0]


def _active_values(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT value FROM claims WHERE subject = ? AND predicate = ?"
        " AND status IN ('active','confirmed','audited')",
        (SUBJ, PRED),
    ).fetchall()
    return sorted(r["value"] for r in rows)


def _superseded_pair(conn: sqlite3.Connection) -> tuple[str, str]:
    old_id = _store(conn, "uses MySQL 5.7")
    head_id = _store(conn, "uses MySQL 8.0")
    old, head = get_claim(conn, old_id), get_claim(conn, head_id)
    assert old is not None and old.status is ClaimStatus.SUPERSEDED  # precondition
    assert head is not None and head.status is ClaimStatus.ACTIVE
    return old_id, head_id


def test_correcting_superseded_claim_redirects_to_active_head(conn):
    old_id, head_id = _superseded_pair(conn)

    result = handle_memory_correct(
        conn, claim_id=old_id, action="correct", new_value="uses MySQL 5.6",
    )

    assert "error" not in result, result
    assert result["corrected_claim_id"] == head_id
    assert result["old_claim_id"] == head_id
    assert result["requested_claim_id"] == old_id
    # One slot, one current value: no fork.
    assert _active_values(conn) == ["uses MySQL 5.6"]
    head = get_claim(conn, head_id)
    assert head is not None and head.status is ClaimStatus.SUPERSEDED
    edge = conn.execute(
        "SELECT old_claim_id, edge_type FROM supersession_edges WHERE new_claim_id = ?",
        (result["new_claim_id"],),
    ).fetchone()
    assert edge["old_claim_id"] == head_id
    assert edge["edge_type"] == "user_correction"


def test_correcting_dismissed_claim_is_refused(conn):
    cid = _store(conn, "uses MySQL 5.7")
    handle_memory_correct(conn, claim_id=cid, action="dismiss")
    n_before = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]

    result = handle_memory_correct(
        conn, claim_id=cid, action="correct", new_value="uses MySQL 5.6",
    )

    assert "error" in result
    assert "dismissed" in result["error"]
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == n_before
    claim = get_claim(conn, cid)
    assert claim is not None and claim.status is ClaimStatus.DISMISSED
    assert _active_values(conn) == []


def test_correcting_superseded_claim_whose_head_was_dismissed_is_refused(conn):
    old_id, head_id = _superseded_pair(conn)
    handle_memory_correct(conn, claim_id=head_id, action="dismiss")
    n_before = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]

    result = handle_memory_correct(
        conn, claim_id=old_id, action="correct", new_value="uses MySQL 5.6",
    )

    assert "error" in result
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == n_before
    assert _active_values(conn) == []


def test_correction_gets_its_own_user_turn_in_same_session_and_namespace(conn):
    cid = _store(conn, "uses MySQL 5.7", sid="s9", namespace="tenant_a")
    old = get_claim(conn, cid)
    assert old is not None

    result = handle_memory_correct(
        conn, claim_id=cid, action="correct", new_value="uses MySQL 5.6",
    )

    new = get_claim(conn, result["new_claim_id"])
    assert new is not None
    assert new.source_turn_id != old.source_turn_id
    assert result["source_turn_id"] == new.source_turn_id
    turn = get_turn(conn, new.source_turn_id)
    assert turn is not None
    assert turn.speaker is Speaker.USER
    assert turn.session_id == "s9"
    assert "uses MySQL 5.6" in turn.text
    ns = conn.execute(
        "SELECT namespace FROM turns WHERE turn_id = ?", (turn.turn_id,),
    ).fetchone()[0]
    assert ns == "tenant_a"
    # Provenance quotes the corrected value, not the stale source text.
    assert new.char_start is not None and new.char_end is not None
    assert turn.text[new.char_start:new.char_end] == "uses MySQL 5.6"
    trace = handle_memory_trace(conn, claim_id=new.claim_id)
    assert "uses MySQL 5.6" in trace["source_turn"]["text"]


def test_correction_does_not_inherit_low_trust_of_old_source(conn):
    web_turn = Turn(
        turn_id=new_turn_id(), session_id="s2", speaker=Speaker.USER,
        text="Scraped: the project uses MySQL 5.7", ts=now_ns(),
        source_type=SourceType.BROWSER,
    )
    insert_turn(conn, web_turn)
    old = insert_fact(
        conn, session_id="s2", source_turn_id=web_turn.turn_id, confidence=0.6,
        subject=SUBJ, predicate=PRED, value="uses MySQL 5.7",
    )

    result = handle_memory_correct(
        conn, claim_id=old.claim_id, action="correct", new_value="uses MySQL 5.6",
    )

    trust = conn.execute(
        "SELECT source_trust FROM claim_metadata WHERE claim_id = ?",
        (result["new_claim_id"],),
    ).fetchone()[0]
    assert trust == TRUSTED_USER


def test_corrected_claim_gets_valid_until(conn):
    cid = _store(conn, "uses MySQL 5.7")

    result = handle_memory_correct(
        conn, claim_id=cid, action="correct", new_value="uses MySQL 5.6",
    )

    old = get_claim(conn, cid)
    assert old is not None and old.valid_until_ts is not None
    edge_ts = conn.execute(
        "SELECT created_ts FROM supersession_edges WHERE edge_id = ?", (result["edge_id"],),
    ).fetchone()[0]
    assert old.valid_until_ts == edge_ts
    assert old.valid_from_ts is None or old.valid_from_ts < old.valid_until_ts


def test_nl_only_correction_turn_text(conn):
    turn = Turn(
        turn_id=new_turn_id(), session_id="s3", speaker=Speaker.USER,
        text="some context", ts=now_ns(),
    )
    insert_turn(conn, turn)
    fact = insert_fact(
        conn, session_id="s3", source_turn_id=turn.turn_id, confidence=0.8,
        text="the old note",
    )

    result = handle_memory_correct(
        conn, claim_id=fact.claim_id, action="correct", new_value="the corrected note",
    )

    new = get_claim(conn, result["new_claim_id"])
    assert new is not None and new.text == "the corrected note"
    new_turn = get_turn(conn, new.source_turn_id)
    assert new_turn is not None
    assert new_turn.text == "Correction: the corrected note"
