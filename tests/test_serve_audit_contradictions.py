from __future__ import annotations

import sqlite3

from memcontext.claims import get_claim, insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import (
    handle_memory_contradictions,
    handle_memory_query,
    handle_memory_verify,
)
from memcontext.on_new_turn import on_new_turn
from memcontext.schema import ClaimStatus, EdgeType, Speaker, Turn, open_database
from memcontext.supersession import detect_pass1


def _conn() -> sqlite3.Connection:
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _turn(conn: sqlite3.Connection, session_id: str, speaker: Speaker, text: str) -> Turn:
    t = Turn(
        turn_id=new_turn_id(),
        session_id=session_id,
        speaker=speaker,
        text=text,
        ts=now_ns(),
        asr_confidence=None,
    )
    insert_turn(conn, t)
    return t


def test_memory_query_records_serve_events_and_verify_checks_them() -> None:
    conn = _conn()
    on_new_turn(
        conn,
        session_id="s1",
        speaker=Speaker.USER,
        text="I strongly prefer coffee during morning planning sessions",
        extractor=PassthroughExtractor(
            [{"subject": "user", "predicate": "user_preference", "value": "likes coffee"}]
        ),
    )
    cid = conn.execute("SELECT claim_id FROM claims").fetchone()["claim_id"]

    result = handle_memory_query(conn, session_id="s1", query="coffee", top_k=5)

    assert result["claims"][0]["claim_id"] == cid
    assert result["serve_event_ids"]
    assert conn.execute("SELECT COUNT(*) FROM serve_events").fetchone()[0] == 1
    assert handle_memory_verify(conn, session_id="s1", claim_ids=[cid])["verified"] is True
    assert handle_memory_verify(
        conn, session_id="other", claim_ids=[cid]
    )["verified"] is False


def test_contradiction_keeps_both_claims_active_and_is_reported() -> None:
    conn = _conn()
    t1 = _turn(conn, "s1", Speaker.ASSISTANT, "You live in Seattle")
    old = insert_claim(
        conn,
        session_id="s1",
        subject="user",
        predicate="user_fact",
        value="lives in Seattle",
        confidence=0.8,
        source_turn_id=t1.turn_id,
    )
    t2 = _turn(conn, "s1", Speaker.USER, "No, I live in Portland")
    new = insert_claim(
        conn,
        session_id="s1",
        subject="user",
        predicate="user_fact",
        value="lives in Portland",
        confidence=0.9,
        source_turn_id=t2.turn_id,
    )

    edge = detect_pass1(conn, new)

    assert edge is not None
    assert edge.edge_type == EdgeType.CONTRADICTS
    assert get_claim(conn, old.claim_id).status == ClaimStatus.ACTIVE
    assert get_claim(conn, new.claim_id).status == ClaimStatus.ACTIVE
    report = handle_memory_contradictions(conn, session_id="s1")
    assert report["count"] == 1
    assert report["contradictions"][0]["old"]["claim_id"] == old.claim_id
    assert report["contradictions"][0]["new"]["claim_id"] == new.claim_id
