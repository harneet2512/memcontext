"""Cycle D (working context): assemble the task-relevant memory for a session
within a token budget, cued by recent turns -- scoped + bounded, beating a dump
of all active memory on precision + tokens.
"""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_claim
from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.schema import ClaimStatus, Speaker, open_database
from memcontext.working_context import build_working_context


def _conn():
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    return c


def _say(conn, value, session="s1"):
    on_new_turn(
        conn, session_id=session, speaker=Speaker.USER,
        text=f"I use {value} for the project",
        extractor=PassthroughExtractor(
            [{"subject": "user", "predicate": "user_fact", "value": value, "confidence": 0.9}]),
    )


def test_working_context_is_scoped_and_budget_bounded():
    conn = _conn()
    for v in ("postgres", "redis", "kafka", "docker", "nginx"):
        _say(conn, v)
    total = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE session_id='s1'"
        " AND status IN ('active','confirmed','audited')"
    ).fetchone()[0]

    ctx = build_working_context(conn, "s1", token_budget=10, recent_turns=5)

    assert 0 < ctx.tokens_used <= 10, "non-empty, respects the token budget"
    assert ctx.excluded_for_budget > 0, "bounded: not everything returned (beats all-active)"
    assert ctx.total_active == total
    assert ctx.recent_turn_ids  # salient_entities is best-effort (see dedicated test)


def test_working_context_captures_salient_entities():
    conn = _conn()
    on_new_turn(
        conn, session_id="s1", speaker=Speaker.USER, text="I work with Alice on the launch",
        extractor=PassthroughExtractor(
            [{"subject": "user", "predicate": "user_fact", "value": "alice", "confidence": 0.9}]),
    )
    ctx = build_working_context(conn, "s1", token_budget=2000)
    assert "alice" in ctx.salient_entities, "proper-noun entities in recent turns are captured"


def test_working_context_excludes_superseded():
    conn = _conn()
    _say(conn, "munich")
    turn = conn.execute("SELECT source_turn_id FROM claims LIMIT 1").fetchone()["source_turn_id"]
    insert_claim(
        conn, session_id="s1", subject="user", predicate="user_fact",
        value="berlin", confidence=0.9, source_turn_id=turn, status=ClaimStatus.SUPERSEDED)

    ctx = build_working_context(conn, "s1", token_budget=2000)
    texts = " ".join((h.text or "") for h, _ in ctx.facts).lower()
    assert "berlin" not in texts, "superseded fact stays out of the working context"


def test_door_exposes_working_context():
    from memcontext.mcp_tools import handle_memory_working_context

    conn = _conn()
    _say(conn, "postgres")
    res = handle_memory_working_context(conn, session_id="s1", token_budget=2000)
    assert set(res) >= {
        "session_id", "facts", "salient_entities", "token_budget",
        "tokens_used", "total_active", "included", "excluded_for_budget",
    }
