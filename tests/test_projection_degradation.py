"""Projection degradation: episodes back the projection when facts are absent.

The graceful-degradation floor — a session whose facts are absent or still
pending (async extraction) must still project something (its recent episodes),
and switch to facts once they land.
"""
from __future__ import annotations

import sqlite3

from memcontext.claims import insert_fact, insert_turn, new_turn_id
from memcontext.projections import rebuild_active_projection
from memcontext.schema import Speaker, Turn, open_database


def _conn() -> sqlite3.Connection:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _episode(conn: sqlite3.Connection, sid: str, text: str, ts: int) -> Turn:
    turn = Turn(
        turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER,
        text=text, ts=ts,
    )
    insert_turn(conn, turn)
    return turn


def test_degrades_to_recent_episodes_when_no_facts():
    conn = _conn()
    sid = "s1"
    _episode(conn, sid, "first episode, no facts extracted yet", ts=100)
    _episode(conn, sid, "second episode, also pending", ts=200)

    proj = rebuild_active_projection(conn, sid)
    assert proj.claims == ()  # no facts
    assert proj.is_episode_backed
    assert len(proj.episodes) == 2
    # Most-recent first.
    assert proj.episodes[0].text == "second episode, also pending"


def test_uses_facts_when_present_and_no_episode_fallback():
    conn = _conn()
    sid = "s2"
    turn = _episode(conn, sid, "I prefer dark mode", ts=100)
    insert_fact(
        conn, session_id=sid, source_turn_id=turn.turn_id, confidence=0.9,
        subject="user", predicate="user_preference", value="dark mode",
    )
    proj = rebuild_active_projection(conn, sid)
    assert len(proj.claims) == 1
    assert proj.episodes == ()  # no degradation when facts exist
    assert not proj.is_episode_backed


def test_switches_from_episodes_to_facts_after_extraction():
    conn = _conn()
    sid = "s3"
    turn = _episode(conn, sid, "I prefer dark mode", ts=100)

    # Before extraction: episode-backed.
    proj_before = rebuild_active_projection(conn, sid)
    assert proj_before.is_episode_backed and len(proj_before.episodes) == 1

    # A fact lands (as async extraction would produce).
    insert_fact(
        conn, session_id=sid, source_turn_id=turn.turn_id, confidence=0.9,
        subject="user", predicate="user_preference", value="dark mode",
    )
    proj_after = rebuild_active_projection(conn, sid)
    assert len(proj_after.claims) == 1
    assert not proj_after.is_episode_backed
    assert proj_after.episodes == ()


def test_empty_session_projects_nothing():
    conn = _conn()
    proj = rebuild_active_projection(conn, "empty")
    assert proj.claims == () and proj.episodes == ()
    assert not proj.is_episode_backed


def test_episode_fallback_k_caps_count():
    conn = _conn()
    sid = "s5"
    for i in range(15):
        _episode(conn, sid, f"episode {i}", ts=100 + i)
    proj = rebuild_active_projection(conn, sid, episode_fallback_k=5)
    assert len(proj.episodes) == 5
    # The 5 most recent (ts 110..114), most-recent first.
    assert proj.episodes[0].text == "episode 14"
