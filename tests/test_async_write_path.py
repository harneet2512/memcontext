"""on_new_turn async branch: deferrable extractor never blocks the write path.

A deferrable extractor + a queue makes on_new_turn return immediately with the
episode stored (status `pending`) and NO facts; the facts appear only after the
queue drains. A non-deferrable extractor runs inline as before. Uses a fake
deferrable extractor (pure Python, no LLM / no network).
"""
from __future__ import annotations

import sqlite3

from memcontext.extraction_queue import InlineQueue
from memcontext.on_new_turn import ExtractedClaim, on_new_turn
from memcontext.schema import ExtractionStatus, Speaker, Turn, open_database


class _FakeDeferrableExtractor:
    is_deferrable = True

    def __call__(self, turn: Turn) -> list[ExtractedClaim]:
        return [ExtractedClaim("user", "user_preference", "dark mode", 0.9)]


class _SyncExtractor:
    is_deferrable = False

    def __call__(self, turn: Turn) -> list[ExtractedClaim]:
        return [ExtractedClaim("user", "user_preference", "dark mode", 0.9)]


def _conn() -> sqlite3.Connection:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _status(conn: sqlite3.Connection, turn_id: str) -> str:
    return conn.execute(
        "SELECT extraction_status FROM turns WHERE turn_id = ?", (turn_id,)
    ).fetchone()["extraction_status"]


def _claim_count(conn: sqlite3.Connection, sid: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM claims WHERE session_id = ?", (sid,)
    ).fetchone()["n"]


def test_deferrable_extractor_does_not_block_write_path():
    conn = _conn()
    sid = "s1"
    ext = _FakeDeferrableExtractor()
    q = InlineQueue(conn, extractor=ext)

    result = on_new_turn(
        conn, session_id=sid, speaker=Speaker.USER,
        text="I prefer dark mode", extractor=ext, queue=q,
    )
    # Returned immediately: episode stored, but NO facts yet.
    assert result.turn is not None
    assert result.created_claims == ()
    assert _claim_count(conn, sid) == 0
    assert _status(conn, result.turn.turn_id) == ExtractionStatus.PENDING.value

    # Facts materialise only after the queue is drained.
    q.drain()
    assert _claim_count(conn, sid) == 1
    assert _status(conn, result.turn.turn_id) == ExtractionStatus.DONE.value


def test_deferrable_without_queue_falls_back_to_inline():
    """Back-compat: a deferrable extractor with no queue still runs inline."""
    conn = _conn()
    sid = "s2"
    ext = _FakeDeferrableExtractor()
    result = on_new_turn(
        conn, session_id=sid, speaker=Speaker.USER,
        text="I prefer dark mode", extractor=ext,  # no queue
    )
    assert len(result.created_claims) == 1
    assert _claim_count(conn, sid) == 1


def test_non_deferrable_extractor_runs_inline():
    conn = _conn()
    sid = "s3"
    ext = _SyncExtractor()
    q = InlineQueue(conn, extractor=ext)  # queue present but extractor is sync
    result = on_new_turn(
        conn, session_id=sid, speaker=Speaker.USER,
        text="I prefer dark mode", extractor=ext, queue=q,
    )
    # Sync path: facts created inline, queue untouched, status structured.
    assert result.turn is not None
    assert len(result.created_claims) == 1
    assert _claim_count(conn, sid) == 1
    assert _status(conn, result.turn.turn_id) == ExtractionStatus.STRUCTURED.value
