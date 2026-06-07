"""InlineQueue mechanics: deferred extraction runs on drain(), failures isolated.

Uses a fake deferrable extractor (pure Python, no LLM / no network) so the queue
behaviour is asserted deterministically — honouring the "zero model downloads in
CI" rule.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memcontext.claims import insert_turn, new_turn_id, now_ns
from memcontext.extraction_queue import ExtractionJob, InlineQueue, ThreadedQueue
from memcontext.on_new_turn import ExtractedClaim
from memcontext.schema import ExtractionStatus, Speaker, Turn, open_database


class _FakeDeferrableExtractor:
    """Stand-in for LLMExtractor: deferrable, but pure-Python and instant."""

    is_deferrable = True

    def __init__(self, claims: list[tuple[str, str, str, float]]) -> None:
        self._claims = claims

    def __call__(self, turn: Turn) -> list[ExtractedClaim]:
        return [
            ExtractedClaim(subject=s, predicate=p, value=v, confidence=c)
            for (s, p, v, c) in self._claims
        ]


def _conn() -> sqlite3.Connection:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _episode(conn: sqlite3.Connection, sid: str, text: str) -> Turn:
    turn = Turn(
        turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER,
        text=text, ts=now_ns(),
    )
    insert_turn(conn, turn)
    return turn


def _status(conn: sqlite3.Connection, turn_id: str) -> str:
    return conn.execute(
        "SELECT extraction_status FROM turns WHERE turn_id = ?", (turn_id,)
    ).fetchone()["extraction_status"]


def _claim_count(conn: sqlite3.Connection, sid: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM claims WHERE session_id = ?", (sid,)
    ).fetchone()["n"]


def test_enqueue_defers_until_drain():
    conn = _conn()
    sid = "s1"
    ext = _FakeDeferrableExtractor([("user", "user_preference", "dark mode", 0.9)])
    q = InlineQueue(conn, extractor=ext)

    ep = _episode(conn, sid, "I prefer dark mode")
    q.enqueue(ep.turn_id, sid)

    # Nothing extracted yet — the job only ran-records, it didn't process.
    assert _claim_count(conn, sid) == 0

    q.drain()
    # Now the facts exist and the episode is marked done.
    assert _claim_count(conn, sid) == 1
    assert _status(conn, ep.turn_id) == ExtractionStatus.DONE.value


def test_drain_marks_skipped_when_no_facts():
    conn = _conn()
    sid = "s2"
    ext = _FakeDeferrableExtractor([])  # extractor yields nothing
    q = InlineQueue(conn, extractor=ext)
    ep = _episode(conn, sid, "small talk")
    q.enqueue(ep.turn_id, sid)
    q.drain()
    assert _claim_count(conn, sid) == 0
    assert _status(conn, ep.turn_id) == ExtractionStatus.SKIPPED.value


def test_missing_episode_job_does_not_crash():
    conn = _conn()
    sid = "s3"
    ext = _FakeDeferrableExtractor([("user", "user_preference", "x", 0.9)])
    q = InlineQueue(conn, extractor=ext)
    q.enqueue("tu_does_not_exist", sid)
    q.drain()  # must not raise
    assert _claim_count(conn, sid) == 0


def test_one_failing_job_does_not_stop_the_rest():
    conn = _conn()
    sid = "s4"

    class _BoomOnFirst:
        is_deferrable = True

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, turn: Turn) -> list[ExtractedClaim]:
            self.calls += 1
            if "boom" in turn.text:
                raise RuntimeError("extractor blew up")
            return [ExtractedClaim("user", "user_preference", "ok", 0.9)]

    ext = _BoomOnFirst()
    q = InlineQueue(conn, extractor=ext)
    bad = _episode(conn, sid, "boom turn")
    good = _episode(conn, sid, "good turn")
    q.enqueue(bad.turn_id, sid)
    q.enqueue(good.turn_id, sid)

    q.drain()  # bad job raises inside run_extraction, caught; good job still runs
    assert ext.calls == 2
    assert _claim_count(conn, sid) == 1


def test_extraction_job_is_a_value():
    job = ExtractionJob("tu_1", "s")
    assert job.episode_id == "tu_1" and job.session_id == "s"


# --- ThreadedQueue (requires a file-backed DB; worker runs on its own thread) --


def test_threaded_queue_rejects_memory_db():
    ext = _FakeDeferrableExtractor([])
    with pytest.raises(ValueError, match="file-backed"):
        ThreadedQueue(":memory:", extractor=ext)


def test_threaded_queue_processes_on_background_thread(tmp_path: Path):
    path = str(tmp_path / "tq.db")
    conn = open_database(path)
    conn.row_factory = sqlite3.Row
    sid = "s1"
    ep = _episode(conn, sid, "I prefer dark mode")

    ext = _FakeDeferrableExtractor([("user", "user_preference", "dark mode", 0.9)])
    tq = ThreadedQueue(path, extractor=ext)
    try:
        tq.enqueue(ep.turn_id, sid)
        tq.drain()  # block until the worker finishes the queued job
        # The worker wrote facts (on its own connection) to the same file DB;
        # the main connection sees them after the commit.
        assert _claim_count(conn, sid) == 1
        assert _status(conn, ep.turn_id) == ExtractionStatus.DONE.value
    finally:
        tq.close()
        conn.close()


def test_threaded_queue_close_joins_cleanly(tmp_path: Path):
    path = str(tmp_path / "tq2.db")
    conn = open_database(path)
    conn.row_factory = sqlite3.Row
    ext = _FakeDeferrableExtractor([])
    tq = ThreadedQueue(path, extractor=ext)
    assert tq.worker_alive
    tq.close()  # must return (worker thread joined) without hanging
    assert not tq.worker_alive
    conn.close()
