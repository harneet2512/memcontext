"""Tier-2 async fact-extraction queue.

The write path (`on_new_turn`) must never block on an LLM extractor. When the
injected extractor is deferrable (an `LLMExtractor`), the episode is stored +
embedded synchronously (Tier-1 floor) and a job is enqueued here; the actual
extract -> insert-facts -> supersede -> project tail runs later via
`on_new_turn.run_extraction`.

Two implementations:
- `InlineQueue` (default): deterministic, single-connection, `:memory:`-safe.
  Jobs accumulate and run on `drain()` — so the write path returns before facts
  exist (status `pending`), and a later `drain()` produces them. Used in tests
  and as the safe default.
- `ThreadedQueue` (added later): a background worker with its own file-backed
  connection, for CLI/server.

The "unit of work" (`run_extraction`) lives in `on_new_turn` to keep
`ExtractedClaim` and the claim-loop in one place; it is imported lazily here to
avoid an import cycle.
"""
from __future__ import annotations

import queue as _queue
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import structlog

if TYPE_CHECKING:
    from memcontext.event_bus import EventBus
    from memcontext.on_new_turn import ExtractorFn
    from memcontext.supersession_semantic import SemanticSupersession

log = structlog.get_logger(__name__)


class _Sentinel:
    """Distinct shutdown token so the worker queue's element type is a clean union."""


_SHUTDOWN: Final[_Sentinel] = _Sentinel()


@dataclass(frozen=True, slots=True)
class ExtractionJob:
    """A deferred fact-extraction unit: extract facts from this episode."""

    episode_id: str
    session_id: str


class ExtractionQueue(Protocol):
    """Minimal queue surface the write path depends on."""

    def enqueue(self, episode_id: str, session_id: str) -> None: ...
    def drain(self) -> None: ...
    def close(self) -> None: ...


class InlineQueue:
    """Synchronous, single-connection queue (the default).

    `enqueue` only records the job; `drain` runs them on the caller's connection.
    Deferring to `drain` is what lets the write path return with the episode in
    `pending` state and the facts appear only after an explicit drain — the same
    observable lifecycle as a real background worker, but deterministic and
    `:memory:`-safe for tests.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        extractor: ExtractorFn,
        semantic: SemanticSupersession | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self._conn = conn
        self._extractor = extractor
        self._semantic = semantic
        self._bus = bus
        self._jobs: list[ExtractionJob] = []

    def enqueue(self, episode_id: str, session_id: str) -> None:
        self._jobs.append(ExtractionJob(episode_id, session_id))
        log.debug("substrate.extraction_enqueued", episode_id=episode_id)

    def drain(self) -> None:
        from memcontext.on_new_turn import ExtractionStatus, run_extraction

        pending, self._jobs = self._jobs, []
        for job in pending:
            try:
                run_extraction(
                    self._conn,
                    episode_id=job.episode_id,
                    session_id=job.session_id,
                    extractor=self._extractor,
                    semantic=self._semantic,
                    bus=self._bus,
                    done_status=ExtractionStatus.DONE,
                )
            except Exception:  # noqa: BLE001 — one bad job must not stop the rest
                log.exception(
                    "substrate.extraction_job_failed", episode_id=job.episode_id
                )

    def close(self) -> None:  # nothing to tear down
        return None


class ThreadedQueue:
    """Background-worker queue for CLI/server — runs extraction off the main thread.

    The worker opens its OWN connection to the same file-backed DB (sqlite3
    connections are not shareable across threads, and a `:memory:` DB is private
    to its connection — so a file path is required). The main thread writes the
    episode in autocommit/WAL mode, so the worker's connection sees it before it
    pulls the job. Jobs are processed serially; one failing job does not stop the
    worker. `drain()` blocks until the currently-queued jobs are done; `close()`
    stops the worker and joins it.
    """

    def __init__(
        self,
        db_path: str,
        *,
        extractor: ExtractorFn,
        semantic: SemanticSupersession | None = None,
        bus: EventBus | None = None,
    ) -> None:
        if str(db_path) == ":memory:":
            raise ValueError(
                "ThreadedQueue requires a file-backed DB path; a ':memory:' "
                "database cannot be shared with a worker thread. Use InlineQueue."
            )
        self._db_path = str(db_path)
        self._extractor = extractor
        self._semantic = semantic
        self._bus = bus
        self._q: _queue.Queue[ExtractionJob | _Sentinel] = _queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, name="memcontext-extraction", daemon=True
        )
        self._thread.start()

    @property
    def worker_alive(self) -> bool:
        """Whether the background worker thread is still running."""
        return self._thread.is_alive()

    def enqueue(self, episode_id: str, session_id: str) -> None:
        self._q.put(ExtractionJob(episode_id, session_id))
        log.debug("substrate.extraction_enqueued", episode_id=episode_id)

    def drain(self) -> None:
        """Block until all currently-queued jobs have been processed."""
        self._q.join()

    def close(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self._q.put(_SHUTDOWN)
        self._thread.join()

    def _worker(self) -> None:
        from memcontext.on_new_turn import ExtractionStatus, run_extraction
        from memcontext.schema import open_database

        conn = open_database(self._db_path)
        try:
            while True:
                item = self._q.get()
                try:
                    if isinstance(item, _Sentinel):
                        return
                    run_extraction(
                        conn,
                        episode_id=item.episode_id,
                        session_id=item.session_id,
                        extractor=self._extractor,
                        semantic=self._semantic,
                        bus=self._bus,
                        done_status=ExtractionStatus.DONE,
                    )
                except Exception:  # noqa: BLE001 — isolate one bad job
                    log.exception(
                        "substrate.extraction_job_failed",
                        episode_id=getattr(item, "episode_id", None),
                    )
                finally:
                    self._q.task_done()
        finally:
            conn.close()


# A run_extraction-shaped callable, for typing the worker entry point.
RunExtractionFn = Callable[..., object]
