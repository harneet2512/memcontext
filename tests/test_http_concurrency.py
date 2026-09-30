"""The HTTP server shares one SQLite connection across threadpool workers.

Regression: with 8 concurrent writers, interleaved use of that connection raised
sqlite3.InterfaceError ("bad parameter or other API misuse") and
"None is not a valid ClaimStatus", surfacing as HTTP 500s.
"""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from memcontext import http_server

THREADS = 8
POSTS_PER_THREAD = 15
TOKEN = "concurrency-test-token"


@pytest.fixture
def file_db(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMCONTEXT_HTTP_TOKEN", TOKEN)
    monkeypatch.setattr(http_server, "_http_token", None)
    http_server.init_db(str(tmp_path / "concurrency.db"))
    conn = http_server._conn  # type: ignore[attr-defined]
    yield conn
    conn.close()
    monkeypatch.setattr(http_server, "_conn", None)


def _writer(worker: int, statuses: list[int], lock: threading.Lock) -> None:
    headers = {"authorization": f"Bearer {TOKEN}"}
    with TestClient(http_server.app, raise_server_exceptions=False) as client:
        for i in range(POSTS_PER_THREAD):
            body = {
                "text": f"worker {worker} message {i}: service{i % 3} database is engine{worker}",
                "session_id": f"session{worker % 2}",
                # Shared subjects across workers exercise supersession concurrently.
                "claims": [{"subject": f"service{i % 3}", "predicate": "user_fact",
                            "value": f"database is engine{worker} rev {i}"}],
            }
            r = client.post("/api/memory/store", json=body, headers=headers)
            q = client.post("/api/memory/query", headers=headers,
                            json={"query": f"service{i % 3} database", "top_k": 5})
            with lock:
                statuses.extend((r.status_code, q.status_code))


def test_concurrent_writers_never_500_and_every_turn_lands(file_db):
    statuses: list[int] = []
    lock = threading.Lock()
    threads = [threading.Thread(target=_writer, args=(w, statuses, lock)) for w in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads)
    bad = [s for s in statuses if s != 200]
    assert bad == [], f"{len(bad)} non-200 responses: {sorted(set(bad))}"
    assert len(statuses) == 2 * THREADS * POSTS_PER_THREAD
    turns = file_db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    assert turns == THREADS * POSTS_PER_THREAD
