"""Real-time hook visibility, and pre-tool queries that see what is about to be written.

- PreToolUse built its query from the file name and command only, so editing
  retrieval.py to change tie ranking never surfaced the tie-ranking decision. The
  text being written (Edit new_string / Write content) now joins the query.
- With MEMCONTEXT_HOOK_ACTIVITY_LOG set, every hook call that queries or stores
  memory appends one JSON line: what it queried, what it injected (even nothing),
  and how long it took. `memcontext watch --activity` shows it live.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from memcontext import http_server
from memcontext.mcp_tools import handle_memory_store
from memcontext.schema import open_database
from memcontext.watch import render_memory_view

DECISION = "Tied ranking scores share a rank; ties are ordered by source trust, then newest."
EDIT = {"tool_name": "Edit", "session_id": "s1", "tool_input": {
    "file_path": "D:/repo/memcontext/retrieval.py",
    "old_string": "return ranks",
    "new_string": "# tied ranking scores share a rank, ordered by source trust then newest\nreturn ranks",
}}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMCONTEXT_HTTP_TOKEN", "tok")
    monkeypatch.setattr(http_server, "_http_token", None)
    monkeypatch.setattr(http_server, "_hook_extractor", http_server._episode_only)
    log = tmp_path / "hook_activity.jsonl"
    monkeypatch.setenv("MEMCONTEXT_HOOK_ACTIVITY_LOG", str(log))
    http_server.init_db(":memory:")
    handle_memory_store(http_server._conn, text=f"We decided: {DECISION}", session_id="har",  # type: ignore[arg-type]
                        claims=[{"subject": "gap9 tie ranking", "predicate": "user_fact",
                                 "value": DECISION}])
    c = TestClient(http_server.app)
    c.headers.update({"authorization": "Bearer tok"})
    return c, log


def _events(log):
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_pre_tool_query_includes_the_text_being_written(client):
    c, _ = client
    out = c.post("/api/hooks/pre_tool_use", json=EDIT).json()
    assert "Tied ranking scores share a rank" in out["hookSpecificOutput"]["additionalContext"]


def test_every_memory_hook_call_is_recorded(client):
    c, log = client
    c.post("/api/hooks/user_prompt_submit",
           json={"session_id": "s1", "prompt": "How should tied ranking scores be ordered now?"})
    c.post("/api/hooks/pre_tool_use", json=EDIT)
    c.post("/api/hooks/post_tool_use", json=EDIT)
    events = _events(log)
    assert [e["event"] for e in events] == ["UserPromptSubmit", "PreToolUse", "PostToolUse"]
    pre = events[1]
    assert pre["tool"] == "Edit" and pre["target"] == "retrieval.py"
    assert "ranking" in pre["query"] and pre["injected"] and pre["ms"] >= 0
    assert events[2]["stored"] is True


def test_queries_that_find_nothing_are_recorded_too(client):
    c, log = client
    c.post("/api/hooks/pre_tool_use", json={"tool_name": "Bash", "session_id": "s1",
                                             "tool_input": {"command": "npm run lint"}})
    (event,) = _events(log)
    assert event["event"] == "PreToolUse" and event["injected"] == []


def test_watch_shows_live_hook_activity(client):
    c, log = client
    c.post("/api/hooks/pre_tool_use", json=EDIT)
    view = render_memory_view(open_database(":memory:"), activity_path=log)
    assert "LIVE HOOKS" in view and "PreToolUse" in view and "retrieval.py" in view
    assert "gap9_tie_ranking" in view  # subjects are shown normalised
