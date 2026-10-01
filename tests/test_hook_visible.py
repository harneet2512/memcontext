"""Opt-in visible hook messages: show the user, inside Claude Code, what memory did.

With MEMCONTEXT_HOOK_VISIBLE=1 each hook response also carries a `systemMessage`
(Claude Code shows it to the user in the terminal): what was retrieved for a prompt,
what a memory_store/memory_correct call stored and what it superseded, and which
current decisions were in context before an edit. Off by default: the model-facing
behaviour (additionalContext) is unchanged either way.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from memcontext import http_server
from memcontext.mcp_tools import handle_memory_store
from memcontext.predicate_packs import active_pack

V1 = "memory_query returns each claim's source_turn_id by default; full provenance via memory_trace."
V2 = "memory_query includes provenance metadata directly; full source text stays behind memory_trace."


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ACTIVE_PACK", "general,developer")
    active_pack.cache_clear()
    monkeypatch.setenv("MEMCONTEXT_HTTP_TOKEN", "tok")
    monkeypatch.setattr(http_server, "_http_token", None)
    monkeypatch.setattr(http_server, "_hook_extractor", http_server._episode_only)
    monkeypatch.setenv("MEMCONTEXT_HOOK_VISIBLE", "1")
    http_server.init_db(":memory:")
    c = TestClient(http_server.app)
    c.headers.update({"authorization": "Bearer tok"})
    yield c
    active_pack.cache_clear()


def _decide(value: str) -> dict:
    return handle_memory_store(http_server._conn, text=f"Decision: {value}", session_id="har",  # type: ignore[arg-type]
                               claims=[{"subject": "memory_query provenance",
                                        "predicate": "decision_made", "value": value}])


def test_prompt_shows_what_memory_retrieved(client):
    _decide(V1)
    r = client.post("/api/hooks/user_prompt_submit", json={
        "session_id": "s1", "prompt": "What did we decide about memory_query provenance?"}).json()
    msg = r["systemMessage"]
    assert msg.startswith("[MemContext]") and "memory_query_provenance" in msg and "source_turn_id" in msg
    assert "source_turn_id" in r["hookSpecificOutput"]["additionalContext"]  # model still gets it


def test_prompt_with_nothing_relevant_says_so(client):
    r = client.post("/api/hooks/user_prompt_submit", json={
        "session_id": "s1", "prompt": "Please run the whole test suite now."}).json()
    assert "nothing relevant" in r["systemMessage"]


def test_memory_store_shows_stored_and_superseded(client):
    _decide(V1)
    stored = _decide(V2)
    tool_response = [{"type": "text", "text": json.dumps(stored)}]  # MCP content blocks
    r = client.post("/api/hooks/post_tool_use", json={
        "session_id": "s1", "tool_name": "mcp__memcontext__memory_store",
        "tool_input": {"text": "Decision changed"}, "tool_response": tool_response}).json()
    msg = r["systemMessage"]
    assert "stored" in msg and "provenance metadata directly" in msg
    assert "superseded" in msg and "source_turn_id by default" in msg


def test_visible_messages_are_off_by_default(client, monkeypatch):
    monkeypatch.delenv("MEMCONTEXT_HOOK_VISIBLE")
    _decide(V1)
    r = client.post("/api/hooks/user_prompt_submit", json={
        "session_id": "s1", "prompt": "What did we decide about memory_query provenance?"}).json()
    assert "systemMessage" not in r and "hookSpecificOutput" in r


def test_edit_shows_decisions_in_context(client):
    _decide(V1)
    r = client.post("/api/hooks/pre_tool_use", json={
        "session_id": "s1", "tool_name": "Edit", "tool_input": {
            "file_path": "D:/repo/memcontext/mcp_tools.py", "old_string": "x",
            "new_string": "# memory_query provenance: return source_turn_id by default"}}).json()
    assert r["systemMessage"].startswith("[MemContext]") and "memory_query_provenance" in r["systemMessage"]


def test_shell_commands_get_no_visible_message(client):
    _decide(V1)
    r = client.post("/api/hooks/pre_tool_use", json={
        "session_id": "s1", "tool_name": "Bash",
        "tool_input": {"command": "python -m pytest tests -q -k memory_query provenance"}}).json()
    assert "systemMessage" not in r and "hookSpecificOutput" in r  # model context unchanged
