"""attach/detach the memcontext entry in Claude (JSON) and Codex (TOML) configs."""
from __future__ import annotations

import json

from memcontext import client_config as cc


def test_attach_detach_claude_idempotent_and_preserves_others(tmp_path):
    p = tmp_path / ".mcp.json"
    db = str(tmp_path / "m.db")
    # Seed an existing config with another server so we can check preservation + backup.
    p.write_text(json.dumps({"mcpServers": {"other": {"command": "x", "args": []}}}), encoding="utf-8")

    assert cc.attach_claude(p, "py", db) is True
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "memcontext" in data["mcpServers"] and "other" in data["mcpServers"]
    assert (tmp_path / ".mcp.json.bak").exists()  # existing file was backed up
    assert cc.attach_claude(p, "py", db) is False  # idempotent re-attach

    assert cc.detach_claude(p) is True
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "memcontext" not in data.get("mcpServers", {})
    assert "other" in data["mcpServers"]  # other server preserved
    assert cc.detach_claude(p) is False  # idempotent


def test_attach_detach_codex_idempotent_and_preserves_others(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[mcp_servers.other]\ncommand = 'x'\nargs = []\n", encoding="utf-8")
    db = str(tmp_path / "m.db")

    assert cc.attach_codex(p, "py", db) is True
    text = p.read_text(encoding="utf-8")
    assert "[mcp_servers.memcontext]" in text and "[mcp_servers.other]" in text
    assert cc.attach_codex(p, "py", db) is False  # idempotent re-attach

    assert cc.detach_codex(p) is True
    text = p.read_text(encoding="utf-8")
    assert "[mcp_servers.memcontext]" not in text
    assert "[mcp_servers.other]" in text  # other server preserved
    assert cc.detach_codex(p) is False  # idempotent
