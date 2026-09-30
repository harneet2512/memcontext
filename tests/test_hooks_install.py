"""`memcontext hooks install/uninstall` must produce hooks the server accepts and
must never destroy hooks the user configured themselves.

Regressions covered:
- install wrote HTTP hooks with no Authorization header while every /api/* route
  requires a bearer token -> every hook call was a silent 401 in Claude Code.
- install replaced the whole ``hooks`` object and uninstall deleted it, wiping
  unrelated user hooks.
"""
from __future__ import annotations

import json
import os
import re

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from memcontext import http_server
from memcontext.cli import main

EVENTS = ("PostToolUse", "UserPromptSubmit", "PreToolUse", "Stop")
FOREIGN = {"type": "command", "command": "my-security-guard"}


def _settings(project) -> dict:
    with open(os.path.join(project, ".claude", "settings.json"), encoding="utf-8") as f:
        return json.load(f)


def _write_settings(project, settings: dict) -> None:
    os.makedirs(os.path.join(project, ".claude"), exist_ok=True)
    with open(os.path.join(project, ".claude", "settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f)


def _memcontext_hooks(settings: dict) -> list[dict]:
    return [
        h
        for groups in settings.get("hooks", {}).values()
        for g in groups
        for h in g.get("hooks", [])
        if "/api/hooks/" in h.get("url", "")
    ]


def _install(project, *extra):
    r = CliRunner().invoke(main, ["hooks", "install", "--project-dir", str(project), *extra])
    assert r.exit_code == 0, r.output
    return r


def test_install_writes_bearer_header_for_every_hook(tmp_path):
    _install(tmp_path)
    hooks = _memcontext_hooks(_settings(tmp_path))
    assert len(hooks) == len(EVENTS)
    for h in hooks:
        assert h["headers"]["Authorization"] == "Bearer $MEMCONTEXT_HTTP_TOKEN"
        assert h["allowedEnvVars"] == ["MEMCONTEXT_HTTP_TOKEN"]


def test_installed_hooks_authenticate_against_the_server(tmp_path, monkeypatch):
    _install(tmp_path)
    monkeypatch.setenv("MEMCONTEXT_HTTP_TOKEN", "tok-hooks-test")
    monkeypatch.setattr(http_server, "_http_token", None)
    http_server.init_db(":memory:")
    client = TestClient(http_server.app)

    stop_hook = next(h for h in _memcontext_hooks(_settings(tmp_path)) if h["url"].endswith("/stop"))
    # Resolve headers the way Claude Code does: only allow-listed $VARs expand.
    headers = {
        k: re.sub(r"\$([A-Z_]+)", lambda m: os.environ[m.group(1)]
                  if m.group(1) in stop_hook["allowedEnvVars"] else "", v)
        for k, v in stop_hook["headers"].items()
    }
    assert client.post("/api/hooks/stop", json={}, headers=headers).status_code == 200
    assert client.post("/api/hooks/stop", json={}).status_code == 401


def test_install_preserves_foreign_hooks_and_is_idempotent(tmp_path):
    _write_settings(tmp_path, {
        "model": "opus",
        "hooks": {
            "PreToolUse": [{"matcher": "Bash", "hooks": [FOREIGN]}],
            "SessionStart": [{"matcher": "", "hooks": [FOREIGN]}],
        },
    })
    _install(tmp_path)
    _install(tmp_path)  # re-running must not duplicate entries

    s = _settings(tmp_path)
    assert s["model"] == "opus"
    assert FOREIGN in s["hooks"]["PreToolUse"][0]["hooks"]
    assert s["hooks"]["SessionStart"] == [{"matcher": "", "hooks": [FOREIGN]}]
    assert len(_memcontext_hooks(s)) == len(EVENTS)


def test_uninstall_removes_only_memcontext_hooks(tmp_path):
    _write_settings(tmp_path, {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [FOREIGN]}]}})
    _install(tmp_path)
    r = CliRunner().invoke(main, ["hooks", "uninstall", "--project-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output

    s = _settings(tmp_path)
    assert _memcontext_hooks(s) == []
    assert s["hooks"] == {"PreToolUse": [{"matcher": "Bash", "hooks": [FOREIGN]}]}


def test_uninstall_without_memcontext_hooks_leaves_user_hooks(tmp_path):
    original = {"hooks": {"Stop": [{"matcher": "", "hooks": [FOREIGN]}]}}
    _write_settings(tmp_path, original)
    CliRunner().invoke(main, ["hooks", "uninstall", "--project-dir", str(tmp_path)])
    assert _settings(tmp_path) == original


@pytest.mark.parametrize("port", [8100, 8765])
def test_install_uses_requested_port(tmp_path, port):
    _install(tmp_path, "--port", str(port))
    assert all(h["url"].startswith(f"http://localhost:{port}/api/hooks/")
               for h in _memcontext_hooks(_settings(tmp_path)))
