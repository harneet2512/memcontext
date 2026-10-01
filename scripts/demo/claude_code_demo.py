#!/usr/bin/env python
"""Claude Code + MemContext demo: one command to (re)build a clean, isolated demo.

    python scripts/demo/claude_code_demo.py setup [--dir D:/demo/orders-service] [--port 8100]
    python scripts/demo/claude_code_demo.py stop  [--dir ...]

`setup` wipes and recreates a small project with a fresh memory database, wires
MemContext into it (MCP tools + Claude Code hooks, lexical mode), starts the
memory server in the background, waits until it is ready, and writes
`start-claude.cmd`, which launches Claude Code in the project with the hook token
set. Nothing is pre-seeded: every memory in the demo is created live.

Isolation: the demo project loads only its own CLAUDE.md and MCP server, with
Claude Code's built-in auto-memory off, so what the fresh session "remembers"
can only have come from MemContext.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PACK = "general,developer"

PROJECT_CLAUDE_MD = """\
# orders-service

Small Python service that stores customer orders.

## Project memory (MemContext)
Durable project decisions live in MemContext (the `memcontext` MCP tools), not in
this file. Relevant decisions are shown to you automatically at the start of a
prompt as "[MemContext] Current project memory".

- When we make or change a project decision, save it with `memory_store`, with one
  structured claim: `{"subject": "<component> <topic>", "predicate": "decision_made",
  "value": "<the decision in one sentence>"}`, e.g. subject "orders-service database".
- One topic per subject. If memory already has a decision on the same topic, reuse
  that exact subject so the new decision replaces the old one.
- To explain how a decision changed, use `memory_trace` with that subject and
  predicate `decision_made`.
- Keep answers short. Don't run or test code unless I ask.
"""

FILES = {
    "README.md": "# orders-service\n\nStores and serves customer orders.\n",
    "orders/__init__.py": '"""orders-service."""\n',
    "orders/config.py": '"""Service configuration."""\n\nSERVICE_NAME = "orders-service"\n',
    "orders/models.py": (
        '"""Order model."""\n\nfrom dataclasses import dataclass\n\n\n@dataclass\n'
        "class Order:\n    order_id: str\n    customer_id: str\n    total_cents: int\n"
    ),
}


def _posix(p: Path) -> str:
    return str(p).replace("\\", "/")


def _env(token: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "MEMCONTEXT_HTTP_TOKEN": token, "ACTIVE_PACK": PACK,
        "MEMCONTEXT_EMBED_EPISODES": "0",  # lexical: no model load, nothing to hang
        "TRANSFORMERS_NO_TF": "1", "USE_TF": "0", "PYTHONUTF8": "1",
    })
    return env


def _run(args: list[str], env: dict[str, str]) -> None:
    subprocess.run([sys.executable, "-m", "memcontext.cli", *args], env=env, check=True,
                   capture_output=True, text=True)


def _isolation(project: Path) -> dict:
    home = Path.home() / ".claude"
    excludes = [_posix(home / "CLAUDE.md"), _posix(home / "rules") + "/**"]
    foreign_servers: list[str] = []
    for parent in project.resolve().parents:
        for name in ("CLAUDE.md", "CLAUDE.local.md"):
            if (parent / name).exists():
                excludes.append(_posix(parent / name))
        with contextlib.suppress(OSError, ValueError, KeyError, AttributeError):
            servers = json.loads((parent / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
            foreign_servers += [n for n in servers if n != "memcontext"]
    return {
        "enableAllProjectMcpServers": True,
        "permissions": {"allow": ["mcp__memcontext", "Read", "Edit", "Write"]},
        "claudeMdExcludes": excludes,
        "disabledMcpjsonServers": sorted(set(foreign_servers)),
        "autoMemoryEnabled": False,
    }


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_ready(proc: subprocess.Popen, port: int, token: str, timeout_s: float = 90) -> None:
    """Ready = OUR server process is alive and accepts OUR token.

    A bare /health check would also be satisfied by a stale server left on the
    port, while every hook got 401 from it.
    """
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/memory/status",
                                 headers={"Authorization": f"Bearer {token}"})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"memory server exited (code {proc.returncode}); see server.log")
        with (contextlib.suppress(urllib.error.URLError, OSError),
              urllib.request.urlopen(req, timeout=2) as r):
            if r.status == 200:
                return
        time.sleep(0.5)
    raise SystemExit(f"memory server did not become ready on port {port}; see server.log")


def stop(project: Path) -> None:
    pid_file = project / ".memcontext-server.pid"
    if not pid_file.exists():
        return
    with contextlib.suppress(OSError, ValueError):
        pid = int(pid_file.read_text())
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    pid_file.unlink(missing_ok=True)


def _start_server(workdir: Path, db: Path, port: int, env: dict[str, str], token: str) -> None:
    """Start serve-http in the background (pid + log in ``workdir``) and wait for it."""
    log = (workdir / "server.log").open("w", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, "-m", "memcontext.cli", "serve-http", "--db", str(db), "--port", str(port)],
        env={**env, "PYTHONUNBUFFERED": "1"}, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, cwd=workdir, creationflags=flags,
    )
    (workdir / ".memcontext-server.pid").write_text(str(proc.pid))
    _wait_ready(proc, port, token)


def _require_free_port(port: int) -> None:
    if _port_in_use(port):
        raise SystemExit(
            f"port {port} is already in use (another demo's memory server?). Run "
            f"`claude_code_demo.py stop --dir <that demo dir>` or pass --port.")


def setup(project: Path, port: int) -> None:
    stop(project)
    if _port_in_use(port):
        raise SystemExit(
            f"port {port} is already in use (another demo's memory server?). Run "
            f"`claude_code_demo.py stop --dir <that demo dir>` or pass --port.")
    if project.exists():
        shutil.rmtree(project)
    project.mkdir(parents=True)
    for rel, text in {**FILES, "CLAUDE.md": PROJECT_CLAUDE_MD}.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(text, encoding="utf-8")
    with contextlib.suppress(OSError):
        subprocess.run(["git", "init", "-q"], cwd=project, capture_output=True, check=False)

    token = secrets.token_urlsafe(24)
    env = _env(token)
    db = project / ".memcontext" / "memory.db"
    db.parent.mkdir()
    _run(["init", "--db", str(db), "--pack", PACK], env)
    _run(["hooks", "install", "--port", str(port), "--project-dir", str(project)], env)

    mcp = {"mcpServers": {"memcontext": {
        "command": _posix(Path(sys.executable)),
        "args": ["-m", "memcontext.mcp_server", "--db", _posix(db)],
        "env": {k: env[k] for k in ("ACTIVE_PACK", "MEMCONTEXT_EMBED_EPISODES",
                                     "TRANSFORMERS_NO_TF", "USE_TF", "PYTHONUTF8")},
    }}}
    (project / ".mcp.json").write_text(json.dumps(mcp, indent=2), encoding="utf-8")
    (project / ".claude" / "settings.local.json").write_text(
        json.dumps(_isolation(project), indent=2), encoding="utf-8")

    # The token lives only in the (untracked) demo dir; it is never printed.
    (project / "start-claude.cmd").write_text(
        f"@echo off\r\nset MEMCONTEXT_HTTP_TOKEN={token}\r\ncd /d \"{project}\"\r\n"
        "claude --setting-sources project,local %*\r\n", encoding="utf-8")
    (project / ".gitignore").write_text(
        ".memcontext/\n.memcontext-server.pid\nserver.log\nstart-claude.cmd\n.mcp.json\n.claude/\n",
        encoding="utf-8")

    _start_server(project, db, port, env, token)

    print(f"demo ready: {project}")
    print(f"  memory db : {db}")
    print(f"  start     : {project / 'start-claude.cmd'}")
    print(f"  watch     : memcontext trace --db \"{db}\" --subject \"orders-service database\""
          " --predicate decision_made")


LIVE_BRIEF = ("Keep answers brief (3-6 sentences or a short list). Use the memcontext "
              "tools as CLAUDE.local.md describes.")


def _live_dir(db: Path) -> Path:
    return db.parent / "live-session"


def live(repo: Path, db: Path, port: int) -> None:
    """Attach MemContext to a REAL repo and its existing memory DB, for a recording.

    Nothing is reset: the session shows real memory. A consistent backup of the DB
    is taken first; `restore` rolls a failed take back to it.
    """
    if not db.exists():
        raise SystemExit(f"no memory database at {db}")
    work = _live_dir(db)
    work.mkdir(exist_ok=True)
    stop(work)
    _require_free_port(port)

    backup = work / "memory-before-take.db"
    with sqlite3.connect(db) as src, sqlite3.connect(backup) as dst:
        src.backup(dst)

    token = secrets.token_urlsafe(24)
    env = _env(token)
    activity = work / "hook_activity.jsonl"
    activity.write_text("", encoding="utf-8")  # each take starts with an empty feed
    env["MEMCONTEXT_HOOK_ACTIVITY_LOG"] = str(activity)
    _run(["hooks", "install", "--port", str(port), "--project-dir", str(repo)], env)

    mcp_path = repo / ".mcp.json"
    mcp = json.loads(mcp_path.read_text(encoding="utf-8")) if mcp_path.exists() else {}
    mcp.setdefault("mcpServers", {})["memcontext"] = {
        "command": _posix(Path(sys.executable)),
        "args": ["-m", "memcontext.mcp_server", "--db", _posix(db)],
        "env": {k: env[k] for k in ("ACTIVE_PACK", "MEMCONTEXT_EMBED_EPISODES",
                                     "TRANSFORMERS_NO_TF", "USE_TF", "PYTHONUTF8")},
    }
    mcp_path.write_text(json.dumps(mcp, indent=2), encoding="utf-8")

    # Claude Code's built-in auto-memory off for this repo, so anything recalled
    # on camera provably came from MemContext. `restore` turns it back on.
    local_path = repo / ".claude" / "settings.local.json"
    local = json.loads(local_path.read_text(encoding="utf-8")) if local_path.exists() else {}
    local.update({"enableAllProjectMcpServers": True, "autoMemoryEnabled": False})
    perms = local.setdefault("permissions", {}).setdefault("allow", [])
    if "mcp__memcontext" not in perms:
        perms.append("mcp__memcontext")
    local_path.parent.mkdir(exist_ok=True)
    local_path.write_text(json.dumps(local, indent=2), encoding="utf-8")

    _start_server(work, db, port, env, token)
    (work / "start-claude.cmd").write_text(
        f"@echo off\r\nset MEMCONTEXT_HTTP_TOKEN={token}\r\ncd /d \"{repo}\"\r\n"
        f"claude --setting-sources project,local --append-system-prompt \"{LIVE_BRIEF}\" %*\r\n",
        encoding="utf-8")

    _check_claude_sees_memcontext(repo)
    print(f"live session ready: {repo}")
    print(f"  memory db : {db}  (backup: {backup})")
    print(f"  start     : {work / 'start-claude.cmd'}")
    print(f"  watch     : memcontext watch --db \"{db}\" --activity \"{activity}\"")
    print(f"  good take : python {Path(__file__).name} finish --db \"{db}\"   (keeps memory)")
    print(f"  bad take  : python {Path(__file__).name} restore --db \"{db}\"  (rolls memory back)")


def _check_claude_sees_memcontext(repo: Path) -> None:
    """Ask Claude Code itself whether the memcontext MCP server connects in this repo."""
    claude = shutil.which("claude")
    if claude is None:
        print("  check     : claude CLI not found; skipped the MCP connection check")
        return
    try:
        out = subprocess.run([claude, "mcp", "list"], cwd=repo, capture_output=True,
                             text=True, encoding="utf-8", timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"MCP connection check failed to run: {type(exc).__name__}") from exc
    line = next((ln for ln in out.splitlines() if ln.startswith("memcontext")), "")
    if "Connected" not in line:
        raise SystemExit(f"Claude Code cannot connect to memcontext in {repo}: {line or out[-300:]}")
    print("  check     : Claude Code -> memcontext MCP: connected")


def restore(repo: Path, db: Path) -> None:
    """A bad take: stop the server and roll memory back to the pre-take backup."""
    work = _live_dir(db)
    stop(work)
    backup = work / "memory-before-take.db"
    if backup.exists():
        with sqlite3.connect(backup) as src, sqlite3.connect(db) as dst:
            src.backup(dst)
        print(f"memory restored from {backup}")
    finish(repo, db)


def finish(repo: Path, db: Path) -> None:
    """A good take: stop the server, keep the new memory, re-enable auto-memory."""
    stop(_live_dir(db))
    local_path = repo / ".claude" / "settings.local.json"
    if local_path.exists():
        local = json.loads(local_path.read_text(encoding="utf-8"))
        local.pop("autoMemoryEnabled", None)
        local_path.write_text(json.dumps(local, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build or stop the Claude Code + MemContext demo.")
    ap.add_argument("command", choices=["setup", "stop", "live", "finish", "restore"])
    ap.add_argument("--dir", default="D:/demo/orders-service", type=Path)
    ap.add_argument("--repo", default="D:/memcontext", type=Path, help="live/restore: repo")
    ap.add_argument("--db", default="D:/memcontext-data/work.db", type=Path,
                    help="live/restore: the real memory database")
    ap.add_argument("--port", default=8100, type=int)
    a = ap.parse_args()
    if a.command == "setup":
        setup(a.dir, a.port)
    elif a.command == "live":
        live(a.repo, a.db, a.port)
    elif a.command == "restore":
        restore(a.repo, a.db)
    elif a.command == "finish":
        finish(a.repo, a.db)
        print("stopped; memory kept")
    elif a.command == "stop":
        stop(a.dir)
        stop(_live_dir(a.db))
        print("stopped")


if __name__ == "__main__":
    main()
