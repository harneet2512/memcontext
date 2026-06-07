"""Attach/detach the MemContext MCP server in AI-client config files.

No external dependencies. Every write backs up the prior file to ``<path>.bak``,
merges idempotently, and never touches other servers' entries.

- Claude Code: JSON with an ``mcpServers`` object — project ``.mcp.json`` or user
  ``~/.claude.json``.
- Codex: TOML ``[mcp_servers.<name>]`` in ``~/.codex/config.toml``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

SERVER_NAME = "memcontext"


def claude_project_path(base: str = ".") -> Path:
    return Path(base).resolve() / ".mcp.json"


def claude_user_path() -> Path:
    return Path.home() / ".claude.json"


def codex_path() -> Path:
    return Path.home() / ".codex" / "config.toml"


def launch_args(db: str) -> list[str]:
    return ["-m", "memcontext.mcp_server", "--db", str(Path(db).resolve())]


def _backup(p: Path) -> None:
    if p.exists():
        shutil.copy2(p, p.with_name(p.name + ".bak"))


# ── Claude Code (JSON) ───────────────────────────────────────────────────────
def attach_claude(path: Path, py: str, db: str) -> bool:
    """Merge the memcontext server into a Claude JSON config. True if changed."""
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            data = {}
    servers = data.setdefault("mcpServers", {})
    entry = {"command": py, "args": launch_args(db)}
    if servers.get(SERVER_NAME) == entry:
        return False  # idempotent: already attached identically
    _backup(path)
    servers[SERVER_NAME] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


def detach_claude(path: Path) -> bool:
    """Remove the memcontext server from a Claude JSON config. True if changed."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return False
    servers = data.get("mcpServers", {})
    if SERVER_NAME not in servers:
        return False
    _backup(path)
    del servers[SERVER_NAME]
    if not servers:
        data.pop("mcpServers", None)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


# ── Codex (TOML) — manual section handling, no toml-writer dependency ─────────
_CODEX_HEADER = f"[mcp_servers.{SERVER_NAME}]"


def _codex_block(py: str, db: str) -> str:
    args = ", ".join(f"'{a}'" for a in launch_args(db))  # single-quoted: Windows-path safe
    return f"{_CODEX_HEADER}\ncommand = '{py}'\nargs = [{args}]\n"


def _strip_codex_section(text: str) -> str:
    """Drop an existing [mcp_servers.memcontext] section (until the next [ or EOF)."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == _CODEX_HEADER:
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("["):
                i += 1
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out).strip("\n")


def attach_codex(path: Path, py: str, db: str) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    stripped = _strip_codex_section(existing)
    new = (stripped + "\n\n" if stripped else "") + _codex_block(py, db)
    if existing.strip() == new.strip():
        return False
    if path.exists():
        _backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new if new.endswith("\n") else new + "\n", encoding="utf-8")
    return True


def detach_codex(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if _CODEX_HEADER not in text:
        return False
    _backup(path)
    stripped = _strip_codex_section(text)
    path.write_text((stripped + "\n") if stripped else "", encoding="utf-8")
    return True
