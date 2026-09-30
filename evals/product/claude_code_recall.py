"""Cross-session recall eval: MemContext as a Claude Code user experiences it.

Every step of a scenario is a NEW headless ``claude -p`` session in the same
throwaway project, wired to MemContext exactly as a user would wire it:
``memcontext serve-http`` (hooks) + ``memcontext hooks install`` + a project
``.mcp.json`` running the stdio MCP server on the same DB + a short project
CLAUDE.md with memory conventions (``PROJECT_CLAUDE_MD`` below; identical for
every scenario and part of the system under test).

It asks two questions: does a fresh session get the CURRENT decision without the
user repeating it, and does Claude ever ANSWER or ACT on a stale (superseded)
one? Scoring is deterministic (regex over the final ``CURRENT:`` line and over a
file assignment written by "act" steps; no LLM judge). Every miss is attributed
to one layer: plumbing, capture, storage/supersession, retrieval/injection, or
use (Claude had the right value and still got it wrong).

Diagnostic, not a benchmark: n is small and model runs vary.

    python -m evals.product.claude_code_recall --model haiku --max-cost-usd 5
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "datasets" / "recall_scenarios.json"
RESULTS_DIR = HERE / "results"
DEFAULT_WORKDIR = Path(os.environ.get("TEMP", "/tmp")) / "memcontext_recall_eval"

PACK = "general,developer"
LIVE = frozenset({"active", "confirmed", "audited"})
SCORED_KINDS = frozenset({"recall", "act"})
MAX_TURNS = {"capture": 8, "change": 8, "suggest": 6, "other": 8, "recall": 6, "act": 10}
SESSION_TIMEOUT_S = 300
PER_SESSION_BUDGET_USD = 0.75
UNKNOWN_COST_ESTIMATE_USD = 0.10  # charged to the budget when a session reports no cost

# Appended to recall/act prompts. Topic-free on purpose: its words must not
# create lexical overlap with stored claims (that would inflate injection hits).
ANSWER_SUFFIX = (
    "\n\nEnd your reply with one final line of the form `CURRENT: <value>` stating "
    "only the current value you relied on."
)

PROJECT_CLAUDE_MD = """\
# Project memory

This project keeps long-term memory in the `memcontext` MCP server.

- Relevant memory may be injected into your context as a "[MemContext] ..." note.
  Treat it as the current state of the project.
- If you need a project decision that is not in that note, look it up with
  `memory_query` before answering or writing code. Do not guess.
- When the user makes or changes a project decision, record it with `memory_store`:
  `text` = the user's statement, plus one structured claim with
  `predicate` = "decision_made", `subject` = "<project>/<topic>" in lowercase
  (for example "billing-api/database"), and `value` = the decision in a few words.
- When a decision changes, store the new value with the SAME subject and predicate so
  it replaces the old one. Reuse the subject already in memory; do not invent a new one.
- Record only decisions the user made. Never store your own suggestions as decisions.
"""

_BOUND_L, _BOUND_R = r"(?<![a-z0-9])(?:", r")(?![a-z0-9])"
_CURRENT_RE = re.compile(r"^[\s*_`>#-]*CURRENT[\s*_`]*:[\s*_`]*(.*?)[\s*_`.]*$", re.I | re.M)
_HOOK_LINE_RE = re.compile(r'"POST /api/hooks/(\w+) HTTP/1\.1" (\d{3})')
_MCP_CONNECT_RE = re.compile(r'MCP server "memcontext": Successfully connected .*? in (\d+)ms')


def _posix(p: Path | str) -> str:
    return str(p).replace("\\", "/")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a process we started and its children (Windows launchers spawn a child)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, check=False)
    else:
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=15)


def _cmd_out(cmd: list[str], cwd: Path | None = None) -> str:
    """First line of a helper command's stdout, or "unknown" (git sha, claude version)."""
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60, check=True)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class Patterns:
    """Named value regexes of one scenario, matched case-insensitively on alnum bounds."""

    def __init__(self, values: dict[str, str]):
        self._re = {k: re.compile(_BOUND_L + v + _BOUND_R, re.I) for k, v in values.items()}

    def hit(self, key: str | None, text: str | None) -> bool:
        return bool(key and text and self._re[key].search(text))

    def any_hit(self, keys: list[str], text: str | None) -> bool:
        return any(self.hit(k, text) for k in keys)


def _claude_md_excludes(project: Path) -> list[str]:
    """User-level and parent-dir CLAUDE.md files; loading them would contaminate the test."""
    home_claude = Path.home() / ".claude"
    out = [_posix(home_claude / "CLAUDE.md"), _posix(home_claude / "rules") + "/**"]
    for parent in project.resolve().parents:
        for name in ("CLAUDE.md", "CLAUDE.local.md"):
            if (parent / name).exists():
                out.append(_posix(parent / name))
    return out


def _ancestor_mcp_servers(project: Path) -> list[str]:
    """Servers from .mcp.json files in parent dirs; enableAllProjectMcpServers would load them."""
    names: list[str] = []
    for parent in project.resolve().parents:
        cfg = parent / ".mcp.json"
        with contextlib.suppress(OSError, ValueError, AttributeError):
            names += [n for n in json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]
                      if n != "memcontext"]
    return sorted(set(names))


def _write_project(project: Path, spec: dict) -> None:
    name, pkg = spec["name"], spec["name"].replace("-", "_")
    files = {
        "README.md": f"# {name}\n\n{spec['description']}\n",
        f"src/{pkg}/__init__.py": f'"""{name}."""\n',
        f"src/{pkg}/app.py": (
            f'"""Entry point for {name}."""\n\n\ndef main() -> None:\n'
            '    raise NotImplementedError("not wired yet")\n'
        ),
        "CLAUDE.md": PROJECT_CLAUDE_MD,
    }
    for rel, text in files.items():
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if shutil.which("git"):
        subprocess.run(["git", "init", "-q"], cwd=project, capture_output=True, check=False)


def _write_claude_config(project: Path, db: Path, embed: str) -> None:
    mcp = {"mcpServers": {"memcontext": {
        "command": _posix(sys.executable),
        "args": ["-m", "memcontext.mcp_server", "--db", _posix(db)],
        "env": {"ACTIVE_PACK": PACK, "MEMCONTEXT_EMBED_EPISODES": embed,
                "TRANSFORMERS_NO_TF": "1", "USE_TF": "0", "PYTHONUTF8": "1"},
    }}}
    (project / ".mcp.json").write_text(json.dumps(mcp, indent=2), encoding="utf-8")
    local = {
        "enableAllProjectMcpServers": True,
        "permissions": {"allow": ["mcp__memcontext", "Write", "Edit", "Read"]},
        # Isolation (not part of the product): no outside CLAUDE.md, no built-in
        # auto-memory competing with MemContext across sessions.
        "claudeMdExcludes": _claude_md_excludes(project),
        "disabledMcpjsonServers": _ancestor_mcp_servers(project),
        "autoMemoryEnabled": False,
    }
    (project / ".claude").mkdir(exist_ok=True)
    (project / ".claude" / "settings.local.json").write_text(
        json.dumps(local, indent=2), encoding="utf-8")


def _server_env(token: str, embed: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"MEMCONTEXT_HTTP_TOKEN": token, "ACTIVE_PACK": PACK,
                "MEMCONTEXT_EMBED_EPISODES": embed, "TRANSFORMERS_NO_TF": "1",
                "USE_TF": "0", "PYTHONUTF8": "1"})
    return env


def _wait_health(port: int, proc: subprocess.Popen, timeout_s: float = 180) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        with contextlib.suppress(OSError), urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2) as r:
            if r.status == 200:
                return True
        time.sleep(0.5)
    return False


class Stack:
    """One scenario's project dir, DB, serve-http process and hook install."""

    def __init__(self, root: Path, spec: dict, cli: str, embed: str):
        self.root, self.project, self.db = root, root / "project", root / "memory.db"
        self.port, self.token = _free_port(), f"recall-eval-{os.getpid()}-{_free_port()}"
        self.serve_log = root / "serve-http.log"
        self.env = _server_env(self.token, embed)
        self.cli, self.proc = cli, None
        self._log_fh = None
        _write_project(self.project, spec)
        self._run(cli, "init", "--db", str(self.db), "--pack", PACK)
        self._run(cli, "hooks", "install", "--port", str(self.port),
                  "--project-dir", str(self.project))
        _write_claude_config(self.project, self.db, embed)
        self._start(cli)

    def _run(self, *cmd: str) -> None:
        subprocess.run(cmd, env=self.env, capture_output=True, text=True, check=True)

    def _start(self, cli: str) -> None:
        self._log_fh = open(self.serve_log, "ab")  # noqa: SIM115 - closed in stop()
        self.proc = subprocess.Popen(
            [cli, "serve-http", "--db", str(self.db), "--port", str(self.port)],
            env=self.env, stdout=self._log_fh, stderr=subprocess.STDOUT)
        if not _wait_health(self.port, self.proc):
            self.stop()
            raise RuntimeError(f"serve-http did not become healthy; see {self.serve_log}")

    def ensure_alive(self) -> bool:
        """Restart serve-http if it died (e.g. killed externally). True if restarted."""
        if self.proc is not None and self.proc.poll() is None and _wait_health(
                self.port, self.proc, timeout_s=5):
            return False
        self.stop()
        self._start(self.cli)
        return True

    def log_size(self) -> int:
        return self.serve_log.stat().st_size if self.serve_log.exists() else 0

    def stop(self) -> None:
        if self.proc is not None:
            _kill_tree(self.proc)
        if self._log_fh is not None:
            self._log_fh.close()


def _claude_env(token: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"MEMCONTEXT_HTTP_TOKEN": token, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                "ENABLE_CLAUDEAI_MCP_SERVERS": "false", "TRANSFORMERS_NO_TF": "1",
                "USE_TF": "0", "PYTHONUTF8": "1"})
    return env


def run_session(claude: str, stack: Stack, prompt: str, *, model: str, max_turns: int,
                budget: float, out_base: Path) -> dict:
    transcript, debug = out_base.with_suffix(".jsonl"), out_base.with_suffix(".debug.log")
    cmd = [claude, "-p", prompt, "--model", model, "--setting-sources", "project,local",
           "--output-format", "stream-json", "--verbose", "--include-hook-events",
           "--max-turns", str(max_turns), "--no-session-persistence",
           "--max-budget-usd", f"{budget:.2f}", "--debug-file", str(debug),
           "--disallowedTools", "mcp__claude_ai_Gmail", "mcp__claude_ai_Claude_Docs"]
    restarted = stack.ensure_alive()
    log_start, t0 = stack.log_size(), time.monotonic()
    timed_out = False
    with open(transcript, "wb") as out:
        proc = subprocess.Popen(cmd, cwd=stack.project, env=_claude_env(stack.token),
                                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=SESSION_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
    wall = time.monotonic() - t0
    time.sleep(0.5)  # let uvicorn flush its access log
    rec = parse_transcript(transcript)
    rec.update({"returncode": proc.returncode, "timed_out": timed_out, "wall_s": round(wall, 1),
                "transcript": _posix(transcript), "debug_log": _posix(debug),
                "server_restarted_before": restarted, "mcp_connect_ms": _mcp_connect_ms(debug),
                "hook_http": _hook_statuses(stack.serve_log, log_start)})
    return rec


def _mcp_connect_ms(debug: Path) -> int | None:
    """How long the stdio MCP server took to connect (from Claude Code's debug log)."""
    m = debug.exists() and _MCP_CONNECT_RE.search(debug.read_text("utf-8", errors="replace"))
    return int(m.group(1)) if m else None


def _hook_statuses(log: Path, start: int) -> dict[str, dict[str, int]]:
    with open(log, "rb") as fh:
        fh.seek(start)
        text = fh.read().decode("utf-8", "replace")
    out: dict[str, dict[str, int]] = {}
    for (endpoint, code), n in Counter(_HOOK_LINE_RE.findall(text)).items():
        out.setdefault(endpoint, {})[code] = n
    if "hook.failed" in text:
        out["_hook_failed_log_lines"] = {"count": text.count("hook.failed")}
    return out


def _tool_result_text(block: dict) -> str:
    content = block.get("content")
    return (" ".join(c.get("text", "") for c in content if isinstance(c, dict))
            if isinstance(content, list) else str(content or ""))


def _hook_context(ev: dict) -> tuple[str, str] | None:
    try:
        out = json.loads(ev.get("output") or "{}")
    except json.JSONDecodeError:
        return None
    ctx = (out.get("hookSpecificOutput") or {}).get("additionalContext") if isinstance(out, dict) else None
    return (ev.get("hook_event", ""), ctx) if ctx else None


def parse_transcript(path: Path) -> dict:
    """Pull what the session saw and did out of a stream-json transcript."""
    rec: dict = {"session_id": None, "mcp_servers": None, "injected": [], "hook_events": [],
                 "tool_calls": [], "final": None, "cost_usd": None, "duration_ms": None,
                 "num_turns": None, "is_error": None, "result_subtype": None,
                 "permission_denials": []}
    calls: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        _absorb_event(ev, rec, calls)
    rec["tool_calls"] = list(calls.values())
    return rec


def _absorb_event(ev: dict, rec: dict, calls: dict[str, dict]) -> None:
    etype, sub = ev.get("type"), ev.get("subtype")
    if etype == "system" and sub == "init":
        rec["session_id"] = ev.get("session_id")
        rec["mcp_servers"] = {s.get("name"): s.get("status") for s in ev.get("mcp_servers", [])}
    elif etype == "system" and sub == "hook_response":
        rec["hook_events"].append({"event": ev.get("hook_event"), "outcome": ev.get("outcome"),
                                   "exit_code": ev.get("exit_code")})
        ctx = _hook_context(ev)
        if ctx:
            rec["injected"].append({"event": ctx[0], "context": ctx[1]})
    elif etype == "assistant":
        for c in ev.get("message", {}).get("content", []):
            if c.get("type") == "tool_use":
                calls[c["id"]] = {"name": c.get("name"), "input": c.get("input"),
                                  "result": None, "is_error": None}
    elif etype == "user" and isinstance(ev.get("message", {}).get("content"), list):
        for c in ev["message"]["content"]:
            if c.get("type") == "tool_result" and c.get("tool_use_id") in calls:
                calls[c["tool_use_id"]].update(result=_tool_result_text(c)[:4000],
                                               is_error=bool(c.get("is_error")))
    elif etype == "result":
        rec.update(final=ev.get("result"), cost_usd=ev.get("total_cost_usd"),
                   duration_ms=ev.get("duration_ms"), num_turns=ev.get("num_turns"),
                   is_error=ev.get("is_error"), result_subtype=sub,
                   permission_denials=ev.get("permission_denials") or [])


def db_snapshot(db: Path) -> list[dict]:
    """Every claim with its status and source speaker (read-only)."""
    with contextlib.closing(sqlite3.connect(f"file:{_posix(db)}?mode=ro", uri=True,
                                            timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT c.claim_id, c.subject, c.predicate, c.value, c.status, c.session_id,"
            " t.speaker FROM claims c LEFT JOIN turns t ON t.turn_id = c.source_turn_id"
            " ORDER BY c.created_ts")]


def _live(snapshot: list[dict]) -> list[dict]:
    return [c for c in snapshot if c["status"] in LIVE and c.get("predicate") != "action"]


def db_state(snapshot: list[dict], pats: Patterns, step: dict) -> dict:
    """Does the DB hold the step's current value live, and nothing stale-only live?"""
    live = _live(snapshot)
    cur, stale = step.get("current"), step.get("stale", [])
    cur_claims = [c for c in live if pats.hit(cur, c["value"])]
    stale_only = [c for c in live if pats.any_hit(stale, c["value"])
                  and not pats.hit(cur, c["value"])]
    missing_also = [k for k in step.get("also_live", [])
                    if not any(pats.hit(k, c["value"]) for c in live)]
    return {"current_live": bool(cur_claims), "stale_live": bool(stale_only),
            "also_live_missing": missing_also,
            "ok": bool(cur_claims) and not stale_only and not missing_also,
            "current_claims": cur_claims, "stale_claims": stale_only}


def current_line(answer: str | None) -> str | None:
    matches = _CURRENT_RE.findall(answer or "")
    return matches[-1].strip() if matches else None


def file_value(path: Path, var: str) -> str | None:
    if not path.exists():
        return None
    pat = re.compile(rf"^\s*{re.escape(var)}\s*(?::\s*[\w\[\]]+\s*)?=\s*[\"']([^\"']*)[\"']", re.M)
    found = pat.findall(path.read_text(encoding="utf-8", errors="replace"))
    return found[-1] if found else None


def _verdict(value: str | None, pats: Patterns, step: dict) -> str:
    if value is None:
        return "missing"
    cur = pats.hit(step.get("current"), value)
    stale = pats.any_hit(step.get("stale", []), value)
    if cur and stale:
        return "mixed"
    return "current" if cur else ("stale" if stale else "wrong")


def _is_mc(call: dict) -> bool:
    return str(call.get("name", "")).startswith("mcp__memcontext__")


def evidence(rec: dict, pats: Patterns, step: dict) -> dict:
    """Where (if anywhere) the session was handed the current / a stale value."""
    cur, stale = step.get("current"), step.get("stale", [])
    injected = " \n".join(i["context"] for i in rec["injected"])
    mc_reads = [c for c in rec["tool_calls"] if _is_mc(c) and not c["name"].endswith("memory_store")]
    mc_text = " \n".join(c.get("result") or "" for c in mc_reads)
    other_text = " \n".join(c.get("result") or "" for c in rec["tool_calls"] if not _is_mc(c))
    inj_cur, tool_cur = pats.hit(cur, injected), pats.hit(cur, mc_text)
    source = ("injection" if inj_cur else "tool" if tool_cur
              else "file" if pats.hit(cur, other_text) else "none")
    return {"injected_current": inj_cur,
            "injected_stale": _stale_only_lines(injected, pats, cur, stale),
            "tool_called": bool(mc_reads), "tool_current": tool_cur,
            "tool_errors": [c["name"] for c in rec["tool_calls"] if _is_mc(c) and c["is_error"]],
            "stored": [c["input"] for c in rec["tool_calls"] if str(c["name"]).endswith("memory_store")],
            "source": source}


def _stale_only_lines(text: str, pats: Patterns, cur: str | None, stale: list[str]) -> bool:
    return any(pats.any_hit(stale, ln) and not pats.hit(cur, ln) for ln in text.splitlines())


def _plumbing_problem(rec: dict, *, hard_only: bool = False) -> str | None:
    """Infra failure in the session. Hard = session/hooks broken; soft = MCP tools unusable."""
    if rec["timed_out"]:
        return f"session timed out after {SESSION_TIMEOUT_S}s"
    if rec["result_subtype"] is None:
        return "no result event (claude crashed?)"
    bad = {ep: codes for ep, codes in rec["hook_http"].items()
           if ep.startswith("_") or any(code != "200" for code in codes)}
    if bad:
        return f"hook HTTP failures: {bad}"
    failed = sorted({h["event"] for h in rec["hook_events"] if h["outcome"] != "success"})
    if failed:
        return f"hook calls failed in Claude Code: {failed}"
    if "200" not in rec["hook_http"].get("user_prompt_submit", {}):
        return "serve-http never answered the UserPromptSubmit hook"
    if hard_only:
        return None
    if _mcp_tools_unavailable(rec["tool_calls"]):
        return (f"memcontext MCP tools not loadable (ToolSearch found none; "
                f"MCP connect took {rec.get('mcp_connect_ms')} ms)")
    if rec["evidence"]["tool_errors"]:
        return f"memcontext tool errors: {rec['evidence']['tool_errors']}"
    return None


def _mcp_tools_unavailable(calls: list[dict]) -> bool:
    """Claude searched for memcontext tools, never got one, and never called one."""
    searched = [c for c in calls if c["name"] == "ToolSearch"
                and re.search(r"memcontext|memory_", json.dumps(c.get("input") or {}).lower())]
    found = any("mcp__memcontext__" in (c.get("result") or "") for c in searched)
    return bool(searched) and not found and not any(_is_mc(c) for c in calls)


def _same_slot(a: list[dict], b: list[dict]) -> bool:
    slots = {(c["subject"], c["predicate"]) for c in a}
    return any((c["subject"], c["predicate"]) in slots for c in b)


def attribute_scored(rec: dict, before: dict) -> dict:
    """Layer for a failed recall/act step, deepest first: broken session, the DB state it
    was served from (``before``), then unusable MCP tools (only if injection missed)."""
    plumbing = _plumbing_problem(rec, hard_only=True)
    if plumbing:
        return {"layer": "plumbing", "reason": plumbing}
    if not before["current_live"]:
        return {"layer": "capture", "reason": "no live claim holds the current value"}
    if before["stale_live"]:
        if _same_slot(before["current_claims"], before["stale_claims"]):
            return {"layer": "storage/supersession",
                    "reason": "stale claim still live in the same subject+predicate slot"}
        return {"layer": "capture",
                "reason": "subject drift: stale and current live under different subjects"}
    ev, soft = rec["evidence"], _plumbing_problem(rec)
    if soft and not ev["injected_current"]:
        return {"layer": "plumbing", "reason": soft}
    if not (ev["injected_current"] or ev["tool_current"]):
        tools = "tools returned none" if ev["tool_called"] else "no memcontext tool fallback"
        return {"layer": "retrieval/injection",
                "reason": f"no hook injection of the current value; {tools}"}
    if ev["injected_stale"]:
        return {"layer": "retrieval/injection", "reason": "injected context carried a stale value"}
    return {"layer": "use", "reason": f"current value was delivered via {ev['source']} "
                                      "but the answer/action was wrong"}


def attribute_state(rec: dict, after: dict, before_snapshot: list[dict]) -> dict:
    """Layer for a capture/change/suggest/other step whose resulting DB state is wrong."""
    plumbing = _plumbing_problem(rec)
    if plumbing:
        return {"layer": "plumbing", "reason": plumbing}
    stored = rec["evidence"]["stored"]
    if not after["current_live"]:
        return {"layer": "capture", "reason": "memory_store stored a different value" if stored
                else "no memory_store call; the decision was not recorded"}
    if after["stale_live"]:
        before_ids = {c["claim_id"] for c in before_snapshot}
        new_stale = [c for c in after["stale_claims"] if c["claim_id"] not in before_ids]
        if new_stale:
            return {"layer": "capture", "reason": "Claude stored the stale/suggested value as live"}
        if _same_slot(after["current_claims"], after["stale_claims"]):
            return {"layer": "storage/supersession", "reason": "same slot, old value not retired"}
        return {"layer": "capture", "reason": "subject drift: new value stored under a different "
                                              "subject, so the old one could not be superseded"}
    return _attribute_lost_unrelated(rec, after)


def _attribute_lost_unrelated(rec: dict, after: dict) -> dict:
    """An unrelated decision stopped being live. Same subject reused by Claude -> capture
    (convention broken, Pass-1 did what it is told); distinct subjects -> storage bug."""
    stored_subjects = {str(c.get("subject", "")).lower()
                       for inp in rec["evidence"]["stored"] if isinstance(inp, dict)
                       for c in inp.get("claims") or [] if isinstance(c, dict)}
    other_subjects = {str(c["subject"]).lower() for c in after["current_claims"]}
    missing = after["also_live_missing"]
    if stored_subjects & other_subjects:
        return {"layer": "capture", "reason": f"{missing} retired: Claude stored two topics "
                                              "under one subject+predicate slot"}
    return {"layer": "storage/supersession",
            "reason": f"{missing} retired although stored under a different subject"}


def score_step(rec: dict, step: dict, pats: Patterns, project: Path) -> dict:
    s: dict = {"answer_line": current_line(rec["final"])}
    s["answer"] = _verdict(s["answer_line"], pats, step)
    if step["kind"] == "act":
        path = project / step["file"]
        s["file_value"] = file_value(path, step["var"])
        s["action"] = _verdict(s["file_value"], pats, step)
        if path.exists():  # keep later sessions from learning the value from the repo
            path.unlink()
    s["ok"] = s["answer"] == "current" and s.get("action", "current") == "current"
    return s


class Budget:
    def __init__(self, limit: float):
        self.limit, self.spent = limit, 0.0

    def charge(self, cost: float | None) -> None:
        self.spent += cost if cost is not None else UNKNOWN_COST_ESTIMATE_USD

    def exhausted(self) -> bool:
        return self.spent >= self.limit

    def session_cap(self) -> float:
        return max(0.05, min(PER_SESSION_BUDGET_USD, self.limit - self.spent))


def run_step(i: int, step: dict, sc: dict, stack: Stack, ctx: dict) -> dict:
    pats, before_snap = ctx["pats"], ctx["snapshot"]
    before = db_state(before_snap, pats, step)
    prompt = step["prompt"] + (ANSWER_SUFFIX if step["kind"] in SCORED_KINDS else "")
    rec = run_session(ctx["claude"], stack, prompt, model=ctx["model"],
                      max_turns=step.get("max_turns", MAX_TURNS[step["kind"]]),
                      budget=ctx["budget"].session_cap(),
                      out_base=stack.root / f"step{i:02d}_{step['kind']}")
    ctx["budget"].charge(rec["cost_usd"])
    after_snap = db_snapshot(stack.db)
    out = {"index": i, "kind": step["kind"], "prompt": prompt, "expect": {
        k: step.get(k) for k in ("current", "stale", "also_live", "file", "var")}, **rec,
        "db_before": _brief(before), "claims_after": after_snap}
    if step["kind"] in SCORED_KINDS:
        out["score"] = score_step(rec, step, pats, stack.project)
    ctx["upstream"] = _judge(out, step, pats, before_snap, ctx.get("upstream"))
    ctx["snapshot"] = after_snap
    _print_step(sc["id"], out)
    return out


def _judge(out: dict, step: dict, pats: Patterns, before_snap: list[dict],
           upstream: dict | None) -> dict | None:
    """(Re)compute evidence, DB state and layer attribution for one step.

    A step that only failed because an earlier capture/change step never got the
    value into memory inherits that step's layer (``upstream``). Returns the
    upstream attribution for the next step."""
    out["evidence"] = evidence(out, pats, step)
    after = db_state(out["claims_after"], pats, step)
    out["db_after"] = _brief(after)
    out.pop("attribution", None)
    if "score" in out and not out["score"]["ok"]:
        out["attribution"] = attribute_scored(out, db_state(before_snap, pats, step))
    elif "score" not in out and not after["ok"]:
        out["attribution"] = attribute_state(out, after, before_snap)
    attr = out.get("attribution")
    if attr and upstream and attr["layer"] != "plumbing" and (
            out["kind"] not in ("capture", "change")
            and not db_state(before_snap, pats, step)["current_live"]):
        out["attribution"] = {**upstream, "reason": f"inherited from step {upstream['step']}: "
                                                    f"{upstream['reason']}"}
    if out["kind"] in ("capture", "change"):
        return {**out["attribution"], "step": out["index"]} if attr else None
    return upstream


def rescore(path: Path, dataset: dict) -> dict:
    """Re-attribute a saved run from its recorded transcripts/DB states (no new sessions)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    specs = {s["id"]: s for s in dataset["scenarios"]}
    for sc in data["scenarios"]:
        pats, before, upstream = Patterns(specs[sc["id"]]["values"]), [], None
        for out in sc["steps"]:
            upstream = _judge(out, specs[sc["id"]]["steps"][out["index"]], pats, before, upstream)
            before = out["claims_after"]
    data["summary"] = summarize(data["scenarios"])
    data["meta"]["rescored_at"] = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return data["summary"]


def _brief(state: dict) -> dict:
    return {k: state[k] for k in ("current_live", "stale_live", "also_live_missing", "ok")}


def _print_step(sid: str, s: dict) -> None:
    sc = s.get("score")
    verdict = (f"answer={sc['answer']}" + (f" action={sc['action']}" if "action" in sc else "")
               if sc else f"db_ok={s['db_after']['ok']}")
    attr = s.get("attribution")
    cost = f"${s['cost_usd']:.3f}" if s["cost_usd"] is not None else "$?"
    print(f"  [{sid} #{s['index']} {s['kind']:<7}] {verdict:<34} src={s['evidence']['source']:<9}"
          f" {cost} {s['wall_s']}s" + (f"  -> {attr['layer']}: {attr['reason']}" if attr else ""),
          flush=True)


def run_scenario(sc: dict, run_dir: Path, ctx: dict, keep: bool) -> dict:
    root = run_dir / sc["id"]
    root.mkdir(parents=True, exist_ok=True)
    result: dict = {"id": sc["id"], "split": sc.get("split"), "tests": sc.get("tests"),
                    "workdir": _posix(root), "steps": []}
    try:
        stack = Stack(root, sc["project"], ctx["cli"], ctx["embed"])
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        result["setup_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [{sc['id']}] SETUP FAILED: {result['setup_error']}", flush=True)
        return result
    ctx.update(pats=Patterns(sc["values"]), snapshot=[], upstream=None)
    try:
        for i, step in enumerate(sc["steps"]):
            if ctx["budget"].exhausted():
                result["aborted"] = f"budget ${ctx['budget'].limit:.2f} exhausted before step {i}"
                break
            result["steps"].append(run_step(i, step, sc, stack, ctx))
    finally:
        stack.stop()
        if not keep:
            shutil.rmtree(stack.project, ignore_errors=True)
            for p in root.glob("memory.db*"):
                p.unlink(missing_ok=True)
    return result


def _rate(num: int, den: int) -> dict:
    return {"n": den, "hits": num, "rate": round(num / den, 3) if den else None}


def summarize(scenarios: list[dict]) -> dict:
    steps = [s for sc in scenarios for s in sc["steps"]]
    scored = [s for s in steps if "score" in s]
    with_stale = [s for s in scored if s["expect"].get("stale")]
    acts = [s for s in scored if s["kind"] == "act"]
    acts_stale = [s for s in acts if s["expect"].get("stale")]
    state_steps = [s for s in steps if "score" not in s]
    costs = [s["cost_usd"] for s in steps if s["cost_usd"] is not None]
    layers = Counter(s["attribution"]["layer"] for s in steps if "attribution" in s)
    return {
        "sessions": len(steps),
        "current_recall": _rate(sum(s["score"]["answer"] == "current" for s in scored), len(scored)),
        "stale_answer": _rate(sum(s["score"]["answer"] == "stale" for s in with_stale), len(with_stale)),
        "action_correct": _rate(sum(s["score"]["action"] == "current" for s in acts), len(acts)),
        "stale_action": _rate(sum(s["score"]["action"] == "stale" for s in acts_stale), len(acts_stale)),
        "step_fully_correct": _rate(sum(s["score"]["ok"] for s in scored), len(scored)),
        "injection_hit": _rate(sum(s["evidence"]["injected_current"] for s in scored), len(scored)),
        "tool_fallback": _rate(sum(s["evidence"]["tool_called"] for s in scored), len(scored)),
        "state_steps_db_ok": _rate(sum(s["db_after"]["ok"] for s in state_steps), len(state_steps)),
        "source_counts": {k: sum(s["evidence"]["source"] == k for s in scored)
                          for k in ("injection", "tool", "file", "none")},
        "failures_by_layer": dict(layers),
        "total_cost_usd": round(sum(costs), 4),
        "mean_cost_usd": round(sum(costs) / len(costs), 4) if costs else None,
        "mean_wall_s": round(sum(s["wall_s"] for s in steps) / len(steps), 1) if steps else None,
    }


def _print_summary(summary: dict) -> None:
    print("\n=== claude_code_recall summary ===")
    for key in ("current_recall", "stale_answer", "action_correct", "stale_action",
                "step_fully_correct", "injection_hit", "tool_fallback", "state_steps_db_ok"):
        r = summary[key]
        rate = f"{r['rate']:.0%}" if r["rate"] is not None else "n/a"
        print(f"  {key:<20} {rate:>5}  ({r['hits']}/{r['n']})")
    print(f"  sources              {summary['source_counts']}\n"
          f"  failures by layer    {summary['failures_by_layer']}")
    print(f"  sessions={summary['sessions']} total=${summary['total_cost_usd']}"
          f" mean=${summary['mean_cost_usd']}/session mean_wall={summary['mean_wall_s']}s")


def _select(dataset: dict, ids: str, split: str) -> list[dict]:
    known = {s["id"] for s in dataset["scenarios"]}
    wanted = known if ids == "all" else {x.strip() for x in ids.split(",") if x.strip()}
    if wanted - known:
        raise SystemExit(f"unknown scenario id(s): {sorted(wanted - known)}")
    return [s for s in dataset["scenarios"]
            if s["id"] in wanted and split in ("all", s.get("split"))]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scenarios", default="all", help="Comma-separated scenario ids, or 'all'.")
    p.add_argument("--split", default="all", choices=["all", "dev", "heldout"])
    p.add_argument("--model", default="haiku", help="Claude model alias (haiku, sonnet, ...).")
    p.add_argument("--max-cost-usd", type=float, default=5.0, help="Abort when spend exceeds this.")
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="Run artifacts dir.")
    p.add_argument("--keep", action="store_true", help="Keep project dirs and DBs.")
    p.add_argument("--semantic", action="store_true", help="MEMCONTEXT_EMBED_EPISODES=1.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--claude", default=shutil.which("claude") or "claude")
    exe = Path(sys.executable).parent / ("memcontext.exe" if os.name == "nt" else "memcontext")
    p.add_argument("--memcontext-cli", default=str(exe) if exe.exists() else "memcontext")
    p.add_argument("--out", type=Path, default=None, help="Results JSON path.")
    p.add_argument("--rescore", type=Path, help="Re-attribute a results JSON in place.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.rescore:
        _print_summary(rescore(args.rescore, dataset))
        return 0
    scenarios = _select(dataset, args.scenarios, args.split)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.workdir / stamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = {"claude": args.claude, "cli": args.memcontext_cli, "model": args.model,
           "embed": "1" if args.semantic else "0", "budget": Budget(args.max_cost_usd)}
    meta = {"timestamp": stamp, "model": args.model,
            "commit": _cmd_out(["git", "rev-parse", "HEAD"], REPO_ROOT),
            "claude_version": _cmd_out([args.claude, "--version"]), "semantic": args.semantic,
            "max_cost_usd": args.max_cost_usd, "run_dir": _posix(run_dir),
            "dataset": _posix(args.dataset), "scenario_ids": [s["id"] for s in scenarios],
            "claude_md": PROJECT_CLAUDE_MD, "answer_suffix": ANSWER_SUFFIX,
            "claude_md_sha256": hashlib.sha256(PROJECT_CLAUDE_MD.encode()).hexdigest()}
    print(f"claude_code_recall: {len(scenarios)} scenarios, model={args.model}, "
          f"run_dir={run_dir}", flush=True)
    results = []
    for sc in scenarios:
        if ctx["budget"].exhausted():
            print(f"  budget exhausted; skipping {sc['id']} and the rest", flush=True)
            break
        results.append(run_scenario(sc, run_dir, ctx, args.keep))
    summary = summarize(results)
    out = args.out or RESULTS_DIR / f"claude_code_recall_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "summary": summary, "scenarios": results},
                              indent=2, default=str), encoding="utf-8")
    _print_summary(summary)
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
