"""Live, read-only view of the memory pipeline (`memcontext watch`).

One screen, four stages, newest first:

  CAPTURED            episodes stored (prompts, tool actions, stored facts)
  CURRENT DECISIONS   structured facts that are live now, with how many earlier
                      versions they replaced
  CHANGES             supersession edges: old value -> new value, with the typed
                      reason (user_correction, semantic_replace, ...)
  SERVED TO AGENT     what memory handed back to the agent, for which request

Reads only; safe to run next to a live server (SQLite WAL).
"""
from __future__ import annotations

import json
import sqlite3
import textwrap
from datetime import datetime
from pathlib import Path

_LIVE = ("active", "confirmed", "audited")


def _clock(ts_ns: int | None) -> str:
    if not ts_ns:
        return "--:--:--"
    return datetime.fromtimestamp(ts_ns / 1e9).strftime("%H:%M:%S")


def _one_line(text: str | None, width: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= width else flat[: max(width - 3, 0)] + "..."


def _section(title: str) -> list[str]:
    return ["", title]


def _captured(conn: sqlite3.Connection, limit: int, width: int) -> list[str]:
    rows = conn.execute(
        "SELECT ts, speaker, source_type, text FROM turns ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    out = _section("CAPTURED (latest first)")
    for r in rows:
        who = f"{r['speaker']}/{r['source_type']}"
        out.append(f"  {_clock(r['ts'])}  {who:<22} {_one_line(r['text'], width - 35)}")
    return out


def _decisions(conn: sqlite3.Connection, limit: int, width: int,
               value_lines: int = 2) -> list[str]:
    rows = conn.execute(
        "SELECT c.subject, c.predicate, c.value, c.created_ts,"
        " (SELECT COUNT(*) FROM claims o WHERE o.subject = c.subject"
        "  AND o.predicate = c.predicate AND o.status = 'superseded') AS replaced"
        " FROM claims c WHERE c.status IN (?,?,?) AND c.predicate IS NOT NULL"
        " ORDER BY c.created_ts DESC LIMIT ?",
        (*_LIVE, limit),
    ).fetchall()
    out = _section("CURRENT DECISIONS (most recent first)")
    for r in rows:
        slot = f"{r['subject']} / {r['predicate']}"
        history = f"  [replaced {r['replaced']}]" if r["replaced"] else ""
        out.append(f"  {_one_line(slot, width - 4)}{history}")
        out.extend(textwrap.wrap(" ".join((r["value"] or "").split()), width=width - 8,
                                 initial_indent="      = ", subsequent_indent="        ",
                                 max_lines=value_lines, placeholder=" ..."))
    return out


def _changes(conn: sqlite3.Connection, limit: int, width: int) -> list[str]:
    rows = conn.execute(
        "SELECT e.created_ts, e.edge_type, o.subject, o.value AS old_value,"
        " n.value AS new_value FROM supersession_edges e"
        " JOIN claims o ON o.claim_id = e.old_claim_id"
        " JOIN claims n ON n.claim_id = e.new_claim_id"
        " ORDER BY e.created_ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = _section("CHANGES (supersession: old -> new, reason)")
    for r in rows:
        out.append(f"  {_clock(r['created_ts'])}  {_one_line(r['subject'], width - 30)}"
                   f"  [{r['edge_type']}]")
        out.append("      was: " + _one_line(r["old_value"], width - 11))
        out.append("      now: " + _one_line(r["new_value"], width - 11))
    return out


def _served(conn: sqlite3.Connection, limit: int, width: int) -> list[str]:
    rows = conn.execute(
        # One request serves several facts, each stamped a few ns apart: group a
        # request's rows by (session, query, second).
        "SELECT MAX(s.served_ts) AS served_ts, s.query, COUNT(*) AS n,"
        " GROUP_CONCAT(COALESCE(c.subject, '(text fact)'), ', ') AS subjects"
        " FROM serve_events s JOIN claims c ON c.claim_id = s.claim_id"
        " GROUP BY s.request_session_id, s.query, s.served_ts / 1000000000"
        " ORDER BY served_ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = _section("SERVED TO AGENT (what memory handed back)")
    for r in rows:
        out.append(f"  {_clock(r['served_ts'])}  {r['n']} fact(s) for "
                   f"\"{_one_line(r['query'], width - 34)}\"")
        out.append("      " + _one_line(r["subjects"], width - 6))
    return out


_HOOK_LABEL = {"UserPromptSubmit": "UserPromptSubmit", "PreToolUse": "PreToolUse",
               "PostToolUse": "PostToolUse"}


def _hooks(activity_path: Path, limit: int, width: int, facts: int = 3) -> list[str]:
    """What the Claude Code hooks just did: queried, injected, captured (newest first)."""
    out = _section("LIVE HOOKS (what memory did for the agent, as it happened)")
    try:
        raw = activity_path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return out + ["  (no hook activity yet)"]
    for line in reversed(raw):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        label = _HOOK_LABEL.get(ev.get("event", ""), ev.get("event", "?"))
        when = _clock(ev.get("ts"))
        if ev.get("event") == "PostToolUse":
            out.append(f"  {when}  {label:<16} {ev.get('tool', '')} {ev.get('target', '')}"
                       "  -> captured as an episode")
            continue
        injected = ev.get("injected") or []
        what = (f"{ev.get('tool', '')} {ev.get('target', '')}" if ev.get("event") == "PreToolUse"
                else f"\"{_one_line(ev.get('prompt'), 32)}\"")
        out.append(f"  {when}  {label:<16} {_one_line(what, 44)}  -> "
                   f"{len(injected)} fact(s) injected  ({ev.get('ms', '?')} ms)")
        if ev.get("query"):
            out.append("      query: " + _one_line(ev["query"], width - 13))
        out.extend("      + " + _one_line(fact, width - 8) for fact in injected[:facts])
        if len(injected) > facts:  # never hide that more was injected
            out.append(f"      (+{len(injected) - facts} more)")
    return out


def render_memory_view(conn: sqlite3.Connection, *, limit: int = 5, width: int = 100,
                       title: str = "", activity_path: Path | str | None = None,
                       compact: bool = False) -> str:
    """Render the pipeline stages as plain ASCII, at most ``width`` columns.

    ``compact`` keeps it to one screen for a side pane: live hooks, current
    decisions and changes only, fewer rows each.
    """
    conn.row_factory = sqlite3.Row
    turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    header = _one_line(f"MEMCONTEXT  .  live memory  .  {title}  .  "
                       f"{datetime.now().strftime('%H:%M:%S')}", width)
    lines = [header]
    if activity_path is not None:
        lines += _hooks(Path(activity_path), 4 if compact else limit, width,
                        facts=2 if compact else 3)
    if not turns:
        return "\n".join(lines + ["", "  (no memory yet)"])
    if compact:
        lines += _decisions(conn, 3, width, value_lines=1)
        lines += _changes(conn, 2, width)
        return "\n".join(line[:width] for line in lines)
    lines += _captured(conn, limit, width)
    lines += _decisions(conn, limit + 3, width)
    lines += _changes(conn, limit, width)
    lines += _served(conn, limit, width)
    return "\n".join(line[:width] for line in lines)
