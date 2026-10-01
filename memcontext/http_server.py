"""MemContext HTTP API — REST interface for any AI platform.

MCP is for Claude Code and Cursor. HTTP is for everything else:
ChatGPT GPTs, Gemini, custom agents.
Also serves Claude Code ambient hooks for silent context capture.

Same database, same memory. Two doors in.

    memcontext serve-http --port 8100 --db memcontext.db
"""
from __future__ import annotations

import contextlib
import functools
import os
import re
import secrets
import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

if TYPE_CHECKING:
    from memcontext.authz import Principal

log = structlog.get_logger(__name__)

# ── Shared-connection serialization ──────────────────────────────────────────
# The server holds ONE sqlite3 connection (check_same_thread=False, autocommit)
# and FastAPI runs sync endpoints / hook workers on a threadpool. A sqlite3
# Connection is not safe for interleaved use from several threads (concurrent
# writers got InterfaceError "bad parameter or other API misuse" and rows read
# mid-write, e.g. "None is not a valid ClaimStatus"), so every use of it goes
# through this lock. Reentrant so a locked helper may call another.
_db_lock = threading.RLock()

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _serialized(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run ``fn`` holding the shared-connection lock (sync callables only)."""
    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with _db_lock:
            return fn(*args, **kwargs)
    return wrapper

app = FastAPI(
    title="MemContext",
    description="Universal AI memory layer. Store, query, and trace structured claims with provenance.",
    version="0.2.0",
)

# CORS default-deny: no cross-origin access unless MEMCONTEXT_HTTP_ORIGINS lists
# explicit origins. Never a wildcard — that plus credentials would expose the
# authenticated store to any web page.
_origins_env = os.environ.get("MEMCONTEXT_HTTP_ORIGINS", "").strip()
_allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip() and o.strip() != "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth: bearer token on every route except /health ─────────────────────────
# Covers /api/* (memory, hooks), the MCP app mounted at /mcp, and the OpenAPI docs.
_PUBLIC_PATHS = frozenset({"/health"})
_http_token: str | None = None


def _configure_auth() -> str:
    """Resolve the bearer token: MEMCONTEXT_HTTP_TOKEN, else generate one once.

    A generated token is printed to stderr so a loopback operator can use it;
    set MEMCONTEXT_HTTP_TOKEN to pin a stable token (required for ``--share``).
    """
    global _http_token
    if _http_token is None:
        env_tok = os.environ.get("MEMCONTEXT_HTTP_TOKEN", "").strip()
        if env_tok:
            _http_token = env_tok
        else:
            _http_token = secrets.token_urlsafe(32)
            print(
                f"[memcontext] generated HTTP bearer token (set MEMCONTEXT_HTTP_TOKEN "
                f"to pin): {_http_token}",
                file=sys.stderr,
            )
    return _http_token


@app.middleware("http")
async def _require_bearer(request: Request, call_next):
    path = request.url.path
    if path not in _PUBLIC_PATHS:
        header = request.headers.get("authorization", "")
        provided = header[7:].strip() if header[:7].lower() == "bearer " else ""
        # Per-principal access control once any principal is registered; otherwise
        # the single shared token applies (backward compatible). The lookup runs in
        # the threadpool: it takes the connection lock, which must never block the
        # event loop while a slow hook holds it.
        enforced, principal = await run_in_threadpool(_lookup_principal, provided)
        if enforced:
            if principal is None:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            request.state.namespace = principal.namespace
            request.state.can_write = principal.can_write
            request.state.principal = principal.name
        else:
            token = _configure_auth()
            if not (provided and secrets.compare_digest(provided, token)):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            request.state.namespace = None  # single shared key = unrestricted
            request.state.can_write = True
            request.state.principal = "shared"
        # The MCP app mounted at /mcp is not namespace-bound, so a tenant-scoped
        # principal must never reach it (it would read every namespace).
        if (path == "/mcp" or path.startswith("/mcp/")) and request.state.namespace is not None:
            return JSONResponse({"error": "MCP over HTTP is single-tenant; use the shared token"},
                                status_code=403)
    return await call_next(request)


@_serialized
def _lookup_principal(provided: str) -> tuple[bool, Principal | None]:
    """(per-principal auth enforced?, principal for the token or None)."""
    from memcontext.authz import any_principals, resolve_principal

    conn = _conn
    if conn is None or not any_principals(conn):
        return False, None
    return True, resolve_principal(conn, provided)


@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception):
    """Return a generic error — never a stack trace, path, or exception message.

    Only the exception type (safe) is logged, to stderr.
    """
    log.error("http.unhandled_error", path=request.url.path, error_type=type(exc).__name__)
    return JSONResponse({"error": "internal error"}, status_code=500)


_conn = None


def get_conn():
    if _conn is None:
        raise HTTPException(500, "Database not initialized")
    return _conn


def init_db(db_path: str):
    global _conn
    from memcontext.schema import open_database
    _conn = open_database(db_path)


# ── Request / Response models ────────────────────────────

class ClaimIn(BaseModel):
    subject: str
    predicate: str
    value: str
    confidence: float = 0.9


class StoreRequest(BaseModel):
    text: str
    speaker: str = "user"
    session_id: str | None = "shared"
    claims: list[ClaimIn] | None = None


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    top_k: int = 10


class TraceRequest(BaseModel):
    claim_id: str


# ── Endpoints ────────────────────────────────────────────

@app.post("/api/memory/store")
@_serialized
def memory_store(req: StoreRequest, request: Request):
    if not getattr(request.state, "can_write", True):
        raise HTTPException(403, "read-only principal")
    from memcontext.mcp_tools import handle_memory_store
    claims = [c.model_dump() for c in req.claims] if req.claims else None
    ns = getattr(request.state, "namespace", None)
    return handle_memory_store(
        get_conn(), text=req.text, speaker=req.speaker,
        session_id=req.session_id, claims=claims,
        namespace=ns or "default",
    )


@app.post("/api/memory/query")
@_serialized
def memory_query(req: QueryRequest, request: Request):
    from memcontext.mcp_tools import handle_memory_query
    ns = getattr(request.state, "namespace", None)
    return handle_memory_query(
        get_conn(), query=req.query,
        session_id=req.session_id, top_k=req.top_k,
        namespace=ns,
    )


def _claim_in_namespace(conn, claim_id: str, namespace: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM claims c JOIN turns t ON t.turn_id = c.source_turn_id"
        " WHERE c.claim_id = ? AND t.namespace = ?",
        (claim_id, namespace),
    ).fetchone() is not None


@app.post("/api/memory/trace")
@_serialized
def memory_trace(req: TraceRequest, request: Request):
    from memcontext.mcp_tools import handle_memory_trace
    conn = get_conn()
    ns = getattr(request.state, "namespace", None)
    if ns is not None and not _claim_in_namespace(conn, req.claim_id, ns):
        # Same answer as a missing id: never confirm another tenant's claim exists.
        return {"error": f"Claim {req.claim_id} not found"}
    return handle_memory_trace(conn, claim_id=req.claim_id)


@app.get("/api/memory/status")
@_serialized
def memory_status(request: Request):
    conn = get_conn()
    ns = getattr(request.state, "namespace", None)
    if ns is None:  # single shared token = whole database
        claims_from, turns_where, args = "claims c", "", ()
    else:
        claims_from = "claims c JOIN turns t ON t.turn_id = c.source_turn_id AND t.namespace = ?"
        turns_where, args = " WHERE namespace = ?", (ns,)
    total = conn.execute(f"SELECT COUNT(*) FROM {claims_from}", args).fetchone()[0]
    active = conn.execute(
        f"SELECT COUNT(*) FROM {claims_from}"
        " WHERE c.status IN ('active','confirmed','audited')", args
    ).fetchone()[0]
    sessions = conn.execute(
        f"SELECT COUNT(DISTINCT c.session_id) FROM {claims_from}", args
    ).fetchone()[0]
    turns = conn.execute(f"SELECT COUNT(*) FROM turns{turns_where}", args).fetchone()[0]
    return {
        "total_claims": total,
        "active_claims": active,
        "sessions": sessions,
        "turns": turns,
    }


_CONTEXT_MAX_CHARS = 1500
_PROMPT_CONTEXT_MAX_LINES = 6
_PROMPT_QUERY_MAX_CHARS = 1000
_LIVE_STATUSES = frozenset({"active", "confirmed", "audited"})
_TOOL_ACTION_PREDICATE = "action"  # tool-use log entries: never useful as injected context

# Words that carry no topical signal, for the lexical relevance gate.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "for", "and", "or",
    "of", "in", "on", "at", "it", "my", "our", "we", "you", "me", "this", "that", "these",
    "those", "with", "from", "by", "as", "do", "does", "did", "can", "could", "should",
    "would", "will", "what", "which", "who", "how", "why", "when", "where", "there", "here",
    "have", "has", "had", "use", "uses", "used", "using", "know", "about", "let", "lets",
    "please", "tell", "some", "any", "all", "not", "now", "then", "into", "out", "get",
    "import", "def", "class", "return", "if", "else", "true", "false", "none", "self",
})


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) > 2 and t not in _STOPWORDS}


def _claim_match_tokens(predicate: str | None, line: str) -> set[str]:
    """Content words of a context line, minus the predicate label ("user_fact" is not content)."""
    return _content_tokens(line) - _content_tokens(predicate or "")


# ── Hook filtering ───────────────────────────────────────

_HOOK_SKIP_TOOLS: set[str] = {
    "Read", "Glob", "Grep", "LS", "LSP",
    "TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskStop", "TaskOutput",
    "Monitor", "AskUserQuestion", "WebSearch", "WebFetch",
    "ToolSearch",  # Claude Code locating its own tools: not project work
}

_BASH_SKIP_PREFIXES: tuple[str, ...] = (
    "cd ", "ls", "pwd", "cat ", "head ", "tail ", "echo ",
    "grep ", "find ", "rg ", "which ", "type ",
    "git status", "git log", "git diff", "git show", "git branch",
)


def _should_skip_tool(tool_name: str, tool_input: dict | str) -> bool:
    if not tool_name:
        return True
    if "memory_" in tool_name or "memcontext" in tool_name.lower():
        return True
    if tool_name in _HOOK_SKIP_TOOLS:
        return True
    if tool_name in ("Bash", "PowerShell"):
        cmd = tool_input.get("command", "") if isinstance(tool_input, dict) else str(tool_input)
        cmd_stripped = cmd.strip().lower()
        if any(cmd_stripped.startswith(p) for p in _BASH_SKIP_PREFIXES):
            return True
    return False


def _summarize_tool(tool_name: str, tool_input: dict | str) -> str:
    """One-line description of a tool use, for its episode text."""
    if isinstance(tool_input, dict):
        fp = tool_input.get("file_path", "")
        cmd = tool_input.get("command", "")
        if fp:
            return f"{tool_name} on {fp}"
        if cmd:
            return str(cmd)[:200]
    return str(tool_input)[:200]


# Directory / file-type words that appear in almost every path and say nothing
# about what is being worked on (they only ever matched other path-bearing text).
_GENERIC_PATH_TOKENS = frozenset({
    "users", "home", "tmp", "temp", "appdata", "local", "roaming", "documents",
    "desktop", "downloads", "src", "lib", "libs", "bin", "usr", "var", "opt", "etc",
    "proj", "project", "projects", "repo", "repos", "workspace", "worktrees", "claude",
    "node", "modules", "site", "packages", "dist", "build", "venv", "scripts",
    "txt", "json", "yaml", "yml", "toml", "html", "css", "tsx", "jsx", "exe",
})
_MAX_QUERY_KEYWORDS = 12
_WRITTEN_TEXT_CHARS = 400


def _tool_query_text(tool_input: dict | str) -> str:
    """The parts of a tool call that name its topic: file basename + command/prompt/query."""
    if not isinstance(tool_input, dict):
        return str(tool_input)
    parts: list[str] = []
    fp = tool_input.get("file_path")
    if isinstance(fp, str) and fp:
        base = re.split(r"[\\/]", fp)[-1]
        parts.append(base.rsplit(".", 1)[0] if "." in base else base)
    for key in ("command", "prompt", "query", "description"):
        val = tool_input.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    # What is about to be written: decisions about THIS change should surface now,
    # even when the file name says nothing about them.
    written = [tool_input.get("new_string"), tool_input.get("content")]
    written += [e.get("new_string") for e in tool_input.get("edits") or [] if isinstance(e, dict)]
    parts += [w[:_WRITTEN_TEXT_CHARS] for w in written if isinstance(w, str) and w]
    return " ".join(parts)


def _extract_query_keywords(tool_name: str, tool_input: dict | str) -> str | None:
    if _should_skip_tool(tool_name, tool_input):
        return None
    keywords: list[str] = []
    for tok in re.findall(r"[a-z0-9]+", _tool_query_text(tool_input).lower()):
        if (len(tok) > 2 and not tok.isdigit() and tok not in _STOPWORDS
                and tok not in _GENERIC_PATH_TOKENS and tok not in keywords):
            keywords.append(tok)
        if len(keywords) >= _MAX_QUERY_KEYWORDS:
            break
    return " ".join(keywords) if keywords else None


# ── Hook activity log (opt-in, for watching hooks in real time) ──────────────

def _record_activity(event: str, **fields: object) -> None:
    """Append one JSON line per memory-touching hook call to
    MEMCONTEXT_HOOK_ACTIVITY_LOG (unset = off). Never fails a hook."""
    path = os.environ.get("MEMCONTEXT_HOOK_ACTIVITY_LOG", "").strip()
    if not path:
        return
    import json

    with contextlib.suppress(OSError, TypeError, ValueError), open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time_ns(), "event": event, **fields}) + "\n")


def _injected_lines(response: dict) -> list[str]:
    ctx = (response.get("hookSpecificOutput") or {}).get("additionalContext") or ""
    return [line[2:] for line in ctx.splitlines() if line.startswith("- ")]


def _target(tool_input: object) -> str:
    fp = tool_input.get("file_path") if isinstance(tool_input, dict) else None
    if isinstance(fp, str) and fp:
        return re.split(r"[\\/]", fp)[-1]
    cmd = tool_input.get("command") if isinstance(tool_input, dict) else None
    return " ".join(str(cmd or "").split())[:60]


# ── Hook endpoints ───────────────────────────────────────
#
# Each handler parses the (possibly malformed) body on the event loop, then runs
# the blocking work — extraction, embedding, SQLite — in the threadpool so one slow
# hook never stalls the server (Claude Code cancels hooks after 5-10s).
# Hooks honour the caller's principal exactly like /api/memory/*: writes need
# can_write, and both reads and writes are confined to the caller's namespace.

_hook_extractor = None


def _episode_only(_turn: object) -> list:
    """Extractor that keeps the turn as an episode and derives no claims from it."""
    return []


def _get_hook_extractor():
    """Select the text extractor once per process, not once per hook call.

    auto_extractor() probes for a local Ollama with a network timeout, which cost
    seconds on every captured prompt.

    With no LLM backend, auto_extractor() falls back to the regex SimpleExtractor,
    which turns free-form prompts into junk claims (subject "we"/"the", values cut
    at the first '.', so "MySQL 5.7" -> "MySQL 5"). Prompts are then kept as
    episodes only — still searchable, no fabricated facts.
    """
    global _hook_extractor
    if _hook_extractor is None:
        from memcontext import mcp_tools
        from memcontext.extractors import SimpleExtractor
        selected = mcp_tools.auto_extractor()
        if isinstance(selected, SimpleExtractor):
            log.warning(
                "hook.prompt_claims_disabled",
                reason="no LLM extractor; prompts are stored as episodes only",
                fix="set MEMCONTEXT_EXTRACTOR_BACKEND to enable claim extraction from prompts",
            )
            selected = _episode_only
        _hook_extractor = selected
    return _hook_extractor


def _hook_scope(request: Request) -> tuple[str | None, bool]:
    return (
        getattr(request.state, "namespace", None),
        getattr(request.state, "can_write", True),
    )


_FORBIDDEN = JSONResponse({"status": "forbidden"}, status_code=403)


@_serialized
def _capture_tool_use(body: dict, namespace: str) -> dict:
    tool_name = body.get("tool_name", "")
    tool_input = body.get("tool_input", {})
    session_id = body.get("session_id", "hooks")

    if _should_skip_tool(tool_name, tool_input):
        return {"status": "skipped"}

    value = _summarize_tool(tool_name, tool_input)

    from memcontext import admission
    if not admission.admit(value).admitted:
        return {"status": "filtered"}

    # Only store edits/writes — the actions that change state
    if tool_name not in ("Edit", "Write", "Bash", "PowerShell"):
        return {"status": "skipped"}

    # Episode only: the turn is the searchable, provenance-bearing record of what
    # was done. A structured "action" claim per call duplicated identical writes
    # and let unrelated commands supersede each other on token overlap.
    from memcontext import mcp_tools
    mcp_tools.handle_memory_store(
        get_conn(),
        text=f"[source: tool] {tool_name}: {value}"[:500],
        speaker="assistant",
        session_id=session_id,
        namespace=namespace,
        extractor=_episode_only,
    )
    _record_activity("PostToolUse", tool=tool_name, target=_target(tool_input), stored=True)
    return {"status": "ok"}


def _claim_line(subject: str | None, predicate: str | None, fact: str) -> str:
    """One context line: ``subject / predicate: <fact>`` for structured facts, else the fact."""
    fact = " ".join((fact or "").split())
    if not (subject and predicate):
        return fact
    synthesized = f"{subject} {predicate} "  # structured facts' NL text is "s p v"
    body = fact[len(synthesized):] if fact.startswith(synthesized) else fact
    return f"{subject} / {predicate}: {body}"


def _render_context(header: str, lines: list[str], max_lines: int) -> str | None:
    out: list[str] = []
    size = len(header)
    for line in lines[:max_lines]:
        entry = f"\n- {line}"
        if size + len(entry) > _CONTEXT_MAX_CHARS:
            break
        out.append(entry)
        size += len(entry)
    return header + "".join(out) if out else None


_CONFLICT_MAX_VALUES = 4


def _conflict_entry(rows: list) -> str:
    """One context entry flagging several current values for one decision, newest first."""
    from memcontext.source_trust import QUARANTINE_THRESHOLD

    out = [f"CONFLICT: {len(rows)} current values recorded for the same decision (newest first)."
           " Prefer the newest trusted value unless the user says otherwise, and store the"
           " resolution with memory_store:"]
    for i, r in enumerate(rows[:_CONFLICT_MAX_VALUES], start=1):
        tags = (["newest"] if i == 1 else []) + (
            ["untrusted source"] if r["trust"] < QUARANTINE_THRESHOLD else [])
        line = _claim_line(r["subject"], r["predicate"], r["text"] or r["value"] or "")
        out.append(f"    {i}. {line}" + (f"  ({', '.join(tags)})" if tags else ""))
    return "\n".join(out)


def _with_conflicts(candidates: list[tuple[str, str | None, str | None, str]],
                    namespace: str | None) -> list[str]:
    """Context entries for ranked candidates ``(claim_id, subject, predicate, line)``.

    A candidate that has another CURRENT value of the same kind of decision (see
    ``memcontext.conflicts``) becomes one CONFLICT entry listing every such value
    newest first, at the candidate's rank. Its partners are looked up in the store,
    not just among the ranked candidates: the newer value is often the one the
    query did not match. Each value appears once.
    """
    from memcontext.conflicts import live_same_kind

    conn = get_conn()
    entries: list[str] = []
    consumed: set[str] = set()
    for claim_id, subject, predicate, line in candidates:
        if claim_id in consumed:
            continue
        group = live_same_kind(conn, subject=subject, predicate=predicate, namespace=namespace)
        if len(group) >= 2:
            consumed |= {r["claim_id"] for r in group}
            entries.append(_conflict_entry(group))
        elif line not in entries:
            entries.append(line)
    return entries


def _hook_context(event: str, context: str | None) -> dict:
    if not context:
        return {}
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": context}}


def _prompt_context(prompt: str, namespace: str | None,
                    session_id: str = "hooks") -> str | None:
    """Current (non-superseded) facts relevant to the prompt, across all sessions.

    Scores live claims directly by shared content words, like PreToolUse, instead of
    filtering the general ranker's top-k: on real memory that ranker's near-uniform
    lexical scores (ties broken by insertion order, GAP-9) filled the top-k with
    superseded versions and unrelated decisions, so the current decision the user
    asked about never reached the agent. Strongest matches first (newest wins ties);
    when any claim shares 2+ content words, single-word matches are dropped. Episodes
    are skipped: raw text can restate a superseded value.
    """
    from memcontext.claims import row_to_claim

    prompt_tokens = _content_tokens(prompt)
    if not prompt_tokens:
        return None
    scored: list[tuple[int, tuple[str, str | None, str | None, str]]] = []
    for row in _active_claim_rows(namespace):  # newest first
        c = row_to_claim(row)
        line = _claim_line(c.subject, c.predicate, c.text or c.value or "")
        overlap = len(prompt_tokens & _claim_match_tokens(c.predicate, line))
        if overlap:
            scored.append((overlap, (c.claim_id, c.subject, c.predicate, line)))
    if not scored:
        return None
    floor = 2 if max(score for score, _ in scored) >= 2 else 1
    scored.sort(key=lambda x: -x[0])  # stable: newest-first within equal overlap
    candidates = [cand for score, cand in scored if score >= floor][: _PROMPT_CONTEXT_MAX_LINES * 2]
    entries = _with_conflicts(candidates, namespace)

    from memcontext import mcp_tools
    mcp_tools._record_serve_events(  # what reached the agent stays verifiable
        get_conn(), request_session_id=session_id,
        claim_ids=[cand[0] for cand in candidates[:_PROMPT_CONTEXT_MAX_LINES]],
        query=prompt[:_PROMPT_QUERY_MAX_CHARS],
    )
    return _render_context(
        "[MemContext] Current project memory relevant to this prompt:",
        entries, _PROMPT_CONTEXT_MAX_LINES,
    )


@_serialized
def _capture_prompt(body: dict, namespace: str | None) -> dict:
    prompt = body.get("prompt", "")
    session_id = body.get("session_id", "hooks")

    if not isinstance(prompt, str) or not prompt or prompt.startswith("/"):
        return {}

    from memcontext import admission
    if not admission.admit(prompt).admitted:
        return {}  # never query with (and so never log) text admission rejected

    # Retrieve before storing, so the prompt never retrieves itself.
    started = time.perf_counter()
    context = _prompt_context(prompt, namespace, session_id)
    response = _hook_context("UserPromptSubmit", context)
    _record_activity("UserPromptSubmit", prompt=" ".join(prompt.split())[:200],
                     injected=_injected_lines(response),
                     ms=round((time.perf_counter() - started) * 1000, 1))

    from memcontext import mcp_tools
    mcp_tools.handle_memory_store(
        get_conn(),
        text=prompt[:2000],
        speaker="user",
        session_id=session_id,
        namespace=namespace or "default",
        extractor=_get_hook_extractor(),
    )
    return response


_TOOL_CONTEXT_MAX_LINES = 5
_TOOL_CONTEXT_ROW_LIMIT = 500


def _active_claim_rows(namespace: str | None) -> list:
    """Most recent live claims, minus tool-action log entries (they only restate paths)."""
    conn = get_conn()
    live = "c.status IN ('active','confirmed','audited') AND (c.predicate IS NULL OR c.predicate != ?)"
    if namespace is None:  # single shared token = unrestricted
        return conn.execute(
            f"SELECT c.* FROM claims c WHERE {live} ORDER BY c.created_ts DESC LIMIT ?",
            (_TOOL_ACTION_PREDICATE, _TOOL_CONTEXT_ROW_LIMIT),
        ).fetchall()
    return conn.execute(
        "SELECT c.* FROM claims c JOIN turns t ON t.turn_id = c.source_turn_id"
        f" WHERE {live} AND t.namespace = ? ORDER BY c.created_ts DESC LIMIT ?",
        (_TOOL_ACTION_PREDICATE, namespace, _TOOL_CONTEXT_ROW_LIMIT),
    ).fetchall()


@_serialized
def _context_for_tool(body: dict, namespace: str | None) -> dict:
    tool_name = body.get("tool_name", "")
    tool_input = body.get("tool_input", {})

    keywords = _extract_query_keywords(tool_name, tool_input)
    if not keywords:
        return {}

    start = time.perf_counter()
    response = _tool_context(keywords, namespace, start)
    _record_activity("PreToolUse", tool=tool_name, target=_target(tool_input), query=keywords,
                     injected=_injected_lines(response),
                     ms=round((time.perf_counter() - start) * 1000, 1))
    return response


def _tool_context(keywords: str, namespace: str | None, start: float) -> dict:
    from memcontext.claims import row_to_claim
    rows = _active_claim_rows(namespace)

    if time.perf_counter() - start > 0.15:
        return {}

    query_tokens = set(keywords.split())
    scored: list[tuple[float, tuple[str, str | None, str | None, str]]] = []
    for row in rows:
        c = row_to_claim(row)
        line = _claim_line(c.subject, c.predicate, c.text or c.value or "")
        overlap = len(query_tokens & _claim_match_tokens(c.predicate, line))
        if overlap > 0:
            scored.append((overlap / len(query_tokens), (c.claim_id, c.subject, c.predicate, line)))

    if time.perf_counter() - start > 0.2:
        return {}

    scored.sort(key=lambda x: -x[0])  # stable: ties keep newest-first order
    # same conflict flag as the prompt hook: this is the context right before an edit
    top = [cand for _score, cand in scored][: _TOOL_CONTEXT_MAX_LINES * 2]
    entries = _with_conflicts(top, namespace)
    return _hook_context(
        "PreToolUse",
        _render_context("[MemContext] Relevant context:", entries, _TOOL_CONTEXT_MAX_LINES),
    )


async def _hook_body(request: Request) -> dict:
    body = await request.json()
    return body if isinstance(body, dict) else {}


def _hook_failed(hook: str, exc: Exception) -> None:
    # Type only (never the message): hook payloads carry user content.
    log.warning("hook.failed", hook=hook, error_type=type(exc).__name__)


@app.post("/api/hooks/post_tool_use")
async def hook_post_tool_use(request: Request):
    """Capture meaningful tool actions silently."""
    namespace, can_write = _hook_scope(request)
    if not can_write:
        return _FORBIDDEN
    try:
        body = await _hook_body(request)
        return await run_in_threadpool(_capture_tool_use, body, namespace or "default")
    except Exception as exc:
        _hook_failed("post_tool_use", exc)
        return {"status": "error"}


@app.post("/api/hooks/user_prompt_submit")
async def hook_user_prompt_submit(request: Request):
    """Capture the prompt and inject current memory relevant to it."""
    namespace, can_write = _hook_scope(request)
    if not can_write:
        return _FORBIDDEN
    try:
        body = await _hook_body(request)
        return await run_in_threadpool(_capture_prompt, body, namespace)
    except Exception as exc:
        _hook_failed("user_prompt_submit", exc)
        return {"status": "error"}


@app.post("/api/hooks/pre_tool_use")
async def hook_pre_tool_use(request: Request):
    """Inject relevant memory context before tool calls."""
    namespace, _ = _hook_scope(request)
    try:
        body = await _hook_body(request)
        return await run_in_threadpool(_context_for_tool, body, namespace)
    except Exception as exc:
        _hook_failed("pre_tool_use", exc)
        return {}


@app.post("/api/hooks/stop")
async def hook_stop(request: Request):
    """Session boundary marker. No-op."""
    return {"status": "ok"}


# ── Core ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "memcontext"}


def run_server(*, db_path: str = "memcontext.db", port: int = 8100, host: str = "127.0.0.1"):
    import uvicorn
    init_db(db_path)
    _configure_auth()  # resolve/print the bearer token before serving

    # Pay model load + extractor selection once at startup, not inside the first
    # hook call (Claude Code cancels hooks after 5-10s).
    from memcontext.mcp_server import prewarm_embedder
    prewarm_embedder()
    _get_hook_extractor()
    print("[memcontext] HTTP API ready (models loaded)", file=sys.stderr, flush=True)

    # Mount MCP Streamable HTTP endpoint — ChatGPT connects here via Developer Mode
    try:
        from memcontext.mcp_server import create_http_app
        mcp_app = create_http_app(db_path)
        app.mount("/mcp", mcp_app)
    except ImportError:
        pass

    uvicorn.run(app, host=host, port=port, log_level="info")
