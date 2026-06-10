"""MemContext HTTP API — REST interface for any AI platform.

MCP is for Claude Code and Cursor. HTTP is for everything else:
ChatGPT GPTs, Gemini, custom agents, browser extensions.
Also serves Claude Code ambient hooks for silent context capture.

Same database, same memory. Two doors in.

    memcontext serve-http --port 8100 --db memcontext.db
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import time

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

app = FastAPI(
    title="MemContext",
    description="Universal AI memory layer. Store, query, and trace structured claims with provenance.",
    version="0.1.0",
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


# ── Auth: bearer token on every /api/* route (memory, hooks, sessions) ────────
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
    if request.url.path.startswith("/api/"):
        header = request.headers.get("authorization", "")
        provided = header[7:].strip() if header[:7].lower() == "bearer " else ""
        # Per-principal access control once any principal is registered; otherwise
        # the single shared token applies (backward compatible).
        from memcontext.authz import any_principals, resolve_principal

        conn = _conn
        if conn is not None and any_principals(conn):
            principal = resolve_principal(conn, provided)
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
    return await call_next(request)


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
def memory_query(req: QueryRequest, request: Request):
    from memcontext.mcp_tools import handle_memory_query
    ns = getattr(request.state, "namespace", None)
    return handle_memory_query(
        get_conn(), query=req.query,
        session_id=req.session_id, top_k=req.top_k,
        namespace=ns,
    )


@app.post("/api/memory/trace")
def memory_trace(req: TraceRequest):
    from memcontext.mcp_tools import handle_memory_trace
    return handle_memory_trace(get_conn(), claim_id=req.claim_id)


@app.get("/api/memory/status")
def memory_status():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    active = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE status IN ('active','confirmed','audited')"
    ).fetchone()[0]
    sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM claims").fetchone()[0]
    turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    return {
        "total_claims": total,
        "active_claims": active,
        "sessions": sessions,
        "turns": turns,
    }


# ── Hook filtering ───────────────────────────────────────

_HOOK_SKIP_TOOLS: set[str] = {
    "Read", "Glob", "Grep", "LS", "LSP",
    "TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskStop", "TaskOutput",
    "Monitor", "AskUserQuestion", "WebSearch", "WebFetch",
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


def _summarize_tool(tool_name: str, tool_input: dict | str) -> tuple[str, str]:
    """Return (subject, value) for a tool use claim."""
    if isinstance(tool_input, dict):
        fp = tool_input.get("file_path", "")
        cmd = tool_input.get("command", "")
        if fp:
            return fp, f"{tool_name} on {fp}"
        if cmd:
            return tool_name, cmd[:200]
        return tool_name, str(tool_input)[:200]
    return tool_name, str(tool_input)[:200]


def _extract_query_keywords(tool_name: str, tool_input: dict | str) -> str | None:
    if _should_skip_tool(tool_name, tool_input):
        return None
    if isinstance(tool_input, dict):
        raw = (tool_input.get("file_path", "") or
               tool_input.get("command", "") or
               tool_input.get("prompt", "") or
               tool_input.get("query", "") or
               str(tool_input))
    else:
        raw = str(tool_input)
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", raw)
    noise = {"the", "a", "an", "is", "are", "to", "for", "and", "or", "of",
             "in", "on", "at", "it", "my", "this", "that", "with", "from",
             "import", "def", "class", "return", "if", "else", "true",
             "false", "none", "self", "not"}
    seen: set[str] = set()
    keywords: list[str] = []
    for t in tokens:
        tl = t.lower()
        if tl not in noise and len(t) > 2 and tl not in seen:
            seen.add(tl)
            keywords.append(t)
        if len(keywords) >= 5:
            break
    return " ".join(keywords) if keywords else None


# ── Hook endpoints ───────────────────────────────────────

@app.post("/api/hooks/post_tool_use")
async def hook_post_tool_use(request: Request):
    """Capture meaningful tool actions silently."""
    try:
        body = await request.json()
        tool_name = body.get("tool_name", "")
        tool_input = body.get("tool_input", {})
        session_id = body.get("session_id", "hooks")

        if _should_skip_tool(tool_name, tool_input):
            return {"status": "skipped"}

        subject, value = _summarize_tool(tool_name, tool_input)

        from memcontext import admission
        if not admission.admit(value).admitted:
            return {"status": "filtered"}

        # Only store edits/writes — the actions that change state
        if tool_name not in ("Edit", "Write", "Bash", "PowerShell"):
            return {"status": "skipped"}

        from memcontext.mcp_tools import handle_memory_store
        handle_memory_store(
            get_conn(),
            text=f"[source: tool] {tool_name}: {value}"[:500],
            speaker="assistant",
            session_id=session_id,
            claims=[{
                "subject": subject.lower().replace("\\", "/").split("/")[-1] if "/" in subject or "\\" in subject else subject.lower(),
                "predicate": "action",
                "value": f"{tool_name}: {value}"[:200],
                "confidence": 0.7,
            }],
        )
        return {"status": "ok"}
    except Exception:
        return {"status": "error"}


@app.post("/api/hooks/user_prompt_submit")
async def hook_user_prompt_submit(request: Request):
    """Capture user decisions and intent silently."""
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        session_id = body.get("session_id", "hooks")

        if not prompt or prompt.startswith("/"):
            return {"status": "skipped"}

        from memcontext import admission
        if not admission.admit(prompt).admitted:
            return {"status": "skipped"}

        from memcontext.mcp_tools import handle_memory_store
        handle_memory_store(
            get_conn(),
            text=prompt[:2000],
            speaker="user",
            session_id=session_id,
        )
        return {"status": "ok"}
    except Exception:
        return {"status": "error"}


@app.post("/api/hooks/pre_tool_use")
async def hook_pre_tool_use(request: Request):
    """Inject relevant memory context before tool calls."""
    try:
        body = await request.json()
        tool_name = body.get("tool_name", "")
        tool_input = body.get("tool_input", {})

        keywords = _extract_query_keywords(tool_name, tool_input)
        if not keywords:
            return {}

        start = time.monotonic()

        from memcontext.claims import row_to_claim
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM claims"
            " WHERE status IN ('active','confirmed','audited')"
            " ORDER BY created_ts DESC LIMIT 500",
        ).fetchall()

        if time.monotonic() - start > 0.15:
            return {}

        query_tokens = set(re.findall(r"[a-z0-9]+", keywords.lower()))
        scored = []
        for row in rows:
            c = row_to_claim(row)
            claim_text = f"{c.subject} {c.predicate} {c.value}".lower()
            claim_tokens = set(re.findall(r"[a-z0-9]+", claim_text))
            overlap = len(query_tokens & claim_tokens)
            if overlap > 0:
                score = overlap / max(len(query_tokens), 1)
                scored.append((c, score))

        if time.monotonic() - start > 0.2:
            return {}

        if not scored:
            return {}

        scored.sort(key=lambda x: -x[1])
        lines = []
        char_count = 0
        for c, score in scored[:5]:
            line = f"- {c.subject}: {c.value}"
            if char_count + len(line) > 1500:
                break
            lines.append(line)
            char_count += len(line)

        if not lines:
            return {}

        context = "[MemContext] Relevant context:\n" + "\n".join(lines)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": context,
            }
        }
    except Exception:
        return {}


@app.post("/api/hooks/stop")
async def hook_stop(request: Request):
    """Session boundary marker. No-op."""
    return {"status": "ok"}


# ── Session propagation (Chrome extension → Agent browser) ──

AGENT_PROFILE = None

def _get_cookie_cache():
    import os
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".agent_chrome_profile")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "_stolen_cookies.json")

@app.post("/api/sessions/export")
async def sessions_export(request: Request):
    body = await request.json()
    cookies = body.get("cookies", [])
    if not cookies:
        raise HTTPException(400, "No cookies provided")

    path = _get_cookie_cache()
    with open(path, "w") as f:
        json.dump(cookies, f)

    domains = {c.get("domain", "") for c in cookies}
    return {
        "status": "ok",
        "cookies_received": len(cookies),
        "domains": len(domains),
        "saved_to": path,
    }

@app.get("/api/sessions/status")
def sessions_status():
    import os
    path = _get_cookie_cache()
    if os.path.exists(path):
        with open(path) as f:
            cookies = json.load(f)
        domains = {c.get("domain", "") for c in cookies}
        return {"has_sessions": True, "cookies": len(cookies), "domains": len(domains)}
    return {"has_sessions": False}


# ── Core ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "memcontext"}


def run_server(*, db_path: str = "memcontext.db", port: int = 8100, host: str = "127.0.0.1"):
    import uvicorn
    init_db(db_path)
    _configure_auth()  # resolve/print the bearer token before serving

    # Mount MCP Streamable HTTP endpoint — ChatGPT connects here via Developer Mode
    try:
        from memcontext.mcp_server import create_http_app
        mcp_app = create_http_app(db_path)
        app.mount("/mcp", mcp_app)
    except ImportError:
        pass

    uvicorn.run(app, host=host, port=port, log_level="info")
