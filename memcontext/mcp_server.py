"""MemContext MCP server — exposes memory tools over Model Context Protocol.

Requires the 'mcp' package: pip install mcp
All MCP-specific imports are lazy so mcp_tools.py works standalone.
"""
from __future__ import annotations

import json

import structlog

log = structlog.get_logger(__name__)


def run_server(
    *,
    db_path: str = "memcontext.db",
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
    oauth: bool = False,
    public_url: str | None = None,
    oauth_password: str | None = None,
) -> None:
    """Start the MCP server. Called by `memcontext serve`.

    transport="stdio" (default) for local clients (Claude Code / Desktop, one config
    line). transport="http" serves Streamable HTTP so a REMOTE client (claude.ai web,
    via a tunnel) can connect by URL — the shape Membase/GBrain use. HTTP auth is either
    a bearer ``token`` (simple) or, with ``oauth=True``, the full OAuth 2.1 flow
    (metadata + dynamic client registration + PKCE + a password login gate) that
    claude.ai's "add custom connector → log in" expects.
    """
    if transport == "http":
        if oauth:
            _run_http_oauth(db_path=db_path, host=host, port=port,
                            public_url=public_url, password=oauth_password)
        else:
            _run_http(db_path=db_path, host=host, port=port, token=token)
        return
    try:
        import mcp.server.stdio
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError:
        raise ImportError(
            "MCP server requires the 'mcp' package. Install with: pip install memcontext[mcp]"
        ) from None

    import asyncio
    import logging
    import sys
    import structlog
    # Redirect structlog to stderr so it doesn't pollute the stdio JSON-RPC transport
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )

    from memcontext.schema import open_database
    from memcontext.mcp_tools import (
        handle_brain,
        handle_memory_correct,
        handle_memory_digest,
        handle_memory_entity_graph,
        handle_memory_events,
        handle_memory_life_events,
        handle_memory_observe,
        handle_memory_observe_url,
        handle_memory_payload,
        handle_memory_profile,
        handle_memory_query,
        handle_memory_stats,
        handle_memory_store,
        handle_memory_trace,
        handle_memory_tuples,
        handle_memory_volatility,
        handle_memory_working_context,
        handle_memory_procedures,
        handle_memory_output_provenance,
        handle_memory_forget,
        handle_memory_trust_status,
    )

    conn = open_database(db_path)

    # Persistent extractor + background extraction queue so a deferrable (LLM)
    # extractor runs off the request path. ThreadedQueue needs a file-backed DB
    # and is only useful when the extractor actually defers; otherwise stay inline.
    from memcontext.extractors import auto_extractor
    store_extractor = auto_extractor()
    store_queue = None
    if db_path != ":memory:" and getattr(store_extractor, "is_deferrable", False):
        from memcontext.extraction_queue import ThreadedQueue
        from memcontext.retrieval import semantic_supersession
        store_queue = ThreadedQueue(
            db_path, extractor=store_extractor, semantic=semantic_supersession()
        )

    server = Server("memcontext")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="memory_store",
                description="Store a conversation turn and extract claims into memory.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text to store"},
                        "speaker": {"type": "string", "enum": ["user", "assistant"], "default": "user"},
                        "session_id": {"type": "string", "description": "Session ID (auto-generated if omitted)"},
                        "claims": {
                            "type": "array",
                            "description": "Pre-extracted claims. If provided, bypasses automatic extraction.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "subject": {"type": "string"},
                                    "predicate": {"type": "string"},
                                    "value": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["value"],
                            },
                        },
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="memory_query",
                description="Query the user's personal memory -- decisions, observations, bug tracking, project status, and context from their coding sessions, browser observations, and cross-tool workflows. Use this for anything about the user's own projects, preferences, or work history.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "session_id": {"type": "string", "default": "default"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="memory_trace",
                description=(
                    "Trace a fact's source turn and typed supersession lineage. "
                    "Pass a claim_id, or a (subject, predicate) pair to trace the "
                    "current value of that slot. Returns the active claim on top "
                    "and the superseded chain beneath, each with its source span "
                    "and edge type (e.g. user_correction)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "description": "Claim ID to trace (optional if subject+predicate given)."},
                        "subject": {"type": "string", "description": "Subject to trace (with predicate)."},
                        "predicate": {"type": "string", "description": "Predicate to trace (with subject)."},
                        "session_id": {"type": "string", "default": "default"},
                    },
                },
            ),
            Tool(
                name="brain",
                description=(
                    "Return the deterministic world-state projection grouped by "
                    "subject. Every fact carries its value, status (ACTIVE/"
                    "SUPERSEDED), confidence, and a provenance handle (source turn "
                    "id + character span). Also reports per-subject gaps: "
                    "vocabulary predicates with no active claim. No LLM in this path."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "default": "default"},
                    },
                },
            ),
            Tool(
                name="memory_payload",
                description=(
                    "Return the memory payload for a question in one of three "
                    "modes, to compare what different memories hand the same "
                    "reader: 'summary' (raw transcript blob), 'vector' (top-k "
                    "statements by similarity), or 'memcontext' (structured "
                    "projection with current value, provenance, and typed "
                    "supersession). Used by the differentiator demo."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to answer."},
                        "mode": {"type": "string", "enum": ["summary", "vector", "memcontext"]},
                        "session_id": {"type": "string", "default": "default"},
                    },
                    "required": ["question", "mode"],
                },
            ),
            Tool(
                name="memory_correct",
                description="Correct or dismiss an existing claim.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "action": {"type": "string", "enum": ["dismiss", "correct"]},
                        "new_value": {"type": "string", "description": "Required when action is 'correct'"},
                    },
                    "required": ["claim_id", "action"],
                },
            ),
            Tool(
                name="memory_observe",
                description="Store browser observation claims from a page accessibility snapshot.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Page URL"},
                        "title": {"type": "string", "description": "Page title"},
                        "accessibility_tree": {"type": "object", "description": "Playwright accessibility tree snapshot"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["url"],
                },
            ),
            Tool(
                name="memory_observe_url",
                description=(
                    "Observe a live URL — capture its accessibility tree, extract "
                    "structured claims, store with provenance. PREFERRED auth: set "
                    "connect_browser=true to read from the user's running Chrome "
                    "(inherits all auth — SSO, 2FA, OAuth — and never handles raw "
                    "passwords). Raw login_email/password is a security hazard and is "
                    "DISABLED unless allow_password_login=true; prefer connect_browser. "
                    "Works on any page the user can see. Re-observing detects changes."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to observe"},
                        "session_id": {
                            "type": "string",
                            "description": "Session ID. Omit to use shared default session for cross-app queries.",
                        },
                        "login_email": {
                            "type": "string",
                            "description": "Email/username for form login. Prefer connect_browser instead.",
                        },
                        "login_password": {
                            "type": "string",
                            "description": (
                                "Raw password for form login — SECURITY HAZARD. Ignored "
                                "unless allow_password_login=true. Prefer connect_browser."
                            ),
                        },
                        "login_url": {
                            "type": "string",
                            "description": "Login page URL if different from the target URL.",
                        },
                        "connect_browser": {
                            "type": "boolean",
                            "description": (
                                "PREFERRED. Attach to the user's running Chrome (port 9222) instead of "
                                "launching headless. Inherits all auth sessions — no credentials needed."
                            ),
                            "default": False,
                        },
                        "allow_password_login": {
                            "type": "boolean",
                            "description": (
                                "Explicit opt-in required to use login_email/login_password. "
                                "Leave false and use connect_browser unless you truly must."
                            ),
                            "default": False,
                        },
                    },
                    "required": ["url"],
                },
            ),
            Tool(
                name="memory_profile",
                description="Get the smart profile for a subject (default: 'user'). Returns a structured summary of key facts, preferences, and changes.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "default": "user"},
                        "max_tokens": {"type": "integer", "default": 500},
                    },
                },
            ),
            Tool(
                name="memory_stats",
                description="Get storage statistics: active claims, superseded claims, turns, profiles, digests, life events.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="memory_digest",
                description="Build and return the session digest (the summary layer): top key facts by importance, supersession updates (old->new), and a remaining-fact count. Deterministic, zero-LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                },
            ),
            Tool(
                name="memory_life_events",
                description="Detect and return life events for a subject: bursts of diverse predicate changes within a time window (e.g. a move, a new job). Deterministic, zero-LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "default": "user"},
                        "window_hours": {"type": "integer", "default": 24},
                        "min_predicates": {"type": "integer", "default": 3},
                    },
                },
            ),
            Tool(
                name="memory_events",
                description="Assemble and return event frames for a session: co-referent claims grouped into multi-slot event records (who/what/where/when/amount). Deterministic, zero-LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                },
            ),
            Tool(
                name="memory_volatility",
                description="Classify how volatile a (subject, predicate) slot is from its supersession history: stable / evolving / volatile. Deterministic, zero-LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "default": "user"},
                        "predicate": {"type": "string"},
                    },
                    "required": ["predicate"],
                },
            ),
            Tool(
                name="memory_working_context",
                description="Assemble the task-relevant memory for a session within a token budget, cued by recent turns (query-free) instead of all active memory. Deterministic, zero-LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "token_budget": {"type": "integer", "default": 2000},
                    },
                    "required": ["session_id"],
                },
            ),
            Tool(
                name="memory_procedures",
                description="Detect recurring procedures (ordered action sequences) across sessions. EXPERIMENTAL: returns disabled unless MEMCONTEXT_EXPERIMENTAL_PROCEDURAL=1. Deterministic, zero-LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {"min_sessions": {"type": "integer", "default": 2}},
                },
            ),
            Tool(
                name="memory_output_provenance",
                description="Output-sentence provenance (audit): record which generated sentences cite which claims, and trace claim<->sentence<->turn links. Deterministic, zero-LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "record": {"type": "array"},
                        "claim_id": {"type": "string"},
                        "turn_id": {"type": "string"},
                        "sentence_id": {"type": "string"},
                    },
                },
            ),
            Tool(
                name="memory_forget",
                description="Right-to-be-forgotten: hard-delete memory and cascade along provenance+supersession (no residual), audited. Specify one of claim_id/subject/session_id/predicate.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "subject": {"type": "string"},
                        "session_id": {"type": "string"},
                        "predicate": {"type": "string"},
                        "reason": {"type": "string", "default": "user_request"},
                    },
                },
            ),
            Tool(
                name="memory_trust_status",
                description="Trust observability: source-trust distribution, contradiction rate, forgetting + drift audit, tenant distribution, staleness proxy. Measures the trust layer, not recall.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="memory_tuples",
                description="Project a session's active facts into event tuples (subject, action, object, validity window). Pure read projection, zero-LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                },
            ),
            Tool(
                name="memory_entity_graph",
                description="Return an entity's co-occurrence neighbors within a session's claim graph (entities mentioned together in the same turn). Deterministic, zero-LLM.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "entity": {"type": "string"},
                        "max_hops": {"type": "integer", "default": 1},
                    },
                    "required": ["session_id", "entity"],
                },
            ),
            Tool(
                name="tool_discover",
                description=(
                    "Curate the agent's tool set: return the top-K most relevant tools "
                    "from the registry for a query, instead of exposing the whole "
                    "toolset (cuts prompt bloat, sharpens selection). Query-only by "
                    "default; set use_memory=true to also condition on the user's "
                    "memory. The agent still chooses the tool."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The task/query to find tools for"},
                        "session_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Memory sessions to condition on (use_memory only)",
                        },
                        "top_k": {"type": "integer", "default": 10},
                        "use_memory": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        try:
            if name == "memory_store":
                result = handle_memory_store(
                    conn, extractor=store_extractor, queue=store_queue, **arguments
                )
            elif name == "memory_query":
                result = handle_memory_query(conn, **arguments)
            elif name == "brain":
                result = handle_brain(conn, **arguments)
            elif name == "memory_payload":
                result = handle_memory_payload(conn, **arguments)
            elif name == "memory_trace":
                result = handle_memory_trace(conn, **arguments)
            elif name == "memory_correct":
                result = handle_memory_correct(conn, **arguments)
            elif name == "memory_observe":
                result = handle_memory_observe(conn, **arguments)
            elif name == "memory_observe_url":
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: handle_memory_observe_url(conn, **arguments)
                )
            elif name == "memory_profile":
                result = handle_memory_profile(conn, **arguments)
            elif name == "memory_stats":
                result = handle_memory_stats(conn)
            elif name == "memory_digest":
                result = handle_memory_digest(conn, **arguments)
            elif name == "memory_life_events":
                result = handle_memory_life_events(conn, **arguments)
            elif name == "memory_events":
                result = handle_memory_events(conn, **arguments)
            elif name == "memory_volatility":
                result = handle_memory_volatility(conn, **arguments)
            elif name == "memory_working_context":
                result = handle_memory_working_context(conn, **arguments)
            elif name == "memory_procedures":
                result = handle_memory_procedures(conn, **arguments)
            elif name == "memory_output_provenance":
                result = handle_memory_output_provenance(conn, **arguments)
            elif name == "memory_forget":
                result = handle_memory_forget(conn, **arguments)
            elif name == "memory_trust_status":
                result = handle_memory_trust_status(conn, **arguments)
            elif name == "memory_tuples":
                result = handle_memory_tuples(conn, **arguments)
            elif name == "memory_entity_graph":
                result = handle_memory_entity_graph(conn, **arguments)
            elif name == "tool_discover":
                from memcontext.mcp_tools import handle_tool_discover

                result = handle_tool_discover(conn, **arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as exc:  # malformed/hostile input or handler error
            # Never leak the exception message/traceback/path to the client or
            # the logs — only the exception type (safe) goes to stderr.
            log.error("mcp.tool_error", tool=name, error_type=type(exc).__name__)
            result = {"error": "internal error", "tool": name}
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _run():
        async with mcp.server.stdio.stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    try:
        asyncio.run(_run())
    finally:
        if store_queue is not None:
            store_queue.close()  # drain in-flight extraction + join the worker


def create_http_app(db_path: str = "memcontext.db"):
    """Create a Starlette ASGI app that serves MCP over Streamable HTTP.

    Mount this at /mcp on your FastAPI app:
        app.mount("/mcp", create_http_app("memcontext.db"))

    ChatGPT connects via: https://your-ngrok-url/mcp
    """
    from contextlib import asynccontextmanager

    import anyio
    from mcp.server import Server
    from mcp.server.streamable_http import StreamableHTTPServerTransport
    from mcp.types import TextContent, Tool
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

    from memcontext.mcp_tools import (
        handle_brain,
        handle_memory_correct,
        handle_memory_digest,
        handle_memory_entity_graph,
        handle_memory_events,
        handle_memory_life_events,
        handle_memory_observe,
        handle_memory_observe_url,
        handle_memory_payload,
        handle_memory_profile,
        handle_memory_query,
        handle_memory_stats,
        handle_memory_store,
        handle_memory_trace,
        handle_memory_tuples,
        handle_memory_volatility,
        handle_memory_working_context,
        handle_memory_procedures,
        handle_memory_output_provenance,
        handle_memory_forget,
        handle_memory_trust_status,
    )
    from memcontext.schema import open_database

    conn = open_database(db_path)
    transports: dict[str, StreamableHTTPServerTransport] = {}

    from memcontext.extractors import auto_extractor
    store_extractor = auto_extractor()
    store_queue = None
    if db_path != ":memory:" and getattr(store_extractor, "is_deferrable", False):
        from memcontext.extraction_queue import ThreadedQueue
        from memcontext.retrieval import semantic_supersession
        store_queue = ThreadedQueue(
            db_path, extractor=store_extractor, semantic=semantic_supersession()
        )

    def _build_server():
        """Create a fresh MCP Server instance with all tools registered."""
        server = Server("memcontext")

        @server.list_tools()
        async def list_tools():
            return [
                Tool(name="memory_store", description="Store a conversation turn and extract claims into memory.",
                     inputSchema={"type":"object","properties":{"text":{"type":"string"},"speaker":{"type":"string","enum":["user","assistant"],"default":"user"},"session_id":{"type":"string"},"claims":{"type":"array","items":{"type":"object","properties":{"subject":{"type":"string"},"predicate":{"type":"string"},"value":{"type":"string"},"confidence":{"type":"number"}},"required":["value"]}}},"required":["text"]}),
                Tool(name="memory_query", description="Query the user's personal memory -- decisions, observations, bug tracking, project status, and context from their coding sessions, browser observations, and cross-tool workflows. Use this for anything about the user's own projects, preferences, or work history.",
                     inputSchema={"type":"object","properties":{"query":{"type":"string"},"session_id":{"type":"string"},"top_k":{"type":"integer","default":10}},"required":["query"]}),
                Tool(name="memory_trace", description="Trace a fact's source turn and typed supersession lineage. Pass a claim_id, or a (subject, predicate) pair to trace the current value of that slot.",
                     inputSchema={"type":"object","properties":{"claim_id":{"type":"string"},"subject":{"type":"string"},"predicate":{"type":"string"},"session_id":{"type":"string"}}}),
                Tool(name="brain", description="Deterministic world-state grouped by subject -- value, status, confidence, provenance span, and per-subject gaps (no LLM).",
                     inputSchema={"type":"object","properties":{"session_id":{"type":"string","default":"default"}}}),
                Tool(name="memory_payload", description="Return the memory payload for a question in mode 'summary', 'vector', or 'memcontext' (the differentiator demo).",
                     inputSchema={"type":"object","properties":{"question":{"type":"string"},"mode":{"type":"string","enum":["summary","vector","memcontext"]},"session_id":{"type":"string"}},"required":["question","mode"]}),
                Tool(name="memory_correct", description="Correct or dismiss an existing claim.",
                     inputSchema={"type":"object","properties":{"claim_id":{"type":"string"},"action":{"type":"string","enum":["dismiss","correct"]},"new_value":{"type":"string"}},"required":["claim_id","action"]}),
                Tool(name="memory_profile", description="Get the smart profile for a subject.",
                     inputSchema={"type":"object","properties":{"subject":{"type":"string","default":"user"},"max_tokens":{"type":"integer","default":500}}}),
                Tool(name="memory_stats", description="Get storage statistics.",
                     inputSchema={"type":"object","properties":{}}),
                Tool(name="memory_digest", description="Build and return the session digest (summary layer): key facts, supersession updates, remaining count.",
                     inputSchema={"type":"object","properties":{"session_id":{"type":"string"}},"required":["session_id"]}),
                Tool(name="memory_life_events", description="Detect life events for a subject: bursts of diverse predicate changes in a time window.",
                     inputSchema={"type":"object","properties":{"subject":{"type":"string","default":"user"},"window_hours":{"type":"integer","default":24},"min_predicates":{"type":"integer","default":3}}}),
                Tool(name="memory_events", description="Assemble event frames for a session: co-referent claims grouped into multi-slot event records.",
                     inputSchema={"type":"object","properties":{"session_id":{"type":"string"}},"required":["session_id"]}),
                Tool(name="memory_volatility", description="Classify a (subject, predicate) slot's volatility from supersession history: stable/evolving/volatile.",
                     inputSchema={"type":"object","properties":{"subject":{"type":"string","default":"user"},"predicate":{"type":"string"}},"required":["predicate"]}),
                Tool(name="memory_working_context", description="Assemble task-relevant memory for a session within a token budget, cued by recent turns (query-free) not all active memory.",
                     inputSchema={"type":"object","properties":{"session_id":{"type":"string"},"token_budget":{"type":"integer","default":2000}},"required":["session_id"]}),
                Tool(name="memory_procedures", description="Detect recurring procedures (ordered action sequences) across sessions. EXPERIMENTAL: disabled unless MEMCONTEXT_EXPERIMENTAL_PROCEDURAL=1.",
                     inputSchema={"type":"object","properties":{"min_sessions":{"type":"integer","default":2}}}),
                Tool(name="memory_output_provenance", description="Output-sentence provenance (audit): record which sentences cite which claims; trace claim<->sentence<->turn links.",
                     inputSchema={"type":"object","properties":{"session_id":{"type":"string"},"record":{"type":"array"},"claim_id":{"type":"string"},"turn_id":{"type":"string"},"sentence_id":{"type":"string"}}}),
                Tool(name="memory_forget", description="Right-to-be-forgotten: hard-delete memory + cascade (no residual), audited. One of claim_id/subject/session_id/predicate.",
                     inputSchema={"type":"object","properties":{"claim_id":{"type":"string"},"subject":{"type":"string"},"session_id":{"type":"string"},"predicate":{"type":"string"},"reason":{"type":"string"}}}),
                Tool(name="memory_trust_status", description="Trust observability: source-trust distribution, contradiction rate, forgetting + drift audit, tenant distribution, staleness proxy.",
                     inputSchema={"type":"object","properties":{}}),
                Tool(name="memory_tuples", description="Project a session's active facts into event tuples (subject, action, object, validity).",
                     inputSchema={"type":"object","properties":{"session_id":{"type":"string"}},"required":["session_id"]}),
                Tool(name="memory_entity_graph", description="Co-occurrence neighbors of an entity within a session's claim graph.",
                     inputSchema={"type":"object","properties":{"session_id":{"type":"string"},"entity":{"type":"string"},"max_hops":{"type":"integer","default":1}},"required":["session_id","entity"]}),
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict):
            import asyncio  # noqa: F401  (kept for parity with stdio dispatcher)
            try:
                if name == "memory_store":
                    result = handle_memory_store(
                        conn, extractor=store_extractor, queue=store_queue,
                        **arguments,
                    )
                elif name == "memory_query":
                    result = handle_memory_query(conn, **arguments)
                elif name == "brain":
                    result = handle_brain(conn, **arguments)
                elif name == "memory_payload":
                    result = handle_memory_payload(conn, **arguments)
                elif name == "memory_trace":
                    result = handle_memory_trace(conn, **arguments)
                elif name == "memory_correct":
                    result = handle_memory_correct(conn, **arguments)
                elif name == "memory_profile":
                    result = handle_memory_profile(conn, **arguments)
                elif name == "memory_stats":
                    result = handle_memory_stats(conn)
                elif name == "memory_digest":
                    result = handle_memory_digest(conn, **arguments)
                elif name == "memory_life_events":
                    result = handle_memory_life_events(conn, **arguments)
                elif name == "memory_events":
                    result = handle_memory_events(conn, **arguments)
                elif name == "memory_volatility":
                    result = handle_memory_volatility(conn, **arguments)
                elif name == "memory_working_context":
                    result = handle_memory_working_context(conn, **arguments)
                elif name == "memory_procedures":
                    result = handle_memory_procedures(conn, **arguments)
                elif name == "memory_output_provenance":
                    result = handle_memory_output_provenance(conn, **arguments)
                elif name == "memory_forget":
                    result = handle_memory_forget(conn, **arguments)
                elif name == "memory_trust_status":
                    result = handle_memory_trust_status(conn, **arguments)
                elif name == "memory_tuples":
                    result = handle_memory_tuples(conn, **arguments)
                elif name == "memory_entity_graph":
                    result = handle_memory_entity_graph(conn, **arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}
            except Exception as exc:  # malformed/hostile input or handler error
                log.error("mcp.tool_error", tool=name, error_type=type(exc).__name__)
                result = {"error": "internal error", "tool": name}
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        return server

    state = {"transport": None, "task": None}

    async def ensure_transport():
        """Lazy-init: create transport + server on first request."""
        if state["transport"] is not None:
            return state["transport"]

        transport = StreamableHTTPServerTransport(
            mcp_session_id=None,
            is_json_response_enabled=True,
        )
        server = _build_server()

        async def run_session():
            async with transport.connect() as (rs, ws):
                await server.run(rs, ws, server.create_initialization_options())

        import asyncio
        state["task"] = asyncio.get_event_loop().create_task(run_session())
        await anyio.sleep(0.05)
        state["transport"] = transport
        return transport

    async def _app(scope, receive, send):
        if scope["type"] == "lifespan":
            await receive()
            await send({"type": "lifespan.startup.complete"})
            await receive()
            await send({"type": "lifespan.shutdown.complete"})
            return
        if scope["type"] != "http":
            return
        transport = await ensure_transport()
        await transport.handle_request(scope, receive, send)

    return _app


def _with_bearer_auth(app, token: str | None):
    """Wrap an ASGI app with a bearer-token gate. No token → pass-through (unauth).

    Accepts the token either as ``Authorization: Bearer <token>`` (preferred) or a
    ``?token=<token>`` query param (some connector UIs only allow a URL). Lifespan and
    non-HTTP scopes pass through untouched.
    """
    if not token:
        return app
    import json as _json

    expected = f"Bearer {token}"

    async def _guarded(scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers") or [])
            authz = headers.get(b"authorization", b"").decode()
            qs = scope.get("query_string", b"").decode()
            if authz != expected and f"token={token}" not in qs:
                body = _json.dumps({"error": "unauthorized"}).encode()
                await send({
                    "type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())],
                })
                await send({"type": "http.response.body", "body": body})
                return
        await app(scope, receive, send)

    return _guarded


def _run_http(*, db_path: str, host: str, port: int, token: str | None) -> None:
    """Serve MCP over Streamable HTTP for remote clients (claude.ai web via a tunnel)."""
    import sys

    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "HTTP transport needs uvicorn + starlette: pip install 'memcontext[mcp]'"
            " (or: pip install uvicorn starlette)"
        ) from None

    app = _with_bearer_auth(create_http_app(db_path), token)
    auth_note = "bearer token required" if token else "NO AUTH"
    print(
        f"[memcontext] MCP Streamable HTTP → http://{host}:{port}/mcp  (auth: {auth_note})",
        file=sys.stderr,
    )
    if not token:
        print(
            "[memcontext] WARNING: no --token / MEMCONTEXT_MCP_TOKEN set — this endpoint is"
            " UNAUTHENTICATED. Do not expose it on a public tunnel.",
            file=sys.stderr,
        )
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _run_http_oauth(
    *, db_path: str, host: str, port: int, public_url: str | None, password: str | None
) -> None:
    """Serve MCP over Streamable HTTP behind OAuth 2.1 (the claude.ai connector flow)."""
    import sys

    if not public_url:
        raise SystemExit(
            "[memcontext] --public-url is required for OAuth: it's the https URL the"
            " client sees (your tunnel/domain), e.g. https://xxx.trycloudflare.com."
        )
    if not password:
        raise SystemExit(
            "[memcontext] OAuth needs a login gate: pass --oauth-password or set"
            " MEMCONTEXT_OAUTH_PASSWORD. Without it the public endpoint is open."
        )
    try:
        import uvicorn

        from memcontext.mcp_oauth import build_oauth_http_app
    except ImportError:
        raise ImportError(
            "OAuth/HTTP needs uvicorn + starlette + mcp auth: pip install 'memcontext[mcp]'"
        ) from None

    app = build_oauth_http_app(db_path=db_path, public_url=public_url, password=password)
    base = public_url.rstrip("/")
    print(
        f"[memcontext] MCP + OAuth 2.1 → connector URL: {base}/mcp  (login gate ON)",
        file=sys.stderr,
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m memcontext.mcp_server` — starts the stdio server.

    PATH-independent: an MCP client launches the server as
    `<python> -m memcontext.mcp_server --db <path>`, so it does not depend on
    the `memcontext` console script being installed on PATH.
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="memcontext.mcp_server",
        description="MemContext MCP server (stdio JSON-RPC transport).",
    )
    parser.add_argument("--db", default="memcontext.db", help="SQLite database path.")
    parser.add_argument(
        "--transport", default="stdio", choices=["stdio", "http"],
        help="stdio (local clients) or http (Streamable HTTP for remote / claude.ai-web).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host (transport=http).")
    parser.add_argument("--port", default=8765, type=int, help="HTTP port (transport=http).")
    parser.add_argument(
        "--token", default=None,
        help="Bearer token required on http (or set MEMCONTEXT_MCP_TOKEN).",
    )
    parser.add_argument(
        "--oauth", action="store_true",
        help="Serve OAuth 2.1 on http (claude.ai connector flow) instead of a bearer token.",
    )
    parser.add_argument(
        "--public-url", default=None,
        help="OAuth issuer = the https URL the client sees (tunnel/domain).",
    )
    parser.add_argument(
        "--oauth-password", default=None,
        help="Login-gate password for OAuth (or set MEMCONTEXT_OAUTH_PASSWORD).",
    )
    parser.add_argument(
        "--pack", default=None, help="Predicate pack(s) to activate (sets ACTIVE_PACK)."
    )
    args = parser.parse_args(argv)
    if args.pack:
        os.environ["ACTIVE_PACK"] = args.pack
    token = args.token or os.environ.get("MEMCONTEXT_MCP_TOKEN")
    oauth_password = args.oauth_password or os.environ.get("MEMCONTEXT_OAUTH_PASSWORD")

    from memcontext.schema import DatabaseUnavailableError

    try:
        run_server(db_path=args.db, transport=args.transport,
                   host=args.host, port=args.port, token=token,
                   oauth=args.oauth, public_url=args.public_url,
                   oauth_password=oauth_password)
    except DatabaseUnavailableError as exc:
        import sys
        print(f"[memcontext] {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
