"""MemContext MCP server — exposes memory tools over Model Context Protocol.

Requires the 'mcp' package: pip install mcp
All MCP-specific imports are lazy so mcp_tools.py works standalone.
"""
from __future__ import annotations

import json

def run_server(*, db_path: str = "memcontext.db", transport: str = "stdio") -> None:
    """Start the MCP server. Called by `memcontext serve`."""
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
        handle_memory_observe,
        handle_memory_observe_url,
        handle_memory_payload,
        handle_memory_profile,
        handle_memory_query,
        handle_memory_stats,
        handle_memory_store,
        handle_memory_trace,
    )

    conn = open_database(db_path)
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
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        try:
            if name == "memory_store":
                result = handle_memory_store(conn, **arguments)
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
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as exc:  # malformed input or handler error → structured result
            result = {"error": str(exc), "tool": name}
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _run():
        async with mcp.server.stdio.stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())


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
        handle_memory_observe,
        handle_memory_observe_url,
        handle_memory_payload,
        handle_memory_profile,
        handle_memory_query,
        handle_memory_stats,
        handle_memory_store,
        handle_memory_trace,
    )
    from memcontext.schema import open_database

    conn = open_database(db_path)
    transports: dict[str, StreamableHTTPServerTransport] = {}

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
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict):
            import asyncio  # noqa: F401  (kept for parity with stdio dispatcher)
            try:
                if name == "memory_store":
                    result = handle_memory_store(conn, **arguments)
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
                else:
                    result = {"error": f"Unknown tool: {name}"}
            except Exception as exc:  # malformed input or handler error → structured result
                result = {"error": str(exc), "tool": name}
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
        "--transport", default="stdio", choices=["stdio"], help="Transport (stdio only)."
    )
    parser.add_argument(
        "--pack", default=None, help="Predicate pack(s) to activate (sets ACTIVE_PACK)."
    )
    args = parser.parse_args(argv)
    if args.pack:
        os.environ["ACTIVE_PACK"] = args.pack
    run_server(db_path=args.db, transport=args.transport)


if __name__ == "__main__":
    main()
