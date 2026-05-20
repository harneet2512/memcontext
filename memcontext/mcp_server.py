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
        handle_memory_correct,
        handle_memory_observe,
        handle_memory_observe_url,
        handle_memory_query,
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
                description="Query active memory claims matching a question.",
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
                description="Trace a claim back to its source turn and supersession history.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "description": "The claim ID to trace"},
                    },
                    "required": ["claim_id"],
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
                    "structured claims, store with provenance. Set connect_browser=true "
                    "to read from the user's running Chrome (inherits all auth — SSO, "
                    "2FA, OAuth). Or provide login_email/password for form-based auth. "
                    "Works on any page the user can see: internal tools, ChatGPT, "
                    "GitHub, SAP, ServiceNow, etc. Re-observing detects changes."
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
                            "description": "Email/username for authentication. Agent fills the login form automatically.",
                        },
                        "login_password": {
                            "type": "string",
                            "description": "Password for authentication.",
                        },
                        "login_url": {
                            "type": "string",
                            "description": "Login page URL if different from the target URL.",
                        },
                        "connect_browser": {
                            "type": "boolean",
                            "description": (
                                "Attach to the user's running Chrome (port 9222) instead of "
                                "launching headless. Inherits all auth sessions — no credentials needed."
                            ),
                            "default": False,
                        },
                    },
                    "required": ["url"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "memory_store":
            result = handle_memory_store(conn, **arguments)
        elif name == "memory_query":
            result = handle_memory_query(conn, **arguments)
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
        else:
            result = {"error": f"Unknown tool: {name}"}
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _run():
        async with mcp.server.stdio.stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())
