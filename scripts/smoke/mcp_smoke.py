#!/usr/bin/env python
"""MCP tool handler + stdio protocol smoke test. No conftest dependency.

Usage: python scripts/smoke/mcp_smoke.py
"""
from __future__ import annotations

import json
import sqlite3
import sys

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [{name}] PASS")
    else:
        FAIL += 1
        print(f"  [{name}] FAIL — {detail}")


print("=== MCP Tool Handler Smoke Test ===\n")

from memcontext.schema import open_database
from memcontext.mcp_tools import (
    handle_memory_store,
    handle_memory_query,
    handle_memory_trace,
    handle_memory_correct,
)
from memcontext.claims import get_claim

conn = open_database(":memory:")
conn.row_factory = sqlite3.Row

print("1. memory_store (passthrough)")
r = handle_memory_store(
    conn, text="I live in Toronto and prefer dark mode", session_id="s1",
    claims=[
        {"subject": "user", "predicate": "user_fact", "value": "lives in Toronto", "confidence": 0.9},
        {"subject": "user", "predicate": "user_preference", "value": "prefers dark mode", "confidence": 0.8},
    ],
)
check("store admitted", r["admitted"] is True)
check("store 2 claims", r["claims_created"] == 2, f"got {r['claims_created']}")

print("\n2. memory_store (SimpleExtractor)")
r2 = handle_memory_store(conn, text="I prefer using Python for backend dev", session_id="s1")
check("simple admitted", r2["admitted"] is True)
check("simple >= 1 claim", r2["claims_created"] >= 1, f"got {r2['claims_created']}")

print("\n3. memory_query")
q = handle_memory_query(conn, query="dark mode", session_id="s1")
check("query finds claims", q["total"] >= 2, f"total={q['total']}")
values = [c["value"] for c in q["claims"]]
check("query has dark mode", any("dark" in v.lower() for v in values), f"values={values}")

print("\n4. memory_trace")
cid = r["claim_ids"][0]
t = handle_memory_trace(conn, claim_id=cid)
check("trace has claim", t.get("claim", {}).get("claim_id") == cid)
check("trace has source_turn", t.get("source_turn") is not None)
check("trace has chain", "supersession_chain" in t)

print("\n5. memory_correct (dismiss)")
cid_dismiss = r["claim_ids"][1]
d = handle_memory_correct(conn, claim_id=cid_dismiss, action="dismiss")
check("dismiss action", d.get("action") == "dismissed")
check("dismiss state", get_claim(conn, cid_dismiss).status.value == "dismissed")

print("\n6. memory_correct (correct)")
c = handle_memory_correct(conn, claim_id=cid, action="correct", new_value="lives in Vancouver")
check("correct action", c.get("action") == "corrected")
check("correct new claim", c.get("new_claim_id") is not None)
old = get_claim(conn, cid)
check("old superseded", old.status.value == "superseded")
new = get_claim(conn, c["new_claim_id"])
check("new active", new.status.value == "active")
check("new value", new.value == "lives in Vancouver")

print("\n=== MCP Protocol Smoke Test ===\n")

try:
    import asyncio
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters

    async def _test_stdio():
        server_params = StdioServerParameters(
            command="memcontext",
            args=["serve", "--transport", "stdio", "--db", ":memory:"],
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_names = {t.name for t in tools.tools}
                check("stdio lists tools", len(tool_names) >= 4, f"got {tool_names}")
                check("has memory_store", "memory_store" in tool_names)
                check("has memory_query", "memory_query" in tool_names)
                check("has memory_trace", "memory_trace" in tool_names)
                check("has memory_correct", "memory_correct" in tool_names)

                store_result = await session.call_tool(
                    "memory_store",
                    {"text": "I prefer dark mode for coding", "session_id": "proto_test"},
                )
                body = json.loads(store_result.content[0].text)
                check("stdio store works", body.get("admitted") is True, str(body))

    asyncio.run(_test_stdio())
except ImportError:
    print("  mcp package not installed — skipping protocol test")
    print("  LIMITATION: tool handlers verified, stdio transport not tested")
except Exception as exc:
    FAIL += 1
    print(f"  MCP protocol test FAILED: {exc}")

print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
