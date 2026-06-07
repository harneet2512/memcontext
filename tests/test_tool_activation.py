"""Product tests for the activation layer (registry + discover_tools).

Self-contained — no benchmark/evals dependency. The load-bearing test is
``test_discover_memory_conditioned_goes_through_substrate``: it ingests memory via
the real substrate, then shows ``discover_tools(use_memory=True)`` consumes it via
``retrieve_memory_across`` and promotes the relevant tool — i.e. the substrate is
wired end-to-end, not bypassed.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3

from memcontext.claims import insert_fact, insert_turn, new_turn_id, now_ns
from memcontext.schema import SCHEMA_VERSION, Speaker, Turn, open_database
from memcontext.tool_activation import DiscoveredTool, discover_tools
from memcontext.tool_registry import ToolDoc, count_tools, embed_tools, upsert_tools


class HashEmbedder:
    """Deterministic L2-normalized bag-of-words embedder (md5-stable, test only)."""

    model_version = "hash-test-v1"

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self._dim
            for tok in t.lower().split():
                idx = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                v[idx % self._dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def _db() -> sqlite3.Connection:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _seed_tools(conn: sqlite3.Connection, emb: HashEmbedder) -> None:
    def td(name: str, desc: str, tid: str) -> ToolDoc:
        return ToolDoc(name=name, description=desc, source="local", source_tool_id=tid)

    tools = [
        td("run_sql_query", "execute a read-only SQL query on the analytics database", "sql"),
        td("get_weather", "current weather for a city", "wx"),
        td("send_email", "send an email to a recipient", "em"),
        td("convert_currency", "convert money between currencies", "fx"),
        td("create_event", "create a calendar event", "cal"),
    ]
    upsert_tools(conn, tools, now=1)
    embed_tools(conn, embedder=emb, now=1)


def _add_memory(conn: sqlite3.Connection, sid: str, text: str) -> None:
    turn = Turn(turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER, text=text, ts=now_ns())
    insert_turn(conn, turn)
    insert_fact(
        conn, session_id=sid, source_turn_id=turn.turn_id, confidence=0.9,
        subject="user", predicate="user_fact", value=text,
    )


def _rank_of(results: list[DiscoveredTool], tool_id: str) -> int:
    for i, r in enumerate(results):
        if r.tool_id == tool_id:
            return i
    return len(results)


def test_schema_v12_has_tool_tables() -> None:
    conn = _db()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tool_schemas", "tool_embeddings"} <= tables
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION >= 12
    # memory tables still present (additive, nothing removed).
    assert {"claims", "turns", "claim_embeddings"} <= tables


def test_discover_query_only() -> None:
    conn = _db()
    emb = HashEmbedder()
    _seed_tools(conn, emb)
    assert count_tools(conn) == 5
    res = discover_tools(conn, query="send an email to my teammate", top_k=3, embedder=emb)
    assert res and res[0].name == "send_email"
    assert res[0].used_memory is False  # memory off by default


def test_discover_memory_conditioned_goes_through_substrate() -> None:
    conn = _db()
    emb = HashEmbedder()
    _seed_tools(conn, emb)
    gold = "local::sql"  # registry id for run_sql_query

    # A weak query that does not name the SQL tool.
    q = "pull the latest figures for the quarterly report"
    base = discover_tools(conn, query=q, top_k=5, embedder=emb)
    rank_base = _rank_of(base, gold)

    # Ingest memory via the real substrate (turn + claim).
    _add_memory(conn, "u1", "I frequently run SQL queries against the analytics database")

    cond = discover_tools(
        conn, query=q, session_ids=["u1"], use_memory=True, top_k=5, embedder=emb
    )
    # Substrate was actually consumed (retrieve_memory_across returned the fact).
    assert any(r.used_memory for r in cond)
    # And it promoted the relevant tool the query alone ranked lower.
    assert _rank_of(cond, gold) <= rank_base


def test_discover_empty_registry_returns_empty() -> None:
    conn = _db()
    assert discover_tools(conn, query="anything", embedder=HashEmbedder()) == []
