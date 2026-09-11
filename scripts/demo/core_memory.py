#!/usr/bin/env python
"""MemContext core demo -- the auditable memory loop, narrated.

Walks the verified core flow with real output:

  1. init a database
  2. store a conversation turn -> structured claims with provenance
  3. store a correction -> typed supersession (old claim kept, not overwritten)
  4. query -> current state only
  5. trace -> provenance + supersession lineage for the answer
  6. brain -> deterministic world-state projection

No LLM, no embeddings, no network. Uses PassthroughExtractor with
pre-extracted claims so it runs anywhere the package installs.

Usage:
    python -m pip install -e .
    python scripts/demo/core_memory.py
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from memcontext.claims import list_active_claims
from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import handle_brain, handle_memory_query, handle_memory_trace
from memcontext.on_new_turn import on_new_turn
from memcontext.schema import Speaker, open_database

LINE = "-" * 64


def show_claim(c) -> None:
    print(f"    [{c.status.value:>10}] {c.subject} | {c.predicate} | {c.value} "
          f"(conf {c.confidence:.2f})")


def main() -> None:
    db = Path(tempfile.mkdtemp()) / "demo.db"
    conn = open_database(str(db))
    conn.row_factory = sqlite3.Row
    sid = "demo"

    print(LINE)
    print("1. INIT -- fresh SQLite substrate")
    print(LINE)
    print(f"    database: {db}")

    print()
    print(LINE)
    print('2. STORE -- "I live in Toronto and work as a data engineer."')
    print(LINE)
    on_new_turn(
        conn, session_id=sid, speaker=Speaker.USER,
        text="I live in Toronto and work as a data engineer.",
        extractor=PassthroughExtractor([
            {"subject": "user", "predicate": "user_fact",
             "value": "lives in Toronto", "confidence": 0.95},
            {"subject": "user", "predicate": "user_fact",
             "value": "works as a data engineer", "confidence": 0.93},
        ]),
    )
    for c in list_active_claims(conn, sid):
        show_claim(c)

    print()
    print(LINE)
    print('3. CORRECT -- "Actually I moved to Vancouver last month."')
    print(LINE)
    r = on_new_turn(
        conn, session_id=sid, speaker=Speaker.USER,
        text="Actually I moved to Vancouver last month.",
        extractor=PassthroughExtractor([
            {"subject": "user", "predicate": "user_fact",
             "value": "lives in Vancouver", "confidence": 0.96},
        ]),
    )
    for e in r.supersession_edges:
        print(f"    supersession edge: {e.edge_type.value}")
    print("    active claims now:")
    for c in list_active_claims(conn, sid):
        show_claim(c)

    print()
    print(LINE)
    print('4. QUERY -- "where does the user live"')
    print(LINE)
    q = handle_memory_query(conn, query="where does the user live", session_id=sid)
    for c in q["claims"][:3]:
        print(f"    -> {c['subject']} | {c['predicate']} | {c['value']}")

    print()
    print(LINE)
    print("5. TRACE -- provenance + lineage of the current answer")
    print(LINE)
    t = handle_memory_trace(conn, subject="user", predicate="user_fact",
                            session_id=sid)
    for step in t.get("lineage", []):
        edge = step["edge_type"] or "origin"
        print(f"    [{step['status']:>10}] {step['value']}  (via {edge})")
        if step.get("text"):
            print(f"                 said: {step['text']!r}")

    print()
    print(LINE)
    print("6. BRAIN -- deterministic world-state projection")
    print(LINE)
    brain = handle_brain(conn, session_id=sid)
    print(json.dumps(brain, indent=4, default=str)[:2000])

    print()
    print(LINE)
    print("Done. The superseded Toronto claim is still in the DB -- queryable")
    print("via memory_trace -- but excluded from active retrieval.")
    print(LINE)


if __name__ == "__main__":
    main()
