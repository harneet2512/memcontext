"""One-command runner for the MemContext differentiator demo.

Builds a fresh demo database, seeds the corrected-fact transcript, and prints
four sections end to end:

  1. brain()  - the deterministic world-state projection (by subject, with
     provenance and a gaps report);
  2. trace    - the supersession lineage for the database slot (active on top,
     the superseded value beneath with its typed edge);
  3. payloads - the same question answered from three memories (summary / vector
     / memcontext) with a structural verdict for each;
  4. live     - how to run the same comparison inside an MCP client, where the
     host model is the reader.

Run it::

    memcontext demo                  # developer pack (default)
    memcontext demo --pack general
    python -m demo.run_demo
"""
from __future__ import annotations

import logging
import os
import sys

import structlog

from memcontext.mcp_tools import handle_memory_payload, handle_memory_trace
from memcontext.schema import open_database
from memcontext.trace_view import format_world_state, render_trace_table
from demo.scenario import QUESTION, TRANSCRIPT, pack_active, seed_demo

_RULE = "=" * 72


def _quiet_logs() -> None:
    """Silence substrate debug/info logging so the demo output stays clean."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def _fresh_db(db: str) -> None:
    if db == ":memory:":
        return
    for suffix in ("", "-wal", "-shm"):
        path = db + suffix
        if os.path.exists(path):
            os.remove(path)


def _print_scenario() -> None:
    print(_RULE)
    print("MemContext demo  .  one corrected fact, three memories")
    print(_RULE)
    print("Transcript:")
    for i, turn in enumerate(TRANSCRIPT, 1):
        print(f"  Turn {i} ({turn['speaker']}): {turn['text']}")
    print(f"\nQuestion: {QUESTION}")


def _print_payloads(db_conn, manifest: dict) -> None:
    session_id = manifest["session_id"]
    labels = manifest["turn_labels"]

    print(_RULE)
    print("3. THREE MEMORIES, ONE READER  (same question, different payload)")
    print(_RULE)

    # --- summary ---
    summary = handle_memory_payload(db_conn, question=QUESTION, mode="summary", session_id=session_id)
    print("\n[ summary payload ]  raw transcript blob")
    print("  " + summary["payload"].replace("\n", "\n  "))
    print("  verdict: NO current value with a source - the payload is unstructured")
    print("           text; there is no current-value or provenance field to cite.")

    # --- vector ---
    vector = handle_memory_payload(db_conn, question=QUESTION, mode="vector", session_id=session_id)
    print("\n[ vector payload ]  top-k raw statements by cosine similarity")
    if vector.get("error"):
        print(f"  (skipped: {vector['error']})")
        print("  verdict: N/A - local embedder not installed.")
    else:
        for item in vector["retrieved"]:
            print(f"  sim {item['similarity']:.3f}  \"{item['text']}\"")
        print("  verdict: NO current value with a source - both the old and current")
        print("           value appear; similarity is not recency or truth (note the")
        print("           stale value can even outrank the current one).")

    # --- memcontext ---
    mc = handle_memory_payload(db_conn, question=QUESTION, mode="memcontext", session_id=session_id)
    support = mc.get("answer_support")
    print("\n[ memcontext payload ]  structured projection")
    if support:
        prov = support["provenance"]
        label = labels.get(prov["source_turn_id"], prov["source_turn_id"])
        span = f"[{prov['char_start']}:{prov['char_end']}]"
        print(f"  current_value: {support['current_value']}  ({support['status'].upper()}, conf {support['confidence']:.2f})")
        print(f"  source: {label} span {span} \"{prov['quote']}\"")
        for old in support["superseded"]:
            old_label = labels.get(old["source_turn_id"], old["source_turn_id"])
            print(f"  superseded: {old['value']}  <-- {old['edge_type']}  ({old_label})")
        print(f"  verdict: YES - {support['current_value']} is current; cite {label} {span}; the")
        print("           prior value is retained and linked by a typed correction edge.")
    else:
        print("  (no answer support found)")


def _print_live_instructions(manifest: dict) -> None:
    print()
    print(_RULE)
    print("4. RUN IT LIVE INSIDE AN MCP CLIENT  (the host model is the reader)")
    print(_RULE)
    print("  1. Seed + serve:  memcontext serve --db memcontext_demo.db")
    print("     (attach the server to Claude Code / Claude Desktop)")
    print("  2. Ask the host model:")
    print(f'       Using the memory_payload tool, answer "{manifest["question"]}"')
    print("       three times - mode=summary, mode=vector, mode=memcontext")
    print(f"       (session_id={manifest['session_id']}) - and show the three answers")
    print("       side by side, citing a source where the payload allows.")
    print("  The same reader produces a traceable answer only from the projection.")


def run(db: str = "memcontext_demo.db", pack: str = "developer") -> dict:
    """Build a fresh demo db, seed it, and print the full demo. Returns the manifest."""
    _quiet_logs()
    _fresh_db(db)
    conn = open_database(db)

    # Scope the pack to the demo so we never leave ACTIVE_PACK mutated in the
    # calling process (seed + every read below need the demo's vocabulary).
    with pack_active(pack):
        manifest = seed_demo(conn, pack=pack)

        _print_scenario()

        print("\n" + _RULE)
        print("1. brain()  -  deterministic world-state (no LLM in this path)")
        print(_RULE)
        from memcontext.brain import brain

        print(format_world_state(brain(conn, session_id=manifest["session_id"])))

        print("\n" + _RULE)
        print("2. trace  -  supersession lineage for the database slot")
        print(_RULE)
        trace = handle_memory_trace(
            conn,
            session_id=manifest["session_id"],
            subject=manifest["subject"],
            predicate=manifest["predicate"],
        )
        print(render_trace_table(trace))

        _print_payloads(conn, manifest)
        _print_live_instructions(manifest)
    conn.close()
    if db != ":memory:":
        print(f"\n(demo database written to {os.path.abspath(db)})")
    return manifest


if __name__ == "__main__":
    run()
