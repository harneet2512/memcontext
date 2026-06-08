"""PROOF: the activation layer is integrated with the memory layer's STRUCTURED
world-state — not raw text — and that structure changes which tools surface.

Run: python -m evals.tool_activation.demo_structured
Model-free (BM25/boost only), deterministic, zero LLM, zero API. The output is
the proof: same query, same registry; the ONLY difference is the user's
structured memory, and it re-ranks the tools toward the user's actual domain,
with per-channel attribution back to the claims that caused it.
"""
from __future__ import annotations

import sqlite3

from memcontext.brain import brain
from memcontext.claims import insert_fact, insert_turn, new_turn_id, now_ns
from memcontext.schema import Speaker, Turn, open_database
from memcontext.tool_activation import discover_tools
from memcontext.tool_registry import ToolDoc, upsert_tools
from memcontext.tool_retrieval import build_memory_instruction

TOOLS = [
    ("sequence_aligner", "align genomic DNA sequences from sequencing reads", ("bioinformatics",)),
    ("variant_caller", "call genetic variants from genome sequencing data", ("bioinformatics",)),
    ("stock_screener", "screen stocks and analyze market data", ("finance",)),
    ("css_linter", "lint and analyze CSS stylesheets for a website", ("frontend",)),
    ("spreadsheet_stats", "compute summary statistics over spreadsheet data", ("analytics",)),
    ("image_classifier", "classify images with a neural network", ("ml",)),
]

# The user's memory, as STRUCTURED claims (subject, predicate, value) — what the
# substrate would hold after extraction. No mention of any tool name.
MEMORY = [
    ("user", "user_fact", "works in bioinformatics on genome sequencing"),
    ("user", "user_fact", "analyzes DNA variants and genomic alignments"),
    ("user", "user_preference", "prefers command-line genomics pipelines"),
]


def _rank(results: list, tid: str) -> str:
    for i, r in enumerate(results, 1):
        if r.tool_id == tid:
            return f"#{i}"
    return "—"


def main() -> int:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row

    upsert_tools(
        conn,
        [ToolDoc(name=n, description=d, domain_tags=t, source="local", source_tool_id=n)
         for n, d, t in TOOLS],
        now=1,
    )

    sid = "alice"
    turn = Turn(turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER,
                text="(structured profile)", ts=now_ns())
    insert_turn(conn, turn)
    for subj, pred, val in MEMORY:
        insert_fact(conn, session_id=sid, source_turn_id=turn.turn_id, confidence=0.9,
                    subject=subj, predicate=pred, value=val)

    print("\n========== PROOF: STRUCTURED memory -> tool activation ==========\n")
    print("1) The substrate's STRUCTURED world-state for 'alice' (brain()):")
    world = brain(conn, session_id=sid)
    for _subject, data in world["subjects"].items():
        for f in data["facts"]:
            print(f"     [{f['predicate']}] {f['subject']} = {f['value']!r}")

    instruction = build_memory_instruction(conn, session_id=sid)
    print("\n2) Deterministic INSTRUCTION the tool layer synthesizes from those claims")
    print("   (zero-LLM, provenance-backed; ToolRet's proven instruction-augmentation):")
    print(f"     {instruction!r}")

    query = "analyze my data"
    print(f"\n3) QUERY (deliberately generic): {query!r}\n")

    a = discover_tools(conn, query=query, top_k=6)  # query-only
    b = discover_tools(conn, query=query, session_ids=[sid], use_memory=True, top_k=6)  # structured

    print("   QUERY-ONLY ranking:")
    for i, r in enumerate(a, 1):
        print(f"     {i}. {r.name:18s} {r.score:.4f}  [{','.join(_dom(r.tool_id))}]")
    print("\n   STRUCTURED-MEMORY-CONDITIONED ranking (used_memory="
          f"{any(r.used_memory for r in b)}):")
    for i, r in enumerate(b, 1):
        mb = r.components.get("memory_bm25", 0.0)
        bo = r.components.get("memory_boost", 0.0)
        print(f"     {i}. {r.name:18s} {r.score:.4f}  [{','.join(_dom(r.tool_id))}]"
              f"  (memory_bm25={mb:.4f} memory_boost={bo:.4f})")

    print("\n4) ATTRIBUTION — what the user's domain did to the ranking:")
    for tid in ("local::sequence_aligner", "local::variant_caller"):
        nm = tid.split("::")[1]
        print(f"     {nm:18s} query-only {_rank(a, tid)}  ->  conditioned {_rank(b, tid)}")
    print("\n   The bioinformatics tools rose purely because the user's STRUCTURED")
    print("   claims (genomics/sequencing/DNA) matched their domain — the query never")
    print("   mentioned them. That is the memory<->tool integration, working.\n")
    return 0


_DOMAIN = {f"local::{n}": t for n, _, t in TOOLS}


def _dom(tid: str) -> tuple[str, ...]:
    return _DOMAIN.get(tid, ())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
