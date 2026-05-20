#!/usr/bin/env python
"""Pyright Observation Demo — MemContext as a developer memory layer.

Demonstrates the core product loop:
1. OBSERVE: Run pyright on the codebase, capture diagnostics
2. EXTRACT: Convert diagnostics into structured claims
3. STORE: Persist claims with provenance (file, line, rule)
4. QUERY: Ask questions about code health through MemContext
5. RE-OBSERVE: Run pyright again after a fix, detect changes via supersession

No browser needed — pyright outputs JSON directly.
Uses the developer predicate pack.

Usage:
    python scripts/demo/pyright_observe.py [path_to_check]
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

# Use developer pack for developer-context predicates
os.environ["ACTIVE_PACK"] = "general,developer"

from memcontext.claims import list_active_claims
from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import handle_memory_query, handle_memory_trace
from memcontext.on_new_turn import on_new_turn
from memcontext.predicate_packs import active_pack
from memcontext.schema import Speaker, open_database

active_pack.cache_clear()


def run_pyright(target: str) -> list[dict]:
    """Run pyright and return diagnostics as dicts."""
    try:
        result = subprocess.run(
            ["pyright", "--outputjson", target],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=120,
        )
        data = json.loads(result.stdout)
        return data.get("generalDiagnostics", [])
    except (json.JSONDecodeError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"  pyright failed: {exc}")
        return []


def diagnostics_to_claims(diags: list[dict]) -> list[dict]:
    """Convert pyright diagnostics into structured claim dicts."""
    claims = []
    seen = set()

    for d in diags:
        filepath = d.get("file", "unknown")
        filename = os.path.basename(filepath)
        rule = d.get("rule", "unknown")
        severity = d.get("severity", "error")
        message = d.get("message", "")
        start = d.get("range", {}).get("start", {})
        line = start.get("line", 0)

        dedup_key = f"{filename}:{line}:{rule}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        claims.append({
            "subject": filename,
            "predicate": "observation",
            "value": f"[{severity}] line {line}: {rule} - {message[:150]}",
            "confidence": 0.95 if severity == "error" else 0.80,
        })

    return claims


def summarize_by_file(diags: list[dict]) -> list[dict]:
    """Create file-level summary claims."""
    by_file: dict[str, dict] = {}
    for d in diags:
        filename = os.path.basename(d.get("file", "unknown"))
        if filename not in by_file:
            by_file[filename] = {"errors": 0, "warnings": 0, "info": 0}
        severity = d.get("severity", "error")
        by_file[filename][severity + "s" if severity != "info" else "info"] = (
            by_file[filename].get(severity + "s" if severity != "info" else "info", 0) + 1
        )

    claims = []
    for filename, counts in by_file.items():
        total = sum(counts.values())
        claims.append({
            "subject": filename,
            "predicate": "project_status",
            "value": f"pyright: {total} issues ({counts.get('errors', 0)} errors)",
            "confidence": 0.98,
        })
    return claims


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "memcontext/"

    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    sid = "pyright_observe"

    print("=" * 60)
    print("MemContext Pyright Observation Demo")
    print("=" * 60)
    print()

    # --- Step 1: OBSERVE ---
    print(f"1. OBSERVE: Running pyright on {target}")
    diags = run_pyright(target)
    print(f"   Found {len(diags)} diagnostics")
    print()

    # --- Step 2: EXTRACT ---
    print("2. EXTRACT: Converting diagnostics to structured claims")
    detail_claims = diagnostics_to_claims(diags)
    summary_claims = summarize_by_file(diags)
    all_claims = summary_claims + detail_claims[:50]
    print(f"   {len(summary_claims)} file summaries + {min(len(detail_claims), 50)} detail claims")
    print()

    # --- Step 3: STORE ---
    print("3. STORE: Persisting claims with provenance")
    ext = PassthroughExtractor(all_claims)
    observation_text = f"[Pyright observation of {target}] {len(diags)} diagnostics found"
    result = on_new_turn(
        conn, session_id=sid, speaker=Speaker.ASSISTANT,
        text=observation_text, extractor=ext,
    )
    print(f"   Turn: {result.turn.turn_id}")
    print(f"   Claims stored: {len(result.created_claims)}")
    print(f"   Supersessions: {len(result.supersession_edges)}")
    print()

    # --- Step 4: QUERY ---
    print("4. QUERY: Asking questions about code health")
    print()

    queries = [
        "which files have the most type errors",
        "what are the pyright issues in cli.py",
        "reportMissingImports errors",
    ]
    for q in queries:
        print(f'   Q: "{q}"')
        qr = handle_memory_query(conn, query=q, session_id=sid, top_k=5)
        for c in qr["claims"][:3]:
            print(f'      [{c["subject"]}] {c["value"][:80]}')
        print()

    # --- Step 5: TRACE ---
    print("5. TRACE: Provenance for a claim")
    active = list_active_claims(conn, sid)
    if active:
        trace = handle_memory_trace(conn, claim_id=active[0].claim_id)
        print(f"   Claim: [{active[0].predicate}] {active[0].subject}: {active[0].value[:60]}")
        if trace.get("source_turn"):
            print(f"   Source: {trace['source_turn']['text'][:80]}")
        print()

    # --- Step 6: RE-OBSERVE (simulate) ---
    print("6. RE-OBSERVE: Simulating a fix (2 errors resolved)")
    fixed_claims = [c for c in all_claims if "cli.py" not in c.get("subject", "")]
    fixed_claims.append({
        "subject": "cli.py",
        "predicate": "project_status",
        "value": "pyright: 0 issues (all fixed)",
        "confidence": 0.98,
    })
    ext2 = PassthroughExtractor(fixed_claims)
    r2 = on_new_turn(
        conn, session_id=sid, speaker=Speaker.ASSISTANT,
        text=f"[Pyright re-observation of {target}] fixes applied",
        extractor=ext2,
    )
    print(f"   Claims updated: {len(r2.created_claims)}")
    print(f"   Supersessions: {len(r2.supersession_edges)}")
    if r2.supersession_edges:
        for edge in r2.supersession_edges[:3]:
            old = handle_memory_trace(conn, claim_id=edge.old_claim_id)
            new = handle_memory_trace(conn, claim_id=edge.new_claim_id)
            old_val = old.get("claim", {}).get("value", "?")[:50]
            new_val = new.get("claim", {}).get("value", "?")[:50]
            print(f"   Superseded: {old_val}")
            print(f"   By:         {new_val}")
    print()

    # --- Final state ---
    final = list_active_claims(conn, sid)
    print(f"Final memory: {len(final)} active claims")
    print()

    conn.close()
    print("Demo complete.")


if __name__ == "__main__":
    main()
