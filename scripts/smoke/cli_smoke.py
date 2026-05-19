#!/usr/bin/env python
"""CLI black-box smoke test. Run from any directory outside the repo.

Usage: python scripts/smoke/cli_smoke.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

PASS = 0
FAIL = 0


def run(cmd: str, expect_exit: int = 0) -> str:
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != expect_exit:
        print(f"  FAIL: exit={result.returncode}, expected={expect_exit}")
        print(f"  stdout: {result.stdout[:500]}")
        print(f"  stderr: {result.stderr[:500]}")
        return ""
    return result.stdout


def check(name: str, ok: bool) -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [{name}] PASS")
    else:
        FAIL += 1
        print(f"  [{name}] FAIL")


with tempfile.TemporaryDirectory() as tmpdir:
    db = os.path.join(tmpdir, "smoke.db")

    print("=== CLI Smoke Test ===\n")

    print("1. memcontext init")
    out = run(f'memcontext init --db "{db}" --pack general,developer')
    check("init creates db", os.path.exists(db))
    check("init reports predicates", "19 predicates" in out)

    print("\n2. memcontext status (empty)")
    out = run(f'memcontext status --db "{db}"')
    check("status shows 0 claims", "0 total" in out)

    print("\n3. memcontext ingest")
    out = run(f'memcontext ingest "I prefer dark mode for my code editor" --db "{db}" --session smoke')
    check("ingest creates claim", "Claims created: 1" in out)
    check("ingest preference predicate", "user_preference" in out)

    print("\n4. memcontext status (after ingest)")
    out = run(f'memcontext status --db "{db}"')
    check("status shows 1 active", "1 active" in out)

    print("\n5. memcontext query")
    out = run(f'memcontext query "dark mode" --db "{db}" --session smoke')
    check("query finds claims", "Found" in out and "claim" in out.lower())
    check("query returns dark mode", "dark" in out.lower())

    print("\n6. memcontext ingest (noise rejected)")
    out = run(f'memcontext ingest "uh um ok" --db "{db}" --session smoke')
    check("noise rejected", "rejected" in out.lower())

    print("\n7. memcontext serve --help")
    out = run("memcontext serve --help")
    check("serve help shows transport", "transport" in out.lower() or "stdio" in out.lower())

print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
