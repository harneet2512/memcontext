#!/usr/bin/env python
"""Browser observation smoke test. No conftest dependency.

Usage: python scripts/smoke/observe_smoke.py
"""
from __future__ import annotations

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


from memcontext.schema import open_database
from memcontext.observe.browser import PageSnapshot, observe_page
from memcontext.observe.extractors import AccessibilityTreeExtractor
from memcontext.observe.revisit import diff_snapshots, apply_changes
from memcontext.claims import list_active_claims

conn = open_database(":memory:")
conn.row_factory = sqlite3.Row

print("=== Observation Smoke Test ===\n")

# Realistic a11y tree from a sprint board
tree_v1 = {
    "role": "WebArea", "name": "Sprint Board", "children": [
        {"role": "heading", "name": "Sprint 42", "children": []},
        {"role": "text", "name": "Migration to PostgreSQL is 75% complete today", "children": []},
        {"role": "link", "name": "TICKET-123: Auth refactor task", "children": []},
        {"role": "textbox", "name": "Search", "value": "auth", "children": []},
    ]
}

print("1. Extract claims from page")
ext = AccessibilityTreeExtractor()
snap_v1 = PageSnapshot(
    url="http://board.test/sprint", title="Sprint Board",
    timestamp="2026-01-01T00:00:00Z", accessibility_tree=tree_v1,
)
claims_v1 = ext.extract(snap_v1)
check("extracts >= 4 claims", len(claims_v1) >= 4, f"got {len(claims_v1)}")
check("all have obs_key", all("obs_key" in c for c in claims_v1))
values = [c["value"] for c in claims_v1]
check("has title", any("Sprint Board" in v for v in values))
check("has heading", any("Sprint 42" in v for v in values))
check("has text", any("PostgreSQL" in v for v in values))
check("has link", any("TICKET-123" in v for v in values))

print("\n2. Store in DB via observe_page")
result = observe_page(conn, snapshot=snap_v1, session_id="obs")
check("turn created", result.turn_id is not None)
stored = list_active_claims(conn, "obs")
check("claims in DB", len(stored) >= 4, f"got {len(stored)}")

print("\n3. Re-visit with changes")
tree_v2 = {
    "role": "WebArea", "name": "Sprint Board", "children": [
        {"role": "heading", "name": "Sprint 42", "children": []},
        {"role": "text", "name": "Migration to PostgreSQL is 100% complete shipped", "children": []},
        {"role": "link", "name": "TICKET-123: Auth refactor task", "children": []},
        {"role": "link", "name": "TICKET-456: New dashboard feature added", "children": []},
        {"role": "textbox", "name": "Search", "value": "dashboard", "children": []},
    ]
}
snap_v2 = PageSnapshot(
    url="http://board.test/sprint", title="Sprint Board",
    timestamp="2026-01-01T01:00:00Z", accessibility_tree=tree_v2,
)
claims_v2 = ext.extract(snap_v2)

report = diff_snapshots(claims_v1, claims_v2, "http://board.test/sprint")
check("detects addition", len(report.added_claims) >= 1, f"added={len(report.added_claims)}")
check("new ticket added", any("TICKET-456" in c["value"] for c in report.added_claims))
check("detects change", len(report.changed_claims) >= 1, f"changed={len(report.changed_claims)}")
check("unchanged preserved", report.unchanged_count >= 2, f"unchanged={report.unchanged_count}")

print("\n4. Apply changes")
stats = apply_changes(conn, change_report=report, session_id="obs")
check("changes applied", stats["added"] >= 1 or stats["changed"] >= 1)
final = list_active_claims(conn, "obs")
check("final claims >= initial", len(final) >= len(stored))

print("\n5. Playwright optional")
try:
    import playwright
    print(f"  Playwright installed: {playwright.__file__}")
except ImportError:
    print("  Playwright NOT installed — observation works without it")
check("playwright optional", True)

print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
