#!/usr/bin/env python
"""Real Playwright browser observation smoke test.

Launches Chromium, navigates to local HTML pages, captures real accessibility
trees via CDP, extracts claims, stores them, queries them, then tests re-visit
change detection. NOT a mock test.

If Playwright cannot launch: exits with BLOCKED status.
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PASS_COUNT = 0
FAIL_COUNT = 0
BLOCKED = False


def check(name, ok, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if ok:
        PASS_COUNT += 1
        print(f"  [{name}] PASS")
    else:
        FAIL_COUNT += 1
        print(f"  [{name}] FAIL — {detail}")


def block(reason):
    global BLOCKED
    BLOCKED = True
    print(f"\n  BLOCKED: {reason}")
    print(f"\n=== Browser Agent Verification: BLOCKED ===")
    sys.exit(2)


def build_a11y_tree(page):
    """Build a dict-based accessibility tree from the CDP Accessibility API.

    Playwright 1.60 removed the deprecated page.accessibility.snapshot().
    We use a CDP session to get the full AX tree and rebuild it as a nested
    dict with {role, name, value, children} — the format expected by
    AccessibilityTreeExtractor.
    """
    cdp = page.context.new_cdp_session(page)
    result = cdp.send("Accessibility.getFullAXTree")
    nodes = result.get("nodes", [])
    if not nodes:
        return {}

    # Map CDP AX roles to the simpler roles the extractor expects
    role_map = {
        "RootWebArea": "WebArea",
        "StaticText": "text",
        "LineBreak": "none",
    }

    node_map = {}
    for n in nodes:
        nid = n["nodeId"]
        role_val = n.get("role", {}).get("value", "none")
        name_val = n.get("name", {}).get("value", "")
        value_val = ""
        if isinstance(n.get("value"), dict):
            value_val = n["value"].get("value", "")
        node_map[nid] = {
            "role": role_map.get(role_val, role_val),
            "name": name_val,
            "value": value_val,
            "children": [],
            "_childIds": list(n.get("childIds", [])),
        }

    # Link children
    for nid, node in node_map.items():
        for cid in node["_childIds"]:
            if cid in node_map:
                node["children"].append(node_map[cid])
        del node["_childIds"]

    root = node_map[nodes[0]["nodeId"]]
    return root


# Locate test pages
SCRIPT_DIR = Path(__file__).resolve().parent
PAGE_V1 = SCRIPT_DIR / "test_page_v1.html"
PAGE_V2 = SCRIPT_DIR / "test_page_v2.html"

if not PAGE_V1.exists() or not PAGE_V2.exists():
    block(f"Test pages not found: {PAGE_V1}, {PAGE_V2}")

print("=== Browser Agent Verification (Real Playwright) ===\n")

# Step 1: Launch Playwright
print("1. Launch Chromium headless")
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    block("playwright package not installed. Run: python -m pip install playwright")

try:
    pw_manager = sync_playwright()
    pw = pw_manager.start()
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page()
    check("chromium launched", True)
except Exception as exc:
    block(f"Chromium failed to launch: {exc}")

# Import memcontext
from memcontext.schema import open_database
from memcontext.observe.browser import PageSnapshot, observe_page
from memcontext.observe.extractors import AccessibilityTreeExtractor
from memcontext.observe.revisit import diff_snapshots, apply_changes
from memcontext.mcp_tools import handle_memory_query, handle_memory_trace

conn = open_database(":memory:")
conn.row_factory = sqlite3.Row

# Step 2: Navigate to v1 page
print("\n2. Navigate to test page v1")
page_url = PAGE_V1.as_uri()
page.goto(page_url)
page.wait_for_load_state("domcontentloaded")
check("page loaded", page.title() == "MemContext Test App - Sprint Dashboard", page.title())

# Step 3: Capture real accessibility snapshot via CDP
print("\n3. Capture real accessibility tree")
a11y_tree = build_a11y_tree(page)
check("a11y tree captured", a11y_tree is not None and isinstance(a11y_tree, dict))
check(
    "a11y tree has children",
    len(a11y_tree.get("children", [])) > 0,
    f"tree keys: {list(a11y_tree.keys()) if a11y_tree else 'None'}",
)

# Build PageSnapshot from real data
content = page.content()
snapshot_v1 = PageSnapshot(
    url=page_url,
    title=page.title(),
    timestamp=datetime.now(timezone.utc).isoformat(),
    accessibility_tree=a11y_tree,
    dom_hash=hashlib.sha256(content.encode()).hexdigest(),
)
check("snapshot created", snapshot_v1.url == page_url)

# Step 4: Extract claims from real tree
print("\n4. Extract claims from real page")
ext = AccessibilityTreeExtractor()
claims_v1 = ext.extract(snapshot_v1)
print(f"   Extracted {len(claims_v1)} claims:")
for c in claims_v1[:8]:
    print(f"     [{c.get('obs_key', '?')}] {c['value'][:80]}")
check("extracted >= 3 claims", len(claims_v1) >= 3, f"got {len(claims_v1)}")

# Step 5: Store claims
print("\n5. Store claims via observe_page")
result = observe_page(conn, snapshot=snapshot_v1, session_id="browser-smoke")
check("claims stored", result.turn_id is not None)
check("stored count matches", len(result.claims) == len(claims_v1))

# Step 6: Query stored claims
print("\n6. Query observed memory")
q_result = handle_memory_query(conn, query="PostgreSQL migration status", session_id="browser-smoke")
check("query returns claims", q_result["total"] > 0, f"total={q_result['total']}")
# Print what was found
for c in q_result.get("claims", [])[:3]:
    print(f"     score={c['score']} value={c['value'][:60]}")

# Step 7: Trace a claim
print("\n7. Trace observed claim provenance")
from memcontext.claims import list_active_claims

active = list_active_claims(conn, "browser-smoke")
if active:
    trace = handle_memory_trace(conn, claim_id=active[0].claim_id)
    check("trace has source turn", trace.get("source_turn") is not None)
    check(
        "trace has claim data",
        trace.get("claim", {}).get("claim_id") == active[0].claim_id,
    )
else:
    check("trace has source turn", False, "no active claims to trace")

# Step 8: Navigate to v2 page (modified content)
print("\n8. Navigate to modified page v2")
page_url_v2 = PAGE_V2.as_uri()
page.goto(page_url_v2)
page.wait_for_load_state("domcontentloaded")
check("v2 loaded", "Sprint Dashboard" in page.title())

# Step 9: Re-capture
print("\n9. Re-capture accessibility tree")
a11y_tree_v2 = build_a11y_tree(page)
content_v2 = page.content()
snapshot_v2 = PageSnapshot(
    url=page_url,  # Same logical URL for diff comparison
    title=page.title(),
    timestamp=datetime.now(timezone.utc).isoformat(),
    accessibility_tree=a11y_tree_v2,
    dom_hash=hashlib.sha256(content_v2.encode()).hexdigest(),
)
claims_v2 = ext.extract(snapshot_v2)
print(f"   V2 extracted {len(claims_v2)} claims")
check("v2 claims extracted", len(claims_v2) >= 3)

# Step 10: Diff snapshots
print("\n10. Detect changes between visits")
report = diff_snapshots(claims_v1, claims_v2, page_url)
print(f"   Added: {len(report.added_claims)}")
print(f"   Removed: {len(report.removed_claims)}")
print(f"   Changed: {len(report.changed_claims)}")
print(f"   Unchanged: {report.unchanged_count}")
check("detected additions", len(report.added_claims) >= 1, f"got {len(report.added_claims)}")
check(
    "detected changes or removals",
    len(report.changed_claims) + len(report.removed_claims) >= 1,
)

# Step 11: Apply changes
print("\n11. Apply changes to memory")
stats = apply_changes(conn, change_report=report, session_id="browser-smoke")
check("changes applied", stats["added"] + stats["changed"] >= 1)

# Step 12: Query updated memory
print("\n12. Query updated memory")
q_updated = handle_memory_query(
    conn, query="migration complete shipped", session_id="browser-smoke"
)
check("updated query returns claims", q_updated["total"] > 0)

# Cleanup
browser.close()
pw.stop()

print(f"\n=== Results: {PASS_COUNT} passed, {FAIL_COUNT} failed ===")
if FAIL_COUNT:
    print("Browser Agent Verification: INCOMPLETE (some checks failed)")
    sys.exit(1)
else:
    print("Browser Agent Verification: PASSED (all real Playwright)")
    sys.exit(0)
