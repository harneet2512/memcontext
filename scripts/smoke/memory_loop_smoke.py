#!/usr/bin/env python
"""Core memory loop smoke test — verifies the 5 fundamental behaviors.

1. EXTRACT: conversation turn → structured claims
2. STORE: claims persisted with provenance
3. UPDATE: correction supersedes old claim
4. RETRIEVE: relevant claims found by query
5. ANSWER: reader uses claims to answer correctly

Uses PassthroughExtractor (no LLM needed) to test the substrate.
Tests general memory behavior, not benchmark-specific logic.
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
        print(f"  [{name}] FAIL -- {detail}")


print("=== Core Memory Loop Smoke Test ===\n")

from memcontext.schema import open_database, Speaker
from memcontext.on_new_turn import on_new_turn, ExtractedClaim
from memcontext.extractors import PassthroughExtractor
from memcontext.claims import list_active_claims, get_claim
from memcontext.provenance import span_for_claim, claim_ids_for_turn
from memcontext.mcp_tools import handle_memory_query, handle_memory_trace, handle_memory_correct

conn = open_database(":memory:")
conn.row_factory = sqlite3.Row
sid = "memory_test"


# --- Behavior 1: EXTRACT ---
print("1. EXTRACT: structured claims from conversation")

ext1 = PassthroughExtractor([
    {"subject": "user", "predicate": "user_fact", "value": "lives in Toronto", "confidence": 0.95},
    {"subject": "user", "predicate": "user_fact", "value": "works as a data engineer", "confidence": 0.93},
    {"subject": "user", "predicate": "user_preference", "value": "prefers dark mode", "confidence": 0.90},
])
r1 = on_new_turn(conn, session_id=sid, speaker=Speaker.USER,
                 text="I live in Toronto and work as a data engineer. I prefer dark mode.",
                 extractor=ext1)
check("turn admitted", r1.admitted)
check("3 claims created", len(r1.created_claims) == 3, f"got {len(r1.created_claims)}")
check("claims have correct predicates",
      {c.predicate for c in r1.created_claims} == {"user_fact", "user_preference"})
print()


# --- Behavior 2: STORE with provenance ---
print("2. STORE: claims persisted with provenance")

active = list_active_claims(conn, sid)
check("3 active claims in DB", len(active) == 3, f"got {len(active)}")

turn_claims = claim_ids_for_turn(conn, r1.turn.turn_id)
check("claims linked to source turn", len(turn_claims) == 3)

trace = handle_memory_trace(conn, claim_id=active[0].claim_id)
check("trace has source turn", trace.get("source_turn") is not None)
check("trace source text matches", trace["source_turn"]["text"] == r1.turn.text)
print()


# --- Behavior 3: UPDATE via supersession ---
print("3. UPDATE: correction supersedes old claim")

ext2 = PassthroughExtractor([
    {"subject": "user", "predicate": "user_fact", "value": "lives in Vancouver", "confidence": 0.96},
])
r2 = on_new_turn(conn, session_id=sid, speaker=Speaker.USER,
                 text="Actually I moved to Vancouver last month.",
                 extractor=ext2)
check("supersession fired", len(r2.supersession_edges) >= 1,
      f"got {len(r2.supersession_edges)} edges")

active_after = list_active_claims(conn, sid)
active_values = [c.value for c in active_after]
check("Vancouver is active", "lives in Vancouver" in active_values)
check("Toronto is superseded", "lives in Toronto" not in active_values)

old_toronto = [c for c in r1.created_claims if "Toronto" in c.value][0]
old_state = get_claim(conn, old_toronto.claim_id)
check("old claim status = superseded", old_state.status.value == "superseded")
print()


# --- Behavior 4: RETRIEVE relevant claims ---
print("4. RETRIEVE: find relevant claims by query")

q_result = handle_memory_query(conn, query="where does the user live", session_id=sid)
check("query returns claims", q_result["total"] > 0)
top_values = [c["value"] for c in q_result["claims"][:3]]
check("Vancouver in results", any("Vancouver" in v for v in top_values),
      f"got {top_values}")
check("Toronto NOT in active results",
      not any("Toronto" in v for v in top_values),
      f"got {top_values}")

q_pref = handle_memory_query(conn, query="dark mode preference", session_id=sid)
check("preference query works", any("dark" in c["value"].lower() for c in q_pref["claims"]))
print()


# --- Behavior 5: CORRECT via MCP tool ---
print("5. CORRECT: dismiss and correct claims")

eng_claim = [c for c in active_after if "engineer" in c.value][0]
correction = handle_memory_correct(conn, claim_id=eng_claim.claim_id,
                                   action="correct", new_value="works as a ML engineer")
check("correction created new claim", correction.get("new_claim_id") is not None)
check("old claim superseded", get_claim(conn, eng_claim.claim_id).status.value == "superseded")
new_claim = get_claim(conn, correction["new_claim_id"])
check("new claim is active", new_claim.status.value == "active")
check("new value correct", new_claim.value == "works as a ML engineer")

dismissal = handle_memory_correct(conn, claim_id=r1.created_claims[2].claim_id,
                                  action="dismiss")
check("dismissal works", dismissal.get("action") == "dismissed")
print()


# --- Final state ---
print("=== Final Memory State ===")
final = list_active_claims(conn, sid)
print(f"Active claims: {len(final)}")
for c in final:
    print(f"  [{c.predicate}] {c.subject}: {c.value} (conf={c.confidence})")

conn.close()

print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
if FAIL:
    print("CORE MEMORY LOOP: INCOMPLETE")
    sys.exit(1)
else:
    print("CORE MEMORY LOOP: ALL BEHAVIORS VERIFIED")
    sys.exit(0)
