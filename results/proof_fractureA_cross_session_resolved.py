"""PROOF — Fracture A: the cross-session serve door returns a RESOLVED world-state.

Before fix-A, ``handle_memory_query(session_id=None, ...)`` — the whole-tenant /
multi-session sweep — returned only ranked claims + raw episodes (a top-k dump).
fix-A gives that path the same resolved layer the single-session path has:
``world_state = brain_across(...)`` projecting ONE current value per slot across
the tenant's sessions, with stale superseded values ABSENT.

Deterministic: NullEmbedder / no model (set MEMCONTEXT_EMBED_EPISODES=0). Run:

    python results/proof_fractureA_cross_session_resolved.py

Exits non-zero if any assertion fails.
"""
from __future__ import annotations

import os
import sqlite3
import sys

# Standalone env (these run OUTSIDE pytest, so set what conftest would set).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
# A script run puts THIS dir (results/) on sys.path[0], so a bare ``import
# memcontext`` would resolve through the editable install to whatever repo the
# .pth points at (the MAIN checkout), NOT this worktree. Prepend the worktree
# root so the integrated code under test wins.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PACKS = os.path.join(_ROOT, "predicate_packs")
os.environ.setdefault("SUBSTRATE_PACKS_DIR", _PACKS)
os.environ.setdefault("ACTIVE_PACK", "general")
os.environ.setdefault("MEMCONTEXT_EMBED_EPISODES", "0")

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.mcp_tools import handle_memory_query
from memcontext.schema import Speaker, Turn, open_database


def _turn(db: sqlite3.Connection, sid: str, text: str) -> str:
    t = Turn(turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER,
             text=text, ts=now_ns(), asr_confidence=None)
    insert_turn(db, t)
    return t.turn_id


def _resolved(ws: dict) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for subj, body in ws["subjects"].items():
        for f in body["facts"]:
            out.setdefault((subj, f["predicate"]), []).append(f["value"])
    return out


def main() -> int:
    db = open_database(":memory:")
    db.row_factory = sqlite3.Row

    # Two sessions, same tenant. home: Boston(s1) -> Seattle(s3); employer: Acme ->
    # Globex. insert_claim does NOT run cross-session supersession, so per the
    # product BOTH the stale (s1) and current (s3) rows are active in the store —
    # the exact situation a top-k dump would surface BOTH of.
    tj = _turn(db, "s1", "I live in Boston and work at Acme.")
    insert_claim(db, session_id="s1", subject="home", predicate="user_fact",
                 value="Boston", confidence=0.9, source_turn_id=tj)
    insert_claim(db, session_id="s1", subject="employer", predicate="user_fact",
                 value="Acme", confidence=0.9, source_turn_id=tj)
    tm = _turn(db, "s3", "Moved to Seattle, new job at Globex.")
    insert_claim(db, session_id="s3", subject="home", predicate="user_fact",
                 value="Seattle", confidence=0.9, source_turn_id=tm)
    insert_claim(db, session_id="s3", subject="employer", predicate="user_fact",
                 value="Globex", confidence=0.9, source_turn_id=tm)

    failures: list[str] = []

    # ---- The cross-session door (session_id=None) now carries world_state. ----
    out = handle_memory_query(db, query="where does the user live now",
                              session_id=None, include_resolved=True)
    if "world_state" not in out:
        failures.append("cross-session door (session_id=None) returned NO world_state")
        print("RESULT keys:", sorted(out.keys()))
        return _report(failures)

    res = _resolved(out["world_state"])
    print("Resolved cross-session world_state slots:")
    for k, v in sorted(res.items()):
        print(f"  {k} -> {v}")

    # CURRENT value per slot; stale superseded values ABSENT.
    if res.get(("home", "user_fact")) != ["Seattle"]:
        failures.append(f"home not resolved to current Seattle: {res.get(('home','user_fact'))}")
    if res.get(("employer", "user_fact")) != ["Globex"]:
        failures.append(f"employer not resolved to current Globex: {res.get(('employer','user_fact'))}")
    # Stale values must NOT appear anywhere in the resolved view.
    all_vals = [v for vs in res.values() for v in vs]
    for stale in ("Boston", "Acme"):
        if stale in all_vals:
            failures.append(f"STALE value {stale!r} present in resolved cross-session world_state")
    print("Stale values (Boston/Acme) absent from resolved view:",
          not any(s in all_vals for s in ("Boston", "Acme")))

    # ---- Additive: opting out returns the OLD top-k-dump shape (no world_state). ----
    bare = handle_memory_query(db, query="where does the user live now",
                               session_id=None, include_resolved=False)
    if "world_state" in bare:
        failures.append("include_resolved=False still attached world_state (not additive)")
    print("include_resolved=False suppresses world_state (additive):",
          "world_state" not in bare)

    return _report(failures)


def _report(failures: list[str]) -> int:
    print()
    if failures:
        print("FRACTURE A PROOF: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("FRACTURE A PROOF: PASS — cross-session (session_id=None) path returns a "
          "resolved world_state; stale superseded values ABSENT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
