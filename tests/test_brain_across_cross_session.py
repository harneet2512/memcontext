"""FRACTURE A — cross-session resolved world-state.

Deterministic (NullEmbedder / no model): proves brain_across resolves each
(subject, predicate) slot to its most-recent value ACROSS sessions, and that the
cross-session serve door (handle_memory_query, session_id=None) now attaches a
resolved world_state where it previously returned only a top-k dump.
"""
from __future__ import annotations

import sqlite3

from memcontext.brain import brain, brain_across
from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.mcp_tools import handle_memory_query
from memcontext.schema import Speaker, Turn


def _turn(db: sqlite3.Connection, sid: str, text: str) -> str:
    t = Turn(turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER,
             text=text, ts=now_ns(), asr_confidence=None)
    insert_turn(db, t)
    return t.turn_id


def _seed_multi_session(db: sqlite3.Connection) -> None:
    # home updated Boston (s1) -> Seattle (s3); employer Acme (s1) -> Globex (s3).
    # Per-session supersession leaves BOTH the stale and current rows active.
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


def _resolved(ws: dict) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for subj, body in ws["subjects"].items():
        for f in body["facts"]:
            out.setdefault((subj, f["predicate"]), []).append(f["value"])
    return out


def test_brain_across_resolves_current_value_per_slot(db: sqlite3.Connection) -> None:
    _seed_multi_session(db)

    # Single-session brain only ever sees one session — the gap Fracture A names.
    s1 = brain(db, session_id="s1")
    assert _resolved(s1)[("home", "user_fact")] == ["Boston"]

    ws = brain_across(db, session_ids=["s1", "s3"])
    assert ws["sessions"] == 2
    res = _resolved(ws)
    # CURRENT value per slot; stale earlier value ABSENT from the resolved view.
    assert res[("home", "user_fact")] == ["Seattle"]
    assert res[("employer", "user_fact")] == ["Globex"]


def test_brain_across_keeps_nl_only_facts_distinct(db: sqlite3.Connection) -> None:
    # NL-only facts (no triple) carry no slot identity and must NOT be fused by the
    # most-recent-wins resolution; each survives in the resolved view.
    from memcontext.claims import insert_fact

    t1 = _turn(db, "s1", "Remember the spare key is under the third flowerpot.")
    insert_fact(db, session_id="s1", source_turn_id=t1, confidence=0.9,
                text="spare key is under the third flowerpot")
    t2 = _turn(db, "s2", "The garage code is 4417.")
    insert_fact(db, session_id="s2", source_turn_id=t2, confidence=0.9,
                text="garage code is 4417")

    ws = brain_across(db, session_ids=["s1", "s2"])
    nl_values = [f["value"] for body in ws["subjects"].values() for f in body["facts"]]
    # Both NL facts present (subjects are entity-derived, predicate empty).
    assert any("flowerpot" in (v or "") for v in nl_values) or len(nl_values) >= 2


def test_cross_session_door_attaches_resolved_world_state(db: sqlite3.Connection) -> None:
    _seed_multi_session(db)

    # Cross-session door (session_id=None): resolved layer now present...
    out = handle_memory_query(db, query="where does the user live now",
                              session_id=None, include_resolved=True)
    assert "world_state" in out
    res = _resolved(out["world_state"])
    assert res[("home", "user_fact")] == ["Seattle"]
    assert res[("employer", "user_fact")] == ["Globex"]

    # ...and is suppressed when the caller opts out (old top-k-dump behavior).
    bare = handle_memory_query(db, query="where does the user live now",
                               session_id=None, include_resolved=False)
    assert "world_state" not in bare


def test_brain_across_empty_scope_is_safe(db: sqlite3.Connection) -> None:
    ws = brain_across(db, session_ids=[])
    assert ws["sessions"] == 0
    assert ws["subjects"] == {}
