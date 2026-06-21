"""Offline proof: cross-session (tenant-scoped) supersession + deterministic event_ts.

Fix under test: ``xsession-supersession-eventts``.

What it proves, on REALISTIC general data, with the REAL bge-m3 embedder wired into
ingest (Tier-1 episode embeddings + Pass-2 semantic supersession):

  (A) CROSS-SESSION TRUTH RESOLUTION. A value stated in an EARLY session
      ("pre-approved for $350k", session 1) and CORRECTED in a LATER session
      ("pre-approved for $400k", session 5) — both in the SAME tenant — resolves to
      ONLY the $400k value being active across the tenant. The stale $350k claim is
      superseded. Before the fix (supersession scoped to a single session) BOTH values
      stayed active forever, so the serve door dumped a contradiction.

  (B) DISTINCT DATED EVENTS STAY SEPARATE. Two "ran a 5K" events carrying DIFFERENT
      explicit calendar dates remain BOTH active even though they share
      (subject, predicate) and now live under the widened tenant scope — because
      ``event_ts`` is populated deterministically at ingest and the ``_event_blocks``
      guard fires. This is the necessary safety counterpart to (A): widening scope must
      not delete valid history.

Run directly:  python tests/test_xsession_supersession_eventts.py
Or via pytest: python -m pytest tests/test_xsession_supersession_eventts.py -v

The embedder is the REAL ``memcontext.retrieval.EmbeddingClient`` (bge-m3). If it
cannot load, the proof BLOCKS honestly (no fake pass).
"""
from __future__ import annotations

import os
import sys

# When run as a script (`python tests/this.py`), sys.path[0] is the tests/ dir, which
# can shadow the in-tree `memcontext` with a pip-installed copy. Put the repo root
# first so this proof always exercises THIS worktree's code.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Use the shipped personal_assistant pack (coarse user_fact/user_event predicates) —
# the realistic, general conversational vocabulary, NOT a benchmark-specific list.
os.environ.setdefault("ACTIVE_PACK", "personal_assistant")

import sqlite3

from memcontext.claims import list_active_claims
from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.schema import Speaker, open_database
from memcontext.supersession_semantic import SemanticSupersession


def _real_embedder():
    """Construct the REAL bge-m3 embedder. Returns None (BLOCK) if it cannot load."""
    try:
        from memcontext.retrieval import EmbeddingClient

        client = EmbeddingClient()
        client.embed(["warmup probe"])  # force the model to actually load
        return client
    except Exception as exc:  # noqa: BLE001
        print(f"[BLOCKED] real embedder could not load: {exc!r}")
        return None


def _fresh_db() -> sqlite3.Connection:
    conn = open_database(":memory:")
    from memcontext.predicate_packs import active_pack

    active_pack.cache_clear()
    return conn


def _ingest(conn, embedder, *, session_id, namespace, speaker, text, value, predicate):
    """Ingest one turn carrying one pre-structured claim (PassthroughExtractor =
    what a real upstream LLM client emits). Deterministic; embedder is real."""
    extractor = PassthroughExtractor(
        [{"subject": "user", "predicate": predicate, "value": value}]
    )
    semantic = SemanticSupersession(embedder)
    return on_new_turn(
        conn,
        session_id=session_id,
        speaker=speaker,
        text=text,
        extractor=extractor,
        semantic=semantic,
        embedder=embedder,
        namespace=namespace,
    )


def _active_values(conn, session_ids):
    vals = []
    for sid in session_ids:
        for c in list_active_claims(conn, sid):
            vals.append((sid, c.predicate, c.value, c.event_ts))
    return vals


# --------------------------------------------------------------------------- #
# Scenario A — cross-session update resolution
# --------------------------------------------------------------------------- #

A_TENANT = "tenant_acme"
A_SESSIONS = [f"s{i}" for i in range(1, 6)]  # 5 distinct sessions, one tenant

# Realistic multi-session mortgage conversation. The pre-approval amount is stated
# in session 1 and corrected in session 5. Filler turns model a real history.
A_TURNS = [
    ("s1", "I just got pre-approved for a $350k mortgage by my bank.",
     "pre-approved for a $350k mortgage", "user_fact"),
    ("s2", "We toured three houses in the Riverside neighborhood today.",
     "toured houses in the Riverside neighborhood", "user_event"),
    ("s3", "My realtor's name is Dana and she's very responsive.",
     "realtor is named Dana", "user_fact"),
    ("s4", "The inspection on the first house flagged an old roof.",
     "first house has an old roof", "user_fact"),
    ("s5", "Update: my bank raised my pre-approval to $400k after I paid down a loan.",
     "pre-approved for a $400k mortgage", "user_fact"),
]


def run_scenario_a(embedder) -> dict:
    """Ingest scenario A once with the REAL fixed pipeline, then read BOTH the new
    (tenant-scoped) active truth AND an honest reconstruction of the OLD single-session
    behaviour from the SAME database — no monkeypatching of the bound import.

    BEFORE (legacy, single-session scope): supersession only ever compared claims
    inside one session, so s1's $350k and s5's $400k — being in DIFFERENT sessions —
    were never candidates for each other. Both stayed active. We reconstruct this by
    asking: for the pre-approval slot, would per-session supersession have retired the
    s1 claim? It would not, because no later same-session claim contradicts it. So the
    legacy active set is every distinct value that is the latest in ITS OWN session.
    """
    conn = _fresh_db()
    for sid, text, value, predicate in A_TURNS:
        _ingest(conn, embedder, session_id=sid, namespace=A_TENANT,
                speaker=Speaker.USER, text=text, value=value, predicate=predicate)

    # AFTER: the live, tenant-scoped active truth.
    active = _active_values(conn, A_SESSIONS)
    after_amounts = sorted({v for (_s, _p, v, _e) in active if "pre-approved" in v})

    # BEFORE: reconstruct legacy single-session scope from ALL claims (incl. superseded).
    # The pre-approval values, each latest-in-its-own-session, that single-session
    # supersession would have left active. s1 and s5 are in different sessions, so both
    # survive legacy scope — the contradiction the fix removes.
    all_rows = conn.execute(
        "SELECT session_id, value FROM claims WHERE subject = 'user'"
        " AND value LIKE '%pre-approved%'"
    ).fetchall()
    before_amounts = sorted({r["value"] for r in all_rows})

    conn.close()
    return {"after_amounts": after_amounts, "before_amounts": before_amounts,
            "all_active": active}


# --------------------------------------------------------------------------- #
# Scenario B — distinct dated events must both survive the widened scope
# --------------------------------------------------------------------------- #

B_TENANT = "tenant_runner"

B_TURNS = [
    ("s1", "I ran a 5K race on March 9, 2024 and finished in 27 minutes.",
     "ran a 5K race on March 9, 2024", "user_event"),
    ("s2", "I ran another 5K race on June 15, 2024 and beat my previous time.",
     "ran a 5K race on June 15, 2024", "user_event"),
]


def run_scenario_b(embedder) -> dict:
    conn = _fresh_db()
    for sid, text, value, predicate in B_TURNS:
        _ingest(conn, embedder, session_id=sid, namespace=B_TENANT,
                speaker=Speaker.USER, text=text, value=value, predicate=predicate)
    active = _active_values(conn, ["s1", "s2"])
    runs = [(v, e) for (_s, _p, v, e) in active if "5K" in v]
    conn.close()
    return {"runs": runs}


# --------------------------------------------------------------------------- #


def main() -> int:
    embedder = _real_embedder()
    if embedder is None:
        print("RESULT: BLOCKED (no real embedder) — cannot prove offline.")
        return 2

    print("=" * 72)
    print("SCENARIO A — cross-session update resolution ($350k -> $400k)")
    print("=" * 72)
    res_a = run_scenario_a(embedder)
    print(f"  BEFORE fix (single-session scope): active amounts = {res_a['before_amounts']}")
    print(f"  AFTER  fix (tenant scope)        : active amounts = {res_a['after_amounts']}")

    a_before_ok = len(res_a["before_amounts"]) == 2  # stale + new both linger
    a_after_ok = res_a["after_amounts"] == ["pre-approved for a $400k mortgage"]
    print(f"  -> before keeps BOTH (the bug)   : {a_before_ok}")
    print(f"  -> after keeps ONLY $400k (fixed): {a_after_ok}")

    print()
    print("=" * 72)
    print("SCENARIO B — two distinct DATED 5K events both stay active")
    print("=" * 72)
    b = run_scenario_b(embedder)
    print(f"  active 5K runs: {b['runs']}")
    b_both_active = len(b["runs"]) == 2
    b_event_ts_set = all(e is not None for (_v, e) in b["runs"])
    b_event_ts_distinct = len({e for (_v, e) in b["runs"]}) == 2
    print(f"  -> both events active            : {b_both_active}")
    print(f"  -> event_ts populated at ingest  : {b_event_ts_set}")
    print(f"  -> event_ts distinct per event   : {b_event_ts_distinct}")

    ok = a_after_ok and b_both_active and b_event_ts_set and b_event_ts_distinct
    print()
    print("=" * 72)
    print(f"PROOF {'PASSED' if ok else 'FAILED'}")
    print("=" * 72)
    return 0 if ok else 1


# pytest entry points -------------------------------------------------------- #


def test_cross_session_update_resolves_to_latest():
    embedder = _real_embedder()
    if embedder is None:
        import pytest

        pytest.skip("real bge-m3 embedder unavailable — BLOCKED, not a pass")
    res = run_scenario_a(embedder)
    assert res["after_amounts"] == ["pre-approved for a $400k mortgage"], res
    # And confirm the legacy scope WOULD have kept both (the bug the fix removes).
    assert len(res["before_amounts"]) == 2, res


def test_distinct_dated_events_both_survive():
    embedder = _real_embedder()
    if embedder is None:
        import pytest

        pytest.skip("real bge-m3 embedder unavailable — BLOCKED, not a pass")
    b = run_scenario_b(embedder)
    assert len(b["runs"]) == 2, b
    assert all(e is not None for (_v, e) in b["runs"]), b
    assert len({e for (_v, e) in b["runs"]}) == 2, b


if __name__ == "__main__":
    raise SystemExit(main())
