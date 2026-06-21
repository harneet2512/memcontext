"""Offline PROOF — FRACTURE B (identity collapse under coarse predicates).

Runs as a plain script (no pytest needed):  python tests/proof_fractureB_identity.py

It proves, with the REAL local embedder driving Pass-2 semantic supersession,
that deriving a deterministic ATTRIBUTE slot from the value
(memcontext.attribute_key) fixes identity collapse when extraction emits ONE
coarse predicate ('user_fact', single_valued=∅ — verified live) for every
personal fact, WITHOUT regressing the fine-grained / slot-less paths.

It measures four things BEFORE (attribute disabled — i.e. today's product) vs
AFTER (attribute enabled — this fix), on the SAME realistic, general scenario:

  M1  Projection collapse: how many of N distinct personal facts survive the
      newest-wins (subject, predicate) collapse.
  M2  False supersession: do two DISTINCT facts under 'user_fact' wrongly
      supersede each other on ingest (Pass-1 jaccard / Pass-2 semantic).
  M3  Genuine UPDATE: does a real change of ONE slot
      ('employer: Acme' -> 'employer: Globex') still supersede (must stay true).
  M4  Enumeration: counting "favorite restaurants" must count THAT slot's
      distinct instances, not the user's whole corpus.

BLOCK honestly if the real embedder can't load — no fake pass.
"""
from __future__ import annotations

import os
import sys

# Make the package importable when run as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memcontext.schema import open_database  # noqa: E402
from memcontext.predicate_packs import active_pack  # noqa: E402


def _load_real_embedder():
    """Return the REAL EmbeddingClient or None (honest BLOCK)."""
    try:
        import sentence_transformers  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"BLOCKED: sentence_transformers unavailable: {exc}")
        return None
    try:
        from memcontext.retrieval import EmbeddingClient
        emb = EmbeddingClient(modal_url=None)
        probe = emb.embed(["probe one", "probe two"])
        assert probe and len(probe[0]) > 0
        return emb
    except Exception as exc:  # pragma: no cover
        print(f"BLOCKED: real embedder failed to load: {exc}")
        return None


# A realistic, GENERAL personal-assistant history. Every value is the coarse
# 'user_fact' slot a real extractor emits (see extractors.py examples:
# "home city: Toronto", "office located in Brooklyn"). NOT LongMemEval verbatim.
DISTINCT_FACTS = [
    "home city: Toronto",
    "employer: Acme",
    "favorite restaurant: Nopa",
    "commute time: 25 minutes by bike",
    "allergic to peanuts",
    "owns a golden retriever named Biscuit",
]


def _ingest(conn, emb, value, *, session, attribute_enabled):
    """Ingest one coarse user_fact through the FULL pipeline (Pass-1 + Pass-2,
    real embedder). When attribute_enabled is False we monkeypatch attribute_key
    to always-empty to reproduce TODAY's behaviour on the identical input."""
    import memcontext.attribute_key as ak
    from memcontext.mcp_tools import handle_memory_store

    saved = ak.attribute_key
    if not attribute_enabled:
        ak.attribute_key = lambda value: ""  # type: ignore[assignment]
    try:
        return handle_memory_store(
            conn,
            text=f"For the record, my {value}.",
            session_id=session,
            claims=[{"subject": "user", "predicate": "user_fact",
                     "value": value, "confidence": 0.9}],
        )
    finally:
        ak.attribute_key = saved  # restore


def _active_user_facts(conn, session):
    from memcontext.claims import list_active_claims
    return [c for c in list_active_claims(conn, session)
            if c.predicate == "user_fact" and c.subject == "user"]


def _run_scenario(emb, *, attribute_enabled):
    """Ingest the 6 distinct facts, then a genuine employer UPDATE. Return
    (surviving_active_facts, total_supersessions, update_superseded)."""
    # patch attribute_key for the WHOLE scenario (Pass-1, Pass-2, projection)
    import memcontext.attribute_key as ak
    saved = ak.attribute_key
    if not attribute_enabled:
        ak.attribute_key = lambda value: ""  # type: ignore[assignment]
    try:
        conn = open_database(":memory:")
        import sqlite3
        conn.row_factory = sqlite3.Row
        session = "fractureB-proof"

        total_super = 0
        for v in DISTINCT_FACTS:
            r = _ingest(conn, emb, v, session=session, attribute_enabled=True)
            total_super += r.get("supersessions", 0)

        survivors_before_update = _active_user_facts(conn, session)

        # M3 genuine update: same 'employer' slot, new value -> MUST supersede.
        r_upd = _ingest(conn, emb, "employer: Globex",
                        session=session, attribute_enabled=True)
        update_superseded = r_upd.get("supersessions", 0) >= 1

        # projection collapse (M1): the newest-wins identity grouping
        from memcontext.projections import claims_grouped_by_subject_predicate
        grouped = claims_grouped_by_subject_predicate(
            _active_user_facts(conn, session)
        )

        return {
            "active_after_distinct_ingest": len(survivors_before_update),
            "supersessions_during_distinct_ingest": total_super,
            "projection_rows": len(grouped),
            "update_superseded": update_superseded,
            "conn": conn,
            "session": session,
        }
    finally:
        ak.attribute_key = saved


def _enumeration_count(conn, session, emb, *, attribute_enabled):
    """M4: count distinct gym-class occurrences (the canonical 'how many times'
    enumeration case) under the COARSE predicate, with the corpus ALSO holding
    unrelated coarse facts. True distinct count = 3. Each occurrence is mentioned
    with TWO paraphrases (the bimodal within/cross signal the separator needs).
    BEFORE (no slot isolation) the dominant slot is the user's whole corpus, so
    the count folds in unrelated facts; AFTER it isolates the gym slot."""
    import memcontext.attribute_key as ak
    from memcontext.enumeration import enumerate_retrieved
    saved = ak.attribute_key
    if not attribute_enabled:
        ak.attribute_key = lambda value: ""  # type: ignore[assignment]
    try:
        from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
        from memcontext.schema import Speaker, Turn
        t = Turn(turn_id=new_turn_id(), session_id=session, speaker=Speaker.USER,
                 text="gym", ts=now_ns())
        insert_turn(conn, t)
        # 3 distinct gym occurrences, each in two phrasings (label: value form).
        gym_vals = [
            "gym class: spin class on Monday morning",
            "gym class: Monday morning cycling session",
            "gym class: yoga on Wednesday evening",
            "gym class: Wednesday night yoga session",
            "gym class: Saturday afternoon CrossFit",
            "gym class: CrossFit workout Saturday afternoon",
        ]
        # Unrelated coarse facts that BEFORE would fold into the dominant slot.
        other_vals = [
            "employer: Globex", "home city: Toronto",
            "allergic to peanuts", "commute time: 25 minutes by bike",
        ]
        retrieved = []
        for v in gym_vals + other_vals:
            insert_claim(conn, session_id=session, subject="user",
                         predicate="user_fact", value=v, confidence=0.9,
                         source_turn_id=t.turn_id)
            retrieved.append({"subject": "user", "predicate": "user_fact", "value": v})
        res = enumerate_retrieved(conn, session, retrieved, emb)
        served = res.distinct_count if res else None

        # Slot-isolation oracle: the count FRACTURE B should produce is exactly the
        # count over the gym slot ALONE. If served == oracle, the served path
        # isolated the right slot (any residual vs the true 3 is the enumeration
        # separator on divergent paraphrases — a separate concern, equal in both).
        from memcontext.enumeration import count_distinct_instances
        oracle = count_distinct_instances(
            conn, session, "user", "user_fact", emb, attribute="gym_class"
        ).distinct_count
        return served, oracle
    finally:
        ak.attribute_key = saved


def main() -> int:
    print("== FRACTURE B identity proof (REAL embedder) ==")
    print(f"active pack: user_fact in families = "
          f"{'user_fact' in active_pack().predicate_families}, "
          f"single_valued = {set(active_pack().single_valued)}")
    print(f"distinct coarse facts ingested: {len(DISTINCT_FACTS)}")
    print()

    emb = _load_real_embedder()
    if emb is None:
        print("RESULT: BLOCKED (no real embedder) — not a pass, not a fail.")
        return 2

    before = _run_scenario(emb, attribute_enabled=False)
    after = _run_scenario(emb, attribute_enabled=True)

    before_enum, _ = _enumeration_count(before["conn"], before["session"], emb,
                                        attribute_enabled=False)
    after_enum, after_oracle = _enumeration_count(
        after["conn"], after["session"], emb, attribute_enabled=True)

    n = len(DISTINCT_FACTS)
    print(f"{'metric':45} {'BEFORE':>10} {'AFTER':>10}")
    print("-" * 67)
    print(f"{'M1 distinct facts surviving collapse (of '+str(n)+')':45}"
          f" {before['projection_rows']:>10} {after['projection_rows']:>10}")
    print(f"{'M2 false supersessions on distinct ingest':45}"
          f" {before['supersessions_during_distinct_ingest']:>10}"
          f" {after['supersessions_during_distinct_ingest']:>10}")
    print(f"{'M3 genuine employer UPDATE supersedes':45}"
          f" {str(before['update_superseded']):>10}"
          f" {str(after['update_superseded']):>10}")
    print(f"{'M4 gym count: served (slot-isolated)':45}"
          f" {str(before_enum):>10} {str(after_enum):>10}")
    print(f"    (true distinct=3; AFTER slot-isolation oracle={after_oracle}; "
          f"served==oracle => isolation correct, residual is the separator)")
    print()

    ok = (
        after["projection_rows"] == n          # all distinct facts survive
        and after["supersessions_during_distinct_ingest"] == 0  # no false fuse
        and after["update_superseded"] is True  # genuine update still works
        and after_enum == after_oracle          # served count == gym-slot-only count
        and after_enum < before_enum            # no longer counts the whole corpus
    )
    improved = (
        after["projection_rows"] > before["projection_rows"]
        or after["supersessions_during_distinct_ingest"]
        < before["supersessions_during_distinct_ingest"]
        or after["update_superseded"] != before["update_superseded"]
        or after_enum < before_enum
    )
    print(f"AFTER correct on all four: {ok}")
    print(f"measurably better than BEFORE: {improved}")
    if ok and improved:
        print("RESULT: PASS — identity collapse fixed, no regression on update.")
        return 0
    if ok and not improved:
        print("RESULT: NEUTRAL — correct, but BEFORE was already correct here.")
        return 0
    print("RESULT: FAIL — fix did not achieve the necessary condition.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
