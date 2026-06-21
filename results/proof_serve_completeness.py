"""PROOF — serve-completeness (fix-S, reconciled with fix-A + Fracture-B identity).

The served claims channel of ``handle_memory_query`` becomes a RESOLVED,
self-justifying view instead of a top-k pile:

  1. SLOT-DEDUP: two competing residence mentions sharing the canonical Fracture-B
     slot (``attribute_key`` -> ``reside``) collapse to ONE served claim
     (residence 2 -> 1). The dedup keys on ``memcontext.attribute_key.attribute_key``
     — the SAME slot taxonomy supersession uses (C1 SLOT UNIFY), not the old
     ``supersession._attribute_of``.
  2. PROVENANCE LINEAGE: every surviving served claim carries ``provenance.quote``
     (the source span it was read from) and the typed supersession chain it retired.
  3. AGGREGATION: a counting query yields ``enumeration.distinct_count`` AND a
     count-aware ``reader_hint`` (answer from the DISTINCT count).
  4. ADDITIVE: a non-aggregation single-fact query is byte-identical with vs.
     without an embedder — the new machinery never perturbs the simple path.

Model-free: a deterministic concept embedder (the proven enumeration stub) makes
``semantic_enabled()`` True with ZERO model download. Run:

    python results/proof_serve_completeness.py

Exits non-zero on any failed assertion.
"""
from __future__ import annotations

import math
import os
import sqlite3
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
# Script run -> sys.path[0] is results/; a bare ``import memcontext`` would resolve
# via the editable install to the MAIN checkout, not this worktree. Prepend the
# worktree root so the integrated code under test wins.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("SUBSTRATE_PACKS_DIR", os.path.join(_ROOT, "predicate_packs"))
os.environ.setdefault("ACTIVE_PACK", "general")
os.environ.setdefault("MEMCONTEXT_EMBED_EPISODES", "0")

import memcontext.retrieval as R
from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.mcp_tools import handle_memory_query
from memcontext.schema import Speaker, Turn, open_database
from memcontext.supersession import detect_pass1


# --- Deterministic, model-free embedder (concept buckets -> bimodal cosine). ---
# Same construction as tests/test_handle_memory_query_enumeration.py.
class _ConceptEmbedder:
    model_version = "concept-serve-proof-v1"
    CONCEPTS = {
        "sushi": ("ate sushi", "had sushi for lunch", "grabbed some sushi"),
        "ramen": ("ate ramen", "had a bowl of ramen", "got ramen for dinner"),
    }

    def __init__(self) -> None:
        self._dim = len(self.CONCEPTS) + 4
        self._idx = {c: i for i, c in enumerate(self.CONCEPTS)}

    def _concept_of(self, text: str) -> str | None:
        low = text.lower()
        for c, phr in self.CONCEPTS.items():
            if any(p in low for p in phr):
                return c
        return None

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self._dim
            c = self._concept_of(t)
            if c is not None:
                v[self._idx[c]] = 1.0
            else:
                for k in range(self._dim):
                    h = int.from_bytes(
                        hashlib.sha256((t + str(k)).encode()).digest()[:4], "big"
                    )
                    v[k] = (h % 1000) / 1000.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def _turn(db: sqlite3.Connection, sid: str, text: str) -> str:
    t = Turn(turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER,
             text=text, ts=now_ns(), asr_confidence=None)
    insert_turn(db, t)
    return t.turn_id


def _fresh() -> sqlite3.Connection:
    db = open_database(":memory:")
    db.row_factory = sqlite3.Row
    return db


def prove_dedup_and_lineage(failures: list[str]) -> None:
    """(1) residence 2->1 deduped on the canonical reside slot; (2) survivor
    carries provenance.quote + supersession lineage."""
    db = _fresh()
    sid = "serve-res"

    # Two residence mentions sharing the canonical Fracture-B slot (reside). The
    # source TEXT carries the span so provenance.quote is populated; detect_pass1
    # writes the typed supersession edge so the survivor reports its retired chain.
    t1 = _turn(db, sid, "I lives in Boston.")
    c1 = insert_claim(db, session_id=sid, subject="user", predicate="user_fact",
                      value="lives in Boston", confidence=0.9, source_turn_id=t1,
                      char_start=2, char_end=17)  # span over "lives in Boston"
    t2 = _turn(db, sid, "I moved to Seattle.")
    c2 = insert_claim(db, session_id=sid, subject="user", predicate="user_fact",
                      value="moved to Seattle", confidence=0.9, source_turn_id=t2,
                      char_start=2, char_end=18)
    detect_pass1(db, c2)  # supersede the older residence value -> typed edge

    out = handle_memory_query(db, query="where does the user live",
                              session_id=sid, include_resolved=True)
    claims = out["claims"]
    res_claims = [c for c in claims if c["predicate"] == "user_fact"
                  and ("Boston" in c["value"] or "Seattle" in c["value"])]
    print(f"[dedup] residence claims served: {len(res_claims)} "
          f"(values={[c['value'] for c in res_claims]})")
    if len(res_claims) != 1:
        failures.append(
            f"residence not deduped 2->1: served {len(res_claims)} residence claims "
            f"{[c['value'] for c in res_claims]}")
        return
    survivor = res_claims[0]
    # Survivor should be the current (active) value, Seattle.
    if "Seattle" not in survivor["value"]:
        failures.append(f"dedup survivor is not the current value: {survivor['value']!r}")

    # (2) Provenance lineage on the survivor.
    prov = survivor.get("provenance")
    if not prov:
        failures.append("served survivor carries NO provenance block")
        return
    print(f"[lineage] provenance.quote = {prov.get('quote')!r}")
    if not prov.get("quote"):
        failures.append(f"provenance.quote empty/missing: {prov!r}")
    superseded = prov.get("superseded") or []
    print(f"[lineage] supersession chain = {superseded}")
    if not superseded:
        failures.append("served survivor carries NO supersession lineage")
    else:
        old_vals = [s.get("old_value") for s in superseded]
        if not any("Boston" in (v or "") for v in old_vals):
            failures.append(f"retired value 'Boston' not in lineage: {old_vals}")


def prove_aggregation_count(failures: list[str]) -> None:
    """(3) counting query -> enumeration.distinct_count + count-aware reader_hint."""
    db = _fresh()
    sid = "serve-meals"
    # 5 raw mentions of 2 distinct meal instances.
    for v in ("ate sushi", "had sushi for lunch", "grabbed some sushi",
              "ate ramen", "had a bowl of ramen"):
        t = _turn(db, sid, v)
        insert_claim(db, session_id=sid, subject="user", predicate="user_event",
                     value=v, confidence=0.9, source_turn_id=t)

    # Real embedder -> semantic_enabled() True, zero model download.
    R.episode_embedder = lambda: _ConceptEmbedder()  # type: ignore[assignment]
    try:
        out = handle_memory_query(db, query="how many different meals have I eaten?",
                                  session_id=sid, include_resolved=True)
    finally:
        pass

    enum = out.get("enumeration")
    print(f"[count] enumeration = "
          f"{ {k: enum[k] for k in ('distinct_count','t_dup')} if enum else None }")
    if not enum:
        failures.append("aggregation query produced NO enumeration block")
    elif enum["distinct_count"] != 2:
        failures.append(f"distinct_count {enum['distinct_count']} != expected 2")

    hint = out.get("reader_hint", "")
    print(f"[count] reader_hint = {hint!r}")
    if "DISTINCT" not in hint and "distinct" not in hint:
        failures.append(f"reader_hint is not count-aware: {hint!r}")


def prove_additive_single_fact(failures: list[str]) -> None:
    """(4) non-aggregation single-fact query: byte-identical with vs without an
    embedder — the new dedup/provenance/enumeration machinery never perturbs it."""
    db = _fresh()
    sid = "serve-single"
    t = _turn(db, sid, "My favorite color is blue.")
    insert_claim(db, session_id=sid, subject="user", predicate="user_fact",
                 value="favorite color is blue", confidence=0.9, source_turn_id=t,
                 char_start=3, char_end=25)

    q = "what is my favorite color?"  # no aggregation keyword
    R.episode_embedder = lambda: _ConceptEmbedder()  # type: ignore[assignment]
    with_emb = handle_memory_query(db, query=q, session_id=sid, include_resolved=True)
    R.episode_embedder = lambda: None  # type: ignore[assignment]
    no_emb = handle_memory_query(db, query=q, session_id=sid, include_resolved=True)

    print(f"[additive] claims served: {len(with_emb['claims'])}; "
          f"'enumeration' present: {'enumeration' in with_emb}")
    if "enumeration" in with_emb:
        failures.append("single-fact (non-aggregation) query wrongly got an enumeration block")
    if with_emb["claims"] != no_emb["claims"]:
        failures.append("single-fact claims differ with vs without embedder (NOT additive)")
    if with_emb["episodes"] != no_emb["episodes"]:
        failures.append("single-fact episodes differ with vs without embedder (NOT additive)")
    if with_emb["total"] != no_emb["total"]:
        failures.append("single-fact total differs with vs without embedder (NOT additive)")
    # Survivor still self-justifying (provenance attached) — additive ADDS, not removes.
    served = with_emb["claims"]
    if served and not any(c.get("provenance", {}).get("quote") for c in served):
        failures.append("single-fact served claim lost its provenance.quote")
    print(f"[additive] byte-identical core payload (claims/episodes/total): "
          f"{with_emb['claims'] == no_emb['claims'] and with_emb['episodes'] == no_emb['episodes'] and with_emb['total'] == no_emb['total']}")


def main() -> int:
    failures: list[str] = []
    print("== (1)(2) slot-dedup residence 2->1 + provenance lineage ==")
    prove_dedup_and_lineage(failures)
    print("\n== (3) aggregation distinct_count + count-aware reader_hint ==")
    prove_aggregation_count(failures)
    print("\n== (4) additive: single-fact byte-identical ==")
    prove_additive_single_fact(failures)

    print()
    if failures:
        print("SERVE-COMPLETENESS PROOF: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("SERVE-COMPLETENESS PROOF: PASS — residence 2->1 deduped on canonical "
          "attribute_key slot; served claims carry provenance.quote + supersession "
          "lineage; counting query yields distinct_count + count-aware reader_hint; "
          "single-fact query byte-identical (additive).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
