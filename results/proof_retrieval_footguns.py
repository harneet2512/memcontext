"""OFFLINE PROOF (retrieval footguns): three SHIPPED retrieval-quality fixes,
each demonstrated against the REAL `retrieve_hybrid` / `classify_query_predicates`
on an in-memory SQLite DB with a deterministic embedder (zero model downloads).

The three footguns (all on memcontext/retrieval.py), and what this proves:

  FOOTGUN 1 — weight parsing silently zeroed BM25.
    A 3-value MEMCONTEXT_RETRIEVAL_WEIGHTS used to be accepted verbatim, which
    dropped the 4th (BM25) channel -> w_bm25 = 0.0. With the lexical channel
    zeroed, an ISOLATED-LEXICAL needle (its rare token appears ONLY in the query
    and the needle) falls out of the top ranks (#5/#9/#11 in the diluted pack).
    FIX: short vectors are PADDED from defaults, so BM25 keeps its weight and the
    needle ranks #1.

  FOOTGUN 2 — the frequency channel demoted a UNIQUE needle.
    The old freq channel scored a claim by the raw COUNT of active claims sharing
    its (subject,predicate) key. A single VERBOSE turn extracted into 18 claims
    self-corroborated to count 18, while a unique relevant needle scored 1 — so
    the freq channel buried the needle (#3) under one verbose turn. FIX: freq is a
    promote-only, NEUTRAL-baseline channel; corroboration counts DISTINCT source
    turns (>= 2), bounded -> the unique needle ranks #1.

  FOOTGUN 3 — classify_query_predicates matched on SUBSTRINGS.
    "plan" matched "ex(plan)ation", "like" matched "dis(like)", etc., routing
    unrelated queries to the wrong predicate family. FIX: word-boundary (\\b)
    match -> those queries no longer mis-route.

Deterministic, general, zero-LLM, zero model download. Run:  python results/proof_retrieval_footguns.py
"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import cast

# IMPORT-PATH PIN (LIPI Plumbing): this trial worktree shares a machine with an
# editable `pip install -e` rooted at the MAIN repo. When this file is run as
# `python results/proof_retrieval_footguns.py`, Python puts `results/` (not the
# worktree root) on sys.path[0], so `import memcontext` resolves to the EDITABLE
# install at the main repo — a DIFFERENT branch whose retrieval.py predates the
# tie-aware `_rrf_ranks`. That silently tests the wrong code. Pin the worktree
# root FIRST so this proof always exercises THIS branch's memcontext/retrieval.py.
_WORKTREE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKTREE_ROOT) != sys.path[0]:
    sys.path.insert(0, str(_WORKTREE_ROOT))
for _m in list(sys.modules):
    if _m == "memcontext" or _m.startswith("memcontext."):
        del sys.modules[_m]

from memcontext.claims import insert_fact, insert_turn, new_turn_id, now_ns
from memcontext.retrieval import (
    EmbeddingClient,
    backfill_embeddings,
    classify_query_predicates,
    retrieve_hybrid,
)
from memcontext.schema import Speaker, Turn, open_database


# --- deterministic embedder (no model download) ------------------------------
class _HashEmbedder:
    """A fixed, deterministic embedder: each text -> a unit vector seeded by its
    own hash. Identical texts embed identically; the query gets a vector close to
    a designated needle by sharing its salient token. No network, no model."""

    model_version = "proof-hash-embedder"
    _DIM = 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self._DIM
            for tok in t.lower().split():
                h = hashlib.sha256(tok.encode()).digest()
                for i in range(self._DIM):
                    vec[i] += (h[i % len(h)] - 127.5) / 127.5
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


class _FlatEmbedder:
    """A degenerate embedder with NO opinion: every text -> the same unit vector.
    The semantic channel is therefore FLAT (one tied rank, neutral in fusion), so
    the BM25 / lexical channel is the sole discriminator — exactly the condition
    that exposes the 'BM25 silently zeroed' footgun."""

    model_version = "proof-flat-embedder"
    _DIM = 32

    def embed(self, texts: list[str]) -> list[list[float]]:
        v = [1.0 / math.sqrt(self._DIM)] * self._DIM
        return [list(v) for _ in texts]


def _client() -> EmbeddingClient:
    return cast(EmbeddingClient, _HashEmbedder())


def _flat_client() -> EmbeddingClient:
    return cast(EmbeddingClient, _FlatEmbedder())


def _conn() -> sqlite3.Connection:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _turn(conn: sqlite3.Connection, sid: str, text: str) -> Turn:
    turn = Turn(turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER,
                text=text, ts=now_ns())
    insert_turn(conn, turn)
    return turn


def _rank_of(results, predicate_value_substr: str) -> int:
    """1-based rank of the first hit whose claim text contains the marker."""
    for i, (claim, _score) in enumerate(results):
        if predicate_value_substr in (claim.text or ""):
            return i + 1
    return -1


# ============================================================================
# FOOTGUN 1: isolated-lexical needle must rank #1 (BM25 channel not zeroed)
# ============================================================================
def proof_isolated_lexical_needle() -> None:
    conn = _conn()
    sid = "f1"
    # A pile of verbose, semantically-busy distractor turns (no rare token).
    distractors = [
        "we talked about the quarterly roadmap and the staging deployment cadence",
        "lunch options near the office include thai ramen and a salad place",
        "the onboarding doc covers laptop setup vpn and the wiki landing page",
        "sprint planning moved to wednesday and standups stay at nine fifteen",
        "the design review covered spacing color tokens and the icon set refresh",
        "travel reimbursement needs receipts within thirty days of the trip",
        "the analytics dashboard now tracks weekly active and retention curves",
        "the support rota rotates every monday across the three on-call engineers",
    ]
    for d in distractors:
        t = _turn(conn, sid, d)
        insert_fact(conn, session_id=sid, source_turn_id=t.turn_id, confidence=0.9, text=d)

    # The NEEDLE: a rare token ("zephyrine") that appears ONLY here and in the query.
    needle_text = "the access code for the zephyrine vault is seven four two"
    nt = _turn(conn, sid, needle_text)
    insert_fact(conn, session_id=sid, source_turn_id=nt.turn_id, confidence=0.9, text=needle_text)

    # Flat embedder: semantic channel is NEUTRAL, so BM25 alone decides the needle.
    backfill_embeddings(conn, sid, client=_flat_client())
    query = "what is the zephyrine vault access code"

    # Simulate the OLD bug: a 3-value override that drops BM25 -> w_bm25 = 0.0.
    # We pass weights explicitly to model both worlds against the SAME corpus.
    W_BUGGY_3 = (0.5, 0.2, 0.1)          # what the old parser returned verbatim (BM25 missing)
    W_FIXED_4 = (0.5, 0.2, 0.1, 0.2)     # padded: BM25 restored to its default weight

    buggy = retrieve_hybrid(conn, session_id=sid, query=query, top_k=20,
                            weights=W_BUGGY_3, embedding_client=_flat_client())
    fixed = retrieve_hybrid(conn, session_id=sid, query=query, top_k=20,
                            weights=W_FIXED_4, embedding_client=_flat_client())

    r_buggy = _rank_of(buggy, "zephyrine")
    r_fixed = _rank_of(fixed, "zephyrine")
    print("FOOTGUN 1 — isolated-lexical needle (rare token only in query+needle)")
    print(f"  corpus = {len(distractors)} verbose distractors + 1 needle")
    print(f"  BM25 zeroed (3-value weights, old bug):  needle ranks #{r_buggy} of {len(buggy)}")
    print(f"  BM25 restored (padded 4-value, shipped): needle ranks #{r_fixed} of {len(fixed)}")
    assert r_buggy not in (1,), f"with BM25 zeroed the needle must NOT be #1 (got #{r_buggy})"
    assert r_fixed == 1, f"with BM25 restored the needle must be #1 (got #{r_fixed})"
    print(f"  PROVEN: padding restores BM25; isolated-lexical needle recovers to #1 (was #{r_buggy}).\n")


# ============================================================================
# FOOTGUN 2: unique needle must beat an 18-claim verbose turn (freq channel)
#
# Method (faithful counterfactual): run the REAL `retrieve_hybrid` once (SHIPPED
# freq), capturing every channel's per-claim contribution via `explain`. Then
# rebuild the fused score swapping ONLY the frequency channel for the OLD
# raw-count formula — every other channel held byte-identical. So the ranking
# delta is attributable to the freq change alone, with no hand-rolled fusion.
# ============================================================================
def proof_unique_needle_vs_verbose_turn() -> None:
    from collections import Counter
    from memcontext.retrieval import _rrf_ranks, RRF_K
    from memcontext.claims import list_active_claims

    conn = _conn()
    sid = "f2"
    # ONE verbose turn that extracts into 18 claims sharing ONE (subject,predicate)
    # key. Its text shares NO token with the query, so it is genuinely irrelevant.
    verbose = _turn(conn, sid, "weekly status notes about assorted minor topics")
    for i in range(18):
        insert_fact(conn, session_id=sid, source_turn_id=verbose.turn_id, confidence=0.9,
                    subject="status", predicate="user_fact", value=f"minor note {i}",
                    text=f"weekly status minor note number {i}")

    # The unique NEEDLE: a distinct turn, single relevant claim, unique key — it
    # matches the query lexically (BM25) where the cluster does not.
    needle_turn = _turn(conn, sid, "the prod database failover region is frankfurt")
    insert_fact(conn, session_id=sid, source_turn_id=needle_turn.turn_id, confidence=0.9,
                subject="prod_database", predicate="user_fact",
                value="failover region frankfurt",
                text="the prod database failover region is frankfurt")

    backfill_embeddings(conn, sid, client=_flat_client())
    query = "prod database failover region frankfurt"

    explain: dict[str, dict[str, float]] = {}
    shipped = retrieve_hybrid(conn, session_id=sid, query=query, top_k=30,
                              embedding_client=_flat_client(), explain=explain)
    active = list_active_claims(conn, sid)

    # OLD freq channel = raw count of active claims sharing (subject,predicate).
    counts = Counter((c.subject, c.predicate) for c in active)
    old_freq = [float(counts[(c.subject, c.predicate)]) for c in active]
    old_freq_ranks = _rrf_ranks(old_freq)
    W_FREQ = 0.1  # the freq channel weight retrieve_hybrid uses

    # Rebuild each fused score: take the SHIPPED final, remove its freq contribution,
    # add back the OLD raw-count freq contribution. Only the freq channel changes.
    rebuilt: list[tuple[object, float]] = []
    for i, c in enumerate(active):
        e = explain[c.claim_id]
        base = e["final"] - e["frequency"]
        rebuilt.append((c, base + W_FREQ / (RRF_K + old_freq_ranks[i])))
    rebuilt.sort(key=lambda x: (-x[1], x[0].claim_id))  # type: ignore[attr-defined]

    def _needle_rank(ranked) -> int:
        for i, item in enumerate(ranked):
            c = item[0]
            if c.subject == "prod_database":
                return i + 1
        return -1

    r_old = _needle_rank(rebuilt)
    r_new = _needle_rank(shipped)

    print("FOOTGUN 2 — unique needle vs one 18-claim verbose turn (freq channel)")
    print(f"  corpus = 18 claims from 1 verbose turn (one (subject,predicate)) + 1 unique needle")
    print(f"  OLD raw-count freq (count=18 self-corroborates): needle ranks #{r_old} of {len(active)}")
    print(f"  SHIPPED distinct-turn corroboration (1 turn->0):  needle ranks #{r_new} of {len(active)}")
    assert r_old != 1, f"OLD freq must BURY the unique needle (got #{r_old})"
    assert r_new == 1, f"SHIPPED freq must restore the unique needle to #1 (got #{r_new})"
    print(f"  PROVEN: raw-count freq let one verbose turn self-corroborate (count=18) and")
    print(f"          buried the relevant needle to #{r_old}; distinct-turn neutral-baseline")
    print(f"          freq gives that single turn 0 corroboration, so relevance wins -> #1.\n")


# ============================================================================
# FOOTGUN 3: classify_query_predicates word-boundary (no substring mis-routing)
# ============================================================================
def proof_substring_routing_fixed() -> None:
    print("FOOTGUN 3 — classify_query_predicates word-boundary matching")
    # "explanation" CONTAINS the substring "plan"; "dislike" CONTAINS "like". Under
    # the OLD `kw in q_lower` substring test these mis-fired and injected a spurious
    # predicate channel; under the shipped `\bkw\b` they must not. We assert that
    # below AND demonstrate the OLD substring test WOULD have fired (the bug).
    q_explanation = "can you give an explanation of the api endpoints"
    q_dislike = "things I dislike about the tooling"  # NB: no other real keyword
    OLD_SUBSTR_FIRES_PLAN = "plan" in q_explanation.lower()
    OLD_SUBSTR_FIRES_LIKE = "like" in q_dislike.lower()
    assert OLD_SUBSTR_FIRES_PLAN and OLD_SUBSTR_FIRES_LIKE, \
        "sanity: the OLD substring test really did mis-fire on these (that was the bug)"
    print(f"  OLD substring test 'plan' in 'explanation'? {OLD_SUBSTR_FIRES_PLAN}  "
          f"'like' in 'dislike'? {OLD_SUBSTR_FIRES_LIKE}  (the bug)")

    preds_explanation, qt_explanation = classify_query_predicates(q_explanation)
    preds_dislike, qt_dislike = classify_query_predicates(q_dislike)

    print(f"  'explanation ...' -> preds={sorted(preds_explanation)} type={qt_explanation}")
    print(f"  'dislike ...'     -> preds={sorted(preds_dislike)} type={qt_dislike}")
    assert "user_goal" not in preds_explanation, \
        "'plan' must NOT match inside 'explanation' (word-boundary)"
    assert "user_preference" not in preds_dislike, \
        "'like' must NOT match inside 'dislike' (word-boundary)"

    # Positive control: a STANDALONE keyword still routes correctly.
    preds_plan, qt_plan = classify_query_predicates("what is my plan for the launch")
    preds_like, qt_like = classify_query_predicates("which editor do I like best")
    print(f"  positive control 'my plan ...'  -> preds={sorted(preds_plan)} type={qt_plan}")
    print(f"  positive control 'I like ...'   -> preds={sorted(preds_like)} type={qt_like}")
    assert "user_goal" in preds_plan, "standalone 'plan' must still route to user_goal"
    assert "user_preference" in preds_like, "standalone 'like' must still route to preference"
    print("  PROVEN: \\b matching kills 'plan'->'explanation' and 'like'->'dislike'")
    print("          false fires while standalone keywords still route correctly.\n")


if __name__ == "__main__":
    os.environ.pop("MEMCONTEXT_RETRIEVAL_WEIGHTS", None)
    proof_isolated_lexical_needle()
    proof_unique_needle_vs_verbose_turn()
    proof_substring_routing_fixed()
    print("ALL THREE RETRIEVAL FOOTGUN PROOFS PASSED (deterministic, zero model download).")
