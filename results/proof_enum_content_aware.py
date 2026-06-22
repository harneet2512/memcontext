"""Proof: content-aware distinct counting survives shared-boilerplate phrasing.

THE FRACTURE (real embedder, MiniLM — the product default): when most instances
of a slot share a phrasing TEMPLATE ("need to return a <X> that ...", "bought a
<Y>"), the repeated boilerplate dominates the sentence embedding and inflates the
pairwise cosine between genuinely DISTINCT objects above the data-driven valley,
so distinct objects MERGE and the distinct-count UNDER-counts.

THE FIX (memcontext/enumeration.py): make the cluster IDENTITY content-aware —
two instances are blocked from merging when each carries a distinguishing content
token the other lacks AND those distinguishing tokens are NOT synonyms of each
other (laptop/phone, jacket/dress); a shared distinguishing object also bridges a
drifted paraphrase of the SAME object. The only learned quantity is the
EMBEDDER's own word-level synonym floor, probed live from fixed domain-neutral
word pairs — no benchmark coupling, no tuned threshold.

This runs the FULL integrated product path: ingest via on_new_turn (Passthrough
extractor + SemanticSupersession + real embedder), then handle_memory_query with
an aggregation-intent query, and reads res["enumeration"]["distinct_count"].

Run:
    SUBSTRATE_PACKS_DIR=<worktree>/predicate_packs ACTIVE_PACK=general \
    MEMCONTEXT_EMBED_EPISODES=1 python results/proof_enum_content_aware.py
"""
from __future__ import annotations

import os
import sys

# Import the WORKTREE's memcontext, not a sibling/parent checkout. When this file
# is run directly (``python results/proof_enum_content_aware.py``) the worktree
# ROOT is not on sys.path, so ``import memcontext`` can resolve to a DIFFERENT
# (parent-repo) copy that lacks this branch's fix — silently proving nothing.
# Prepend the worktree root (this file's parent's parent) so its own memcontext/
# wins. No effect when run via ``-m`` (the root is already path[0]).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# The integrated path requires a real embedder for the enumeration block to
# attach (semantic_enabled()). Make that explicit and fail loud if misconfigured.
os.environ.setdefault("MEMCONTEXT_EMBED_EPISODES", "1")
os.environ.setdefault("ACTIVE_PACK", "general")

from memcontext.schema import Speaker, open_database
from memcontext.extractors import PassthroughExtractor
from memcontext.on_new_turn import on_new_turn
from memcontext.predicate_packs import active_pack
from memcontext.retrieval import EmbeddingClient
from memcontext.supersession_semantic import SemanticSupersession
from memcontext.mcp_tools import handle_memory_query

NAMESPACE = "user"
AGG_QUERY = "how many things"  # 'how many' => aggregation intent => enumeration


def _fresh_db():
    """A fresh in-memory store; clear the pack cache per the development rule."""
    active_pack.cache_clear()
    conn = open_database(":memory:")
    import sqlite3

    conn.row_factory = sqlite3.Row
    return conn


def _ingest(conn, emb, values: list[str], *, session_prefix: str) -> None:
    """Ingest each value as one user_event instance, one session per value (the
    product keeps one session per ingested document)."""
    from memcontext.retrieval import backfill_embeddings

    sem = SemanticSupersession(emb)
    for i, v in enumerate(values):
        sid = f"{session_prefix}-{i}"
        on_new_turn(
            conn,
            session_id=sid,
            speaker=Speaker.USER,
            text=v,
            extractor=PassthroughExtractor(
                [{"subject": "user", "predicate": "user_event", "value": v}]
            ),
            semantic=sem,
            embedder=emb,
            namespace=NAMESPACE,
        )
        # Embed the claim for semantic retrieval (the production serve path relies
        # on claim embeddings; backfilling here makes the aggregation query surface
        # the instances regardless of its exact wording — not a benchmark tweak).
        backfill_embeddings(conn, sid, client=emb)


def _distinct_count(conn, emb, session_prefix: str) -> int:
    res = handle_memory_query(
        conn, query=AGG_QUERY, session_id=None, namespace=NAMESPACE
    )
    enum = res.get("enumeration")
    assert enum is not None, f"no enumeration block attached (res keys: {list(res)})"
    return enum["distinct_count"]


def _warm_serve_path(emb) -> None:
    """Run ONE throwaway ingest+aggregation-query cycle so the serve path's
    embedder and retrieval caches are hot. The very first query in a cold process
    can be served before the lazily-loaded retrieval embedder is ready (no
    semantic hits, empty claims_out, so the additive enumeration block is skipped)
    — this warms it end-to-end so every measured case is served warm. Throwaway
    DB + throwaway namespace, so it touches none of the measured cases."""
    conn = _fresh_db()
    sem = SemanticSupersession(emb)
    from memcontext.retrieval import backfill_embeddings

    for i, v in enumerate(["warm up one", "warm up two", "warm up three"]):
        sid = f"warm-{i}"
        on_new_turn(
            conn,
            session_id=sid,
            speaker=Speaker.USER,
            text=v,
            extractor=PassthroughExtractor(
                [{"subject": "user", "predicate": "user_event", "value": v}]
            ),
            semantic=sem,
            embedder=emb,
            namespace="warmup",
        )
        backfill_embeddings(conn, sid, client=emb)
    handle_memory_query(conn, query=AGG_QUERY, session_id=None, namespace="warmup")


def _case(emb, name: str, values: list[str], expected: int) -> bool:
    conn = _fresh_db()
    _ingest(conn, emb, values, session_prefix=name)
    got = _distinct_count(conn, emb, name)
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: distinct_count={got} (expected {expected})")
    for v in values:
        print(f"         - {v}")
    return ok


def main() -> int:
    emb = EmbeddingClient(modal_url=None)
    # surface any model load failure immediately (no fake pass)
    probe = emb.embed(["probe one", "probe two"])
    assert probe and len(probe[0]) > 0, "real embedder failed to load"

    # The SERVE path (handle_memory_query) embeds via the GLOBAL episode_embedder(),
    # a different client than `emb`. Warm it now so the very first query is not
    # served by a still-cold retrieval embedder (which would return no semantic
    # hits, no claims_out, and thus no enumeration block on a cold start).
    from memcontext.retrieval import episode_embedder, semantic_enabled

    assert semantic_enabled(), "MEMCONTEXT_EMBED_EPISODES must be 1 (real embedder)"
    ee = episode_embedder()
    assert ee is not None and ee.embed(["warm up the serve-path embedder"]), (
        "serve-path embedder failed to load"
    )
    _warm_serve_path(emb)  # end-to-end warmup so the first measured query is hot

    results = []
    # 1. clothing: three DISTINCT items sharing 'need to return a X' boilerplate.
    results.append(
        _case(
            emb,
            "clothing",
            [
                "need to pick up boots exchanged at Zara",
                "need to return a jacket that was the wrong size",
                "need to return a dress that did not fit",
            ],
            3,
        )
    )
    # 2. purchases: three DISTINCT objects sharing 'bought a Y' boilerplate.
    #    ("new" keeps every value >=3 content words so the product's admission
    #    layer admits all three — it rejects 2-content-word turns; this is the
    #    real product gate, not part of the enumeration fracture.)
    results.append(
        _case(
            emb,
            "purchases",
            ["bought a new laptop", "bought a new phone", "bought new headphones"],
            3,
        )
    )
    # 3. paraphrases of ONE purchase => exactly one occurrence.
    results.append(
        _case(
            emb,
            "paraphrase",
            [
                "bought a new laptop",
                "I purchased a new laptop",
                "got myself a new laptop",
            ],
            1,
        )
    )
    # 4. mixed: two distinct objects + one paraphrase of one of them => 2.
    results.append(
        _case(
            emb,
            "mixed",
            ["bought a new laptop", "bought a new phone", "I purchased a new laptop"],
            2,
        )
    )

    passed = sum(results)
    print(f"\n{passed}/{len(results)} TDD cases passed.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())



