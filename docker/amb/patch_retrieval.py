#!/usr/bin/env python3
"""Build-time patch: opt-in precision re-rank for the product's
``retrieve_memory_across`` (memcontext/retrieval.py @ PRODUCT_REF).

Diagnosis (from the first AMB smoke): the cross-session breadth guarantee serves
~``per_session_k`` x ``n_sessions`` memories (~140 on a real haystack). Recall is
fine — the gold answer was present in the served context — but the reader drowns
and mis-reads/abstains (3 of 4 single-session failures had the gold IN context;
multi-session counting mis-counted with duplicates). This adds an OPT-IN
``rerank_top_k`` that re-ranks the breadth pool by QUERY-TEXT COSINE (comparable
across sessions, unlike the raw hybrid scores the docstring warns about) and
serves only the top-k. Recall preserved (needle already reserved into the pool),
precision restored. Default (no rerank_top_k) is byte-identical to the shipping
product, so this changes nothing until a caller opts in (the AMB provider does,
via patch_provider.py).

Proven offline against this exact code: tests/test_precision_rerank.py
(needle survives + ranks #1, flood cut) + the 10 existing two-tier tests
(recall-starvation guards) stay green.

Asserts every anchor matches exactly once; fails the build loudly on drift.

Usage: python patch_retrieval.py /opt/product/memcontext/retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

EDITS = [
    # 1) signature: add the opt-in parameter (anchored on per_session_k, which is
    #    unique to retrieve_memory_across).
    (
        """    per_session_k: int = DEFAULT_PER_SESSION_K,
    embedding_client: EmbeddingClient | None = None,
    explain: dict[str, dict[str, float]] | None = None,
    include_superseded: bool = False,
) -> list[tuple[MemoryHit, float]]:""",
        """    per_session_k: int = DEFAULT_PER_SESSION_K,
    embedding_client: EmbeddingClient | None = None,
    explain: dict[str, dict[str, float]] | None = None,
    include_superseded: bool = False,
    rerank_top_k: int | None = None,
) -> list[tuple[MemoryHit, float]]:""",
        "retrieve_memory_across signature: add opt-in rerank_top_k",
    ),
    # 2) the budget/return block -> breadth pool + opt-in cosine re-rank.
    (
        """    reserved.sort(key=tie)
    overflow.sort(key=tie)
    # Never cut below the per-session guarantee for the queried breadth.
    budget = min(max(top_k, len(reserved)), MAX_ACROSS_HITS)
    return (reserved + overflow)[:budget]""",
        """    reserved.sort(key=tie)
    overflow.sort(key=tie)
    # Never cut below the per-session guarantee for the queried breadth.
    budget = min(max(top_k, len(reserved)), MAX_ACROSS_HITS)
    pool = (reserved + overflow)[:budget]

    # PRECISION re-rank (opt-in via ``rerank_top_k``; legacy callers pass nothing
    # and get the full-breadth pool unchanged). The breadth pool above maximises
    # RECALL — the answer turn is present even from a low-raw-score session — but
    # serving the WHOLE pool (~per_session_k x n_sessions, ~140 on a 47-session
    # haystack) floods the reader, which then cannot isolate the needle (measured:
    # the gold answer was present in the served context yet mis-read/abstained on
    # 3 of 4 single-session failures). So when a caller asks for a precision cut,
    # re-rank the pool by QUERY-TEXT COSINE and serve only the top ``rerank_top_k``.
    #
    # Why cosine is safe where the docstring's "score-rank the sessions" was not:
    # that warning is about the RAW HYBRID scores (BM25/semantic magnitudes that
    # scale with a session's size, so cross-session comparison drops the answer
    # session). Query-to-text cosine is a NORMALISED similarity in [-1, 1], the
    # SAME quantity regardless of session, so it ranks the needle above noise
    # without the size bias. Recall is preserved (the needle was already reserved
    # into the pool); only the serve breadth shrinks. Degrades safely: any
    # embedding failure falls back to a plain truncation of the breadth order.
    if rerank_top_k is not None and embedding_client is not None and len(pool) > rerank_top_k:
        try:
            qv = embedding_client.embed([apply_query_prefix(query)])[0]
            tvs = embedding_client.embed([h.text for h, _ in pool])

            def _rel(v: list[float]) -> float:
                s = _cosine_normalised(qv, v)
                return s if -1.01 <= s <= 1.01 else _cosine_fallback(qv, v)

            order = sorted(
                range(len(pool)),
                key=lambda i: (-_rel(tvs[i]), pool[i][0].kind != "fact", pool[i][0].id),
            )
            return [pool[i] for i in order[:rerank_top_k]]
        except Exception:  # noqa: BLE001 — never break serve on an embed hiccup
            return pool[:rerank_top_k]
    return pool""",
        "retrieve_memory_across body: breadth pool + opt-in cosine precision re-rank",
    ),
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_retrieval.py <path-to memcontext/retrieval.py>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    if "rerank_top_k" in text:
        print("[patch_retrieval] already applied")
        return 0
    for anchor, replacement, label in EDITS:
        count = text.count(anchor)
        if count != 1:
            print(
                f"ERROR: anchor for [{label}] found {count}x (expected 1). "
                "Product retrieval.py drifted from PRODUCT_REF — refusing to patch.",
                file=sys.stderr,
            )
            return 1
        text = text.replace(anchor, replacement, 1)
        print(f"  patched: {label}")
    path.write_text(text, encoding="utf-8")
    print(f"[patch_retrieval] precision re-rank applied to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
