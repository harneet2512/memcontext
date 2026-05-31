"""Retrieval latency benchmark for MemContext.

Measures the component breakdown that matters for a fair comparison against a
hosted memory API (e.g. Supermemory's "sub-300ms recall"):

  * query-embed cost  — fixed per query (local model inference)
  * scan cost         — O(n) over the corpus: decode every embedding + cosine
  * total retrieval   — what a caller actually waits for (NO network here)

It sweeps the corpus size N so you can see the O(n) curve and find the point
where the linear scan crosses a flat ANN-backed service.

IMPORTANT — this measures *in-process* latency (no network). A hosted API's
number includes a network round-trip; do not compare the two directly. To
compare apples-to-apples, either (a) also run MemContext behind its HTTP server
(`memcontext serve-http`) and measure client->server->client wall-clock, or
(b) compare only the *scaling shape* (this curve vs the API's flat latency).

Usage:
    python -m evals.latency_bench --sizes 100,1000,10000 --iters 50
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("ACTIVE_PACK", "general")

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.predicate_packs import active_pack
from memcontext.retrieval import (
    EmbeddingClient,
    backfill_embeddings,
    retrieve_hybrid,
    retrieve_relevant_claims,
)
from memcontext.schema import Speaker, Turn, open_database

active_pack.cache_clear()

SESSION = "bench"
QUERIES = [
    "What database do they use?",
    "What is the deployment schedule?",
    "Which CI system is configured?",
    "What did the user decide about caching?",
    "What is the current project status?",
]


def _seed(conn, n: int, client: EmbeddingClient) -> None:
    """Insert n active claims with varied text and embed them all."""
    for i in range(n):
        topic = i % 50
        turn = Turn(
            turn_id=new_turn_id(),
            session_id=SESSION,
            speaker=Speaker.USER,
            text=f"Fact {i} about topic {topic}: the value for item {i} is option_{i % 7}.",
            ts=now_ns(),
            asr_confidence=None,
        )
        insert_turn(conn, turn)
        insert_claim(
            conn,
            session_id=SESSION,
            subject=f"topic_{topic}",
            predicate="user_fact",
            value=f"item {i} resolves to option_{i % 7} in context {i % 13}",
            confidence=0.9,
            source_turn_id=turn.turn_id,
        )
    backfill_embeddings(conn, SESSION, client=client)


def _percentiles(samples_ms: list[float]) -> tuple[float, float, float]:
    s = sorted(samples_ms)
    p = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return p(0.50), p(0.95), p(0.99)


def _time(fn, iters: int) -> list[float]:
    out = []
    for j in range(iters):
        q = QUERIES[j % len(QUERIES)]
        t0 = time.perf_counter()
        fn(q)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _quiet_logs() -> None:
    import logging
    import sys

    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def run(sizes: list[int], iters: int) -> None:
    _quiet_logs()
    client = EmbeddingClient()
    # Warm the model so the first call's load cost isn't billed to a query.
    client.embed(["warmup"])

    print(f"{'N':>8} {'embed_p50':>10} {'scan_p50':>9} "
          f"{'sem_p50':>8} {'sem_p95':>8} {'hyb_p50':>8} {'hyb_p95':>8}   (ms)")
    print("-" * 72)

    for n in sizes:
        conn = open_database(":memory:")
        t_setup = time.perf_counter()
        _seed(conn, n, client)
        setup_s = time.perf_counter() - t_setup

        # warm the retrieval path on this corpus
        retrieve_relevant_claims(conn, session_id=SESSION, question=QUERIES[0], k=10, client=client)

        embed = _time(lambda q: client.embed([q]), iters)
        sem = _time(
            lambda q: retrieve_relevant_claims(conn, session_id=SESSION, question=q, k=10, client=client),
            iters,
        )
        hyb = _time(
            lambda q: retrieve_hybrid(conn, session_id=SESSION, query=q, top_k=10),
            iters,
        )

        e50, _, _ = _percentiles(embed)
        s50, s95, _ = _percentiles(sem)
        h50, h95, _ = _percentiles(hyb)
        scan50 = max(0.0, s50 - e50)  # scan ≈ semantic total − query embed
        print(f"{n:>8} {e50:>10.1f} {scan50:>9.1f} {s50:>8.1f} {s95:>8.1f} "
              f"{h50:>8.1f} {h95:>8.1f}   (corpus embed setup {setup_s:.1f}s)")
        conn.close()

    print("\nNotes:")
    print("  embed_p50 = fixed local query-embedding cost (model inference, no network)")
    print("  scan_p50  = O(n) cosine scan over the corpus (semantic_total - embed)")
    print("  sem_*     = retrieve_relevant_claims (pure semantic, the O(n) path)")
    print("  hyb_*     = retrieve_hybrid (semantic+BM25+entity+temporal RRF; what memory_query uses)")
    print("  All numbers are IN-PROCESS (no network). A hosted API adds a round-trip.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="100,1000,10000",
                    help="comma-separated corpus sizes (claims)")
    ap.add_argument("--iters", type=int, default=50, help="timed iterations per size")
    args = ap.parse_args()
    run([int(x) for x in args.sizes.split(",")], args.iters)
