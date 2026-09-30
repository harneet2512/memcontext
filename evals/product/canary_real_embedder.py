"""Real-embedder canary: is semantic memory actually ON when the product says it is?

A type-checker "fix" once made the local embedder raise on every call for 19 days while
CI (NullEmbedder) and ``memcontext status`` both said ON: Pass-2 and episode embedding
swallow their exceptions by design, so nothing failed loudly. This canary exercises the
REAL configured model (BAAI/bge-m3 by default) end to end and exits non-zero on any
failure. Diagnostic only; it never mutates product state (``:memory:`` DBs, temp dir).

    python -m evals.product.canary_real_embedder            # all checks
    python -m evals.product.canary_real_embedder --skip-cli # skip the status --verify subprocess
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK = "general,developer"
SEMANTIC_THRESHOLD = 0.88
FAILURE_EVENTS = frozenset({
    "substrate.semantic_supersession_failed", "substrate.embed_episode_failed",
    "substrate.episode_embed_failed", "substrate.semantic_embed_length_mismatch",
})
_events: list[str] = []

# Retrieval probe: the query shares NO BM25 token with the target episode, so a lexical-only
# engine cannot find it; only the dense channel can.
TARGET = "Penicillin triggers severe hives."
QUERY = "what antibiotic am I allergic to"
DISTRACTORS = (
    "The staging deploy runs every Friday at 3pm.",
    "My sister lives in Toronto with her two kids.",
    "We use Postgres for the orders database.",
    "I prefer aisle seats on long flights.",
    "The quarterly planning offsite is in Denver this year.",
)


def _setup_env(embed_on: bool) -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ["ACTIVE_PACK"] = PACK
    os.environ.setdefault("SUBSTRATE_PACKS_DIR", str(REPO_ROOT / "predicate_packs"))
    os.environ["MEMCONTEXT_EMBED_EPISODES"] = "1" if embed_on else "0"
    from memcontext.predicate_packs import active_pack

    active_pack.cache_clear()


def _quiet_logs() -> None:
    import structlog

    def _capture(_l: Any, _n: str, ev: dict) -> dict:
        _events.append(str(ev.get("event", "")))
        return ev

    structlog.configure(
        processors=[_capture, structlog.processors.KeyValueRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(file=open(os.devnull, "w")),  # noqa: SIM115
        cache_logger_on_first_use=False,
    )


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _store(conn: Any, text: str, *, session: str = "s1", claims: list[dict] | None = None) -> dict:
    from memcontext.mcp_tools import handle_memory_store

    return handle_memory_store(conn, text=text, speaker="user", session_id=session,
                               namespace="canary", claims=claims)


# --------------------------------------------------------------------- checks ---
# Each check returns a one-line detail on success and raises AssertionError on failure.


def check_semantic_enabled() -> str:
    from memcontext.retrieval import episode_embedder, semantic_enabled

    assert semantic_enabled(), "semantic_enabled() is False with MEMCONTEXT_EMBED_EPISODES=1"
    emb = episode_embedder()
    assert emb is not None and emb.backend_available(), "no importable embedding backend"
    return f"backend importable ({type(emb).__name__})"


def check_probe() -> str:
    from memcontext.retrieval import BGE_M3_EMBED_DIM, BGE_M3_VERSION_TAG, probe_embedder

    probe = probe_embedder()
    assert probe is not None, "probe_embedder() returned None (semantic OFF)"
    assert probe.ok, f"probe failed: {probe.error}"
    assert probe.dim == BGE_M3_EMBED_DIM, f"dim {probe.dim} != expected {BGE_M3_EMBED_DIM}"
    return f"{BGE_M3_VERSION_TAG} {probe.dim}-d, first embed {probe.seconds:.1f}s (includes model load)"


def check_embed_one() -> str:
    from memcontext.retrieval import BGE_M3_EMBED_DIM, episode_embedder

    vecs = episode_embedder().embed(["We deploy the API on Fridays."])  # type: ignore[union-attr]
    assert len(vecs) == 1, f"expected 1 vector, got {len(vecs)}"
    v = vecs[0]
    assert len(v) == BGE_M3_EMBED_DIM, f"dim {len(v)}"
    assert all(math.isfinite(x) for x in v), "non-finite component"
    assert abs(_norm(v) - 1.0) < 1e-2, f"vector not unit-normalised (norm={_norm(v):.4f})"
    return f"1 text -> 1x{len(v)} unit vector"


def check_embed_batch() -> str:
    from memcontext.retrieval import BGE_M3_EMBED_DIM, episode_embedder
    from memcontext.supersession_semantic import cosine

    texts = [TARGET, *DISTRACTORS]
    vecs = episode_embedder().embed(texts)  # type: ignore[union-attr]
    assert len(vecs) == len(texts), f"{len(vecs)} vectors for {len(texts)} texts"
    assert all(len(v) == BGE_M3_EMBED_DIM for v in vecs), "dim mismatch in batch"
    worst = max(cosine(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs)))
    assert worst < 0.99, f"distinct texts embed ~identically (max cos {worst:.3f}): constant/degenerate embedder"
    again = episode_embedder().embed([texts[0]])[0]  # type: ignore[union-attr]
    assert cosine(again, vecs[0]) > 0.999, "same text embeds differently across calls (non-deterministic)"
    return f"{len(texts)} texts -> {len(vecs)}x{BGE_M3_EMBED_DIM}; max pairwise cos {worst:.3f}; deterministic"


def check_semantic_geometry() -> str:
    from memcontext.retrieval import episode_embedder
    from memcontext.supersession_semantic import cosine

    a, para, unrel = episode_embedder().embed([  # type: ignore[union-attr]
        "The staging deploy runs every Friday afternoon.",
        "Staging deployments happen on Friday afternoons.",
        "My sister adopted a golden retriever.",
    ])
    cp, cu = cosine(a, para), cosine(a, unrel)
    assert cp > cu + 0.15, f"paraphrase cos {cp:.3f} not clearly above unrelated cos {cu:.3f}"
    return f"paraphrase cos {cp:.3f} vs unrelated {cu:.3f}"


def check_turn_embeddings_written() -> str:
    from memcontext.retrieval import BGE_M3_EMBED_DIM, BGE_M3_VERSION_TAG, _decode_vector
    from memcontext.schema import open_database

    conn = open_database(":memory:")
    n0 = len(_events)
    for t in (TARGET, *DISTRACTORS[:2]):
        _store(conn, t)
    rows = conn.execute("SELECT embedding, embedding_model_version FROM turn_embeddings").fetchall()
    fails = [e for e in _events[n0:] if e in FAILURE_EVENTS]
    assert not fails, f"swallowed embedding failures: {fails}"
    assert len(rows) == 3, f"{len(rows)} turn_embeddings rows for 3 stored turns"
    assert all(r[1] == BGE_M3_VERSION_TAG for r in rows), "wrong embedding_model_version"
    assert all(len(_decode_vector(r[0])) == BGE_M3_EMBED_DIM for r in rows), "stored vector dim mismatch"
    return f"3 stores -> 3 turn_embeddings rows ({BGE_M3_VERSION_TAG})"


def check_pass2_wired() -> str:
    from memcontext.retrieval import semantic_supersession
    from memcontext.supersession_semantic import NullEmbedder

    sem = semantic_supersession()
    assert sem is not None, "semantic_supersession() is None -> Pass-2 not wired into ingest"
    inner = getattr(sem, "_embedder", None)
    assert not isinstance(inner, NullEmbedder), "Pass-2 wired to NullEmbedder (cos always 1.0)"
    return f"Pass-2 wired with {type(inner).__name__}"


def _nl_restatement_edges(conn: Any) -> list[Any]:
    fact = "The staging deploy runs every Friday at 3pm"
    for _ in range(2):
        out = _store(conn, fact + ".", claims=[{"subject": "user", "predicate": "deploy_schedule", "value": fact}])
        assert out["claims_created"] == 1, f"store failed: {out}"
    return conn.execute("SELECT edge_type, identity_score FROM supersession_edges").fetchall()


def check_pass2_fires_beyond_pass1() -> str:
    """An NL-only fact (no triple) restated verbatim: Pass-1 cannot match it by construction
    (no subject/predicate), so any edge must come from Pass-2 -- and lexical mode must differ."""
    from memcontext.schema import open_database

    n0 = len(_events)
    edges = _nl_restatement_edges(open_database(":memory:"))
    fails = [e for e in _events[n0:] if e in FAILURE_EVENTS]
    assert not fails, f"Pass-2 raised and was swallowed: {fails}"
    sem = [e for e in edges if e[0] == "semantic_replace"]
    assert sem, f"no SEMANTIC_REPLACE edge on a verbatim NL restatement (edges={[tuple(e) for e in edges]})"
    assert sem[0][1] is not None and sem[0][1] >= SEMANTIC_THRESHOLD, f"identity_score {sem[0][1]}"
    _setup_env(embed_on=False)
    try:
        lexical = _nl_restatement_edges(open_database(":memory:"))
    finally:
        _setup_env(embed_on=True)
    assert not lexical, f"lexical-only mode also produced edges {lexical}; the check does not isolate Pass-2"
    return f"semantic: SEMANTIC_REPLACE (cos {sem[0][1]:.3f}); lexical-only: no edge"


def _paraphrase_db() -> Any:
    from memcontext.retrieval import _bm25_over_docs, _tokenize_for_bm25
    from memcontext.schema import open_database

    # Target in a MIDDLE position: RRF ties are broken by list position (retrieval._rrf_ranks),
    # so first- or last-stored would bias the lexical baseline (oldest wins ties, newest wins recency).
    docs = [*DISTRACTORS[:3], TARGET, *DISTRACTORS[3:]]
    overlap = set(_tokenize_for_bm25(QUERY)) & set(_tokenize_for_bm25(TARGET))
    assert not overlap, f"probe is invalid: query shares tokens {overlap} with the target"
    bm25 = _bm25_over_docs(_tokenize_for_bm25(QUERY), [_tokenize_for_bm25(d) for d in docs])
    assert bm25[docs.index(TARGET)] == 0.0, "BM25 unexpectedly scores the target"
    conn = open_database(":memory:")
    for d in docs:
        _store(conn, d)
    return conn


def check_dense_channel_finds_paraphrase() -> str:
    """ON/OFF signal: with the product's STORED episode vectors and its query prefix, the
    paraphrase-only target (BM25 score 0) is the nearest episode."""
    from memcontext.retrieval import _decode_vector, apply_query_prefix, episode_embedder
    from memcontext.supersession_semantic import cosine

    conn = _paraphrase_db()
    q = episode_embedder().embed([apply_query_prefix(QUERY)])[0]  # type: ignore[union-attr]
    rows = conn.execute(
        "SELECT t.text, e.embedding FROM turns t JOIN turn_embeddings e USING (turn_id)").fetchall()
    ranked = sorted(((cosine(q, _decode_vector(r[1])), r[0]) for r in rows), reverse=True)
    assert len(ranked) == 1 + len(DISTRACTORS), f"only {len(ranked)} stored episode vectors"
    assert ranked[0][1] == TARGET, f"nearest stored vector is {ranked[0][1]!r} ({ranked[0][0]:.3f})"
    return f"BM25(target)=0.0; dense nearest = target (cos {ranked[0][0]:.3f} vs next {ranked[1][0]:.3f})"


def check_fused_retrieval_lifts_paraphrase() -> str:
    """Public episode retrieval with embeddings must rank the paraphrase-only target
    higher than the same DB with its vectors removed (lexical-only), and within top-3."""
    from memcontext.retrieval import retrieve_episodes

    conn = _paraphrase_db()

    def rank() -> int:
        hits = retrieve_episodes(conn, session_id="s1", query=QUERY, top_k=10)
        texts = [t.text for t, _ in hits]
        return texts.index(TARGET) + 1 if TARGET in texts else 99

    with_emb, top1 = rank(), retrieve_episodes(conn, session_id="s1", query=QUERY, top_k=1)[0][0].text
    conn.execute("DELETE FROM turn_embeddings")
    lexical = rank()
    assert with_emb < lexical, f"embeddings did not lift the target (rank {with_emb} vs lexical {lexical})"
    assert with_emb <= 3, f"target only at rank {with_emb} with embeddings"
    note = "" if with_emb == 1 else f"; NOTE fused top-1 is {top1!r} (stopword-only lexical match outvotes dense)"
    return f"target rank {with_emb} with embeddings vs {lexical} lexical-only{note}"


def check_status_verify(timeout_s: int) -> str:
    exe = Path(sys.executable).with_name("memcontext.exe")
    cmd = [str(exe)] if exe.exists() else [sys.executable, "-m", "memcontext.cli"]
    tmp = Path(tempfile.mkdtemp(prefix="mc_canary_"))
    try:
        env = {**os.environ, "MEMCONTEXT_EMBED_EPISODES": "1"}
        proc = subprocess.run([*cmd, "status", "--verify", "--db", str(tmp / "canary.db")],
                              capture_output=True, text=True, timeout=timeout_s, env=env, cwd=str(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("Semantic memory:")), "")
    assert proc.returncode == 0, f"exit {proc.returncode}: {line or proc.stderr[-300:]}"
    assert "ON (verified" in line, f"status line is {line!r}"
    return line


# ----------------------------------------------------------------------- main ---


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-cli", action="store_true", help="skip `memcontext status --verify` (loads the model a 2nd time)")
    ap.add_argument("--cli-timeout", type=int, default=900)
    args = ap.parse_args(argv)
    _setup_env(embed_on=True)
    _quiet_logs()
    checks: list[tuple[str, Callable[[], str]]] = [
        ("semantic_enabled", check_semantic_enabled),
        ("probe_embedder", check_probe),
        ("embed_one_text", check_embed_one),
        ("embed_batch", check_embed_batch),
        ("semantic_geometry", check_semantic_geometry),
        ("turn_embeddings_written", check_turn_embeddings_written),
        ("pass2_wired", check_pass2_wired),
        ("pass2_fires_where_pass1_cannot", check_pass2_fires_beyond_pass1),
        ("dense_channel_finds_paraphrase", check_dense_channel_finds_paraphrase),
        ("fused_retrieval_lifts_paraphrase", check_fused_retrieval_lifts_paraphrase),
    ]
    if not args.skip_cli:
        checks.append(("status_verify_cli", lambda: check_status_verify(args.cli_timeout)))
    t_all = time.perf_counter()
    failed = 0
    for name, fn in checks:
        t0 = time.perf_counter()
        try:
            detail = fn()
            verdict = "PASS"
        except Exception as exc:  # noqa: BLE001 - every failure is a FAIL line, never a crash
            detail, verdict = f"{type(exc).__name__}: {exc}", "FAIL"
            failed += 1
        print(f"{verdict} {name:32s} {time.perf_counter() - t0:6.1f}s  {detail}", flush=True)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed in {time.perf_counter() - t_all:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
