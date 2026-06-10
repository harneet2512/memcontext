"""Build a PERSISTED LongMemEval-30 memory DB — the paid step, run ONCE.

Real memory-layer build (not run_preflight's claims-only diagnostic):
  - extract facts (DeepSeek no-think, pooled session) in parallel
  - ingest EVERY turn as a first-class Episode (incl. claim-less turns — the
    Tier-1 floor that run_preflight silently dropped at `if not claims_data`)
  - embed episodes synchronously (Tier-1 semantic floor) + backfill claim embeds
  - persist to a FILE DB so scoring/retrieval config is tunable for $0 afterward

Asserts the config BEFORE spending, and the DB AFTER, so a mis-config fails
loud and free instead of after a paid extraction.

Usage: python evals/mem_build_db.py <db_path> [n_per_cat]
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\lme30.db"
    n_per_cat = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    dataset_path = "data/longmemeval-s/data/longmemeval_s_cleaned.json"

    # --- FREE config asserts (before any paid call) ---
    assert db_path != ":memory:", "DB must be a file (persisted), not :memory:"
    assert os.environ.get("MEMCONTEXT_EXTRACTOR_MODEL") == "deepseek-v4-flash", \
        "extractor model must be deepseek-v4-flash"
    assert os.environ.get("MEMCONTEXT_EXTRACTOR_NO_THINK") == "1", \
        "NO_THINK must be 1 (no-think is the standard regime)"
    assert os.environ.get("MEMCONTEXT_EMBED_EPISODES", "1") != "0", \
        "MEMCONTEXT_EMBED_EPISODES must be on (Tier-1 semantic floor)"
    assert os.environ.get("MEMCONTEXT_EXTRACTOR_API_KEY"), "extractor API key required"
    print(f"[assert] config OK | db={db_path} | no-think=1 | embed_episodes=on", flush=True)

    os.environ.setdefault("ACTIVE_PACK", "personal_assistant")
    os.environ.setdefault(
        "SUBSTRATE_PACKS_DIR", os.path.join(os.getcwd(), "predicate_packs")
    )

    from evals.longmemeval import load_dataset
    from memcontext.claims import new_turn_id, now_ns
    from memcontext.extractors import LLMExtractor, auto_extractor
    from memcontext.on_new_turn import on_new_turn
    from memcontext.extractors import PassthroughExtractor
    from memcontext.predicate_packs import active_pack
    from memcontext.retrieval import (
        EmbeddingClient,
        backfill_embeddings,
        episode_embedder,
    )
    from memcontext.schema import Speaker, Turn, open_database

    active_pack.cache_clear()

    sessions, questions = load_dataset(dataset_path)
    session_map = {s.session_id: s for s in sessions}

    # Deterministic 5-per-category selection (matches prior runs)
    bycat: "OrderedDict[str, list]" = OrderedDict()
    for q in questions:
        bycat.setdefault(q.category, []).append(q)
    selected = []
    for _cat, qs in bycat.items():
        selected.extend(qs[:n_per_cat])
    print(f"selected {len(selected)} questions ({n_per_cat}/cat from "
          f"{len(bycat)} cats)", flush=True)

    needed: set[str] = set()
    for q in selected:
        needed.update(q.session_ids)

    extractor = auto_extractor()
    assert isinstance(extractor, LLMExtractor), "expected LLMExtractor (DeepSeek)"

    # Build the work list — EVERY non-empty turn (no claim-gating here).
    work: list[tuple[str, str, str, str]] = []
    for sid in needed:
        sess = session_map.get(sid)
        if sess is None:
            continue
        for td in sess.turns:
            role = td.get("role", "user")
            content = td.get("content", "")
            if content and content.strip():
                work.append((sid, role, content, sess.date))
    print(f"turns to extract+ingest: {len(work)}", flush=True)

    def _extract_one(item):
        raw_sid, role, text, date = item
        sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
        t = Turn(turn_id=new_turn_id(), session_id=raw_sid, speaker=sp,
                 text=text, ts=now_ns(), asr_confidence=None)
        claims = extractor(t)
        return (raw_sid, role, text, date, [
            {"subject": c.subject, "predicate": c.predicate,
             "value": c.value, "confidence": c.confidence}
            for c in claims
        ])

    t0 = time.time()
    extracted: list = []
    workers = int(os.environ.get("MEMCONTEXT_EVAL_WORKERS", "64"))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_extract_one, w): w for w in work}
        for fut in as_completed(futs):
            done += 1
            try:
                extracted.append(fut.result())
            except Exception:
                pass
            if done % 500 == 0 or done == len(work):
                print(f"  extracted {done}/{len(work)}", flush=True)
    print(f"extraction done in {time.time()-t0:.0f}s", flush=True)

    # --- Persist DB, ingest EVERY turn as an episode (embedder on) ---
    conn = open_database(db_path)
    conn.row_factory = sqlite3.Row
    emb = episode_embedder()
    assert emb is not None, "episode_embedder() is None — embeddings would be skipped"

    by_session: dict[str, list] = {}
    for raw_sid, role, text, date, claims_data in extracted:
        by_session.setdefault(raw_sid, []).append((role, text, date, claims_data))

    ingested = 0
    for raw_sid in sorted(by_session.keys()):
        for role, text, date, claims_data in by_session[raw_sid]:
            sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
            pt = PassthroughExtractor(claims_data)  # [] is fine -> episode-only
            on_new_turn(conn, session_id=raw_sid, speaker=sp,
                        text=text, extractor=pt, embedder=emb)
            ingested += 1
            if ingested % 1000 == 0:
                print(f"  ingested+embedded {ingested}/{len(work)} episodes",
                      flush=True)

    ec = EmbeddingClient()
    total_claim_emb = 0
    for sid in by_session:
        total_claim_emb += backfill_embeddings(conn, sid, client=ec)

    # --- AFTER-build asserts: the DB is actually populated + embedded ---
    cur = conn.cursor()
    n_turns = cur.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    n_temb = cur.execute("SELECT COUNT(*) FROM turn_embeddings").fetchone()[0]
    n_claims = cur.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    n_cemb = cur.execute("SELECT COUNT(*) FROM claim_embeddings").fetchone()[0]
    conn.commit()

    print("=== BUILD SUMMARY ===", flush=True)
    print(f"db_path           : {db_path}", flush=True)
    print(f"episodes (turns)  : {n_turns}", flush=True)
    print(f"episode embeds    : {n_temb}", flush=True)
    print(f"facts (claims)    : {n_claims}", flush=True)
    print(f"claim embeds      : {n_cemb}", flush=True)
    assert n_turns > 0 and n_temb > 0, "Tier-1 floor empty — episodes not embedded"
    assert n_claims > 0 and n_cemb > 0, "Tier-2 facts not embedded"
    print(f"[assert] DB populated + embedded OK | total {time.time()-t0:.0f}s",
          flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
