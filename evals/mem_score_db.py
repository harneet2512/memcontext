"""Score a persisted LongMemEval-30 DB — the FREE step, run as many times as you like.

Loads the file DB built by mem_build_db.py and, per question, serves the reader
through one of two retrieval paths so you can A/B on identical extracted data:

  MEMCONTEXT_SERVE_PATH=hybrid  -> retrieve_hybrid: legacy claim-source-turn
        selection — the reader reads the SOURCE TURNS of the top-50 fact matches
        (category prompts render excerpts, so structured claims don't reach the
        reader — this mirrors run_preflight's behavior).
  MEMCONTEXT_SERVE_PATH=memory  -> retrieve_memory: the unified two-tier ranking
        — the reader reads EVERY hit's text, i.e. fact NL-text AND episode
        turn-text, ranked by second-level RRF. This is the real product serving
        path; here the two-tier facts genuinely reach the reader.

The A/B therefore measures "legacy claim-source-turn selection" vs "unified
fact+episode serving" on identical extracted data — NOT "facts vs no-facts".

Applies a per-item context cap (the baseline's missing context_char_limit) so
the reader gets a clean payload, not 44k chars. No extraction here — only
retrieve -> read -> judge (~$0.05 of reader+judge).

Usage: python evals/mem_score_db.py <db_path> [serve_path] [context_cap]
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
from collections import OrderedDict
from datetime import datetime

import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def _relative_offset(session_date: str, question_date: str) -> str:
    try:
        sd = datetime.strptime(session_date[:10], "%Y/%m/%d")
        qd = datetime.strptime(question_date[:10], "%Y/%m/%d")
        delta = (qd - sd).days
        if delta <= 0:
            return "same day"
        if delta < 7:
            return f"{delta} days ago"
        if delta < 35:
            return f"~{delta // 7} week(s) ago"
        if delta < 365:
            return f"~{delta // 30} month(s) ago"
        return f"~{delta // 365} year(s) ago"
    except (ValueError, TypeError):
        return ""


def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\lme30.db"
    serve_path = (sys.argv[2] if len(sys.argv) > 2
                  else os.environ.get("MEMCONTEXT_SERVE_PATH", "memory"))
    cap = int(sys.argv[3] if len(sys.argv) > 3
              else os.environ.get("MEMCONTEXT_CONTEXT_CAP", "300"))
    dataset_path = "data/longmemeval-s/data/longmemeval_s_cleaned.json"
    n_per_cat = int(os.environ.get("MEMCONTEXT_N_PER_CAT", "5"))

    assert serve_path in ("hybrid", "memory"), "serve_path must be hybrid|memory"
    assert os.path.exists(db_path), f"DB not found: {db_path} (run mem_build_db.py)"
    assert os.environ.get("MEMCONTEXT_READER_API_KEY"), \
        "reader/judge API key required (MEMCONTEXT_READER_API_KEY) — else every " \
        "item silently skips and you get a bogus 0/30"
    os.environ.setdefault("ACTIVE_PACK", "personal_assistant")
    os.environ.setdefault(
        "SUBSTRATE_PACKS_DIR", os.path.join(os.getcwd(), "predicate_packs")
    )
    # Pin the official LongMemEval judge (gpt-4o-2024-08-06, ~97% human agreement)
    os.environ.setdefault("MEMCONTEXT_JUDGE_MODEL", "openai/gpt-4o-2024-08-06")
    print(f"[assert] serve_path={serve_path} | context_cap={cap} | "
          f"judge={os.environ['MEMCONTEXT_JUDGE_MODEL']} | db={db_path}", flush=True)

    from evals.longmemeval import load_dataset
    from evals.metrics import score_answer
    from evals.runner import ReaderMode, answer_question
    from memcontext.claims import get_claim, get_turn
    from memcontext.predicate_packs import active_pack
    from memcontext.retrieval import (
        EmbeddingClient,
        retrieve_hybrid,
        retrieve_memory,
    )
    from memcontext.schema import open_database

    active_pack.cache_clear()
    sessions, questions = load_dataset(dataset_path)
    session_map = {s.session_id: s for s in sessions}
    sess_date = {s.session_id: s.date for s in sessions}

    bycat: "OrderedDict[str, list]" = OrderedDict()
    for q in questions:
        bycat.setdefault(q.category, []).append(q)
    selected = []
    for _cat, qs in bycat.items():
        selected.extend(qs[:n_per_cat])

    conn = open_database(db_path)
    conn.row_factory = sqlite3.Row
    ec = EmbeddingClient()

    def _clip(s: str) -> str:
        return (s or "")[:cap]

    results = []
    t0 = time.time()
    for qi, q in enumerate(selected, 1):
        q_sids = [sid for sid in q.session_ids if session_map.get(sid)]
        claims: list[dict] = []
        excerpts: list[dict] = []

        if serve_path == "hybrid":
            agg: list = []
            for sid in q_sids:
                agg.extend(retrieve_hybrid(conn, session_id=sid, query=q.question,
                                           top_k=50, embedding_client=ec))
            agg.sort(key=lambda x: (-x[1], x[0].claim_id))
            seen: set[str] = set()
            for c, s in agg[:50]:
                claims.append({"claim_id": c.claim_id, "subject": c.subject,
                               "predicate": c.predicate, "value": _clip(c.value or c.text or ""),
                               "confidence": c.confidence, "status": c.status.value,
                               "score": round(s, 4)})
                if c.source_turn_id in seen:
                    continue
                seen.add(c.source_turn_id)
                t = get_turn(conn, c.source_turn_id)
                if t is None:
                    continue
                sd = sess_date.get(t.session_id, "")
                excerpts.append({"text": _clip(t.text),
                                 "speaker": t.speaker.value if hasattr(t.speaker, "value") else str(t.speaker),
                                 "session_date": sd,
                                 "relative_offset": _relative_offset(sd, q.question_date),
                                 "score": round(s, 4)})
        else:  # memory — serve the unified retrieve_memory ranking to the reader
            # Render EVERY hit's text as a context item (fact NL-text AND episode
            # turn-text), so the two-tier facts genuinely reach the reader — not
            # dropped by _format_context_text the way a `claims` list would be.
            agg = []
            for sid in q_sids:
                agg.extend(retrieve_memory(conn, session_id=sid, query=q.question,
                                           top_k=50, embedding_client=ec))
            agg.sort(key=lambda x: (-x[1], x[0].kind != "fact", x[0].id))
            for hit, s in agg[:50]:
                # source turn carries the session date (facts link back via source_turn_id)
                t = get_turn(conn, hit.source_turn_id)
                sd = sess_date.get(t.session_id, "") if t is not None else ""
                spk = (t.speaker.value if t is not None and hasattr(t.speaker, "value")
                       else "user")
                excerpts.append({"text": _clip(hit.text), "speaker": spk,
                                 "session_date": sd,
                                 "relative_offset": _relative_offset(sd, q.question_date),
                                 "kind": hit.kind, "score": round(s, 4)})

        ar = answer_question(question=q.question, category=q.category, claims=claims,
                             reader=ReaderMode.CONFIGURED, question_date=q.question_date,
                             excerpts=excerpts)
        pred = ar.get("predicted_answer")
        correct = None
        if pred is not None:
            sc = score_answer(predicted=pred, gold=str(q.gold_answer),
                              question=q.question, question_type=q.category,
                              question_id=q.question_id)
            correct = bool(sc.get("correct")) if isinstance(sc, dict) else bool(sc)
        results.append({"category": q.category, "gold": q.gold_answer,
                        "pred": pred, "correct": correct,
                        "n_facts": len(claims), "n_eps": len(excerpts)})
        print(f"  [{qi}/{len(selected)}] {'OK' if correct else 'XX'} {q.category}",
              flush=True)

    # Tally per category
    per_cat: dict[str, list[int]] = {}
    for r in results:
        c = per_cat.setdefault(r["category"], [0, 0])
        c[1] += 1
        if r["correct"]:
            c[0] += 1
    overall = sum(1 for r in results if r["correct"])
    print(f"=== SCORE ({serve_path}, cap={cap}, {time.time()-t0:.0f}s) ===", flush=True)
    for cat, (corr, tot) in per_cat.items():
        print(f"  {cat:28} {corr}/{tot}  ({corr/tot*100:.0f}%)", flush=True)
    print(f"  OVERALL {overall}/{len(results)} ({overall/len(results)*100:.1f}%)",
          flush=True)
    out = db_path.replace(".db", f"_score_{serve_path}.json")
    json.dump({"serve_path": serve_path, "cap": cap, "results": results,
               "overall": f"{overall}/{len(results)}"}, open(out, "w"),
              default=str, indent=2)
    print(f"SCORE_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
