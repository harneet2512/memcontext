"""Score the persisted DB through the PRODUCT retrieval path, with an audit trace.

This is the "product == benchmark, reconstructable score" run:

- Retrieval goes through `retrieve_memory_across` -- the EXACT unified two-tier
  retrieval the live MCP door (`handle_memory_query`) serves (the door calls it
  for the all-sessions case). It's scoped to each question's haystack sessions
  (the door can't subset sessions, so we call its engine directly with the right
  scope -- same algorithm, correct isolation, no cross-question leakage).
- Every question writes a full record to a JSONL trace:
  {question_id, category, gold, retrieved items, full reader prompt, raw reader
   output, predicted answer, judge verdict, correct}. The headline score is
  literally sum(correct)/N over that file -- a constant print() can't fake 30
  coherent pred-vs-gold rows, and anyone can re-judge any row.

Usage: python evals/mem_score_door.py <db_path> [context_cap]
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


def _rel(sd: str, qd: str) -> str:
    try:
        d = (datetime.strptime(qd[:10], "%Y/%m/%d") - datetime.strptime(sd[:10], "%Y/%m/%d")).days
        if d <= 0:
            return "same day"
        if d < 7:
            return f"{d} days ago"
        if d < 35:
            return f"~{d // 7} week(s) ago"
        if d < 365:
            return f"~{d // 30} month(s) ago"
        return f"~{d // 365} year(s) ago"
    except (ValueError, TypeError):
        return ""


def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\lme30.db"
    cap = int(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MEMCONTEXT_CONTEXT_CAP", "300"))
    dataset_path = "data/longmemeval-s/data/longmemeval_s_cleaned.json"
    n_per_cat = int(os.environ.get("MEMCONTEXT_N_PER_CAT", "5"))
    trace_path = db_path.replace(".db", "_door_trace.jsonl")

    assert os.path.exists(db_path), f"DB not found: {db_path}"
    assert os.environ.get("MEMCONTEXT_READER_API_KEY"), "reader/judge key required"
    os.environ.setdefault("ACTIVE_PACK", "personal_assistant")
    os.environ.setdefault("SUBSTRATE_PACKS_DIR", os.path.join(os.getcwd(), "predicate_packs"))
    os.environ.setdefault("MEMCONTEXT_JUDGE_MODEL", "openai/gpt-4o-2024-08-06")
    print(f"[assert] product-path=retrieve_memory_across | cap={cap} | "
          f"judge={os.environ['MEMCONTEXT_JUDGE_MODEL']} | trace={trace_path}", flush=True)

    from evals.longmemeval import load_dataset
    from evals.metrics import score_answer
    from evals.runner import ReaderMode, answer_question
    from memcontext.claims import get_turn
    from memcontext.predicate_packs import active_pack
    from memcontext.retrieval import EmbeddingClient, retrieve_memory_across
    from memcontext.schema import open_database

    active_pack.cache_clear()
    sessions, questions = load_dataset(dataset_path)
    smap = {s.session_id: s for s in sessions}
    sdate = {s.session_id: s.date for s in sessions}
    bycat: "OrderedDict[str, list]" = OrderedDict()
    for q in questions:
        bycat.setdefault(q.category, []).append(q)
    sel = []
    for _c, qs in bycat.items():
        sel += qs[:n_per_cat]

    conn = open_database(db_path)
    conn.row_factory = sqlite3.Row
    ec = EmbeddingClient()

    def _clip(s: str) -> str:
        return (s or "")[:cap]

    results = []
    t0 = time.time()
    with open(trace_path, "w", encoding="utf-8") as tf:
        for qi, q in enumerate(sel, 1):
            q_sids = [s for s in q.session_ids if smap.get(s)]
            # PRODUCT retrieval path -- the engine handle_memory_query serves.
            hits = retrieve_memory_across(
                conn, session_ids=q_sids, query=q.question, top_k=50, embedding_client=ec
            )
            excerpts = []
            retrieved = []
            for hit, s in hits:
                t = get_turn(conn, hit.source_turn_id)
                sd = sdate.get(t.session_id, "") if t is not None else ""
                spk = (t.speaker.value if t is not None and hasattr(t.speaker, "value") else "user")
                excerpts.append({"text": _clip(hit.text), "speaker": spk, "session_date": sd,
                                 "relative_offset": _rel(sd, q.question_date), "kind": hit.kind,
                                 "score": round(s, 4)})
                retrieved.append({"kind": hit.kind, "id": hit.id, "text": hit.text[:160],
                                  "score": round(s, 4)})

            ar = answer_question(question=q.question, category=q.category, claims=[],
                                 reader=ReaderMode.CONFIGURED, question_date=q.question_date,
                                 excerpts=excerpts)
            pred = ar.get("predicted_answer")
            verdict = None
            correct = None
            if pred is not None:
                sc = score_answer(predicted=pred, gold=str(q.gold_answer), question=q.question,
                                  question_type=q.category, question_id=q.question_id)
                verdict = sc if not isinstance(sc, dict) else sc.get("correct")
                correct = bool(verdict)

            rec = {
                "question_id": q.question_id, "category": q.category, "question": q.question,
                "gold_answer": q.gold_answer, "predicted_answer": pred,
                "judge_verdict": verdict, "correct": correct,
                "n_retrieved": len(retrieved),
                "n_facts": sum(1 for r in retrieved if r["kind"] == "fact"),
                "n_episodes": sum(1 for r in retrieved if r["kind"] == "episode"),
                "retrieved": retrieved, "full_prompt": ar.get("full_prompt"),
                "raw_reader_output": ar.get("raw_reader_output"),
            }
            tf.write(json.dumps(rec, default=str) + "\n")
            tf.flush()
            results.append(rec)
            print(f"  [{qi}/{len(sel)}] {'OK' if correct else 'XX'} {q.category} "
                  f"(facts={rec['n_facts']} eps={rec['n_episodes']})", flush=True)

    per_cat: dict[str, list[int]] = {}
    for r in results:
        c = per_cat.setdefault(r["category"], [0, 0])
        c[1] += 1
        if r["correct"]:
            c[0] += 1
    overall = sum(1 for r in results if r["correct"])
    print(f"=== SCORE via product door engine ({time.time()-t0:.0f}s) ===", flush=True)
    for cat, (corr, tot) in per_cat.items():
        print(f"  {cat:28} {corr}/{tot}  ({corr/tot*100:.0f}%)", flush=True)
    print(f"  OVERALL {overall}/{len(results)} ({overall/len(results)*100:.1f}%)", flush=True)
    print(f"  reconstruct: jq -s 'map(.correct)|add' {trace_path}  (== {overall})", flush=True)
    print(f"SCORE_DONE -> trace at {trace_path}", flush=True)


if __name__ == "__main__":
    main()
