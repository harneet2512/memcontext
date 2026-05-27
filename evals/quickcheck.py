"""Stratified subset runner for LongMemEval-S.

Samples 5 questions per category (30 total) using a daily-rotating seed.
Runs the same eval pipeline as longmemeval.py -- same judge, same reader,
same scoring, same extraction. No sampled IDs are persisted.

Usage:
    python evals/quickcheck.py --dataset data/longmemeval_s.json
    python evals/quickcheck.py --dataset data/longmemeval_s.json --reader gpt-4o-mini
    python evals/quickcheck.py --dataset data/longmemeval_s.json --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import date

from evals.longmemeval import load_dataset, run_preflight

SAMPLE_CATEGORIES = [
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]

QUESTIONS_PER_CATEGORY = 3

_INTERNAL_TO_DATASET: dict[str, str | None] = {
    "single_session_user_fact": "single-session-user",
    "single_session_preference": "single-session-preference",
    "single_session_assistant": "single-session-assistant",
    "cross_session_user_fact": "multi-session",
    "cross_session_preference": "multi-session",
    "temporal_ordering": "temporal-reasoning",
    "knowledge_update": "knowledge-update",
    "abstention": None,
}


def _dataset_category(internal_cat: str, question_id: str) -> str | None:
    """Map internal category back to dataset-level category for sampling."""
    if question_id.endswith("_abs"):
        return None
    return _INTERNAL_TO_DATASET.get(internal_cat)


def sample_questions(dataset_path: str, seed: int) -> list[str]:
    """Return sampled question_ids (5 per category, 30 total)."""
    _, questions = load_dataset(dataset_path)

    by_cat: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        ds_cat = _dataset_category(q.category, q.question_id)
        if ds_cat in SAMPLE_CATEGORIES:
            by_cat[ds_cat].append(q.question_id)

    rng = random.Random(seed)
    sampled: list[str] = []
    for cat in SAMPLE_CATEGORIES:
        pool = by_cat.get(cat, [])
        n = min(QUESTIONS_PER_CATEGORY, len(pool))
        if n > 0:
            sampled.extend(rng.sample(pool, n))

    return sampled


def run_quickcheck(
    *,
    dataset_path: str,
    seed: int | None = None,
    reader: str = "configured",
    no_extract: bool = False,
) -> dict[str, object]:
    """Run stratified quickcheck and return results."""
    if seed is None:
        seed = int(date.today().strftime("%Y%m%d"))

    sampled_ids = sample_questions(dataset_path, seed)
    sampled_set = set(sampled_ids)

    result = run_preflight(
        dataset_path=dataset_path,
        limit=len(sampled_ids),
        reader=reader,
        question_ids=sampled_set,
        no_extract=no_extract,
    )

    question_results: list[dict[str, object]] = result.get("questions", [])

    scored = [r for r in question_results if "score" in r]
    per_cat: dict[str, dict[str, object]] = {}
    for r in scored:
        qid = str(r.get("question_id", ""))
        cat_internal = str(r.get("category", ""))
        ds_cat = _dataset_category(cat_internal, qid) or cat_internal
        if ds_cat not in per_cat:
            per_cat[ds_cat] = {"correct": 0, "total": 0, "ids": []}
        entry = per_cat[ds_cat]
        entry["total"] = int(entry["total"]) + 1  # type: ignore[arg-type]
        ids_list: list[str] = entry["ids"]  # type: ignore[assignment]
        ids_list.append(qid)
        if r.get("correct"):
            entry["correct"] = int(entry["correct"]) + 1  # type: ignore[arg-type]

    overall_correct = sum(1 for r in scored if r.get("correct"))
    overall_total = len(scored)

    cat_accuracies = [
        int(v["correct"]) / int(v["total"])
        for v in per_cat.values()
        if int(v["total"]) > 0
    ]
    task_averaged = round(sum(cat_accuracies) / len(cat_accuracies), 4) if cat_accuracies else None

    return {
        "seed": seed,
        "questions_sampled": len(sampled_ids),
        "questions_scored": overall_total,
        "overall_accuracy_raw": round(overall_correct / overall_total, 4) if overall_total else None,
        "overall_accuracy_task_averaged": task_averaged,
        "per_category": {
            cat: {
                "accuracy": round(
                    int(v["correct"]) / int(v["total"]), 4  # type: ignore[arg-type]
                ) if int(v["total"]) else 0,  # type: ignore[arg-type]
                "correct": v["correct"],
                "total": v["total"],
                "question_ids": v["ids"],
            }
            for cat, v in sorted(per_cat.items())
        },
        "extraction_stats": result.get("extraction_stats"),
        "questions": question_results,
        "reader": reader,
        "dataset_path": dataset_path,
    }


def main() -> None:
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser(description="LongMemEval-S stratified quickcheck")
    parser.add_argument("--dataset", required=True, help="Path to LongMemEval-S dataset")
    parser.add_argument("--reader", default="configured", help="Reader model (default: configured)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: today's date as YYYYMMDD)")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip LLM extraction, store raw turns as claims")
    args = parser.parse_args()

    result = run_quickcheck(
        dataset_path=args.dataset,
        seed=args.seed,
        reader=args.reader,
        no_extract=args.no_extract,
    )

    # Auto-save full results to file
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"quickcheck_{result['seed']}_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    # Print summary to stdout
    raw = result.get("overall_accuracy_raw")
    task_avg = result.get("overall_accuracy_task_averaged")
    if raw is not None:
        print(f"\n=== Quickcheck (seed={result['seed']}) ===")
        total = result.get("questions_scored", 0)
        print(f"Raw accuracy: {raw:.1%} ({total} questions)")
        if task_avg is not None:
            print(f"Task-averaged: {task_avg:.1%}")
        cats = result.get("per_category", {})
        if isinstance(cats, dict):
            for cat, v in sorted(cats.items()):
                if isinstance(v, dict):
                    acc = v.get("accuracy", 0)
                    correct = v.get("correct", 0)
                    total_cat = v.get("total", 0)
                    print(f"  {cat}: {acc:.0%} ({correct}/{total_cat})")

    # Print extraction stats
    ext_stats = result.get("extraction_stats")
    if ext_stats:
        print(f"\n--- Extraction Coverage ---")
        print(f"  Total turns: {ext_stats['total_turns']}")
        print(f"  With claims: {ext_stats['turns_with_claims']}")
        print(f"  Empty (fallback): {ext_stats['turns_empty_fallback']}")
        print(f"  Failed: {ext_stats['turns_failed']}")

    # Print all answers with full detail
    questions = result.get("questions", [])
    if questions:
        print(f"\n--- Per-Question Results ({len(questions)}) ---")
        for q in questions:
            if not isinstance(q, dict):
                continue
            correct = q.get("correct")
            status = "CORRECT" if correct else ("WRONG" if correct is False else "UNSCORED")
            tier = q.get("scoring_tier", "?")
            cat = q.get("category", "?")
            qid = q.get("question_id", "?")
            print(f"\n  [{status}] {qid}")
            print(f"    Category: {cat} | Tier: {tier}")
            print(f"    Question: {str(q.get('question', ''))[:150]}")

            gold = q.get("gold_answer", "?")
            if gold and len(str(gold)) > 150:
                gold = str(gold)[:150] + "..."
            print(f"    Gold: {gold}")

            predicted = q.get("predicted_answer", "(none)")
            if predicted and len(str(predicted)) > 200:
                predicted = str(predicted)[:200] + "..."
            print(f"    Predicted: {predicted}")

            judge_verdict = q.get("judge_verdict")
            judge_raw = q.get("judge_raw_response")
            if judge_verdict:
                print(f"    Judge: {judge_verdict} (raw: {judge_raw})")

            claims = q.get("num_claims_retrieved", "?")
            excerpts = q.get("num_excerpts", "?")
            superseded = q.get("claims_superseded", "?")
            total_claims = q.get("total_claims_in_db", "?")
            print(f"    Retrieval: {claims} claims retrieved, {excerpts} excerpts, "
                  f"{superseded} superseded, {total_claims} total in DB")

    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
