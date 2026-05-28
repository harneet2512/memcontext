"""Merge chunk results from parallel GHA benchmark jobs."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge LongMemEval chunk results")
    parser.add_argument("--input-dir", required=True, help="Directory with chunk JSON files")
    parser.add_argument("--output", required=True, help="Output merged JSON file")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    all_questions = []
    total_elapsed = 0
    extraction_stats = {"total_turns": 0, "turns_with_claims": 0, "turns_empty_fallback": 0, "turns_failed": 0}

    chunk_files = sorted(input_dir.rglob("chunk_*.json"))
    print(f"Found {len(chunk_files)} chunk files", flush=True)

    for fp in chunk_files:
        with open(fp, "r", encoding="utf-8") as f:
            chunk = json.load(f)
        questions = chunk.get("questions", [])
        all_questions.extend(questions)
        total_elapsed += chunk.get("elapsed_seconds", 0)
        es = chunk.get("extraction_stats", {})
        for k in extraction_stats:
            extraction_stats[k] += es.get(k, 0)
        print(f"  {fp.name}: {len(questions)} questions", flush=True)

    # Score
    scored = [q for q in all_questions if "score" in q]
    correct = sum(1 for q in scored if q.get("correct"))
    total = len(scored)

    # Per-category
    per_cat: dict[str, dict] = {}
    for q in scored:
        cat = q.get("category", "unknown")
        if cat not in per_cat:
            per_cat[cat] = {"correct": 0, "total": 0}
        per_cat[cat]["total"] += 1
        if q.get("correct"):
            per_cat[cat]["correct"] += 1

    result = {
        "dataset_path": "data/longmemeval-s/data/longmemeval_s_cleaned.json",
        "reader_mode": "configured",
        "total_questions": len(all_questions),
        "scored_questions": total,
        "overall_accuracy_raw": round(correct / total, 4) if total else None,
        "overall_accuracy_task_averaged": round(
            sum(v["correct"] / v["total"] for v in per_cat.values() if v["total"]) / len(per_cat), 4
        ) if per_cat else None,
        "per_category_accuracy": {
            cat: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0,
            }
            for cat, v in sorted(per_cat.items())
        },
        "extraction_stats": extraction_stats,
        "elapsed_seconds": round(total_elapsed, 1),
        "num_chunks": len(chunk_files),
        "scoring_method": "judge",
        "extractor_model": os.environ.get("MEMCONTEXT_EXTRACTOR_MODEL", "unknown"),
        "reader_model": os.environ.get("MEMCONTEXT_READER_MODEL", "unknown"),
        "judge_model": os.environ.get("MEMCONTEXT_JUDGE_MODEL", "unknown"),
        "questions": all_questions,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"", flush=True)
    print(f"=== MERGED RESULTS ===", flush=True)
    print(f"Total questions: {total}", flush=True)
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%" if total else "No scores", flush=True)
    print(f"", flush=True)
    for cat, v in sorted(per_cat.items()):
        print(f"  {cat}: {v['correct']}/{v['total']} ({v['correct']/v['total']*100:.0f}%)", flush=True)
    print(f"", flush=True)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
