"""Run a chunk of LongMemEval questions for parallel GHA execution.

Splits 500 questions into N chunks by index, runs one chunk.
Each chunk is fully independent — extracts only the sessions needed
for its questions.

Usage:
    python evals/run_chunk.py --chunk 0 --total-chunks 5
    python evals/run_chunk.py --chunk 2 --total-chunks 5 --workers 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run a chunk of LongMemEval")
    parser.add_argument("--dataset", default="data/longmemeval-s/data/longmemeval_s_cleaned.json")
    parser.add_argument("--chunk", type=int, required=True, help="Chunk index (0-based)")
    parser.add_argument("--total-chunks", type=int, default=5, help="Total number of chunks")
    parser.add_argument("--workers", type=int, default=30, help="Parallel extraction workers")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    os.environ.setdefault("ACTIVE_PACK", "personal_assistant")
    os.environ.setdefault("MEMCONTEXT_READER_MODEL", "openai/gpt-5-mini")
    os.environ.setdefault("MEMCONTEXT_JUDGE_MODEL", "openai/gpt-4o-2024-08-06")

    from memcontext.predicate_packs import active_pack
    active_pack.cache_clear()

    from evals.longmemeval import load_dataset, run_preflight

    # Load all questions, split into chunks
    _, questions = load_dataset(args.dataset)
    total_q = len(questions)
    chunk_size = total_q // args.total_chunks
    start_idx = args.chunk * chunk_size
    end_idx = start_idx + chunk_size if args.chunk < args.total_chunks - 1 else total_q

    chunk_ids = {q.question_id for q in questions[start_idx:end_idx]}

    print(f"Chunk {args.chunk}/{args.total_chunks}: questions {start_idx}-{end_idx} ({len(chunk_ids)} questions)", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Extractor: {os.environ.get('MEMCONTEXT_EXTRACTOR_MODEL', '?')}", flush=True)
    print(f"Reader: {os.environ.get('MEMCONTEXT_READER_MODEL', '?')}", flush=True)
    print(f"Judge: {os.environ.get('MEMCONTEXT_JUDGE_MODEL', '?')}", flush=True)
    print(flush=True)

    # Patch worker count
    import evals.longmemeval as lme_module
    original_source = None
    # Workers are set in the code — we'll use env var override
    os.environ["MEMCONTEXT_EXTRACTION_WORKERS"] = str(args.workers)

    start = time.time()
    result = run_preflight(
        dataset_path=args.dataset,
        limit=len(chunk_ids),
        reader="configured",
        question_ids=chunk_ids,
    )
    elapsed = time.time() - start

    result["chunk"] = args.chunk
    result["total_chunks"] = args.total_chunks
    result["elapsed_seconds"] = round(elapsed, 1)

    # Save
    output_path = args.output or f"results/chunk_{args.chunk}.json"
    os.makedirs(os.path.dirname(output_path) or "results", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    scored = [q for q in result.get("questions", []) if "score" in q]
    correct = sum(1 for q in scored if q.get("correct"))
    total = len(scored)

    print(f"", flush=True)
    print(f"=== CHUNK {args.chunk} RESULTS ===", flush=True)
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%" if total else "No scores", flush=True)
    print(f"Time: {elapsed:.0f}s", flush=True)
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
