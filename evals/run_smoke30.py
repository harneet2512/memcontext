"""LongMemEval-S 30-question smoke test runner.

Pipeline:
  1. Extraction: Ollama + qwen3:8b (local, $0)
  2. Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, $0)
  3. Active pack: personal_assistant (8 predicates)
  4. Reader: openai/gpt-5-mini via OpenRouter (~$0.02 for 30 calls)
  5. Judge: openai/gpt-4o-2024-08-06 via OpenRouter (~$0.01 for 30 calls)

Total estimated cost: $0.03-0.05 (only reader + judge hit the API)

Usage:
    python evals/run_smoke30.py
    python evals/run_smoke30.py --seed 42 --yes
    python evals/run_smoke30.py --dataset path/to/dataset.json

Environment variables (required):
    MEMCONTEXT_READER_API_KEY   OpenRouter API key for reader + judge

Environment variables (optional, have sane defaults):
    MEMCONTEXT_READER_MODEL     default: openai/gpt-5-mini
    MEMCONTEXT_JUDGE_MODEL      default: openai/gpt-4o-2024-08-06
    MEMCONTEXT_EXTRACTOR_MODEL  default: qwen3:8b
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

_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "longmemeval-s" / "data" / "longmemeval_s_cleaned.json"


# ── Preflight checks ───────────────────────────────────────────────────────

def check_ollama() -> list[str]:
    """Check Ollama is running and return list of pulled model names."""
    import urllib.request
    import urllib.error

    url = os.environ.get("MEMCONTEXT_OLLAMA_URL", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m.get("name", "") for m in data.get("models", [])]
    except (urllib.error.URLError, TimeoutError, OSError):
        return []


def check_model_pulled(models: list[str], target: str = "qwen3:8b") -> bool:
    target_base = target.split(":")[0].lower()
    for m in models:
        if m.lower().startswith(target_base):
            return True
    return False


def check_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def check_dataset(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            return first_char == "["
    except (OSError, UnicodeDecodeError):
        return False


def check_api_key() -> bool:
    return bool(os.environ.get("MEMCONTEXT_READER_API_KEY", ""))


def run_preflight(dataset_path: Path, extractor_model: str) -> bool:
    """Run all preflight checks. Returns True if all pass."""
    print("=" * 60)
    print("PREFLIGHT CHECKS")
    print("=" * 60)

    all_ok = True
    backend = os.environ.get("MEMCONTEXT_EXTRACTOR_BACKEND", "ollama")

    if backend == "ollama":
        # 1. Ollama
        models = check_ollama()
        if models:
            print(f"  [OK] Ollama running ({len(models)} models available)")
        else:
            print("  [FAIL] Ollama not running on localhost:11434")
            print("         Start with: ollama serve")
            all_ok = False

        # 2. Model pulled
        if models and check_model_pulled(models, extractor_model):
            print(f"  [OK] Model '{extractor_model}' available")
        elif models:
            print(f"  [FAIL] Model '{extractor_model}' not pulled")
            print(f"         Run: ollama pull {extractor_model}")
            all_ok = False
        else:
            print("  [SKIP] Cannot check model (Ollama not running)")
    else:
        # OpenRouter extraction
        ext_key = os.environ.get("MEMCONTEXT_EXTRACTOR_API_KEY", "")
        if ext_key:
            print(f"  [OK] Extraction: OpenRouter {extractor_model}")
        else:
            print("  [FAIL] MEMCONTEXT_EXTRACTOR_API_KEY not set for openrouter backend")
            all_ok = False

    # 3. sentence-transformers
    if check_sentence_transformers():
        print("  [OK] sentence-transformers installed")
    else:
        print("  [FAIL] sentence-transformers not installed")
        print("         Run: python -m pip install sentence-transformers")
        all_ok = False

    # 4. Dataset
    if check_dataset(dataset_path):
        size_mb = dataset_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Dataset found ({size_mb:.0f} MB)")
    else:
        print(f"  [FAIL] Dataset not found at {dataset_path}")
        all_ok = False

    # 5. API key
    if check_api_key():
        key = os.environ["MEMCONTEXT_READER_API_KEY"]
        print(f"  [OK] MEMCONTEXT_READER_API_KEY set ({key[:8]}...)")
    else:
        print("  [FAIL] MEMCONTEXT_READER_API_KEY not set")
        print("         Export your OpenRouter API key")
        all_ok = False

    print()
    return all_ok


def print_config() -> None:
    """Print the full configuration and cost estimate."""
    backend = os.environ.get("MEMCONTEXT_EXTRACTOR_BACKEND", "ollama")
    ext_model = os.environ.get("MEMCONTEXT_EXTRACTOR_MODEL", "qwen3:8b")

    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    if backend == "ollama":
        print(f"  Extraction:  Ollama {ext_model} (local, $0)")
    else:
        print(f"  Extraction:  OpenRouter {ext_model}")
    print(f"  Embeddings:  {os.environ.get('MEMCONTEXT_EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')} (local, $0)")
    print(f"  Pack:        {os.environ.get('ACTIVE_PACK', 'personal_assistant')}")
    print(f"  Reader:      {os.environ.get('MEMCONTEXT_READER_MODEL', 'openai/gpt-5-mini')} (30 calls)")
    print(f"  Judge:       {os.environ.get('MEMCONTEXT_JUDGE_MODEL', 'openai/gpt-4o-2024-08-06')} (30 calls)")
    print()
    print("  ESTIMATED COST")
    print("  -------------------------------------")
    if backend == "ollama":
        print("  Extraction (local Ollama):      $0.00")
        print("  Embeddings (local model):       $0.00")
        print("  Reader (30x gpt-5-mini):       ~$0.02")
        print("  Judge  (30x gpt-4o):           ~$0.01")
        print("  -------------------------------------")
        print("  TOTAL:                         ~$0.03")
    else:
        print(f"  Extraction (~14.8k calls):     ~$0.27")
        print("  Embeddings (local model):       $0.00")
        print("  Reader (30x gpt-5-mini):       ~$0.02")
        print("  Judge  (30x gpt-4o):           ~$0.01")
        print("  -------------------------------------")
        print("  TOTAL:                         ~$0.30")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval-S 30-question smoke test")
    parser.add_argument("--dataset", default=str(_DEFAULT_DATASET),
                        help="Path to LongMemEval-S dataset")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--output-dir", default="results", help="Output directory (default: results/)")
    parser.add_argument("--extractor-model", default=None,
                        help="Extraction model (default: qwen3:8b for ollama, openai/gpt-4.1-nano for openrouter)")
    parser.add_argument("--extractor-backend", default=None,
                        choices=["ollama", "openrouter"],
                        help="Extraction backend (default: ollama, or openrouter if MEMCONTEXT_EXTRACTOR_BACKEND is set)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    # Set environment — respect CLI flags > env vars > defaults
    backend = args.extractor_backend or os.environ.get("MEMCONTEXT_EXTRACTOR_BACKEND", "ollama")
    os.environ["MEMCONTEXT_EXTRACTOR_BACKEND"] = backend

    if args.extractor_model:
        os.environ["MEMCONTEXT_EXTRACTOR_MODEL"] = args.extractor_model
    elif backend == "openrouter":
        os.environ.setdefault("MEMCONTEXT_EXTRACTOR_MODEL", "openai/gpt-4.1-nano")
        os.environ.setdefault("MEMCONTEXT_EXTRACTOR_API_KEY",
                              os.environ.get("MEMCONTEXT_READER_API_KEY", ""))
    else:
        os.environ.setdefault("MEMCONTEXT_EXTRACTOR_MODEL", "qwen3:8b")
    os.environ.setdefault("ACTIVE_PACK", "personal_assistant")
    os.environ.setdefault("MEMCONTEXT_READER_MODEL", "openai/gpt-5-mini")
    os.environ.setdefault("MEMCONTEXT_JUDGE_MODEL", "openai/gpt-4o-2024-08-06")

    from memcontext.predicate_packs import active_pack
    active_pack.cache_clear()

    # Preflight
    if not run_preflight(dataset_path, args.extractor_model):
        print("Preflight checks failed. Fix the issues above and retry.")
        sys.exit(1)

    print_config()

    if not args.yes:
        try:
            confirm = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = ""
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    # Reset judge tracking
    from evals.metrics import get_judge_fallback_stats, reset_judge_fallback_stats
    reset_judge_fallback_stats()

    # Run
    print()
    print("=" * 60)
    print(f"RUNNING 30-QUESTION SMOKE TEST (seed={args.seed})")
    print("=" * 60)
    print()

    from evals.quickcheck import run_quickcheck

    start = time.time()
    result = run_quickcheck(
        dataset_path=str(dataset_path),
        seed=args.seed,
        reader="configured",
    )
    elapsed = time.time() - start

    # Post-run checks
    judge_stats = get_judge_fallback_stats()
    result["judge_fallbacks"] = judge_stats["count"]
    result["elapsed_seconds"] = round(elapsed, 1)

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    raw = result.get("overall_accuracy_raw")
    task_avg = result.get("overall_accuracy_task_averaged")
    scored = result.get("questions_scored", 0)
    extraction_failures = 0
    for q in result.get("questions", []):
        if isinstance(q, dict):
            extraction_failures = q.get("extraction_failures", 0)
            break

    print(f"  Questions sampled:  {result.get('questions_sampled', 0)}")
    print(f"  Questions scored:   {scored}")
    print(f"  Time elapsed:       {elapsed:.0f}s")
    print()

    if raw is not None:
        print(f"  Raw accuracy:       {raw:.1%} ({scored} questions)")
    if task_avg is not None:
        print(f"  Task-averaged:      {task_avg:.1%}")
    print()

    cats = result.get("per_category", {})
    if isinstance(cats, dict) and cats:
        print("  Per-category breakdown:")
        for cat, v in sorted(cats.items()):
            if isinstance(v, dict):
                acc = v.get("accuracy", 0)
                correct = v.get("correct", 0)
                total_cat = v.get("total", 0)
                print(f"    {cat:30s}  {acc:5.0%}  ({correct}/{total_cat})")
    print()

    # Warnings
    if judge_stats["count"] > 0:
        print(f"  WARNING: {judge_stats['count']} judge call(s) fell back to fuzzy F1:")
        for qid, err in judge_stats["log"][:3]:
            print(f"    {qid}: {err}")
        print()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"smoke30_{timestamp}.json"
    output_file.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  Full results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
