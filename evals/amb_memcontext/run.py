"""Runtime-register MemContext with AMB and run the benchmark harness."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .provider import MemContextFullProvider
from .router_llm import (
    OPENROUTER_BASE_URL,
    OPENROUTER_READER_MODEL,
    TOKENROUTER_BASE_URL,
    TOKENROUTER_EXTRACTOR_MODEL,
    TOKENROUTER_JUDGE_MODEL,
    OpenRouterReaderLLM,
    TokenRouterJudgeLLM,
)


def _add_amb_to_path(amb_root: Path | None) -> Path | None:
    if amb_root is None:
        return None
    root = amb_root.resolve()
    src = root / "src"
    if not src.exists():
        raise SystemExit(f"AMB root does not look valid; missing src/: {root}")
    sys.path.insert(0, str(src))
    return root


def _git_output(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _verify_amb_read_only(root: Path | None) -> None:
    if root is None:
        return
    try:
        remotes = _git_output(root, "remote", "-v")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Could not inspect AMB git remote at {root}: {exc}") from exc
    lower = remotes.lower()
    if "harneet2512" not in lower and "vectorize-io/agent-memory-benchmark" in lower:
        status = _git_output(root, "status", "--short")
        if status:
            raise SystemExit(
                "Refusing to run against a dirty upstream AMB checkout. "
                f"Remote is vectorize-io and status is:\n{status}"
            )
    elif "harneet2512" not in lower:
        status = _git_output(root, "status", "--short")
        if status:
            raise SystemExit(
                "AMB checkout remote is not harneet2512 and is dirty. "
                f"Treating it as read-only:\n{status}"
            )


def _configure_tokenrouter_models() -> None:
    os.environ.setdefault("MEMCONTEXT_EXTRACTOR_BACKEND", "openrouter")
    os.environ.setdefault(
        "MEMCONTEXT_EXTRACTOR_ENDPOINT", f"{TOKENROUTER_BASE_URL}/chat/completions"
    )
    os.environ.setdefault("MEMCONTEXT_EXTRACTOR_MODEL", TOKENROUTER_EXTRACTOR_MODEL)
    os.environ.setdefault("MEMCONTEXT_EXTRACTOR_REASONING_EFFORT", "none")
    os.environ.setdefault("MEMCONTEXT_EXTRACTOR_REASONING_EXCLUDE", "1")
    extractor_key = (
        os.environ.get("TOKENROUTER_AMB_EXTRACTOR_KEY")
        or os.environ.get("TOKENROUTER_API_KEY")
    )
    if extractor_key and not os.environ.get("MEMCONTEXT_EXTRACTOR_API_KEY"):
        os.environ["MEMCONTEXT_EXTRACTOR_API_KEY"] = extractor_key

    os.environ.setdefault("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)
    os.environ.setdefault("OMB_ANSWER_LLM", "openrouter-reader")
    os.environ.setdefault("OMB_ANSWER_MODEL", OPENROUTER_READER_MODEL)
    os.environ.setdefault("OMB_ANSWER_REASONING_EFFORT", "high")
    os.environ.setdefault("OMB_ANSWER_REASONING_EXCLUDE", "1")
    os.environ.setdefault("TOKENROUTER_BASE_URL", TOKENROUTER_BASE_URL)
    os.environ.setdefault("OMB_JUDGE_LLM", "tokenrouter-judge")
    os.environ.setdefault("OMB_JUDGE_MODEL", TOKENROUTER_JUDGE_MODEL)
    os.environ.setdefault("OMB_JUDGE_REASONING_EFFORT", "low")
    os.environ.setdefault("OMB_JUDGE_REASONING_EXCLUDE", "1")


def _register_provider() -> None:
    try:
        from memory_bench.memory import REGISTRY as MEMORY_REGISTRY
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import AMB's memory registry. Install the AMB checkout and "
            "its dependencies first, for example: `pip install -e /path/to/agent-memory-benchmark`."
        ) from exc

    MEMORY_REGISTRY[MemContextFullProvider.name] = MemContextFullProvider


def _register_router_llms() -> None:
    from memory_bench.llm import REGISTRY as LLM_REGISTRY

    LLM_REGISTRY["openrouter-reader"] = OpenRouterReaderLLM
    LLM_REGISTRY["tokenrouter-judge"] = TokenRouterJudgeLLM


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AMB with MemContext registered at runtime.")
    parser.add_argument("--amb-root", type=Path, default=None, help="Local AMB checkout root.")
    parser.add_argument("--dataset", default="longmemeval")
    parser.add_argument("--split", required=True)
    parser.add_argument("--memory", default=MemContextFullProvider.name)
    parser.add_argument("--mode", default="rag")
    parser.add_argument("--llm", default="gemini", help="Accepted for parity with AMB CLI.")
    parser.add_argument("--category", default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--query-id", default=None)
    parser.add_argument("--doc-limit", type=int, default=None)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--skip-ingested", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("evals") / "amb_outputs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--description", default=None)
    return parser


def main(argv: list[str] | None = None) -> Any:
    args = _build_parser().parse_args(argv)
    amb_root = _add_amb_to_path(args.amb_root)
    _verify_amb_read_only(amb_root)
    _configure_tokenrouter_models()
    _register_router_llms()
    _register_provider()

    from memory_bench.dataset import get_dataset
    from memory_bench.llm import get_answer_llm
    from memory_bench.memory import get_memory_provider
    from memory_bench.modes import get_mode
    from memory_bench.runner import EvalRunner

    dataset = get_dataset(args.dataset)
    if args.split not in dataset.splits:
        raise SystemExit(
            f"Unknown split '{args.split}' for {args.dataset}. "
            f"Available: {', '.join(dataset.splits)}"
        )

    description = args.description or (
        "MemContext full substrate registered at runtime from harneet2512/memcontext; "
        "upstream AMB checkout left unmodified."
    )
    summary = EvalRunner(output_dir=args.output_dir).run(
        dataset=dataset,
        split=args.split,
        memory=get_memory_provider(args.memory),
        mode=get_mode(args.mode, llm=get_answer_llm()),
        category=args.category,
        query_limit=args.query_limit,
        query_id=args.query_id,
        doc_limit=args.doc_limit,
        oracle=args.oracle,
        skip_ingestion=args.skip_ingestion,
        skip_ingested=args.skip_ingested,
        skip_retrieval=args.skip_retrieval,
        skip_answer=args.skip_answer,
        only_failed=args.only_failed,
        show_raw=args.show_raw,
        run_name=args.name,
        description=description,
    )
    print(
        f"AMB complete: total={summary.total_queries} correct={summary.correct} "
        f"accuracy={summary.accuracy:.3f}"
    )
    _verify_amb_read_only(amb_root)
    return summary


if __name__ == "__main__":
    main()
