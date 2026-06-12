from __future__ import annotations

import argparse
import sys
from pathlib import Path

from memcontext.benchmark_bridge import run_branch_harness


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the raw benchmark/amb harness unchanged against the latest "
            "MemContext product code in this checkout."
        )
    )
    parser.add_argument(
        "--branch",
        default="benchmark/amb",
        help="Git branch containing the released raw harness.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the exported harness temp directory for inspection.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path for the reproducibility manifest JSON.",
    )
    parser.add_argument(
        "harness_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the raw evals/amb_runner.py harness.",
    )
    args = parser.parse_args()

    harness_args = args.harness_args
    if harness_args and harness_args[0] == "--":
        harness_args = harness_args[1:]

    repo_root = Path(__file__).resolve().parents[2]
    return run_branch_harness(
        repo_root=repo_root,
        branch=args.branch,
        harness_path="evals/amb_runner.py",
        export_prefix="evals",
        harness_args=harness_args,
        keep_temp=args.keep_temp,
        manifest_path=Path(args.manifest) if args.manifest else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
