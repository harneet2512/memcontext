from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git(repo_root: Path, *args: str, text: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=text,
    )


def list_branch_files(repo_root: Path, branch: str, prefix: str) -> list[str]:
    proc = _git(
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        branch,
        "--",
        prefix,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def export_branch_tree(repo_root: Path, branch: str, prefix: str, dest_root: Path) -> list[Path]:
    written: list[Path] = []
    for rel_path in list_branch_files(repo_root, branch, prefix):
        blob = _git(repo_root, "show", f"{branch}:{rel_path}", text=False).stdout
        out_path = dest_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(blob)
        written.append(out_path)
    return written


def git_rev(repo_root: Path, ref: str) -> str:
    return _git(repo_root, "rev-parse", ref, text=True).stdout.strip()


def git_tree(repo_root: Path, ref_path: str) -> str:
    return _git(repo_root, "rev-parse", ref_path, text=True).stdout.strip()


def git_status(repo_root: Path) -> str:
    return _git(repo_root, "status", "--porcelain=v2", text=True).stdout


def git_diff_hash(repo_root: Path, paths: list[str]) -> str:
    proc = _git(repo_root, "diff", "--binary", "HEAD", "--", *paths, text=False)
    return hashlib.sha256(proc.stdout).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exported_file_hashes(harness_root: Path, files: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(harness_root).as_posix(): file_sha256(path)
        for path in sorted(files)
    }


def build_harness_env(repo_root: Path, harness_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    py_parts = [str(harness_root), str(repo_root)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        py_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(py_parts)
    env.setdefault("SUBSTRATE_PACKS_DIR", str(repo_root / "predicate_packs"))
    return env


def redacted_env(env: dict[str, str]) -> dict[str, Any]:
    keys = [
        "ACTIVE_PACK",
        "SUBSTRATE_PACKS_DIR",
        "MEMCONTEXT_EXTRACTOR_BACKEND",
        "MEMCONTEXT_EXTRACTOR_ENDPOINT",
        "MEMCONTEXT_EXTRACTOR_MODEL",
        "MEMCONTEXT_EXTRACTOR_NO_THINK",
        "MEMCONTEXT_READER_ENDPOINT",
        "MEMCONTEXT_READER_MODEL",
        "MEMCONTEXT_JUDGE_MODEL",
        "MEMCONTEXT_EMBED_EPISODES",
        "MEMCONTEXT_EMBED_MODEL",
        "MEMCONTEXT_RETRIEVAL_WEIGHTS",
    ]
    out: dict[str, Any] = {k: env[k] for k in keys if k in env}
    for secret_key in (
        "MEMCONTEXT_EXTRACTOR_API_KEY",
        "MEMCONTEXT_READER_API_KEY",
        "TOKENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        if secret_key in env:
            out[f"{secret_key}_present"] = bool(env[secret_key])
    return out


def make_run_manifest(
    *,
    repo_root: Path,
    branch: str,
    export_prefix: str,
    harness_path: str,
    harness_args: list[str],
    harness_root: Path,
    exported_files: list[Path],
    env: dict[str, str],
    command: list[str],
    exit_code: int | None,
) -> dict[str, Any]:
    product_paths = ["memcontext", "predicate_packs", "pyproject.toml"]
    status = git_status(repo_root)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_harness": {
            "branch": branch,
            "commit": git_rev(repo_root, branch),
            "export_prefix": export_prefix,
            "export_prefix_tree": git_tree(repo_root, f"{branch}:{export_prefix}"),
            "harness_path": harness_path,
            "exported_file_count": len(exported_files),
            "exported_file_sha256": exported_file_hashes(harness_root, exported_files),
        },
        "product_under_test": {
            "repo_root": str(repo_root),
            "branch": _git(repo_root, "branch", "--show-current", text=True).stdout.strip(),
            "head_commit": git_rev(repo_root, "HEAD"),
            "status_porcelain_v2": status,
            "working_tree_dirty": bool(status.strip()),
            "product_diff_sha256": git_diff_hash(repo_root, product_paths),
            "product_paths_hashed": product_paths,
        },
        "execution": {
            "python": sys.executable,
            "command": command,
            "harness_args": harness_args,
            "cwd": str(repo_root),
            "exit_code": exit_code,
            "env": redacted_env(env),
        },
    }


def default_manifest_path(repo_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "results" / "raw_amb_runs" / f"{stamp}.manifest.json"


def run_branch_harness(
    *,
    repo_root: Path,
    branch: str,
    harness_path: str,
    export_prefix: str,
    harness_args: list[str],
    keep_temp: bool = False,
    manifest_path: Path | None = None,
) -> int:
    repo_root = repo_root.resolve()
    if keep_temp:
        harness_root = Path(tempfile.mkdtemp(prefix="memcontext-bench-"))
        cleanup = False
    else:
        tempdir = tempfile.TemporaryDirectory(prefix="memcontext-bench-")
        harness_root = Path(tempdir.name)
        cleanup = True

    try:
        exported = export_branch_tree(repo_root, branch, export_prefix, harness_root)
        env = build_harness_env(repo_root, harness_root)
        cmd = [sys.executable, str(harness_root / harness_path), *harness_args]
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env)
        manifest = make_run_manifest(
            repo_root=repo_root,
            branch=branch,
            export_prefix=export_prefix,
            harness_path=harness_path,
            harness_args=harness_args,
            harness_root=harness_root,
            exported_files=exported,
            env=env,
            command=cmd,
            exit_code=proc.returncode,
        )
        manifest_out = manifest_path or default_manifest_path(repo_root)
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[memcontext] reproducibility manifest: {manifest_out}")
        if keep_temp:
            print(f"[memcontext] exported raw harness to {harness_root}")
        return proc.returncode
    finally:
        if cleanup:
            tempdir.cleanup()
