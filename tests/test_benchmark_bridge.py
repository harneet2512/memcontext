from __future__ import annotations

from pathlib import Path

from memcontext.benchmark_bridge import (
    build_harness_env,
    export_branch_tree,
    list_branch_files,
    make_run_manifest,
    redacted_env,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_list_branch_files_contains_raw_amb_runner():
    files = list_branch_files(_repo_root(), "benchmark/amb", "evals")
    assert "evals/amb_runner.py" in files


def test_export_branch_tree_writes_exact_raw_amb_runner(tmp_path: Path):
    repo_root = _repo_root()
    export_branch_tree(repo_root, "benchmark/amb", "evals", tmp_path)
    exported = (tmp_path / "evals" / "amb_runner.py").read_bytes()
    expected = __import__("subprocess").run(
        ["git", "show", "benchmark/amb:evals/amb_runner.py"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
    ).stdout
    assert exported == expected


def test_build_harness_env_orders_raw_harness_before_repo(tmp_path: Path):
    repo_root = _repo_root()
    env = build_harness_env(repo_root, tmp_path)
    py_parts = env["PYTHONPATH"].split(__import__("os").pathsep)
    assert py_parts[0] == str(tmp_path)
    assert py_parts[1] == str(repo_root)
    assert env["SUBSTRATE_PACKS_DIR"] == str(repo_root / "predicate_packs")


def test_redacted_env_never_emits_secret_values():
    env = {
        "MEMCONTEXT_READER_API_KEY": "secret",
        "MEMCONTEXT_READER_ENDPOINT": "https://api.example.test/v1/chat/completions",
    }
    redacted = redacted_env(env)
    assert redacted["MEMCONTEXT_READER_API_KEY_present"] is True
    assert "secret" not in repr(redacted)
    assert redacted["MEMCONTEXT_READER_ENDPOINT"] == env["MEMCONTEXT_READER_ENDPOINT"]


def test_run_manifest_records_raw_harness_and_product_identity(tmp_path: Path):
    repo_root = _repo_root()
    exported = export_branch_tree(repo_root, "benchmark/amb", "evals", tmp_path)
    env = build_harness_env(repo_root, tmp_path)
    manifest = make_run_manifest(
        repo_root=repo_root,
        branch="benchmark/amb",
        export_prefix="evals",
        harness_path="evals/amb_runner.py",
        harness_args=["--help"],
        harness_root=tmp_path,
        exported_files=exported,
        env=env,
        command=["python", str(tmp_path / "evals" / "amb_runner.py"), "--help"],
        exit_code=0,
    )
    assert manifest["raw_harness"]["branch"] == "benchmark/amb"
    assert manifest["raw_harness"]["commit"]
    assert manifest["raw_harness"]["export_prefix_tree"]
    assert "evals/amb_runner.py" in manifest["raw_harness"]["exported_file_sha256"]
    assert manifest["product_under_test"]["head_commit"]
    assert manifest["product_under_test"]["product_diff_sha256"]
    assert manifest["execution"]["exit_code"] == 0
