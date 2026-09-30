"""On-screen CLI polish regressions: log routing, ASCII help, trace noise."""
from __future__ import annotations

import re

import pytest
import structlog
from click.testing import CliRunner

from memcontext.cli import main

# A structlog ConsoleRenderer line: "<ts> [debug    ] event ..." / "[info     ]".
_STRUCTLOG_LINE = re.compile(r"\[(debug|info)\s*\]")


@pytest.fixture(autouse=True)
def _reset_structlog():
    """The CLI group configures structlog process-wide; undo it per test."""
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def _runner() -> CliRunner:
    # click >= 8.2 always captures stderr separately (mix_stderr was removed).
    try:
        return CliRunner(mix_stderr=False)  # type: ignore[call-arg]
    except TypeError:
        return CliRunner()


def _run_session(runner: CliRunner, db: str) -> list:
    results = [
        runner.invoke(main, ["init", "--db", db]),
        runner.invoke(main, ["ingest", "I prefer dark mode", "--db", db]),
        runner.invoke(main, ["ingest", "Actually I prefer light mode", "--db", db]),
        runner.invoke(main, ["trace", "--db", db, "--subject", "user",
                             "--predicate", "user_preference"]),
        runner.invoke(main, ["query", "what mode do I prefer", "--db", db]),
    ]
    for r in results:
        assert r.exit_code == 0, (r.output, r.exception)
    return results


def test_cli_stdout_has_no_structlog_debug_or_info_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMCONTEXT_LOG_LEVEL", raising=False)
    results = _run_session(_runner(), str(tmp_path / "t.db"))
    for r in results:
        assert not _STRUCTLOG_LINE.search(r.stdout), r.stdout
        assert not _STRUCTLOG_LINE.search(r.stderr), r.stderr
    assert "TRACE" in results[3].stdout


def test_cli_log_level_env_routes_debug_logs_to_stderr(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMCONTEXT_LOG_LEVEL", "DEBUG")
    results = _run_session(_runner(), str(tmp_path / "t.db"))
    for r in results:
        assert not _STRUCTLOG_LINE.search(r.stdout), r.stdout
    stderr = "".join(r.stderr for r in results)
    assert "substrate.db_opened" in stderr
    assert _STRUCTLOG_LINE.search(stderr)


def test_cli_invalid_log_level_falls_back_to_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMCONTEXT_LOG_LEVEL", "LOUD")
    r = _runner().invoke(main, ["init", "--db", str(tmp_path / "t.db")])
    assert r.exit_code == 0
    assert not _STRUCTLOG_LINE.search(r.stdout + r.stderr)

