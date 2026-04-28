from __future__ import annotations

import os

from click.testing import CliRunner

from memcontext.cli import main


def test_cli_init(tmp_path):
    runner = CliRunner()
    db_path = str(tmp_path / "test.db")
    result = runner.invoke(main, ["init", "--db", db_path])
    assert result.exit_code == 0
    assert "Initialized" in result.output
    assert os.path.exists(db_path)


def test_cli_status_after_init(tmp_path):
    runner = CliRunner()
    db_path = str(tmp_path / "test.db")
    runner.invoke(main, ["init", "--db", db_path])
    result = runner.invoke(main, ["status", "--db", db_path])
    assert result.exit_code == 0
    assert "Claims:" in result.output
    assert "Sessions:" in result.output


def test_cli_ingest(tmp_path):
    runner = CliRunner()
    db_path = str(tmp_path / "test.db")
    runner.invoke(main, ["init", "--db", db_path])
    result = runner.invoke(
        main, ["ingest", "I prefer dark mode for my code editor", "--db", db_path]
    )
    assert result.exit_code == 0
    assert "Claims created:" in result.output


def test_cli_ingest_then_status(tmp_path):
    runner = CliRunner()
    db_path = str(tmp_path / "test.db")
    runner.invoke(main, ["init", "--db", db_path])
    runner.invoke(main, ["ingest", "I prefer dark mode for my code editor", "--db", db_path])
    result = runner.invoke(main, ["status", "--db", db_path])
    assert result.exit_code == 0
    assert "1" in result.output


def test_cli_query_after_ingest(tmp_path):
    runner = CliRunner()
    db_path = str(tmp_path / "test.db")
    runner.invoke(main, ["init", "--db", db_path])
    runner.invoke(main, ["ingest", "I prefer dark mode for my code editor", "--db", db_path])
    result = runner.invoke(main, ["query", "dark mode", "--db", db_path])
    assert result.exit_code == 0
    assert "Found" in result.output


def test_cli_query_empty_session(tmp_path):
    runner = CliRunner()
    db_path = str(tmp_path / "test.db")
    runner.invoke(main, ["init", "--db", db_path])
    result = runner.invoke(main, ["query", "anything", "--db", db_path])
    assert result.exit_code == 0
    assert "No active claims" in result.output


def test_cli_serve_command_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "MCP server" in result.output or "transport" in result.output


def test_cli_ingest_rejected_noise(tmp_path):
    runner = CliRunner()
    db_path = str(tmp_path / "test.db")
    runner.invoke(main, ["init", "--db", db_path])
    result = runner.invoke(main, ["ingest", "uh um ok", "--db", db_path])
    assert result.exit_code == 0
    assert "rejected" in result.output.lower()
