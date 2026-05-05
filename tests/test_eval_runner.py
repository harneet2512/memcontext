from __future__ import annotations

import os
from pathlib import Path

from evals.runner import EvalCase, EvalResult, load_suite, run_case, run_suite

SUITES_DIR = Path(__file__).resolve().parent.parent / "evals" / "suites"


def test_load_suite_extraction():
    cases = load_suite(SUITES_DIR / "extraction.json")
    assert len(cases) == 10
    assert all(isinstance(c, EvalCase) for c in cases)
    assert cases[0].name == "simple_user_fact"


def test_load_suite_retrieval():
    cases = load_suite(SUITES_DIR / "retrieval.json")
    assert len(cases) == 10


def test_load_suite_supersession():
    cases = load_suite(SUITES_DIR / "supersession.json")
    assert len(cases) == 10


def test_run_case_simple():
    import sqlite3
    from memcontext.schema import open_database

    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    case = EvalCase(
        name="test_basic",
        turns=[{"speaker": "user", "text": "I prefer dark mode for my editor"}],
        gold_claims=[
            {"subject": "user", "predicate": "user_preference", "value": "dark mode for my editor"}
        ],
    )
    result = run_case(case, conn, "eval_test")
    assert isinstance(result, EvalResult)
    assert result.case_name == "test_basic"
    assert result.provenance_valid is True


def test_run_suite_extraction():
    results = run_suite(SUITES_DIR / "extraction.json")
    assert len(results) == 10
    assert all(isinstance(r, EvalResult) for r in results)
    passed = sum(1 for r in results if r.provenance_valid and not r.errors)
    assert passed >= 8


def test_eval_case_dataclass():
    case = EvalCase(name="test", turns=[], queries=[], gold_claims=[])
    assert case.name == "test"
    assert case.turns == []
