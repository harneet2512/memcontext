from __future__ import annotations

import sqlite3

import pytest

from evals.metrics import (
    answer_accuracy_exact,
    answer_accuracy_fuzzy,
    extraction_precision_recall,
    provenance_integrity,
    retrieval_mrr,
    retrieval_recall_at_k,
)


def test_extraction_precision_recall_perfect():
    gold = [{"subject": "user", "predicate": "user_fact", "value": "lives in Toronto"}]
    extracted = [{"subject": "user", "predicate": "user_fact", "value": "lives in Toronto"}]
    result = extraction_precision_recall(extracted, gold)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_extraction_precision_recall_partial():
    gold = [
        {"subject": "user", "predicate": "user_fact", "value": "lives in Toronto"},
        {"subject": "user", "predicate": "user_fact", "value": "works at Google"},
    ]
    extracted = [{"subject": "user", "predicate": "user_fact", "value": "lives in Toronto"}]
    result = extraction_precision_recall(extracted, gold)
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["matched"] == 1


def test_extraction_precision_recall_no_match():
    gold = [{"subject": "user", "predicate": "user_fact", "value": "lives in Toronto"}]
    extracted = [{"subject": "user", "predicate": "user_fact", "value": "works at Google"}]
    result = extraction_precision_recall(extracted, gold)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_extraction_empty():
    result = extraction_precision_recall([], [])
    assert result["precision"] == 0.0
    assert result["matched"] == 0


def test_retrieval_recall_at_k():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d", "f"}
    assert retrieval_recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)
    assert retrieval_recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)
    assert retrieval_recall_at_k(retrieved, relevant, 1) == 0.0


def test_retrieval_mrr():
    assert retrieval_mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert retrieval_mrr(["a", "b", "c"], {"a"}) == 1.0
    assert retrieval_mrr(["a", "b", "c"], {"x"}) == 0.0


def test_answer_accuracy_exact():
    assert answer_accuracy_exact("Dark mode", "dark mode") == 1.0
    assert answer_accuracy_exact("light", "dark") == 0.0
    assert answer_accuracy_exact("  Toronto ", "Toronto") == 1.0


def test_answer_accuracy_fuzzy():
    score = answer_accuracy_fuzzy("dark mode editor", "dark mode")
    assert score > 0.5
    assert answer_accuracy_fuzzy("completely different", "nothing alike") == 0.0
    assert answer_accuracy_fuzzy("Toronto", "Toronto") == 1.0


def test_provenance_integrity_valid(db, sample_claim):
    result = provenance_integrity(db, sample_claim.claim_id)
    assert result["valid"] is True
    assert result["has_turn"] is True


def test_provenance_integrity_missing():
    from memcontext.schema import open_database

    conn = open_database(":memory:")
    result = provenance_integrity(conn, "cl_nonexistent")
    assert result["valid"] is False


# --- Two-tier scoring (ported from RobbyMD official protocol) ---


def test_strict_short_answer_exact():
    from evals.metrics import strict_short_answer_check

    assert strict_short_answer_check("3", "3") is True
    assert strict_short_answer_check("3", "The answer is 3 items") is True  # boundary match
    assert strict_short_answer_check("3", "5") is False  # numeric mismatch
    assert strict_short_answer_check("8 days", "8 days") is True


def test_strict_short_answer_fallthrough():
    from evals.metrics import strict_short_answer_check

    # Long gold answers fall through to judge (return None)
    assert strict_short_answer_check(
        "The user prefers dark mode for developer tools", "dark mode"
    ) is None


def test_strict_short_normalization():
    from evals.metrics import _normalize_short

    assert _normalize_short("$400,000") == "400000"
    assert _normalize_short("  3  ") == "3"
    assert _normalize_short("8 Days") == "8 days"


def test_score_answer_without_api_key_uses_fuzzy():
    """Without MEMCONTEXT_READER_API_KEY, score_answer falls back to fuzzy."""
    import os
    from evals.metrics import score_answer

    old_key = os.environ.pop("MEMCONTEXT_READER_API_KEY", None)
    try:
        # Short answer: strict match works without API
        assert score_answer("3", "3", "how many?", "multi-session") == 1.0
        assert score_answer("5", "3", "how many?", "multi-session") == 0.0
    finally:
        if old_key:
            os.environ["MEMCONTEXT_READER_API_KEY"] = old_key


def test_judge_prompts_exist():
    from evals.metrics import _JUDGE_PROMPTS

    assert "default" in _JUDGE_PROMPTS
    assert "temporal-reasoning" in _JUDGE_PROMPTS
    assert "single-session-preference" in _JUDGE_PROMPTS
    assert "knowledge-update" in _JUDGE_PROMPTS
    assert "abstention" in _JUDGE_PROMPTS
    for key, prompt in _JUDGE_PROMPTS.items():
        assert "yes or no" in prompt.lower(), f"Judge prompt {key} missing yes/no instruction"
