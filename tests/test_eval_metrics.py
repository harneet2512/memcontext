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
