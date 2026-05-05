"""Retrieval ceiling analysis for wrong-answer diagnosis.

Classifies each wrong answer into failure categories to determine
whether architecture changes are needed or if prompt/reader fixes suffice.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum


class FailureCategory(StrEnum):
    RETRIEVAL_MISS = "retrieval_miss"
    READER_FAILURE = "reader_failure"
    EXTRACTION_MISS = "extraction_miss"
    SUPERSESSION_FAILURE = "supersession_failure"
    PROMPT_FAILURE = "prompt_failure"
    CORRECT = "correct"


@dataclass
class FailureAnalysis:
    question_id: str
    category: FailureCategory
    evidence: str


def _token_overlap(a: str, b: str) -> float:
    ta = set(a.strip().lower().split())
    tb = set(b.strip().lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def classify_failure(
    *,
    question_id: str = "",
    gold_answer: str,
    predicted_answer: str,
    retrieved_claim_values: list[str],
    all_active_claim_values: list[str],
    all_claim_values_including_superseded: list[str],
) -> FailureAnalysis:
    """Classify why an answer was wrong (or mark it correct)."""
    if _token_overlap(predicted_answer, gold_answer) > 0.5:
        return FailureAnalysis(question_id, FailureCategory.CORRECT, "answer matches gold")

    gold_in_retrieved = any(
        _token_overlap(gold_answer, v) > 0.3 for v in retrieved_claim_values
    )
    gold_in_active = any(
        _token_overlap(gold_answer, v) > 0.3 for v in all_active_claim_values
    )
    gold_in_all = any(
        _token_overlap(gold_answer, v) > 0.3 for v in all_claim_values_including_superseded
    )

    if gold_in_retrieved:
        return FailureAnalysis(
            question_id, FailureCategory.READER_FAILURE,
            "gold info was retrieved but answer is wrong",
        )
    if gold_in_active:
        return FailureAnalysis(
            question_id, FailureCategory.RETRIEVAL_MISS,
            "gold info is active but not in top-k retrieval",
        )
    if gold_in_all:
        return FailureAnalysis(
            question_id, FailureCategory.SUPERSESSION_FAILURE,
            "gold info exists but in a superseded claim",
        )
    return FailureAnalysis(
        question_id, FailureCategory.EXTRACTION_MISS,
        "gold info never became a claim",
    )


def compute_ceiling(analyses: list[FailureAnalysis]) -> dict:
    """Compute ceiling metrics from failure analyses."""
    counts = Counter(a.category.value for a in analyses)
    total = len(analyses)
    if total == 0:
        return {"total": 0, "correct": 0, "by_failure": {}, "retrieval_ceiling": 0.0}
    return {
        "total": total,
        "correct": counts.get("correct", 0),
        "by_failure": {k: v for k, v in counts.items() if k != "correct"},
        "retrieval_ceiling": (total - counts.get("extraction_miss", 0)) / total,
    }
