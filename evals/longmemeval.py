"""LongMemEval benchmark integration scaffold.

The actual dataset download and full benchmark run require:
1. The LongMemEval dataset (public GitHub)
2. A reader LLM for answer generation
3. API keys for the reader

This module provides the scaffold — data loading, session ingestion,
and scoring infrastructure. Full execution is deferred to Phase 6.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class LongMemEvalQuestion:
    question_id: str
    question: str
    category: str
    gold_answer: str
    session_ids: list[str] = field(default_factory=list)


@dataclass
class LongMemEvalSession:
    session_id: str
    turns: list[dict] = field(default_factory=list)


@dataclass
class LongMemEvalResult:
    question_id: str
    category: str
    predicted_answer: str
    gold_answer: str
    score: float
    retrieved_claims: list[dict] = field(default_factory=list)


CATEGORIES = [
    "single_session_user_fact",
    "single_session_preference",
    "cross_session_preference",
    "cross_session_user_fact",
    "temporal_ordering",
    "knowledge_update",
    "abstention",
]


def load_dataset(
    path: str,
) -> tuple[list[LongMemEvalSession], list[LongMemEvalQuestion]]:
    """Load LongMemEval dataset. Raises NotImplementedError until Phase 6."""
    raise NotImplementedError(
        "Dataset loading deferred to Phase 6. Download dataset first."
    )


def score_results(results: list[LongMemEvalResult]) -> dict:
    """Compute per-category and overall scores."""
    if not results:
        return {"overall": 0.0, "by_category": {}, "total_questions": 0}
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r.score)
    return {
        "overall": sum(r.score for r in results) / len(results),
        "by_category": {
            cat: sum(scores) / len(scores) for cat, scores in by_cat.items()
        },
        "total_questions": len(results),
    }
