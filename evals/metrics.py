"""Evaluation metrics for memcontext. Pure functions, no LLM calls."""
from __future__ import annotations

import sqlite3


def extraction_precision_recall(
    extracted: list[dict],
    gold: list[dict],
    match_fields: tuple[str, ...] = ("subject", "predicate", "value"),
) -> dict:
    """Precision, recall, F1 between extracted and gold claims.

    Match is exact on lowercased/stripped specified fields.
    """
    def _key(c: dict) -> tuple:
        return tuple(str(c.get(f, "")).strip().lower() for f in match_fields)

    gold_keys = {_key(g) for g in gold}
    extracted_keys = [_key(e) for e in extracted]
    matched = sum(1 for k in extracted_keys if k in gold_keys)

    n_ext = len(extracted)
    n_gold = len(gold)
    precision = matched / n_ext if n_ext else 0.0
    recall = matched / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": matched,
        "extracted": n_ext,
        "gold": n_gold,
    }


def retrieval_recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int,
) -> float:
    """Fraction of relevant items appearing in top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def retrieval_mrr(
    retrieved_ids: list[str], relevant_ids: set[str],
) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant item, or 0."""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def provenance_integrity(conn: sqlite3.Connection, claim_id: str) -> dict:
    """Check that a claim's provenance is intact."""
    from memcontext.claims import get_claim, get_turn
    from memcontext.provenance import span_for_claim

    claim = get_claim(conn, claim_id)
    if claim is None:
        return {"valid": False, "has_turn": False, "has_span": False, "span_in_bounds": False}

    turn = get_turn(conn, claim.source_turn_id)
    has_turn = turn is not None
    span = span_for_claim(conn, claim_id)
    has_span = span is not None and span.char_start is not None

    span_in_bounds = False
    if has_span and has_turn and span is not None and turn is not None:
        span_in_bounds = (
            0 <= (span.char_start or 0) <= (span.char_end or 0) <= len(turn.text)
        )

    return {
        "valid": has_turn,
        "has_turn": has_turn,
        "has_span": has_span,
        "span_in_bounds": span_in_bounds,
    }


def answer_accuracy_exact(predicted: str, gold: str) -> float:
    """1.0 if lowercased/stripped strings match, else 0.0."""
    return 1.0 if predicted.strip().lower() == gold.strip().lower() else 0.0


def answer_accuracy_fuzzy(predicted: str, gold: str) -> float:
    """Token-overlap F1 between predicted and gold answers."""
    pred_tokens = set(predicted.strip().lower().split())
    gold_tokens = set(gold.strip().lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)
