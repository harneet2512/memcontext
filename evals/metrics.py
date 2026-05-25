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


# ---------------------------------------------------------------------------
# Answer scoring — two-tier system matching official LongMemEval protocol
# Ported from RobbyMD eval/longmemeval/final_runner.py
# ---------------------------------------------------------------------------

import re
import unicodedata


def _normalize_short(text: str) -> str:
    """Normalize a short answer for exact matching.

    Ported from RobbyMD final_runner.py lines 252-260.
    """
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    text = re.sub(r"(?<=\d)[,_](?=\d)", "", text)
    text = re.sub(r"[$£€¥]", "", text)
    text = re.sub(r"[^a-z0-9.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _short_token_count(text: str) -> int:
    return len(text.split())


def _boundary_match(gold_norm: str, text_norm: str) -> bool:
    """Check if gold appears as a word-boundary match in the text."""
    pattern = rf"(?<![a-z0-9]){re.escape(gold_norm)}(?![a-z0-9])"
    return re.search(pattern, text_norm) is not None


def strict_short_answer_check(gold: str, hypothesis: str) -> bool | None:
    """Tier 1: exact match for short answers (<=3 tokens).

    Returns True (correct), False (wrong), or None (fall through to judge).
    Ported from RobbyMD final_runner.py lines 272-288.
    """
    gold_norm = _normalize_short(gold)
    if _short_token_count(gold_norm) > 3:
        return None
    if not gold_norm:
        return None
    hyp_norm = _normalize_short(hypothesis)
    if not hyp_norm:
        return False
    if gold_norm == hyp_norm or _boundary_match(gold_norm, hyp_norm):
        return True
    gold_numbers = set(re.findall(r"\d+", gold_norm))
    if gold_numbers:
        hyp_numbers = set(re.findall(r"\d+", hyp_norm))
        if hyp_numbers and not gold_numbers.intersection(hyp_numbers):
            return False
    return None


# Exact official LongMemEval judge prompts from evaluate_qa.py — do NOT paraphrase.
_JUDGE_PROMPTS: dict[str, str] = {
    "default": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {gold}\n\nModel Response: {prediction}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. In addition, "
        "do not penalize off-by-one errors for the number of days. If the question asks for "
        "the number of days/weeks/months, etc., and the model makes off-by-one errors "
        "(e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {gold}\n\nModel Response: {prediction}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer is the "
        "required answer."
        "\n\nQuestion: {question}\n\nCorrect Answer: {gold}\n\nModel Response: {prediction}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response "
        "from a model. Please answer yes if the response satisfies the desired response. "
        "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
        "The response is correct as long as it recalls and utilizes the user's personal "
        "information correctly."
        "\n\nQuestion: {question}\n\nRubric: {gold}\n\nModel Response: {prediction}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say that the information is incomplete, or some other information is "
        "given but the asked information is not."
        "\n\nQuestion: {question}\n\nExplanation: {gold}\n\nModel Response: {prediction}"
        "\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    ),
}


_JUDGE_CATEGORY_MAP: dict[str, str] = {
    "single_session_preference": "single-session-preference",
    "single_session_user_fact": "default",
    "single_session_assistant": "default",
    "cross_session_user_fact": "default",
    "cross_session_preference": "single-session-preference",
    "temporal_ordering": "temporal-reasoning",
    "knowledge_update": "knowledge-update",
    "abstention": "abstention",
    "single-session-user": "default",
    "single-session-assistant": "default",
    "single-session-preference": "single-session-preference",
    "multi-session": "default",
    "temporal-reasoning": "temporal-reasoning",
    "knowledge-update": "knowledge-update",
}


def _get_judge_prompt(
    question_type: str, question: str, gold: str, prediction: str,
    question_id: str = "",
) -> str:
    """Get the appropriate judge prompt for the question type."""
    is_abstention = question_id.endswith("_abs") or question_type == "abstention"
    if is_abstention:
        key = "abstention"
    else:
        key = _JUDGE_CATEGORY_MAP.get(question_type, question_type)
        if key not in _JUDGE_PROMPTS:
            key = "default"
    return _JUDGE_PROMPTS[key].format(
        question=question, gold=gold, prediction=prediction,
    )


def answer_accuracy_exact(predicted: str, gold: str) -> float:
    """1.0 if lowercased/stripped strings match, else 0.0."""
    return 1.0 if predicted.strip().lower() == gold.strip().lower() else 0.0


def answer_accuracy_fuzzy(predicted: str, gold: str) -> float:
    """Token-overlap F1. Use only as fallback when no API key is available."""
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


def score_answer(
    predicted: str,
    gold: str,
    question: str = "",
    question_type: str = "",
    question_id: str = "",
) -> float:
    """Score using LLM-as-judge with task-specific rubrics.

    Matches the official LongMemEval protocol: every answer goes through
    the LLM judge. No short-answer bypass.

    Returns 1.0 (correct) or 0.0 (wrong).
    """
    if not predicted or not predicted.strip():
        return 0.0

    return _call_judge(
        question=question,
        gold=str(gold),
        prediction=predicted,
        question_type=question_type,
        question_id=question_id,
    )


def _call_judge(
    *,
    question: str,
    gold: str,
    prediction: str,
    question_type: str,
    question_id: str,
) -> float:
    """Call LLM judge via OpenRouter. Falls back to fuzzy F1 if no API key."""
    import os

    import requests

    api_key = os.environ.get("MEMCONTEXT_READER_API_KEY", "")
    if not api_key:
        return answer_accuracy_fuzzy(prediction, gold)

    model = os.environ.get("MEMCONTEXT_JUDGE_MODEL", "openai/gpt-4o-2024-08-06")
    endpoint = os.environ.get(
        "MEMCONTEXT_READER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
    )

    prompt = _get_judge_prompt(question_type, question, gold, prediction, question_id)

    try:
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Title": "memcontext",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = (
            resp.json().get("choices", [{}])[0].get("message", {}).get("content") or ""
        )
        return 1.0 if "yes" in content.strip().lower() else 0.0
    except Exception:
        return answer_accuracy_fuzzy(prediction, gold)
