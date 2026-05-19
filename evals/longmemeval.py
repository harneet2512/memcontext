"""LongMemEval benchmark integration.

Supports real dataset loading from the official LongMemEval-S JSON files
(https://github.com/xiaowu0162/LongMemEval), session ingestion,
prompt routing via category-specific prompts, and scoring infrastructure.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


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
    "single_session_assistant",
    "cross_session_preference",
    "cross_session_user_fact",
    "temporal_ordering",
    "knowledge_update",
    "abstention",
]


# ---------------------------------------------------------------------------
# Scoring methodology metadata
# ---------------------------------------------------------------------------


class ScoringMethod(StrEnum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    JUDGE = "judge"
    PROVISIONAL = "provisional"


CURRENT_SCORING = ScoringMethod.PROVISIONAL
SCORING_NOTES = (
    "The reported 88.4% was obtained with GPT-5-mini reader using fuzzy token-overlap scoring. "
    "This has not been verified against the official LongMemEval evaluation script. "
    "Until methodology is aligned with the official script, treat as PROVISIONAL."
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _resolve_category(question_id: str, question_type: str) -> str:
    """Map dataset question_type + question_id to our internal category name.

    The dataset uses:
      - question_type: single-session-user, single-session-assistant,
        single-session-preference, temporal-reasoning, knowledge-update,
        multi-session
      - Abstention questions have question_id ending with '_abs'
    """
    if question_id.endswith("_abs"):
        return "abstention"

    mapping = {
        "single-session-user": "single_session_user_fact",
        "single-session-assistant": "single_session_assistant",
        "single-session-preference": "single_session_preference",
        "temporal-reasoning": "temporal_ordering",
        "knowledge-update": "knowledge_update",
        "multi-session": "cross_session_user_fact",
    }
    return mapping.get(question_type, question_type)


def load_dataset(
    path: str,
) -> tuple[list[LongMemEvalSession], list[LongMemEvalQuestion]]:
    """Load LongMemEval dataset from a JSON file or directory.

    Accepts either:
      - A path to a specific JSON file (e.g., longmemeval_s_cleaned.json)
      - A directory containing one or more longmemeval_*.json files
        (will prefer longmemeval_s_cleaned.json, then longmemeval_s.json)

    Returns (sessions, questions) parsed into typed dataclass instances.
    """
    p = Path(path)

    # Resolve the JSON file to load
    if p.is_file():
        json_path = p
    elif p.is_dir():
        # Try preferred filenames in order
        candidates = [
            "longmemeval_s_cleaned.json",
            "longmemeval_s.json",
            "longmemeval_oracle.json",
        ]
        json_path = None
        for name in candidates:
            candidate = p / name
            if candidate.exists():
                json_path = candidate
                break
        # Fall back to any longmemeval*.json in the directory
        if json_path is None:
            json_files = sorted(p.glob("longmemeval*.json"))
            if json_files:
                json_path = json_files[0]
        # Also check data/ subdirectory (cloned repo structure)
        if json_path is None:
            data_sub = p / "data"
            if data_sub.is_dir():
                for name in candidates:
                    candidate = data_sub / name
                    if candidate.exists():
                        json_path = candidate
                        break
                if json_path is None:
                    json_files = sorted(data_sub.glob("longmemeval*.json"))
                    if json_files:
                        json_path = json_files[0]
        if json_path is None:
            raise FileNotFoundError(
                f"LongMemEval dataset not found at {path}. "
                "Download from https://github.com/xiaowu0162/LongMemEval"
            )
    else:
        raise FileNotFoundError(
            f"LongMemEval dataset not found at {path}. "
            "Download from https://github.com/xiaowu0162/LongMemEval"
        )

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list of evaluation instances, got {type(raw).__name__}"
        )

    all_sessions: dict[str, LongMemEvalSession] = {}
    questions: list[LongMemEvalQuestion] = []

    for instance in raw:
        qid = instance["question_id"]
        qtype = instance.get("question_type", "")
        category = _resolve_category(qid, qtype)

        # Parse haystack sessions
        haystack_ids = instance.get("haystack_session_ids", [])
        haystack_sessions = instance.get("haystack_sessions", [])
        answer_session_ids = instance.get("answer_session_ids", [])

        for sid, turns in zip(haystack_ids, haystack_sessions):
            # Keyed per-question to avoid collisions across instances
            full_sid = f"{qid}__{sid}"
            if full_sid not in all_sessions:
                all_sessions[full_sid] = LongMemEvalSession(
                    session_id=full_sid,
                    turns=turns if isinstance(turns, list) else [],
                )

        session_refs = [f"{qid}__{sid}" for sid in haystack_ids]
        questions.append(
            LongMemEvalQuestion(
                question_id=qid,
                question=instance["question"],
                category=category,
                gold_answer=instance.get("answer", ""),
                session_ids=session_refs,
            )
        )

    return list(all_sessions.values()), questions


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


# ---------------------------------------------------------------------------
# Preflight runner
# ---------------------------------------------------------------------------


def run_preflight(
    *,
    dataset_path: str,
    limit: int = 5,
    reader: str = "none",
    target_categories: list[str] | None = None,
) -> dict:
    """Run a tiny LongMemEval preflight.

    Loads real dataset, ingests sessions, runs questions with prompt routing.
    reader="none": outputs retrieval context + prompt only (no LLM, no score).

    Returns dict with results per question and overall stats.
    """
    from evals.runner import ReaderMode, answer_question
    from memcontext.mcp_tools import handle_memory_query, handle_memory_store
    from memcontext.schema import open_database

    sessions, questions = load_dataset(dataset_path)

    # Filter by target categories if specified
    if target_categories:
        questions = [q for q in questions if q.category in target_categories]

    # Apply limit
    questions = questions[:limit]

    # Build session lookup
    session_map: dict[str, LongMemEvalSession] = {s.session_id: s for s in sessions}

    reader_mode = ReaderMode(reader)
    question_results = []

    for q in questions:
        # Create a fresh DB per question to avoid cross-contamination
        conn = open_database(":memory:")
        conn.row_factory = sqlite3.Row

        # Ingest the sessions relevant to this question
        ingested_sessions = 0
        ingested_turns = 0
        for sid in q.session_ids:
            sess = session_map.get(sid)
            if sess is None:
                continue
            for turn in sess.turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if not content:
                    continue
                handle_memory_store(
                    conn,
                    text=content,
                    speaker=role,
                    session_id=sid,
                )
                ingested_turns += 1
            ingested_sessions += 1

        # Query memory for relevant claims
        qr = handle_memory_query(
            conn, query=q.question, session_id=q.session_ids[0] if q.session_ids else "default", top_k=10,
        )
        claims = qr.get("claims", [])

        # Route through category prompt
        answer_result = answer_question(
            question=q.question,
            category=q.category,
            claims=claims,
            reader=reader_mode,
        )

        qr_entry: dict = {
            "question_id": q.question_id,
            "category": q.category,
            "gold_answer": q.gold_answer,
            "ingested_sessions": ingested_sessions,
            "ingested_turns": ingested_turns,
            "num_claims_retrieved": len(claims),
            **answer_result,
        }

        predicted = answer_result.get("predicted_answer")
        if predicted is not None:
            from evals.metrics import answer_accuracy_fuzzy
            score = answer_accuracy_fuzzy(predicted, str(q.gold_answer))
            qr_entry["score"] = score
            qr_entry["correct"] = score > 0.3

        question_results.append(qr_entry)
        conn.close()

    # Compute summary stats
    categories_seen = list({r["category"] for r in question_results})
    scored = [r for r in question_results if "score" in r]

    per_cat: dict[str, dict] = {}
    for r in scored:
        cat = r["category"]
        if cat not in per_cat:
            per_cat[cat] = {"correct": 0, "total": 0}
        per_cat[cat]["total"] += 1
        if r.get("correct"):
            per_cat[cat]["correct"] += 1

    return {
        "dataset_path": str(dataset_path),
        "reader_mode": reader,
        "total_questions": len(question_results),
        "scored_questions": len(scored),
        "categories_seen": sorted(categories_seen),
        "per_category_accuracy": {
            cat: {"correct": v["correct"], "total": v["total"],
                  "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0}
            for cat, v in per_cat.items()
        },
        "overall_accuracy": round(
            sum(1 for r in scored if r.get("correct")) / len(scored), 4
        ) if scored else None,
        "scoring_method": str(CURRENT_SCORING),
        "scoring_notes": SCORING_NOTES,
        "questions": question_results,
    }
