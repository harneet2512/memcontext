"""Context formatting for reader LLMs — JSON + Chain-of-Note, matching LongMemEval official protocol."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def format_context_json(
    *,
    claims: list[dict[str, Any]] | None = None,
    turns: list[dict[str, Any]] | None = None,
    query_type: str = "fact_recall",
) -> str:
    """Format retrieved context as JSON array — matches LongMemEval official format.

    Each item is a JSON object with session date, speaker, and content.
    Items are sorted by date for temporal consistency.
    """
    items: list[dict[str, Any]] = []

    if claims:
        for c in claims:
            items.append({
                "session_date": c.get("session_date", ""),
                "speaker": c.get("speaker", "user"),
                "content": c.get("value", ""),
                "predicate": c.get("predicate", ""),
                "confidence": c.get("confidence", 0.0),
                "type": "claim",
            })

    if turns:
        for t in turns:
            items.append({
                "session_date": t.get("session_date", ""),
                "speaker": t.get("speaker", "user"),
                "content": t.get("text", ""),
                "type": "turn",
            })

    items.sort(key=lambda x: x.get("session_date", ""))
    return json.dumps(items, indent=2)


def format_reader_prompt(
    *,
    context_json: str,
    question: str,
    question_date: str = "",
) -> str:
    """Build Chain-of-Note reader prompt: per-item notes, then reasoning, then answer.

    CoN writes a structured note per context item before synthesizing,
    unlike CoT which reasons in one pass. CoN + JSON = up to +10 points
    per LongMemEval paper Section 5.5.
    """
    date_line = f"\nCurrent Date: {question_date}" if question_date else ""
    return (
        "You are given a question and a set of memory items from past conversations.\n\n"
        "Step 1 — Notes: For each memory item below, write one brief note "
        "extracting only the information relevant to the question. "
        "Skip items with no relevant information.\n"
        "Step 2 — Reasoning: Using only your notes, reason step by step toward the answer.\n"
        "Step 3 — Answer: State the final answer concisely.\n\n"
        "Memory items:\n"
        f"{context_json}"
        f"{date_line}\n"
        f"Question: {question}\n\n"
        "Step 1 — Notes:"
    )


def format_context_for_reader(
    *,
    profile_text: str | None = None,
    claims: list[dict[str, Any]] | None = None,
    turns: list[dict[str, Any]] | None = None,
    chains: dict[str, str] | None = None,
    query_type: str = "fact_recall",
) -> str:
    """Format retrieved context as JSON array for reader LLMs."""
    return format_context_json(claims=claims, turns=turns, query_type=query_type)


def format_claim_with_speaker(
    conn: sqlite3.Connection,
    claim_id: str,
    value: str,
    predicate: str,
    subject: str,
    confidence: float,
) -> str:
    """Format a single claim with its source turn's speaker."""
    row = conn.execute(
        "SELECT t.speaker FROM turns t "
        "JOIN claims c ON c.source_turn_id = t.turn_id "
        "WHERE c.claim_id = ?",
        (claim_id,),
    ).fetchone()

    speaker = row["speaker"] if row else "unknown"
    return f"[{predicate}, {speaker}] {subject}: {value} (confidence: {confidence})"
