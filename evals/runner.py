"""Evaluation suite runner.

Loads test cases from JSON, runs them through the memcontext pipeline,
and computes metrics. No LLM calls.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from memcontext.extractors import PassthroughExtractor, SimpleExtractor
from memcontext.mcp_tools import handle_memory_store
from memcontext.retrieval import retrieve_hybrid
from memcontext.schema import open_database

from evals.metrics import extraction_precision_recall, provenance_integrity


@dataclass
class EvalCase:
    """One evaluation test case."""

    name: str
    turns: list[dict]
    queries: list[dict] = field(default_factory=list)
    gold_claims: list[dict] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of running one eval case."""

    case_name: str
    extraction_metrics: dict | None = None
    retrieval_metrics: dict | None = None
    provenance_valid: bool = True
    errors: list[str] = field(default_factory=list)


def load_suite(suite_path: str | Path) -> list[EvalCase]:
    """Load eval cases from a JSON file."""
    data = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    return [
        EvalCase(
            name=c["name"],
            turns=c.get("turns", []),
            queries=c.get("queries", []),
            gold_claims=c.get("gold_claims", []),
        )
        for c in cases
    ]


def run_case(case: EvalCase, conn: sqlite3.Connection, session_id: str) -> EvalResult:
    """Run a single eval case: ingest turns, run queries, compute metrics."""
    result = EvalResult(case_name=case.name)

    all_claim_ids: list[str] = []
    for turn in case.turns:
        speaker = turn.get("speaker", "user")
        text = turn.get("text", "")
        claims_data = turn.get("claims")
        store_result = handle_memory_store(
            conn, text=text, speaker=speaker, session_id=session_id, claims=claims_data
        )
        if store_result.get("claim_ids"):
            all_claim_ids.extend(store_result["claim_ids"])

    if case.gold_claims:
        from memcontext.claims import list_active_claims

        active = list_active_claims(conn, session_id)
        extracted_dicts = [
            {"subject": c.subject, "predicate": c.predicate, "value": c.value}
            for c in active
        ]
        result.extraction_metrics = extraction_precision_recall(
            extracted_dicts, case.gold_claims
        )

    for claim_id in all_claim_ids:
        pi = provenance_integrity(conn, claim_id)
        if not pi["valid"]:
            result.provenance_valid = False
            result.errors.append(f"Broken provenance for {claim_id}")

    if case.queries:
        from evals.metrics import retrieval_recall_at_k

        for q in case.queries:
            question = q.get("question", "")
            hybrid_results = retrieve_hybrid(conn, session_id=session_id, query=question, top_k=10)
            retrieved_ids = [c.claim_id for c, _ in hybrid_results]
            expected_ids = set(q.get("expected_claim_ids", []))
            if expected_ids:
                recall = retrieval_recall_at_k(retrieved_ids, expected_ids, 10)
                result.retrieval_metrics = result.retrieval_metrics or {}
                result.retrieval_metrics[question] = {"recall@10": recall}

    return result


def run_suite(suite_path: str | Path) -> list[EvalResult]:
    """Run all cases in a suite file. Fresh in-memory DB per case."""
    cases = load_suite(suite_path)
    results = []
    for i, case in enumerate(cases):
        conn = open_database(":memory:")
        conn.row_factory = sqlite3.Row
        sid = f"eval_{i}_{case.name}"
        results.append(run_case(case, conn, sid))
        conn.close()
    return results


def print_results(results: list[EvalResult]) -> None:
    """Print results as a readable summary."""
    for r in results:
        status = "PASS" if r.provenance_valid and not r.errors else "FAIL"
        print(f"  [{status}] {r.case_name}")
        if r.extraction_metrics:
            m = r.extraction_metrics
            print(f"    extraction: P={m['precision']} R={m['recall']} F1={m['f1']}")
        if r.errors:
            for e in r.errors:
                print(f"    ERROR: {e}")


# ---------------------------------------------------------------------------
# Reader / prompt-routing layer
# ---------------------------------------------------------------------------


class ReaderMode(StrEnum):
    NONE = "none"  # retrieval context only, no LLM call, no fake answer
    CONFIGURED = "configured"  # call LLM if configured (not implemented yet)


_READER_SYSTEM_PROMPT = (
    "You are a personal assistant with access to past conversation history."
)


def _format_evidence(
    question: str,
    question_date: str,
    excerpts: list[dict],
) -> str:
    """Format retrieved excerpts for the reader, matching baseline format.

    Each excerpt includes session date, relative offset, and temporal gap
    markers between excerpts with large time differences (Mastra pattern).
    """
    from datetime import datetime

    def _parse_date(d: str) -> datetime | None:
        try:
            return datetime.strptime(d[:10], "%Y/%m/%d")
        except (ValueError, TypeError):
            return None

    # Sort excerpts by date for temporal coherence
    def _sort_key(ex: dict) -> str:
        return ex.get("session_date", "") or "9999"

    sorted_excerpts = sorted(excerpts, key=_sort_key)

    parts = []
    if question_date:
        parts.append(f"Question (asked on {question_date}): {question}\n")
    else:
        parts.append(f"Question: {question}\n")

    prev_date: datetime | None = None
    for i, ex in enumerate(sorted_excerpts, 1):
        speaker = ex.get("speaker", "user")
        text = ex.get("text", "")[:1000]
        date_str = ex.get("session_date", "")
        offset = ex.get("relative_offset", "")

        # Temporal gap marker between excerpts (Mastra pattern)
        cur_date = _parse_date(date_str)
        if prev_date and cur_date:
            gap_days = (cur_date - prev_date).days
            if gap_days >= 7:
                weeks = gap_days // 7
                if weeks >= 4:
                    months = gap_days // 30
                    parts.append(f"  --- {months} month{'s' if months > 1 else ''} later ---\n")
                else:
                    parts.append(f"  --- {weeks} week{'s' if weeks > 1 else ''} later ---\n")
            elif gap_days >= 2:
                parts.append(f"  --- {gap_days} days later ---\n")
        if cur_date:
            prev_date = cur_date

        header = f"--- Excerpt {i}"
        if date_str:
            header += f" | {date_str}"
        if offset:
            header += f" ({offset})"
        header += " ---"
        parts.append(f"{header}\n{speaker}: {text}\n")

    return "\n".join(parts)


def _format_context_text(
    claims: list[dict],
    excerpts: list[dict] | None,
) -> str:
    """Format context as numbered text for category-specific prompts."""
    if excerpts:
        lines = []
        for i, ex in enumerate(excerpts, 1):
            date = ex.get("session_date", "")
            speaker = ex.get("speaker", "user")
            text = ex.get("text", "")
            header = f"{i}."
            if date:
                header += f" [{date}]"
            lines.append(f"{header} {speaker}: {text}")
        return "\n".join(lines)

    from evals.longmemeval_prompts import format_claims_for_prompt

    return format_claims_for_prompt(claims)


def answer_question(
    *,
    question: str,
    category: str,
    claims: list[dict],
    reader: ReaderMode = ReaderMode.NONE,
    question_date: str = "",
    excerpts: list[dict] | None = None,
) -> dict:
    """Format evidence and call reader. Routes to category-specific prompts
    when available, falls back to universal Chain-of-Note prompt.

    reader="none": returns retrieval context only. NO fake answer.
    reader="configured": calls reader LLM with category or CoN prompting.
    """
    from evals.longmemeval_prompts import CATEGORY_MAP, PROMPTS, get_prompt
    from memcontext.formatting import format_context_json, format_reader_prompt

    prompt_key = CATEGORY_MAP.get(category, category)
    use_category = prompt_key in PROMPTS

    if use_category:
        claims_text = _format_context_text(claims, excerpts)
        full_prompt = get_prompt(category, claims_text, question)
        if question_date:
            full_prompt = f"Current date: {question_date}\n\n{full_prompt}"
        template_name = f"category_{prompt_key}"
    else:
        context_json = format_context_json(
            claims=claims if not excerpts else None,
            turns=excerpts,
        )
        full_prompt = format_reader_prompt(
            context_json=context_json,
            question=question,
            question_date=question_date,
        )
        template_name = "universal_con"

    result: dict[str, object] = {
        "category": category,
        "prompt_template_used": template_name,
        "full_prompt": full_prompt,
        "num_claims": len(claims),
    }

    if reader == ReaderMode.NONE:
        result["predicted_answer"] = None
        result["reader_mode"] = "none"
    elif reader == ReaderMode.CONFIGURED:
        raw_answer = _call_reader_llm(full_prompt)
        if "Answer:" in raw_answer:
            result["predicted_answer"] = raw_answer.rsplit("Answer:", 1)[1].strip()
        else:
            result["predicted_answer"] = raw_answer
        result["reader_mode"] = "configured"
        result["raw_reader_output"] = raw_answer

    return result


def _call_reader_llm_with_system(system: str, user: str) -> str:
    """Call reader with system + user messages (baseline style)."""
    import os

    import requests

    api_key = os.environ.get("MEMCONTEXT_READER_API_KEY", "")
    if not api_key:
        raise ValueError("MEMCONTEXT_READER_API_KEY not set.")

    model = os.environ.get("MEMCONTEXT_READER_MODEL", "openai/gpt-5-mini")
    endpoint = os.environ.get(
        "MEMCONTEXT_READER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
    )

    resp = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Title": "memcontext",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system, "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": user},
            ],
            "max_tokens": 2048,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    return (content or "").strip()


def _call_reader_llm(prompt: str) -> str:
    """Call the configured reader LLM via OpenRouter-compatible API."""
    import os

    import requests

    api_key = os.environ.get("MEMCONTEXT_READER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "MEMCONTEXT_READER_API_KEY not set. "
            "Export it before running with --reader configured."
        )

    model = os.environ.get("MEMCONTEXT_READER_MODEL", "openai/gpt-5-mini")
    endpoint = os.environ.get(
        "MEMCONTEXT_READER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
    )

    resp = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    return (content or "").strip()
