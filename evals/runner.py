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
from memcontext.mcp_tools import handle_memory_query, handle_memory_store
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
            qr = handle_memory_query(conn, query=question, session_id=session_id, top_k=10)
            retrieved_ids = [c["claim_id"] for c in qr.get("claims", [])]
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


def answer_question(
    *,
    question: str,
    category: str,
    claims: list[dict],
    reader: ReaderMode = ReaderMode.NONE,
) -> dict:
    """Select category prompt and prepare answer context.

    reader="none": returns retrieval context + selected prompt. NO fake answer.
    reader="configured": would call LLM (not yet implemented).
    """
    from evals.longmemeval_prompts import format_claims_for_prompt, get_prompt

    claims_text = format_claims_for_prompt(claims)
    prompt = get_prompt(category, claims_text, question)

    result = {
        "category": category,
        "prompt_template_used": category,
        "formatted_claims": claims_text,
        "full_prompt": prompt,
        "num_claims": len(claims),
    }

    if reader == ReaderMode.NONE:
        result["predicted_answer"] = None  # NO fake answer
        result["reader_mode"] = "none"
    elif reader == ReaderMode.CONFIGURED:
        raise NotImplementedError(
            "reader='configured' requires LLM configuration. "
            "Set MEMCONTEXT_READER_MODEL and MEMCONTEXT_READER_API_KEY."
        )

    return result
