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
    question_date: str = ""


@dataclass
class LongMemEvalSession:
    session_id: str
    turns: list[dict] = field(default_factory=list)
    date: str = ""


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


CURRENT_SCORING = ScoringMethod.JUDGE
SCORING_NOTES = (
    "The 88.4% (442/500) was scored with a two-tier system matching the official "
    "LongMemEval protocol: (1) normalized exact boundary match for short answers "
    "(<=3 tokens), (2) GPT-4o LLM-as-judge for everything else with task-specific "
    "rubrics. Reader: GPT-5-mini. Judge: GPT-4o-2024-11-20. Cost: $10.21. "
    "Source: RobbyMD eval/longmemeval/final_runner.py"
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
        haystack_dates = instance.get("haystack_dates", [])
        answer_session_ids = instance.get("answer_session_ids", [])

        for idx, (sid, turns) in enumerate(zip(haystack_ids, haystack_sessions)):
            full_sid = f"{qid}__{sid}"
            date = haystack_dates[idx] if idx < len(haystack_dates) else ""
            if full_sid not in all_sessions:
                all_sessions[full_sid] = LongMemEvalSession(
                    session_id=full_sid,
                    turns=turns if isinstance(turns, list) else [],
                    date=date,
                )

        session_refs = [f"{qid}__{sid}" for sid in haystack_ids]
        questions.append(
            LongMemEvalQuestion(
                question_id=qid,
                question=instance["question"],
                category=category,
                gold_answer=instance.get("answer", ""),
                session_ids=session_refs,
                question_date=instance.get("question_date", ""),
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
    """Run a LongMemEval preflight using the full MemContext pipeline.

    Per CLAUDE.md: this is a diagnostic, not a target. The pipeline is
    general-purpose and not LongMemEval-specific:

    1. Ingest turns through on_new_turn (admission → extract → supersede)
    2. LLMExtractor with prior-turn context for coreference resolution
    3. Embed structured claims via backfill_embeddings
    4. Retrieve via hybrid RRF (semantic + entity + temporal)
    5. Route through category-specific answer prompt
    6. Score with reader if configured (reader=none → retrieval context only)

    Requires MEMCONTEXT_EXTRACTOR_BACKEND + key for extraction.
    Requires MEMCONTEXT_READER_API_KEY for reader=configured.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import os

    from evals.runner import ReaderMode, answer_question
    from memcontext.extractors import LLMExtractor, auto_extractor
    from memcontext.on_new_turn import ExtractedClaim, on_new_turn
    from memcontext.extractors import PassthroughExtractor
    from memcontext.retrieval import EmbeddingClient, backfill_embeddings
    from memcontext.schema import Speaker, Turn, open_database

    # Use personal_assistant pack (matching baseline) unless overridden
    if not os.environ.get("ACTIVE_PACK"):
        os.environ["ACTIVE_PACK"] = "personal_assistant"
        from memcontext.predicate_packs import active_pack
        active_pack.cache_clear()

    sessions, questions = load_dataset(dataset_path)

    if target_categories:
        questions = [q for q in questions if q.category in target_categories]

    questions = questions[:limit]

    session_map: dict[str, LongMemEvalSession] = {s.session_id: s for s in sessions}

    reader_mode = ReaderMode(reader)
    question_results = []
    embedding_client = EmbeddingClient()
    extractor = auto_extractor()

    for qi, q in enumerate(questions):
        conn = open_database(":memory:")
        conn.row_factory = sqlite3.Row

        unified_sid = f"haystack_{q.question_id}"

        # Collect all turns with session dates
        all_turns: list[tuple[str, str, str]] = []  # (role, content, session_date)
        for sid in q.session_ids:
            sess = session_map.get(sid)
            if sess is None:
                continue
            for turn_data in sess.turns:
                role = turn_data.get("role", "user")
                content = turn_data.get("content", "")
                if content and content.strip():
                    all_turns.append((role, content, sess.date))

        # Parallel extraction: fire all LLM calls concurrently
        from memcontext.claims import new_turn_id, now_ns

        def _extract_one(idx_role_text: tuple[int, str, str]) -> tuple[int, list[dict]]:
            idx, role, text = idx_role_text
            sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
            t = Turn(turn_id=new_turn_id(), session_id=unified_sid,
                     speaker=sp, text=text, ts=now_ns(), asr_confidence=None)
            claims = extractor(t)
            return (idx, [
                {"subject": c.subject, "predicate": c.predicate,
                 "value": c.value, "confidence": c.confidence}
                for c in claims
            ])

        extracted_by_idx: dict[int, list[dict]] = {}
        work = [(i, role, text) for i, (role, text, _date) in enumerate(all_turns)]

        if isinstance(extractor, LLMExtractor):
            with ThreadPoolExecutor(max_workers=20) as pool:
                futures = {pool.submit(_extract_one, w): w[0] for w in work}
                for fut in as_completed(futures):
                    try:
                        idx, claims = fut.result()
                        extracted_by_idx[idx] = claims
                    except Exception:
                        extracted_by_idx[futures[fut]] = []
        else:
            for w in work:
                idx, claims = _extract_one(w)
                extracted_by_idx[idx] = claims

        # Sequential storage: insert in order for supersession to work
        ingested_turns = len(all_turns)
        ingested_sessions = len(set(
            sid for sid in q.session_ids if session_map.get(sid)
        ))
        claims_created = 0

        turn_session_date: dict[str, str] = {}  # turn_id → session_date

        for i, (role, text, sess_date) in enumerate(all_turns):
            claims_data = extracted_by_idx.get(i, [])
            if not claims_data:
                continue
            sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
            pt = PassthroughExtractor(claims_data)
            result = on_new_turn(
                conn, session_id=unified_sid, speaker=sp,
                text=text, extractor=pt,
            )
            claims_created += len(result.created_claims)
            if result.turn is not None:
                turn_session_date[result.turn.turn_id] = sess_date

        embedded_count = backfill_embeddings(conn, unified_sid, client=embedding_client)

        # Hybrid retrieval: 0.7 dense + 0.3 BM25, matching 88.4% baseline
        from memcontext.retrieval import retrieve_hybrid
        top_claims = retrieve_hybrid(
            conn, session_id=unified_sid, query=q.question,
            top_k=50, embedding_client=embedding_client,
            weights=(0.7, 0.0, 0.0, 0.3),  # semantic=0.7, entity=0, temporal=0, BM25=0.3
        )

        # Look up source turns for retrieved claims — reader needs
        # original conversation context with temporal offsets
        from memcontext.claims import get_turn

        def _relative_offset(session_date: str, question_date: str) -> str:
            """Compute relative time offset like '2 weeks ago'."""
            from datetime import datetime
            try:
                fmt = "%Y/%m/%d"
                sd = datetime.strptime(session_date[:10], fmt)
                qd = datetime.strptime(question_date[:10], fmt)
                delta = (qd - sd).days
                if delta == 0:
                    return "same day"
                if delta == 1:
                    return "1 day ago"
                if delta < 7:
                    return f"{delta} days ago"
                weeks = delta // 7
                if weeks < 5:
                    return f"~{weeks} week{'s' if weeks > 1 else ''} ago"
                months = delta // 30
                if months < 12:
                    return f"~{months} month{'s' if months > 1 else ''} ago"
                return f"~{delta // 365} year{'s' if delta > 730 else ''} ago"
            except (ValueError, TypeError):
                return ""

        seen_turns: set[str] = set()
        excerpts: list[dict] = []
        for c, s in top_claims:
            if c.source_turn_id in seen_turns:
                continue
            seen_turns.add(c.source_turn_id)
            turn = get_turn(conn, c.source_turn_id)
            if turn is None:
                continue
            sess_date = turn_session_date.get(c.source_turn_id, "")
            offset = _relative_offset(sess_date, q.question_date) if sess_date and q.question_date else ""
            excerpts.append({
                "text": turn.text,
                "speaker": turn.speaker.value if hasattr(turn.speaker, "value") else str(turn.speaker),
                "session_date": sess_date,
                "relative_offset": offset,
                "claim_value": c.value,
                "score": round(s, 4),
            })

        claims = [
            {
                "claim_id": c.claim_id,
                "subject": c.subject,
                "predicate": c.predicate,
                "value": c.value,
                "confidence": c.confidence,
                "status": c.status.value,
                "score": round(s, 4),
            }
            for c, s in top_claims
        ]

        answer_result = answer_question(
            question=q.question,
            category=q.category,
            claims=claims,
            reader=reader_mode,
            question_date=q.question_date,
            excerpts=excerpts,
        )

        qr_entry: dict = {
            "question_id": q.question_id,
            "category": q.category,
            "gold_answer": q.gold_answer,
            "ingested_sessions": ingested_sessions,
            "ingested_turns": ingested_turns,
            "claims_created": claims_created,
            "embedded_count": embedded_count,
            "num_claims_retrieved": len(claims),
            **answer_result,
        }

        predicted = answer_result.get("predicted_answer")
        if predicted is not None:
            from evals.metrics import score_answer
            score = score_answer(
                predicted=predicted,
                gold=str(q.gold_answer),
                question=q.question,
                question_type=q.category,
                question_id=q.question_id,
            )
            qr_entry["score"] = score
            qr_entry["correct"] = score >= 1.0

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
