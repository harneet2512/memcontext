"""Agent Memory Benchmark (AMB) runner for MemContext.

Supports loading PersonaMem32K and BEAM format datasets from
https://github.com/vectorize-io/agent-memory-benchmark, ingesting
conversations through the MemContext pipeline, and evaluating retrieval
+ reader accuracy on probing questions.

Usage:
    python evals/amb_runner.py --dataset path/to/amb_data.json --reader gpt-4o-mini --limit 10
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AMBConversation:
    """A single conversation from the AMB dataset."""

    conversation_id: str
    messages: list[dict] = field(default_factory=list)
    # [{"role": "user"/"assistant", "content": str}]


@dataclass
class AMBQuestion:
    """A probing question from the AMB dataset."""

    question_id: str
    conversation_id: str
    question: str
    gold_answer: str
    category: str


@dataclass
class AMBResult:
    """Result of evaluating a single AMB question."""

    question_id: str
    category: str
    predicted_answer: str | None
    gold_answer: str
    score: float = 0.0
    num_claims_retrieved: int = 0
    ingested_turns: int = 0
    claims_created: int = 0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_amb_dataset(
    path: str,
) -> tuple[list[AMBConversation], list[AMBQuestion]]:
    """Load an AMB dataset from a JSON file.

    Supports two formats:

    1. PersonaMem32K / BEAM format — a list of objects, each with
       ``conversation_id``, ``messages`` (list of role/content dicts),
       and ``questions`` (list of probing questions with ``question_id``,
       ``question``, ``expected_answer``/``gold_answer``, and optional
       ``category``).

    2. Flat list-of-questions format — each object has ``question_id``,
       ``conversation_id``, ``question``, ``gold_answer``/``expected_answer``,
       ``category``, and a ``conversation`` or ``messages`` field with the
       messages for the conversation.

    Returns (conversations, questions).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"AMB dataset not found at {path}. "
            "Download from https://github.com/vectorize-io/agent-memory-benchmark"
        )

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list, got {type(raw).__name__}"
        )

    conversations: dict[str, AMBConversation] = {}
    questions: list[AMBQuestion] = []

    for item in raw:
        # Detect format by checking for top-level "messages" + "questions"
        # (PersonaMem32K/BEAM grouped format)
        if "messages" in item and "questions" in item:
            conv_id = item.get("conversation_id", item.get("id", ""))
            messages = item.get("messages", [])
            if conv_id not in conversations:
                conversations[conv_id] = AMBConversation(
                    conversation_id=conv_id,
                    messages=messages,
                )

            for q in item.get("questions", []):
                qid = q.get("question_id", q.get("id", ""))
                gold = q.get("gold_answer", q.get("expected_answer", q.get("answer", "")))
                category = q.get("category", q.get("type", "general"))
                questions.append(AMBQuestion(
                    question_id=str(qid),
                    conversation_id=str(conv_id),
                    question=q.get("question", ""),
                    gold_answer=str(gold),
                    category=category,
                ))

        # Flat format: each item is a question with embedded conversation
        elif "question" in item:
            conv_id = str(item.get("conversation_id", item.get("id", "")))
            messages = item.get("conversation", item.get("messages", []))
            if conv_id and conv_id not in conversations:
                conversations[conv_id] = AMBConversation(
                    conversation_id=conv_id,
                    messages=messages if isinstance(messages, list) else [],
                )

            qid = item.get("question_id", item.get("id", ""))
            gold = item.get("gold_answer", item.get("expected_answer", item.get("answer", "")))
            category = item.get("category", item.get("type", "general"))
            questions.append(AMBQuestion(
                question_id=str(qid),
                conversation_id=str(conv_id),
                question=item.get("question", ""),
                gold_answer=str(gold),
                category=category,
            ))

    return list(conversations.values()), questions


# ---------------------------------------------------------------------------
# Preflight runner
# ---------------------------------------------------------------------------


def run_amb_preflight(
    *,
    dataset_path: str,
    limit: int = 5,
    reader: str = "none",
) -> dict:
    """Run an AMB preflight using the full MemContext pipeline.

    For each question:
    1. Create in-memory SQLite DB.
    2. Ingest conversation messages via on_new_turn.
    3. Backfill embeddings.
    4. Retrieve via retrieve_hybrid.
    5. Call the reader LLM (if configured).
    6. Score with score_answer.

    Returns a results dict with per-category accuracy and overall accuracy.
    """
    import os

    from evals.metrics import score_answer
    from evals.runner import ReaderMode, answer_question
    from memcontext.claims import get_turn
    from memcontext.extractors import (
        LLMExtractor,
        PassthroughExtractor,
        auto_extractor,
    )
    from memcontext.on_new_turn import on_new_turn
    from memcontext.retrieval import (
        EmbeddingClient,
        backfill_embeddings,
        retrieve_hybrid,
    )
    from memcontext.schema import Speaker, Turn, open_database

    # Use personal_assistant pack unless overridden
    if not os.environ.get("ACTIVE_PACK"):
        os.environ["ACTIVE_PACK"] = "personal_assistant"
        from memcontext.predicate_packs import active_pack

        active_pack.cache_clear()

    from memcontext.predicate_packs import active_pack as _get_pack
    pack = _get_pack()
    multi_valued = pack.multi_valued_predicates

    conversations, questions = load_amb_dataset(dataset_path)
    conv_map: dict[str, AMBConversation] = {
        c.conversation_id: c for c in conversations
    }

    questions = questions[:limit]

    reader_mode = ReaderMode(reader)
    embedding_client = EmbeddingClient()
    extractor = auto_extractor()

    # Group questions by conversation to avoid redundant ingestion
    questions_by_conv: dict[str, list[AMBQuestion]] = defaultdict(list)
    for q in questions:
        questions_by_conv[q.conversation_id].append(q)

    question_results: list[dict] = []

    for conv_id, conv_questions in questions_by_conv.items():
        conv = conv_map.get(conv_id)
        if conv is None:
            for q in conv_questions:
                question_results.append({
                    "question_id": q.question_id,
                    "category": q.category,
                    "gold_answer": q.gold_answer,
                    "error": f"Conversation {conv_id} not found",
                })
            continue

        conn = open_database(":memory:")
        conn.row_factory = sqlite3.Row

        session_id = f"amb_{conv_id}"

        # Parallel extraction if LLMExtractor, sequential otherwise
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from memcontext.claims import new_turn_id, now_ns

        all_messages: list[tuple[str, str]] = []
        for msg in conv.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content and content.strip():
                all_messages.append((role, content))

        def _extract_one(
            idx_role_text: tuple[int, str, str],
        ) -> tuple[int, list[dict]]:
            idx, role, text = idx_role_text
            sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
            t = Turn(
                turn_id=new_turn_id(),
                session_id=session_id,
                speaker=sp,
                text=text,
                ts=now_ns(),
                asr_confidence=None,
            )
            claims = extractor(t)
            return (
                idx,
                [
                    {
                        "subject": c.subject,
                        "predicate": c.predicate,
                        "value": c.value,
                        "confidence": c.confidence,
                    }
                    for c in claims
                ],
            )

        extracted_by_idx: dict[int, list[dict]] = {}
        work = [(i, role, text) for i, (role, text) in enumerate(all_messages)]

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

        # Sequential storage for supersession ordering
        ingested_turns = len(all_messages)
        claims_created = 0

        for i, (role, text) in enumerate(all_messages):
            claims_data = extracted_by_idx.get(i, [])
            if not claims_data:
                continue
            sp = Speaker.USER if role == "user" else Speaker.ASSISTANT
            pt = PassthroughExtractor(claims_data)
            result = on_new_turn(
                conn,
                session_id=session_id,
                speaker=sp,
                text=text,
                extractor=pt,
                multi_valued_predicates=multi_valued,
            )
            claims_created += len(result.created_claims)

        embedded_count = backfill_embeddings(
            conn, session_id, client=embedding_client
        )

        # Evaluate each question against this conversation's memory
        for q in conv_questions:
            top_claims = retrieve_hybrid(
                conn,
                session_id=session_id,
                query=q.question,
                top_k=50,
                embedding_client=embedding_client,
                weights=(0.7, 0.0, 0.0, 0.3),
            )

            # Build excerpts from source turns
            seen_turns: set[str] = set()
            excerpts: list[dict] = []
            for c, s in top_claims:
                if c.source_turn_id in seen_turns:
                    continue
                seen_turns.add(c.source_turn_id)
                turn = get_turn(conn, c.source_turn_id)
                if turn is None:
                    continue
                excerpts.append({
                    "text": turn.text,
                    "speaker": (
                        turn.speaker.value
                        if hasattr(turn.speaker, "value")
                        else str(turn.speaker)
                    ),
                    "claim_value": c.value,
                    "score": round(s, 4),
                })

            claims_list = [
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
                claims=claims_list,
                reader=reader_mode,
                excerpts=excerpts,
            )

            qr_entry: dict = {
                "question_id": q.question_id,
                "category": q.category,
                "gold_answer": q.gold_answer,
                "conversation_id": q.conversation_id,
                "ingested_turns": ingested_turns,
                "claims_created": claims_created,
                "embedded_count": embedded_count,
                "num_claims_retrieved": len(claims_list),
                **answer_result,
            }

            predicted = answer_result.get("predicted_answer")
            if predicted is not None:
                score = score_answer(
                    predicted=predicted,
                    gold=q.gold_answer,
                    question=q.question,
                    question_type=q.category,
                    question_id=q.question_id,
                )
                qr_entry["score"] = score
                qr_entry["correct"] = score >= 1.0

            question_results.append(qr_entry)

        conn.close()

    # Compute summary stats
    categories_seen = sorted({r["category"] for r in question_results})
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
        "categories_seen": categories_seen,
        "per_category_accuracy": {
            cat: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4)
                if v["total"]
                else 0,
            }
            for cat, v in per_cat.items()
        },
        "overall_accuracy": (
            round(sum(1 for r in scored if r.get("correct")) / len(scored), 4)
            if scored
            else None
        ),
        "questions": question_results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Memory Benchmark runner for MemContext"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to AMB dataset JSON file (PersonaMem32K or BEAM format)",
    )
    parser.add_argument(
        "--reader",
        default="none",
        choices=["none", "configured"],
        help="Reader mode: 'none' for retrieval only, 'configured' for LLM reader",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max number of questions to evaluate (default: 5)",
    )

    args = parser.parse_args()

    print(f"Running AMB preflight: dataset={args.dataset}, reader={args.reader}, limit={args.limit}")
    results = run_amb_preflight(
        dataset_path=args.dataset,
        limit=args.limit,
        reader=args.reader,
    )

    print(f"\nTotal questions: {results['total_questions']}")
    print(f"Scored questions: {results['scored_questions']}")
    if results["overall_accuracy"] is not None:
        print(f"Overall accuracy: {results['overall_accuracy']:.1%}")
    print(f"Categories: {', '.join(results['categories_seen'])}")
    print("\nPer-category accuracy:")
    for cat, stats in results["per_category_accuracy"].items():
        print(f"  {cat}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")

    # Dump full results as JSON
    output_path = Path(args.dataset).with_suffix(".results.json")
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
