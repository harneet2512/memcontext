"""Official LongMemEval benchmark runner.

Outputs hypothesis JSONL for scoring by the official evaluate_qa.py script.
Does NOT judge -- judging is done externally.

Usage:
    python evals/benchmark/run_official.py \
        --dataset data/longmemeval-s/data/longmemeval_s_cleaned.json \
        --output results/hypothesis.jsonl \
        --reader gpt-4o-mini

Config (weights, models, top_k, git commit hash) is recorded to a JSON
sidecar file alongside the hypothesis JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evals.longmemeval import load_dataset, LongMemEvalQuestion, LongMemEvalSession
from evals.runner import ReaderMode, answer_question
from memcontext.claims import insert_turn, new_turn_id, now_ns
from memcontext.extractors import PassthroughExtractor, auto_extractor
from memcontext.on_new_turn import on_new_turn
from memcontext.retrieval import (
    EmbeddingClient,
    backfill_embeddings,
    backfill_episode_embeddings,
    classify_query_depth,
    retrieve_memory,
)
from memcontext.schema import Speaker, Turn, open_database


def _extract_all_parallel(turns: list, workers: int) -> list:
    """Extract claims for many turns concurrently.

    Extraction is independent per turn (the runner sets no prior-turn context),
    so the slow per-turn LLM call parallelizes cleanly. Each worker thread gets
    its own extractor instance (the extractor holds mutable per-call state, so it
    is not shared across threads). Returns a list aligned with ``turns``.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    _tl = threading.local()

    def _extract(t: Turn) -> list:
        e = getattr(_tl, "ext", None)
        if e is None:
            e = auto_extractor()
            _tl.ext = e
        try:
            return e(t)
        except Exception:  # one bad turn must not kill the run
            return []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_extract, turns))


def _git_commit_hash() -> str:
    """Return the current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


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


_CATEGORY_RETRIEVAL_CONFIG: dict[str, dict] = {
    "cross_session_user_fact": {
        "weights": (0.5, 0.2, 0.0, 0.3),
        "top_k": 100,
    },
    "single_session_preference": {
        "weights": (0.5, 0.2, 0.0, 0.3),
        "top_k": 60,
    },
}


def run_benchmark(
    *,
    dataset_path: str,
    output_path: str,
    reader: str = "gpt-4o-mini",
    top_k: int = 50,
    weights: tuple[float, ...] = (0.7, 0.0, 0.0, 0.3),
) -> None:
    """Run the full benchmark and write hypothesis JSONL + config sidecar."""
    # Use personal_assistant pack unless overridden
    if not os.environ.get("ACTIVE_PACK"):
        os.environ["ACTIVE_PACK"] = "personal_assistant"
        from memcontext.predicate_packs import active_pack

        active_pack.cache_clear()

    sessions, questions = load_dataset(dataset_path)
    session_map: dict[str, LongMemEvalSession] = {s.session_id: s for s in sessions}

    embedding_client = EmbeddingClient()
    reader_mode = ReaderMode(reader) if reader in ("none", "configured") else ReaderMode.CONFIGURED

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    config_path = output_file.with_suffix(".config.json")

    # Record config BEFORE the run
    config = {
        "dataset_path": dataset_path,
        "reader": reader,
        "top_k": top_k,
        "weights": list(weights),
        "git_commit": _git_commit_hash(),
        "embedding_model": embedding_client.model_version,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_questions": len(questions),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    with output_file.open("w", encoding="utf-8") as f_out:
        for qi, q in enumerate(questions):
            conn = open_database(":memory:")
            conn.row_factory = sqlite3.Row

            unified_sid = f"haystack_{q.question_id}"

            # Collect all turns with session dates
            all_turns: list[tuple[str, str, str]] = []
            for sid in q.session_ids:
                sess = session_map.get(sid)
                if sess is None:
                    continue
                for turn_data in sess.turns:
                    role = turn_data.get("role", "user")
                    content = turn_data.get("content", "")
                    if content and content.strip():
                        all_turns.append((role, content, sess.date))

            # Extract (parallel, slow per-turn LLM) then ingest (ordered, local).
            # Every turn is ingested as an episode even when it yields no facts,
            # so the Tier-1 episode floor of the unified two-tier retrieval is
            # populated (not just turns that produced claims).
            turn_session_date: dict[str, str] = {}
            built = [
                (Speaker.USER if role == "user" else Speaker.ASSISTANT, text, sess_date)
                for role, text, sess_date in all_turns
            ]
            probe_turns = [
                Turn(
                    turn_id=new_turn_id(), session_id=unified_sid, speaker=sp,
                    text=text, ts=now_ns(), asr_confidence=None,
                )
                for sp, text, _sd in built
            ]
            workers = int(os.environ.get("MEMCONTEXT_EXTRACT_WORKERS", "64"))
            claims_per_turn = _extract_all_parallel(probe_turns, workers)

            for (sp, text, sess_date), claims_data in zip(built, claims_per_turn):
                pt = PassthroughExtractor(
                    [
                        {
                            "subject": c.subject,
                            "predicate": c.predicate,
                            "value": c.value,
                            "confidence": c.confidence,
                        }
                        for c in (claims_data or [])
                    ]
                )
                result = on_new_turn(
                    conn,
                    session_id=unified_sid,
                    speaker=sp,
                    text=text,
                    extractor=pt,
                )
                if result.turn is not None:
                    turn_session_date[result.turn.turn_id] = sess_date

            # Embed claims
            backfill_embeddings(conn, unified_sid, client=embedding_client)
            # Episode (turn) embeddings too — without these the Tier-1 episode
            # channel of the unified two-tier retrieval is BM25-only (semantic dead).
            backfill_episode_embeddings(conn, unified_sid, client=embedding_client)

            cat_config = _CATEGORY_RETRIEVAL_CONFIG.get(q.category, {})
            q_top_k = cat_config.get("top_k", top_k)

            _, depth_k = classify_query_depth(q.question)
            q_top_k = max(q_top_k, depth_k)

            # Unified two-tier retrieval (facts + episodes, RRF-fused) — the
            # shipping path (retrieve_memory). Per-category weight tuning is
            # intentionally dropped: retrieve_memory uses the product's fixed
            # fusion, so this measures shipping behaviour honestly.
            hits = retrieve_memory(
                conn,
                session_id=unified_sid,
                query=q.question,
                top_k=q_top_k,
                embedding_client=embedding_client,
            )

            # Build excerpts for reader: the source turn behind each hit (fact or
            # episode), deduped by turn. This is the reader's actual context.
            from memcontext.claims import get_turn

            seen_turns: set[str] = set()
            excerpts: list[dict] = []
            for h, s in hits:
                if h.source_turn_id in seen_turns:
                    continue
                seen_turns.add(h.source_turn_id)
                turn = get_turn(conn, h.source_turn_id)
                if turn is None:
                    continue
                sess_date = turn_session_date.get(h.source_turn_id, "")
                offset = (
                    _relative_offset(sess_date, q.question_date)
                    if sess_date and q.question_date
                    else ""
                )
                excerpts.append(
                    {
                        "text": turn.text,
                        "speaker": (
                            turn.speaker.value
                            if hasattr(turn.speaker, "value")
                            else str(turn.speaker)
                        ),
                        "session_date": sess_date,
                        "relative_offset": offset,
                        "claim_value": h.text,
                        "score": round(s, 4),
                    }
                )

            claims = [
                {
                    "claim_id": h.id,
                    "kind": h.kind,
                    "value": h.text,
                    "score": round(s, 4),
                }
                for h, s in hits
            ]

            answer_result = answer_question(
                question=q.question,
                category=q.category,
                claims=claims,
                reader=reader_mode,
                question_date=q.question_date,
                excerpts=excerpts,
            )

            predicted = answer_result.get("predicted_answer", "")

            # Write hypothesis line
            hypothesis_line = {
                "question_id": q.question_id,
                "hypothesis": predicted or "",
            }
            f_out.write(json.dumps(hypothesis_line, ensure_ascii=False) + "\n")

            conn.close()

            if (qi + 1) % 50 == 0:
                print(f"  Processed {qi + 1}/{len(questions)} questions")

    # Update config with completion time
    config["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Hypothesis written to {output_path}")
    print(f"Config written to {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Official LongMemEval benchmark runner for MemContext"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to LongMemEval dataset JSON file or directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output hypothesis JSONL file",
    )
    parser.add_argument(
        "--reader",
        default="gpt-4o-mini",
        help="Reader model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of claims to retrieve per question (default: 50)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0.7,0.0,0.0,0.3",
        help="Retrieval weights: semantic,entity,temporal,BM25 (default: 0.7,0.0,0.0,0.3)",
    )
    args = parser.parse_args()

    weights = tuple(float(x.strip()) for x in args.weights.split(","))

    run_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        reader=args.reader,
        top_k=args.top_k,
        weights=weights,
    )


if __name__ == "__main__":
    main()
