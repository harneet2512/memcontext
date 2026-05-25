"""Instant raw-text ceiling check for LongMemEval-S.

Checks whether gold answer text exists in the raw conversation turns
(no extraction, no retrieval). This tells us the information availability
ceiling — if the answer isn't in the conversation, no pipeline can find it.

Usage:
    python evals/ceiling_rawtext.py --dataset data/longmemeval-s/data/longmemeval_s_cleaned.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _token_overlap(a: str, b: str) -> float:
    ta = set(a.strip().lower().split())
    tb = set(b.strip().lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def check_gold_in_turns(
    gold_answer: str,
    turns_text: list[str],
    threshold: float = 0.3,
) -> dict:
    gold_norm = _normalize(gold_answer)
    gold_tokens = set(gold_norm.split())

    exact_substring = any(gold_norm in _normalize(t) for t in turns_text)

    best_overlap = 0.0
    best_turn_idx = -1
    best_turn_text = ""
    for i, t in enumerate(turns_text):
        overlap = _token_overlap(gold_answer, t)
        if overlap > best_overlap:
            best_overlap = overlap
            best_turn_idx = i
            best_turn_text = t[:200]

    per_token_found = {}
    for token in gold_tokens:
        per_token_found[token] = any(
            token in _normalize(t) for t in turns_text
        )
    tokens_found_ratio = (
        sum(per_token_found.values()) / len(per_token_found)
        if per_token_found
        else 0.0
    )

    return {
        "exact_substring": exact_substring,
        "best_token_overlap": round(best_overlap, 4),
        "best_turn_idx": best_turn_idx,
        "best_turn_preview": best_turn_text,
        "tokens_found_ratio": round(tokens_found_ratio, 4),
        "above_threshold": best_overlap >= threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw-text ceiling check")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Filter to specific question_type values")
    args = parser.parse_args()

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    results_by_cat: dict[str, list[dict]] = defaultdict(list)

    for inst in data:
        qid = inst["question_id"]
        qtype = inst.get("question_type", "unknown")
        gold = str(inst.get("answer", ""))
        question = inst.get("question", "")

        if args.categories and qtype not in args.categories:
            continue

        all_turns: list[str] = []
        for session in inst.get("haystack_sessions", []):
            if isinstance(session, list):
                for turn in session:
                    content = turn.get("content", "") if isinstance(turn, dict) else ""
                    if content.strip():
                        all_turns.append(content)

        result = check_gold_in_turns(gold, all_turns, args.threshold)
        result["question_id"] = qid
        result["question_type"] = qtype
        result["question"] = question[:150]
        result["gold_answer"] = gold[:150]
        result["num_turns"] = len(all_turns)
        results_by_cat[qtype].append(result)

    print("=" * 80)
    print("RAW-TEXT CEILING CHECK — LongMemEval-S")
    print("=" * 80)
    print(f"Threshold: Jaccard >= {args.threshold}")
    print()

    overall_above = 0
    overall_total = 0
    overall_exact = 0

    for cat in sorted(results_by_cat.keys()):
        items = results_by_cat[cat]
        above = sum(1 for r in items if r["above_threshold"])
        exact = sum(1 for r in items if r["exact_substring"])
        total = len(items)
        overall_above += above
        overall_total += total
        overall_exact += exact

        print(f"--- {cat} ({total} questions) ---")
        print(f"  Gold found (Jaccard >= {args.threshold}): {above}/{total} ({100*above/total:.1f}%)")
        print(f"  Exact substring match:                    {exact}/{total} ({100*exact/total:.1f}%)")
        print(f"  Avg best token overlap:                   {sum(r['best_token_overlap'] for r in items)/total:.3f}")
        print(f"  Avg tokens-found ratio:                   {sum(r['tokens_found_ratio'] for r in items)/total:.3f}")

        misses = [r for r in items if not r["above_threshold"]]
        if misses:
            print(f"  --- MISSES ({len(misses)} questions where gold NOT found in turns) ---")
            for m in misses:
                print(f"    [{m['question_id']}] overlap={m['best_token_overlap']:.3f} tokens_found={m['tokens_found_ratio']:.3f}")
                print(f"      Q: {m['question']}")
                print(f"      Gold: {m['gold_answer']}")
                print()
        print()

    print("=" * 80)
    print(f"OVERALL: {overall_above}/{overall_total} ({100*overall_above/overall_total:.1f}%) gold answers found in raw turns")
    print(f"         {overall_exact}/{overall_total} ({100*overall_exact/overall_total:.1f}%) exact substring matches")
    print("=" * 80)
    print()
    print("INTERPRETATION:")
    print(f"  - {overall_above}/{overall_total} = raw information ceiling (max possible if extraction + retrieval were perfect)")
    print(f"  - Questions where gold is NOT found = extraction can never help (information not in conversation)")
    print(f"  - Gap between this ceiling and 88.4% = extraction + retrieval + reader losses")


if __name__ == "__main__":
    main()
