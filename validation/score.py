"""Forgetting-aware scorer for the validation experiment.

For each probe, the host model's free-text answer is scored against the known
correction points: using the CURRENT value is rewarded, using a SUPERSEDED value
is penalized. This mirrors forgetting-aware memory accuracy — a system that
serves stale facts is worse than one that abstains.
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def score_probe(answer: str, current_value: str, stale_values: list[str]) -> dict:
    """Score one answer. Returns verdict + points in [0,1]."""
    a = (answer or "").lower()
    used_current = current_value.lower() in a
    used_stale = any(s.lower() in a for s in stale_values)

    if used_current and not used_stale:
        verdict, points = "current", 1.0      # reward: used the live value
    elif used_stale and not used_current:
        verdict, points = "stale", 0.0        # penalize: relied on an invalidated fact
    elif used_current and used_stale:
        verdict, points = "ambiguous", 0.5    # hedged — mentioned both
    else:
        verdict, points = "missing", 0.0      # neither value present

    return {
        "verdict": verdict,
        "points": points,
        "used_current": used_current,
        "used_stale": used_stale,
    }


def score_records(records: list[dict]) -> dict:
    """Score a list of probe records (each with an 'answer' filled in)."""
    scored = []
    for r in records:
        s = score_probe(r.get("answer", ""), r["current_value"], list(r["stale_values"]))
        # MemContext (Block B) can cite a source if the projection carried provenance.
        mem_traceable = bool((r.get("projection") or {}).get("source_turn_id"))
        scored.append({
            "probe_id": r.get("probe_id"),
            "question": r["question"],
            "answer": r.get("answer", ""),
            "current_value": r["current_value"],
            **s,
            "mem_traceable": mem_traceable,
        })

    n = len(scored) or 1
    verdicts = Counter(s["verdict"] for s in scored)
    return {
        "block": records[0].get("block") if records else None,
        "n_probes": len(scored),
        "accuracy": round(sum(s["points"] for s in scored) / n, 4),
        "stale_rate": round(verdicts["stale"] / n, 4),
        "verdicts": dict(verdicts),
        "traceable": round(sum(1 for s in scored if s["mem_traceable"]) / n, 4),
        "per_probe": scored,
    }


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m validation.score <answers.jsonl>", file=sys.stderr)
        raise SystemExit(2)
    records = [json.loads(line) for line in open(argv[0], encoding="utf-8") if line.strip()]
    report = score_records(records)
    print(json.dumps({k: v for k, v in report.items() if k != "per_probe"}, indent=2))
    print("\nPer-probe:")
    for s in report["per_probe"]:
        flag = "OK " if s["verdict"] == "current" else s["verdict"].upper()
        print(f"  [{flag}] {s['question']}  answer={s['answer']!r}  (current={s['current_value']!r})")


if __name__ == "__main__":
    main()
