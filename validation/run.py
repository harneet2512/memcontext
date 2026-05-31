"""One-command runner for a validation block.

    python -m validation.run --block B --db validation.db     # MemContext attached
    python -m validation.run --block A --db validation.db     # native memory only

Block B seeds the task into MemContext (corrections via the typed correction
primitive) and captures the projection state at each probe. Block A captures the
raw transcript only. In both, the HOST MODEL answers the probes (it is the reader);
fill the 'answer' fields, then score with `python -m validation.score`.

`--self-check` fills Block B answers from the deterministic projection and scores
immediately — a pipeline smoke test that verifies the harness end-to-end WITHOUT
running the real experiment (the host model is not invoked).
"""
from __future__ import annotations

import argparse
import json
import os


def run(block: str, db: str, answers_path: str | None, self_check: bool) -> str:
    os.environ.setdefault("ACTIVE_PACK", "general")
    from memcontext.predicate_packs import active_pack
    active_pack.cache_clear()

    from memcontext.schema import open_database
    from validation.harness import ingest_block_b, projection_state, raw_transcript
    from validation.task import PROBES, TURNS

    block = block.upper()
    answers_path = answers_path or f"validation_answers_{block}.jsonl"

    conn = open_database(db)
    if block == "B":
        ingest_block_b(conn, TURNS)

    records: list[dict] = []
    for i, probe in enumerate(PROBES):
        rec: dict = {
            "probe_id": i,
            "block": block,
            "question": probe.question,
            "subject": probe.subject,
            "predicate": probe.predicate,
            "current_value": probe.current_value,
            "stale_values": list(probe.stale_values),
            "answer": None,
        }
        if block == "B":
            rec["projection"] = projection_state(conn, probe.subject, probe.predicate)
            if self_check:
                rec["answer"] = rec["projection"]["current_value"] or ""
        else:
            rec["context"] = raw_transcript(TURNS, probe.after_turn)
        records.append(rec)

    with open(answers_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    seeded = "yes" if block == "B" else "no (native memory only)"
    print(f"[validation] block={block}  seeded={seeded}  probes={len(records)}  -> {answers_path}")
    if block == "B":
        for r in records:
            print(f"  probe {r['probe_id']}: {r['question']}  "
                  f"[MemContext current={r['projection']['current_value']!r}]")
    print()

    if self_check and block == "B":
        from validation.score import score_records
        report = score_records(records)
        print("[validation] --self-check (pipeline smoke, NOT the experiment):")
        print(json.dumps({k: v for k, v in report.items() if k != "per_probe"}, indent=2))
    elif self_check:
        print("[validation] --self-check has no deterministic oracle for Block A "
              "(native memory) — the host model must answer the probes.")
    else:
        print(f"[validation] Answer each probe (the host model is the reader). For Block B you may "
              f"call memory_query/brain. Fill the 'answer' field in {answers_path}, then run:")
        print(f"             python -m validation.score {answers_path}")

    conn.close()
    return answers_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="validation.run", description="Run one validation block.")
    ap.add_argument("--block", required=True, choices=["A", "B", "a", "b"])
    ap.add_argument("--db", default="validation.db")
    ap.add_argument("--answers", default=None, help="answers JSONL path")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="Fill Block B answers from the projection and score (verifies the harness "
             "end-to-end; NOT the real experiment).",
    )
    args = ap.parse_args(argv)
    run(args.block, args.db, args.answers, args.self_check)


if __name__ == "__main__":
    main()
