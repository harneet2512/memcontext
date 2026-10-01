"""Injection-noise eval: does the hook stay quiet when memory is irrelevant, and on-topic when it is?

Eval harness, NOT product code. Deterministic, lexical mode, no LLM, no judge.

Every prompt and every edit in Claude Code goes through MemContext's hooks. A hook that
injects memory into unrelated work costs tokens and attention and can steer the agent
wrong; a hook that injects the wrong decisions into related work is worse. This eval
stores the 44 decision histories of datasets/stale_exposure.json exactly as
stale_exposure.py does, then sends hand-written prompts and tool calls
(datasets/injection_noise.json) through the real hook functions:

  UserPromptSubmit  http_server._prompt_context(prompt, None)
  PreToolUse        http_server._context_for_tool({"tool_name", "tool_input"}, None)

  run  --split dev|heldout   ingest, serve every case, score, write
                             results/injection_noise/<split>.json

Scores
  silence    unrelated prompt/tool call -> anything injected? how many chars/lines?
  precision  related prompt/tool call   -> injected at all? current value present?
             outdated value present? share of injected lines about the asked subject?
Value matching is stale_exposure.mentions (whole value, case-insensitive).

Examples (from the repo root):
  python evals/product/injection_noise.py run --split dev
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stale_exposure as se  # noqa: E402

DATASET = HERE / "datasets" / "injection_noise.json"
MEMORY_DATASET = HERE / "datasets" / "stale_exposure.json"
RESULTS = HERE / "results" / "injection_noise"
MAX_EXAMPLES = 4
TOP_TRIGGERS = 8


# ------------------------------------------------------------------ scoring ---
def context_entries(context: str | None) -> list[str]:
    """Injected entries: each "- " item after the header (a CONFLICT entry spans lines)."""
    if not context:
        return []
    return [e.strip() for e in ("\n" + context).split("\n- ")[1:] if e.strip()]


def entry_is_about(entry: str, subject_key: str) -> bool:
    """True if the entry (or any value line of a CONFLICT entry) is about subject_key."""
    prefix = subject_key + " /"
    for line in entry.splitlines():
        line = line.strip()
        head, _, rest = line.partition(". ")
        if head.isdigit():
            line = rest
        if line.lower().startswith(prefix.lower()):
            return True
    return False


def pct(k: int, n: int) -> float | None:
    return round(100.0 * k / n, 1) if n else None


def score_silence(cases: list[dict]) -> dict:
    """cases: {"context": str|None}. Lower injection on unrelated work is better."""
    injected = [c for c in cases if context_entries(c["context"])]
    chars = [len(c["context"]) for c in injected]
    lines = [len(context_entries(c["context"])) for c in injected]
    return {"n": len(cases), "injected": len(injected), "injected_pct": pct(len(injected), len(cases)),
            "chars_total": sum(chars), "chars_median_when_injected": statistics.median(chars) if chars else 0,
            "lines_total": sum(lines)}


def score_related(cases: list[dict]) -> dict:
    """cases: {"context", "subject_key", "current", "stale": [..]}."""
    n = len(cases)
    injected = cur = stale = own = total = 0
    for c in cases:
        entries = context_entries(c["context"])
        if not entries:
            continue
        injected += 1
        text = c["context"]
        cur += se.mentions(text, c["current"])
        stale += any(se.mentions(text, v) for v in c["stale"])
        own += sum(entry_is_about(e, c["subject_key"]) for e in entries)
        total += len(entries)
    return {"n": n, "injected": injected, "injected_pct": pct(injected, n),
            "current_present": cur, "current_present_pct": pct(cur, n),
            "stale_present": stale, "stale_present_pct": pct(stale, n),
            "on_subject_lines": own, "lines_total": total, "on_subject_pct": pct(own, total)}


def trigger_counts(pairs: list[tuple[set[str], set[str]]]) -> list[tuple[str, int]]:
    """Words shared by a query and an injected entry, most frequent first."""
    c: Counter[str] = Counter()
    for query_tokens, entry_tokens in pairs:
        c.update(query_tokens & entry_tokens)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))


# ------------------------------------------------------------------ running ---
def product_dirty(repo: Path) -> bool:
    try:
        out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--",
                              "memcontext", "predicate_packs"], capture_output=True, text=True, timeout=30)
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return True


def build_memory(mc, workdir: Path) -> tuple[Path, dict]:
    hist = json.loads(MEMORY_DATASET.read_text(encoding="utf-8"))["histories"]
    db = workdir / "memory.db"
    stats = se.ingest(mc, db, se.build_turns(hist, "per-change", False), hist)
    values = {mc.norm_subject(h["subject"]): h["values"] for h in hist}
    return db, {"stats": stats, "values": values, "n_histories": len(hist)}


def tool_context(mc, case: dict) -> tuple[str | None, set[str]]:
    hs = mc.http_server
    resp = hs._context_for_tool({"tool_name": case["tool_name"], "tool_input": case["tool_input"]}, None)
    ctx = (resp.get("hookSpecificOutput") or {}).get("additionalContext")
    kw = hs._extract_query_keywords(case["tool_name"], case["tool_input"]) or ""
    return ctx, set(kw.split())


def serve_all(mc, split: dict, values: dict) -> dict:
    hs = mc.http_server
    out: dict[str, list[dict]] = {k: [] for k in
                                  ("unrelated_prompts", "related_prompts", "unrelated_tools", "related_tools")}
    for p in split["unrelated_prompts"]:
        out["unrelated_prompts"].append({"query": p, "context": hs._prompt_context(p, None),
                                         "tokens": hs._content_tokens(p)})
    for t in split["unrelated_tools"]:
        ctx, toks = tool_context(mc, t)
        out["unrelated_tools"].append({"query": f"{t['tool_name']} {se_target(t)}", "context": ctx, "tokens": toks})
    for p in split["related_prompts"]:
        key = mc.norm_subject(p["subject"])
        out["related_prompts"].append({"query": p["prompt"], "context": hs._prompt_context(p["prompt"], None),
                                       "tokens": hs._content_tokens(p["prompt"]), "subject_key": key,
                                       "current": values[key][-1], "stale": values[key][:-1]})
    for t in split["related_tools"]:
        key = mc.norm_subject(t["subject"])
        ctx, toks = tool_context(mc, t)
        out["related_tools"].append({"query": f"{t['tool_name']} {se_target(t)}", "context": ctx, "tokens": toks,
                                     "subject_key": key, "current": values[key][-1], "stale": values[key][:-1]})
    return out


def se_target(case: dict) -> str:
    ti = case["tool_input"]
    return ti.get("file_path") or ti.get("command", "")[:60]


def triggers_for(mc, cases: list[dict]) -> list[tuple[str, int]]:
    hs = mc.http_server
    pairs = [(c["tokens"], hs._content_tokens(e)) for c in cases for e in context_entries(c["context"])]
    return trigger_counts(pairs)


# ---------------------------------------------------------------- reporting ---
def frac(k: int, n: int) -> str:
    return f"{k}/{n}"


def build_result(mc, served: dict, memory: dict, split: str, repo: Path, sha: str, secs: float) -> dict:
    up, ut = score_silence(served["unrelated_prompts"]), score_silence(served["unrelated_tools"])
    rp, rt = score_related(served["related_prompts"]), score_related(served["related_tools"])
    noise_triggers = triggers_for(mc, served["unrelated_prompts"] + served["unrelated_tools"])
    metrics = [
        {"name": "Unrelated prompts with memory injected", "value": frac(up["injected"], up["n"]),
         "pct": up["injected_pct"], "n": up["n"], "better": "lower",
         "definition": "UserPromptSubmit added any memory to a prompt whose task needs no stored decision"},
        {"name": "Chars injected into unrelated prompts (median when injected)",
         "value": str(up["chars_median_when_injected"]), "pct": None, "n": up["injected"], "better": "lower",
         "definition": "size of the injected block on noisy prompts"},
        {"name": "Unrelated edits/commands with memory injected", "value": frac(ut["injected"], ut["n"]),
         "pct": ut["injected_pct"], "n": ut["n"], "better": "lower",
         "definition": "PreToolUse added memory before an edit or command that needs no stored decision"},
        {"name": "Related prompts: current decision present", "value": frac(rp["current_present"], rp["n"]),
         "pct": rp["current_present_pct"], "n": rp["n"], "better": "higher",
         "definition": "the asked subject's current value appears in the injected block"},
        {"name": "Related prompts: outdated value present", "value": frac(rp["stale_present"], rp["n"]),
         "pct": rp["stale_present_pct"], "n": rp["n"], "better": "lower",
         "definition": "any superseded value of the asked subject appears in the injected block"},
        {"name": "Related prompts: injected lines on the asked subject",
         "value": frac(rp["on_subject_lines"], rp["lines_total"]), "pct": rp["on_subject_pct"],
         "n": rp["lines_total"], "better": "higher",
         "definition": "precision of the injected block: lines about the subject the task depends on"},
        {"name": "Related edits: current decision present", "value": frac(rt["current_present"], rt["n"]),
         "pct": rt["current_present_pct"], "n": rt["n"], "better": "higher",
         "definition": "right before the edit, the decision it depends on is in context"},
        {"name": "Related edits: injected lines on the asked subject",
         "value": frac(rt["on_subject_lines"], rt["lines_total"]), "pct": rt["on_subject_pct"],
         "n": rt["lines_total"], "better": "higher",
         "definition": "precision of the pre-edit block"},
    ]

    def case_rows(key: str, related: bool) -> list[list]:
        rows = []
        for c in served[key]:
            entries = context_entries(c["context"])
            row = [c["query"], len(entries), len(c["context"] or "")]
            if related:
                row += [c["subject_key"], "yes" if se.mentions(c["context"] or "", c["current"]) else "no",
                        sum(entry_is_about(e, c["subject_key"]) for e in entries)]
            else:
                row += ["; ".join(e.split(":", 1)[0] for e in entries)]
            rows.append(row)
        return rows

    worst = sorted(served["unrelated_prompts"] + served["unrelated_tools"],
                   key=lambda c: -len(c["context"] or ""))
    examples = [{"prompt": c["query"], "injected": c["context"]} for c in worst[:MAX_EXAMPLES - 1] if c["context"]]
    missed = [c for c in served["related_prompts"] if not se.mentions(c["context"] or "", c["current"])]
    if missed:
        examples.append({"prompt": missed[0]["query"] + f"  (needs {missed[0]['subject_key']})",
                         "injected": missed[0]["context"] or "(nothing injected)"})

    findings = []
    if up["injected"]:
        top = ", ".join(f"{w} ({k})" for w, k in noise_triggers[:5])
        findings.append(f"{up['injected']}/{up['n']} unrelated prompts got memory injected. The prompt hook "
                        f"injects on a single shared word when no stored claim shares two; the most frequent "
                        f"trigger words were: {top}. (Logic: http_server._prompt_context relevance floor.)")
    if ut["injected"]:
        findings.append(f"{ut['injected']}/{ut['n']} unrelated edits/commands got memory injected before the "
                        "tool ran; PreToolUse injects on any shared keyword. (Logic: http_server._tool_context.)")
    misses = rp["n"] - rp["current_present"]
    if misses:
        findings.append(f"{misses}/{rp['n']} related task prompts did not get the decision they depend on: "
                        "task wording rarely repeats the stored subject's words (lexical match only).")
    return {
        "eval": "injection_noise", "title": "Injection noise",
        "promise": "Memory helps where it is relevant and stays out of the way where it is not.",
        "question": "When a prompt or edit does not need any stored decision, does the hook stay quiet? "
                    "When it does, is the injected context about that decision and current?",
        "method": f"{memory['n_histories']} decision histories stored through the real API "
                  "(stale_exposure.json); hand-written prompts and Edit/Write/Bash payloads sent through "
                  "the real UserPromptSubmit and PreToolUse hook functions; lexical mode; "
                  "deterministic string scoring, no LLM judge.",
        "commit": sha, "product_code_dirty": product_dirty(repo),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": split, "deterministic": True, "seconds": round(secs, 1),
        "headline": {"label": "Unrelated prompts that got memory injected",
                     "value": frac(up["injected"], up["n"]), "pct": up["injected_pct"], "better": "lower"},
        "metrics": metrics,
        "tables": [
            {"title": "Trigger words behind noisy injections", "columns": ["word", "injected entries"],
             "rows": [[w, k] for w, k in noise_triggers[:TOP_TRIGGERS]]},
            {"title": "Unrelated prompts", "columns": ["prompt", "lines", "chars", "injected subjects"],
             "rows": case_rows("unrelated_prompts", False)},
            {"title": "Unrelated edits and commands", "columns": ["tool call", "lines", "chars", "injected subjects"],
             "rows": case_rows("unrelated_tools", False)},
            {"title": "Related prompts", "columns": ["prompt", "lines", "chars", "needs", "current present",
                                                     "lines on subject"],
             "rows": case_rows("related_prompts", True)},
            {"title": "Related edits", "columns": ["tool call", "lines", "chars", "needs", "current present",
                                                   "lines on subject"],
             "rows": case_rows("related_tools", True)},
        ],
        "examples": examples[:MAX_EXAMPLES],
        "limits": [
            "Hand-written prompts and tool calls by one author against a synthetic 44-decision memory.",
            "Measures what the hooks inject, not whether the model is distracted by it.",
            "Lexical mode only; 'unrelated' is labelled from intent, and some prompts deliberately share an "
            "everyday word (timeout, error, format, logging) with a stored subject.",
            "PreToolUse has a 150-200 ms time budget; on this small memory it never trips.",
        ],
        "findings": findings,
        "raw": {"unrelated_prompts": up, "unrelated_tools": ut, "related_prompts": rp, "related_tools": rt,
                "ingest": memory["stats"]},
    }


def print_summary(r: dict) -> None:
    print(f"\nInjection noise ({r['split']}) at {r['commit'][:10]}"
          f"{' + uncommitted product changes' if r['product_code_dirty'] else ''}, {r['seconds']} s")
    for m in r["metrics"]:
        print(f"  {m['name']:<62} {m['value']:>7}" + (f"  ({m['pct']}%)" if m["pct"] is not None else ""))
    for f in r["findings"]:
        print("  - " + f)


def cmd_run(a) -> None:
    split_name = a.split
    data = json.loads(Path(a.dataset).read_text(encoding="utf-8"))[split_name]
    repo = Path(a.repo)
    t0 = time.perf_counter()
    sha = se.repo_sha(repo)
    mc = se.load_memcontext(repo, "lexical")
    workdir = Path(tempfile.mkdtemp(prefix="mc-injection-noise-"))
    try:
        db, memory = build_memory(mc, workdir)
        mc.http_server.init_db(str(db))
        served = serve_all(mc, data, memory["values"])
        result = build_result(mc, served, memory, split_name, repo, sha, time.perf_counter() - t0)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{split_name}.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(result)
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ingest the memory, serve every case through the hooks, score")
    r.add_argument("--split", choices=["dev", "heldout"], default="dev")
    r.add_argument("--repo", default=str(se.REPO), help="MemContext checkout under test")
    r.add_argument("--dataset", default=str(DATASET))
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
