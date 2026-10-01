"""Counterfactual for claude_code_recall: the same real Claude Code sessions, three ways.

Arms (same scenarios, prompts, model, answer suffix and scoring as claude_code_recall):
  self_notes  Claude's own notes, no memory tool: no MCP server, no hooks, a plain CLAUDE.md.
              Claude keeps whatever it writes (README, CLAUDE.md, docs), which is what an
              agent without a memory tool actually does, so this is the honest competitor.
  notes       a hand-kept decision log: after every user decision statement (capture /
              change steps) the harness appends that statement, dated, to DECISIONS.md;
              CLAUDE.md tells the agent to read it and that the newest entry wins. Old entries
              are never deleted. Claude also keeps whatever it writes itself.
  memcontext  exactly what claude_code_recall runs (serve-http hooks + stdio MCP server).

Each step is a NEW headless ``claude -p`` session in a throwaway project under --workdir
(never the user's repo, never port 8100). Scenarios run step-major: step i runs in every arm
before step i+1, so a budget stop cuts all arms at the same step. Scoring is
claude_code_recall's own deterministic scoring; tokens, USD and latency come from each
session. Layer attribution (plumbing / capture / ...) describes MemContext wiring, so it is
only computed for the memcontext arm.

Answer sources: injection (hook context), tool (a memcontext tool result), file (a file the
session read), claude_md (the project CLAUDE.md, which Claude Code loads into every session
without a tool call), none.

Diagnostic, not a benchmark: n is small, one model, one run.

    python evals/product/recall_ab.py --model haiku --max-cost-usd 3 --split dev \
        --dataset long_horizon=evals/product/datasets/recall_long_horizon.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import claude_code_recall as ccr  # noqa: E402

ARMS = ("self_notes", "notes", "memcontext")
NULL_ARMS = frozenset({"self_notes", "notes"})
ARM_LABELS = {"self_notes": "Claude's own notes (no memory tool)",
              "notes": "Decision log kept by the harness (DECISIONS.md)",
              "memcontext": "MemContext"}
LEGACY_ARM_NAMES = {"none": "self_notes"}
SOURCES = ("injection", "tool", "file", "claude_md", "none")
SET_LABELS = {"short": "short (2-5 sessions, one or two topics)",
              "long_horizon": "long-horizon (15 decisions, 28 sessions before the questions)"}
DECISION_KINDS = frozenset({"capture", "change"})
OUT = HERE / "results" / "recall_ab" / "latest.json"
DEFAULT_WORKDIR = Path(os.environ.get("TEMP", "/tmp")) / "memcontext_recall_ab"
NOTES_FILE = "DECISIONS.md"
NOTES_START = date(2026, 9, 1)  # step i is logged on NOTES_START + i days: order is explicit
ROUND_RESERVE_USD = 0.25  # stop before a step round when less than this is left (3 sessions)

SELF_NOTES_CLAUDE_MD = "# Project\n\nWork in this repository as the user asks.\n"
NOTES_CLAUDE_MD = f"""\
# Project decisions

Project decisions are logged in `{NOTES_FILE}`, one dated entry per decision, oldest first.
Before answering a question about a project decision or writing code that depends on one,
read `{NOTES_FILE}`. When entries about the same topic conflict, the newest entry is the
current decision.
"""
NOTES_HEADER = "# Decisions\n\nAppend-only log, oldest first. The newest entry on a topic wins.\n"
NO_DB = {"current_live": False, "stale_live": False, "also_live_missing": [], "ok": False}


# ------------------------------------------------------------------ arm setup ---
def write_arm_files(project: Path, arm: str) -> None:
    """CLAUDE.md, DECISIONS.md and isolation settings for an arm without MemContext."""
    if arm not in NULL_ARMS:
        raise ValueError(f"write_arm_files is for the {sorted(NULL_ARMS)} arms, not {arm!r}")
    (project / "CLAUDE.md").write_text(SELF_NOTES_CLAUDE_MD if arm == "self_notes" else NOTES_CLAUDE_MD,
                                       encoding="utf-8")
    if arm == "notes":
        (project / NOTES_FILE).write_text(NOTES_HEADER, encoding="utf-8")
    local = {
        "permissions": {"allow": ["Write", "Edit", "Read"]},
        # Same isolation as claude_code_recall: no outside CLAUDE.md, no built-in auto-memory,
        # no MCP servers from parent dirs. No hooks and no memcontext server: that is the arm.
        "claudeMdExcludes": ccr._claude_md_excludes(project),
        "disabledMcpjsonServers": ccr._ancestor_mcp_servers(project),
        "autoMemoryEnabled": False,
    }
    (project / ".claude").mkdir(exist_ok=True)
    (project / ".claude" / "settings.local.json").write_text(json.dumps(local, indent=2),
                                                            encoding="utf-8")


def append_decision(project: Path, statement: str, step_index: int) -> None:
    """What a disciplined person does: log the user's own words, dated, never edit old ones."""
    day = NOTES_START + timedelta(days=step_index)
    with open(project / NOTES_FILE, "a", encoding="utf-8") as fh:
        fh.write(f"\n## {day.isoformat()}\n\n{statement.strip()}\n")


def safe_dirname(*parts: str) -> str:
    """A path segment that cannot start a backslash escape the model may "helpfully" decode.

    Seen in a real run: a project under a "none" directory was written to a sibling "one"
    directory (backslash-n read as a newline escape), so the act step's file landed outside
    the project and scored as missing.
    Every segment starts with "p", which is not an escape in JSON, Python or C."""
    return "p-" + "-".join(parts)


class NullStack:
    """A scenario project with no memory wiring; the Stack interface run_session relies on."""

    def __init__(self, root: Path, spec: dict, arm: str):
        self.root, self.project = root, root / "project"
        self.token, self.serve_log = "no-server", root / "no-server.log"
        self.serve_log.write_text("", encoding="utf-8")
        ccr._write_project(self.project, spec)
        write_arm_files(self.project, arm)

    def ensure_alive(self) -> bool:
        return False

    def log_size(self) -> int:
        return 0

    def stop(self) -> None:
        return None


# -------------------------------------------------------------------- running ---
def result_usage(transcript: Path) -> dict:
    """Token usage from the session's stream-json result event."""
    usage: dict = {}
    if transcript.exists():
        for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                usage = ev.get("usage") or {}
    keys = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
    out = {k: int(usage.get(k) or 0) for k in keys}
    out["total_tokens"] = sum(out.values())
    return out


def attribute_source(source: str, claude_md_before: str, pats: ccr.Patterns, current: str | None) -> str:
    """CLAUDE.md is in every session's context without a tool call; credit it when nothing
    the session saw explains the current value."""
    if source == "none" and pats.hit(current, claude_md_before):
        return "claude_md"
    return source


def run_null_step(i: int, step: dict, stack: NullStack, ctx: dict) -> dict:
    """claude_code_recall.run_step without MemContext: no DB state, no layer attribution."""
    pats = ctx["pats"]
    prompt = step["prompt"] + (ccr.ANSWER_SUFFIX if step["kind"] in ccr.SCORED_KINDS else "")
    rec = ccr.run_session(ctx["claude"], stack, prompt, model=ctx["model"],
                          max_turns=step.get("max_turns", ccr.MAX_TURNS[step["kind"]]),
                          budget=ctx["budget"].session_cap(),
                          out_base=stack.root / f"step{i:02d}_{step['kind']}")
    ctx["budget"].charge(rec["cost_usd"])
    out = {"index": i, "kind": step["kind"], "prompt": prompt,
           "expect": {k: step.get(k) for k in ("current", "stale", "also_live", "file", "var")}, **rec}
    out["evidence"] = ccr.evidence(out, pats, step)
    if step["kind"] in ccr.SCORED_KINDS:
        out["score"] = ccr.score_step(rec, step, pats, stack.project)
    out["db_after"] = dict(NO_DB)
    return out


def strip_memcontext_attribution(steps: list[dict]) -> list[dict]:
    """Arms without MemContext have no hooks or DB, so layer attribution is meaningless there
    (it would blame "serve-http never answered" on an arm that has no serve-http)."""
    return [{k: v for k, v in s.items() if k != "attribution"} for s in steps]


def run_arm_step(arm: str, i: int, step: dict, sc: dict, stack, ctx: dict) -> dict:
    claude_md = stack.project / "CLAUDE.md"
    before = claude_md.read_text(encoding="utf-8", errors="replace") if claude_md.exists() else ""
    out = (ccr.run_step(i, step, sc, stack, ctx) if arm == "memcontext"
           else run_null_step(i, step, stack, ctx))
    out["usage"] = result_usage(Path(out["transcript"]))
    out["evidence"]["source"] = attribute_source(out["evidence"]["source"], before, ctx["pats"],
                                                 step.get("current"))
    out["claude_md_chars"] = len(before)
    if arm == "notes" and step["kind"] in DECISION_KINDS:
        append_decision(stack.project, step["prompt"], i)
    if arm != "memcontext":
        _print_null_step(arm, sc["id"], out)
    return out


def _print_null_step(arm: str, sid: str, s: dict) -> None:
    sc = s.get("score")
    verdict = (f"answer={sc['answer']}" + (f" action={sc['action']}" if "action" in sc else "")
               if sc else "unscored")
    cost = f"${s['cost_usd']:.3f}" if s["cost_usd"] is not None else "$?"
    print(f"  ({arm}) [{sid} #{s['index']} {s['kind']:<7}] {verdict:<34} "
          f"src={s['evidence']['source']:<9} {cost} {s['wall_s']}s", flush=True)


def _make_stack(arm: str, root: Path, sc: dict, ctx: dict):
    if arm == "memcontext":
        return ccr.Stack(root, sc["project"], ctx["cli"], ctx["embed"])
    return NullStack(root, sc["project"], arm)


def run_scenario_all_arms(sc: dict, arms: list[str], run_dir: Path, ctx: dict, keep: bool) -> dict:
    """Step-major: step i in every arm, then step i+1. Each arm has its own project and state."""
    results, stacks, actx = {}, {}, {}
    for arm in arms:
        root = run_dir / safe_dirname(arm, sc["id"])
        root.mkdir(parents=True, exist_ok=True)
        results[arm] = {"id": sc["id"], "arm": arm, "split": sc.get("split"), "set": sc.get("set"),
                        "steps": []}
        try:
            stacks[arm] = _make_stack(arm, root, sc, ctx)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            results[arm]["setup_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [{arm}/{sc['id']}] SETUP FAILED: {results[arm]['setup_error']}", flush=True)
            continue
        actx[arm] = {**ctx, "pats": ccr.Patterns(sc["values"]), "snapshot": [], "upstream": None}
    try:
        for i, step in enumerate(sc["steps"]):
            left = ctx["budget"].limit - ctx["budget"].spent
            if left < ROUND_RESERVE_USD:
                for arm in stacks:
                    results[arm]["aborted"] = f"budget: ${left:.2f} left before step {i}"
                print(f"  budget: ${left:.2f} left; stopping {sc['id']} before step {i} in every arm",
                      flush=True)
                break
            for arm in stacks:
                results[arm]["steps"].append(run_arm_step(arm, i, step, sc, stacks[arm], actx[arm]))
    finally:
        for stack in stacks.values():
            stack.stop()
            if not keep:
                shutil.rmtree(stack.project, ignore_errors=True)
                for p in stack.root.glob("*.db*"):
                    p.unlink(missing_ok=True)
    return results


# ------------------------------------------------------------------- scoring ---
def _scored(results: list[dict]) -> list[dict]:
    return [s for sc in results for s in sc["steps"] if "score" in s]


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def arm_summary(results: list[dict]) -> dict:
    """claude_code_recall's own summary, plus tokens, latency, sources and wrong steps."""
    summ = ccr.summarize(results)
    steps = [s for sc in results for s in sc["steps"]]
    scored = _scored(results)
    toks = [s["usage"]["total_tokens"] for s in steps if s.get("usage")]
    summ["mean_tokens"] = round(sum(toks) / len(toks)) if toks else None
    summ["mean_wall_s"] = _mean([s["wall_s"] for s in steps if s.get("wall_s") is not None])
    summ["source_counts"] = {k: sum(s["evidence"]["source"] == k for s in scored) for k in SOURCES}
    summ["wrong_steps"] = {"n": len(scored), "hits": sum(not s["score"]["ok"] for s in scored)}
    return summ


def _fmt(r: dict) -> str:
    return f"{r['hits']}/{r['n']}" if r.get("n") else "n/a"


def _pct(r: dict) -> float | None:
    return round(100.0 * r["hits"] / r["n"], 1) if r.get("n") else None


def _sources(s: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in s["source_counts"].items() if v) or "-"


def pick_examples(by_arm: dict[str, list[dict]], limit: int = 3) -> list[dict]:
    """Steps a baseline arm got wrong next to MemContext on the same step, then MemContext's
    own misses (so the examples show failures on both sides)."""
    mc = {(sc["id"], s["index"]): s for sc in by_arm.get("memcontext", []) for s in sc["steps"]}
    out: list[dict] = []

    def add(arm: str, sid: str, s: dict) -> None:
        sc_ = s["score"]
        out.append({"scenario": sid, "step": f"#{s['index']} {s['kind']}", "arm": arm,
                    "verdict": sc_["answer"] + (f" / action {sc_['action']}" if "action" in sc_ else ""),
                    "source": s.get("evidence", {}).get("source"),
                    "answer_excerpt": (s.get("final") or "")[-400:]})

    for arm in ("notes", "self_notes"):
        for sc in by_arm.get(arm, []):
            for s in sc["steps"]:
                if "score" not in s or s["score"]["ok"] or len(out) >= 2 * limit:
                    continue
                add(arm, sc["id"], s)
                twin = mc.get((sc["id"], s["index"]))
                if twin is not None and "score" in twin:
                    add("memcontext", sc["id"], twin)
    for sc in by_arm.get("memcontext", []):
        for s in sc["steps"]:
            if "score" in s and not s["score"]["ok"] and len(out) < 3 * limit and not any(
                    e["arm"] == "memcontext" and e["scenario"] == sc["id"] and e["step"].startswith(f"#{s['index']} ")
                    for e in out):
                add("memcontext", sc["id"], s)
    return out


def findings(per_set: dict[str, dict[str, dict]]) -> list[str]:
    """Plain statements of who won on each scenario set, including when MemContext did not."""
    out = []
    for set_name, summ in per_set.items():
        if "memcontext" not in summ:
            continue
        mc = summ["memcontext"]["wrong_steps"]
        others = {a: s["wrong_steps"] for a, s in summ.items() if a != "memcontext" and s["wrong_steps"]["n"]}
        if not mc["n"] or not others:
            continue
        label = SET_LABELS.get(set_name, set_name)
        better = [a for a, w in others.items() if mc["hits"] < w["hits"]]
        not_better = [a for a in others if a not in better]
        detail = "; ".join(f"{ARM_LABELS[a]} {_fmt(w)}" for a, w in others.items())
        if not_better:
            out.append(f"{label}: MemContext shows no advantage over "
                       f"{' or '.join(ARM_LABELS[a] for a in not_better)} "
                       f"(wrong steps: MemContext {_fmt(mc)}; {detail}).")
        else:
            out.append(f"{label}: MemContext had fewer wrong steps than every baseline "
                       f"(MemContext {_fmt(mc)}; {detail}). One run, small n: treat as a signal, "
                       "not a measured rate.")
    return out


def _code_under_test() -> dict:
    """The memcontext package the arms actually ran (sys.executable's install)."""
    try:
        import memcontext
        root = Path(memcontext.__file__).resolve().parents[1]
    except ImportError:
        return {"path": None, "commit": "unknown", "dirty": None}
    dirty = ccr._cmd_out(["git", "status", "--porcelain", "--", "memcontext", "predicate_packs"], root)
    return {"path": ccr._posix(root), "commit": ccr._cmd_out(["git", "rev-parse", "HEAD"], root),
            "dirty": dirty not in ("unknown", "")}


def _by_set(by_arm: dict[str, list[dict]]) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {}
    for arm, res in by_arm.items():
        for sc in res:
            out.setdefault(sc.get("set") or "short", {}).setdefault(arm, []).append(sc)
    return out


def _arm_row(arm: str, s: dict) -> list:
    return [ARM_LABELS.get(arm, arm), _fmt(s["wrong_steps"]), _fmt(s["current_recall"]),
            _fmt(s["stale_answer"]), _fmt(s["action_correct"]), _fmt(s["stale_action"]),
            _sources(s), s["sessions"], s["mean_tokens"], s["mean_cost_usd"], s["mean_wall_s"]]


def build_report(by_arm: dict[str, list[dict]], meta: dict) -> dict:
    by_arm = {a: by_arm[a] for a in ARMS if a in by_arm}
    summ = {arm: arm_summary(res) for arm, res in by_arm.items()}
    per_set = {name: {arm: arm_summary(res) for arm, res in arms.items()}
               for name, arms in sorted(_by_set(by_arm).items(), key=lambda kv: kv[0] != "long_horizon")}
    for group in (summ, *per_set.values()):
        for arm, s in group.items():
            if arm != "memcontext":  # layer attribution describes MemContext wiring only
                s.pop("failures_by_layer", None)
                s.pop("state_steps_db_ok", None)
    head_set = "long_horizon" if "long_horizon" in per_set else next(iter(per_set), None)
    head = per_set.get(head_set, summ)
    wrong = {arm: s["wrong_steps"] for arm, s in head.items()}
    arm_cols = ["arm", "wrong steps", "current recall", "stale answers", "correct actions",
                "stale actions", "answer sources", "sessions", "mean tokens/session",
                "mean $/session", "mean s/session"]
    metrics = []
    for set_name, group in per_set.items():
        for arm, s in group.items():
            for key, name, better in (("wrong_steps", "wrong steps", "lower"),
                                      ("stale_answer", "stale answers", "lower"),
                                      ("stale_action", "stale actions", "lower")):
                r = s[key]
                metrics.append({"name": f"{set_name} / {ARM_LABELS[arm]}: {name}", "value": _fmt(r),
                                "pct": _pct(r), "n": r["n"], "definition": METRIC_DEFS[key],
                                "better": better})
    code = meta["code_under_test"]
    return {
        "eval": "recall_ab", "title": "Real Claude Code sessions: with vs without MemContext",
        "promise": "3. It works where the user is: a fresh session acts on the current decision.",
        "question": "Across separate real Claude Code sessions, does the agent answer and act on the "
                    "current project decision with its own notes, with a hand-kept decision log, "
                    "and with MemContext, both on short scenarios and on a long project history?",
        "method": "Same scenarios, prompts, model and deterministic scoring as claude_code_recall "
                  "(regex over a final CURRENT: line and over a file the agent writes). Every step "
                  "is a new headless `claude -p` session; steps run in all arms before the next "
                  "step. The self_notes arm has no memory tool and keeps whatever Claude writes. "
                  "The notes arm also gets every user decision appended, dated, to DECISIONS.md.",
        "commit": meta["commit"], "code_under_test": code,
        "product_code_dirty": bool(code.get("dirty")),
        "timestamp_utc": meta["timestamp_utc"], "split": meta["split"], "deterministic": False,
        "model": meta["model"], "claude_version": meta["claude_version"],
        "headline": {
            "label": f"{SET_LABELS.get(head_set, head_set)}: steps answered or acted on with a "
                     "non-current decision (own notes / decision log / MemContext)",
            "value": " / ".join(_fmt(wrong[a]) for a in ARMS if a in wrong),
            "pct": _pct(wrong["memcontext"]) if "memcontext" in wrong else None,
            "better": "lower", "by_arm": {ARM_LABELS[a]: _fmt(w) for a, w in wrong.items()}},
        "metrics": metrics,
        "tables": [
            {"title": "By arm", "columns": arm_cols,
             "rows": [_arm_row(arm, s) for arm, s in summ.items()]},
            {"title": "By scenario set (short vs long-horizon)", "columns": ["scenario set", *arm_cols],
             "rows": [[SET_LABELS.get(name, name), *_arm_row(arm, s)]
                      for name, group in per_set.items() for arm, s in group.items()]},
        ],
        "examples": pick_examples(by_arm),
        "limits": [
            "One model (haiku), one run per arm: LLM variance is not measured, and a difference of "
            "one or two steps is within noise.",
            "Scenarios are hand-written. The long-horizon set is one project of 37 sessions; real "
            "projects run for months with far more decisions and code.",
            "The notes arm is logged by the harness, perfectly and immediately; real logs are "
            "written late or not at all.",
            "Cost and tokens include each arm's own overhead (MCP tool schemas, hooks, reading "
            "notes files).",
            *meta.get("extra_limits", []),
        ],
        "findings": findings(per_set),
        "totals": {"spent_usd": meta["spent_usd"], "spent_this_run_usd": meta.get("spent_this_run_usd"),
                   "sessions": sum(s["sessions"] for s in summ.values())},
        "imported": meta.get("imported", []),
        "summary_by_arm": summ, "summary_by_set": per_set,
    }


METRIC_DEFS = {
    "wrong_steps": "recall/act steps where the answer or the written file is not the current decision",
    "stale_answer": "steps with an older decision on record whose CURRENT: line names an old value",
    "stale_action": "act steps with an older decision on record where the written file holds an old value",
}


def print_tables(report: dict) -> None:
    for t in report["tables"]:
        rows = [[str(v) for v in r] for r in t["rows"]]
        widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(t["columns"])]
        print(f"\n{t['title']}")
        print("  ".join(c.ljust(w) for c, w in zip(t["columns"], widths, strict=True)))
        for r in rows:
            print("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)))
    print(f"\nheadline: {report['headline']['label']}: {report['headline']['value']}")
    for f in report["findings"]:
        print(f"finding: {f}")
    print(f"spent ${report['totals']['spent_usd']:.3f} over {report['totals']['sessions']} sessions")


# ---------------------------------------------------------------- import raw ---
def load_raw(path: Path, set_name: str) -> dict[str, list[dict]]:
    """A previous run's raw.json, with legacy arm names and MemContext-only attribution fixed."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for arm, scenarios in raw.items():
        arm = LEGACY_ARM_NAMES.get(arm, arm)
        fixed = []
        for sc in scenarios:
            steps = sc["steps"] if arm == "memcontext" else strip_memcontext_attribution(sc["steps"])
            fixed.append({**sc, "arm": arm, "set": sc.get("set") or set_name, "steps": steps})
        out[arm] = fixed
    return out


def _spent(by_arm: dict[str, list[dict]]) -> float:
    return round(sum(s.get("cost_usd") or 0.0 for res in by_arm.values() for sc in res
                     for s in sc["steps"]), 4)


# ----------------------------------------------------------------------- main ---
def _named_path(item: str) -> tuple[str, Path]:
    name, sep, path = item.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {item!r}")
    return name, Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--scenarios", default="all")
    p.add_argument("--split", default="dev", choices=["all", "dev", "heldout"])
    p.add_argument("--model", default="haiku")
    p.add_argument("--max-cost-usd", type=float, default=3.0, help="Total across all arms and sets.")
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--dataset", type=_named_path, action="append",
                   help="SET=PATH, repeatable (default: short=recall_scenarios.json)")
    p.add_argument("--import-raw", type=_named_path, action="append", default=[],
                   help="SET=PATH of an earlier run's raw.json to include in the report (not re-run)")
    p.add_argument("--claude", default=shutil.which("claude") or "claude")
    exe = Path(sys.executable).parent / ("memcontext.exe" if os.name == "nt" else "memcontext")
    p.add_argument("--memcontext-cli", default=str(exe) if exe.exists() else "memcontext")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    args.dataset = args.dataset or [("short", ccr.DEFAULT_DATASET)]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    arms = [LEGACY_ARM_NAMES.get(a.strip(), a.strip()) for a in args.arms.split(",") if a.strip()]
    if set(arms) - set(ARMS):
        raise SystemExit(f"unknown arm(s): {sorted(set(arms) - set(ARMS))}")
    scenarios = []
    for set_name, path in args.dataset:
        data = json.loads(path.read_text(encoding="utf-8"))
        scenarios += [{**sc, "set": sc.get("set") or set_name}
                      for sc in ccr._select(data, args.scenarios, args.split)]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.workdir / safe_dirname(stamp)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = {"claude": args.claude, "cli": args.memcontext_cli, "model": args.model, "embed": "0",
           "budget": ccr.Budget(args.max_cost_usd)}
    print(f"recall_ab: arms={arms} scenarios={[s['id'] for s in scenarios]} model={args.model} "
          f"budget=${args.max_cost_usd} run_dir={run_dir}", flush=True)
    by_arm: dict[str, list[dict]] = {a: [] for a in arms}
    for sc in scenarios:
        if ctx["budget"].limit - ctx["budget"].spent < ROUND_RESERVE_USD:
            print(f"  budget exhausted; skipping {sc['id']} and the rest", flush=True)
            break
        for arm, res in run_scenario_all_arms(sc, arms, run_dir, ctx, args.keep).items():
            by_arm[arm].append(res)
    (run_dir / "raw.json").write_text(json.dumps(by_arm, indent=2, default=str), encoding="utf-8")
    spent_now = round(ctx["budget"].spent, 4)
    imported, extra_limits = [], []
    for set_name, path in args.import_raw:
        prev = load_raw(path, set_name)
        for arm, res in prev.items():
            if arm in by_arm:
                by_arm[arm] += res
        imported.append({"set": set_name, "raw": ccr._posix(path), "spent_usd": _spent(prev)})
        extra_limits.append(f"The '{set_name}' set comes from an earlier run ({ccr._posix(path)}) on "
                            "the same product build and harness scoring, re-judged after the arm "
                            "rename; its project files were not kept, so a CLAUDE.md source shows "
                            "there as 'none'.")
    meta = {"commit": ccr._cmd_out(["git", "rev-parse", "HEAD"], ccr.REPO_ROOT),
            "code_under_test": _code_under_test(), "timestamp_utc": datetime.now(UTC).isoformat(),
            "split": args.split, "model": args.model,
            "claude_version": ccr._cmd_out([args.claude, "--version"]),
            "spent_usd": round(spent_now + sum(i["spent_usd"] for i in imported), 4),
            "spent_this_run_usd": spent_now, "imported": imported, "extra_limits": extra_limits}
    report = build_report(by_arm, meta)
    report["raw_run_dir"] = ccr._posix(run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print_tables(report)
    print(f"\nresults: {args.out}\nraw sessions: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
