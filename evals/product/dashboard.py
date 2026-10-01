"""Product eval dashboard: every eval's status, metrics, method, n, baseline and findings.

Eval harness, NOT product code. Reads the latest results; computes nothing new.

  python evals/product/dashboard.py                 build results/dashboard.html and open it
  python evals/product/dashboard.py --run           re-run the fast deterministic evals first
  python evals/product/dashboard.py --no-open       build only

In Claude Code, `/evals` runs this (.claude/commands/evals.md).

Evals that write the common schema (results/<eval>/latest.json or dev.json) are picked up
as-is; stale_exposure, supersession_matrix and claude_code_recall are adapted here from
their own result files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from dashboard_page import render as render_page  # noqa: E402

ORDER = ("recall_ab", "stale_exposure", "claude_code_recall", "injection_noise", "provenance_trace",
         "supersession_matrix", "scale")
PROMISE = {"stale_exposure": "Current, not stale", "provenance_trace": "Traceable history",
           "injection_noise": "Quiet when irrelevant", "scale": "Holds as memory grows",
           "supersession_matrix": "Right state decisions", "claude_code_recall": "Works where the user is",
           "recall_ab": "Value in real sessions"}
# short, scannable labels for the overview cards (the full label stays in each eval's view)
SHORT = {"stale_exposure": "outdated values served", "claude_code_recall": "outdated answers, fresh sessions",
         "injection_noise": "unrelated prompts given memory", "provenance_trace": "histories fully correct",
         "supersession_matrix": "changes handled correctly", "recall_ab": "wrong steps, MemContext arm"}
VS = {"stale_exposure": "Plain search", "scale": "With no extra decisions"}
TITLE = {"recall_ab": "With vs without MemContext"}
CONDITION_NAMES = {
    "A_bm25_turns_top5": "Plain search, top 5 messages",
    "A_bm25_turns_top1": "Plain search, top message",
    "A_dense_turns_top5": "Vector search, top 5 messages",
    "B_mc_claims": "memory_query, ranked claims",
    "B5_mc_claims_first5": "memory_query, first 5 claims",
    "C_mc_claims_plus_episodes": "memory_query, claims + sources",
    "D_mc_hook_injection": "MemContext prompt hook",
}
LAYER_RE = re.compile(r"\b(Logic|Implementation|Integration|Plumbing)\b")
WHERE_RE = re.compile(r"((?:memcontext|evals)/[\w/.]+\.py(?::\d+(?:-\d+)?)?)")
RANK = {"pass": 0, "info": 0, "watch": 1, "fail": 2}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mtime_iso(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat(timespec="seconds")


def _frac(hits: int, n: int) -> dict:
    return {"value": f"{hits}/{n}", "pct": round(100.0 * hits / n, 1) if n else None}


def grade(pct: float | None, better: str | None) -> str:
    """pass / watch / fail for a rate. Lower-is-better rates fail above 20%; higher-is-better below 70%."""
    if pct is None or better not in ("lower", "higher"):
        return "info"
    if better == "lower":
        return "pass" if pct <= 5 else "watch" if pct <= 20 else "fail"
    return "pass" if pct >= 90 else "watch" if pct >= 70 else "fail"


def _pct_of(value: str) -> float | None:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(value))
    return round(100.0 * int(m[1]) / int(m[2]), 1) if m and int(m[2]) else None


def _ensure_question_view(run_dir: Path, summary: dict) -> None:
    """The per-question page is written by `stale_exposure run --html`; build it if that flag was not used."""
    if (run_dir / "report.html").exists() or not (run_dir / "per_question.json").exists():
        return
    import stale_exposure as se  # harness module; imports no product code at load time

    hist = _load(HERE / "datasets" / "stale_exposure.json")["histories"]
    se.write_report(summary, _load(run_dir / "per_question.json"), se.build_questions(hist), len(hist),
                    run_dir / "report.html")


# --------------------------------------------------------------- adapters ---
def adapt_stale_exposure() -> dict | None:
    runs = [(p, _load(p)) for p in (RESULTS / "stale_exposure").glob("*/summary.json")]
    plain = [(p, s) for p, s in runs if not s["subject_drift"] and s["mode"] == "lexical"]
    if not plain:
        return None
    path, s = max(plain, key=lambda x: x[0].stat().st_mtime)
    drift = max(((p, d) for p, d in runs if d["subject_drift"]), key=lambda x: x[0].stat().st_mtime,
                default=(None, None))[1]
    _ensure_question_view(path.parent, s)
    m = s["metrics"]
    n = m["D_mc_hook_injection"]["n_changed_q"]

    def cnt(cond: str, key: str) -> int:
        return round(m[cond][key] * n / 100)

    hook, base = m["D_mc_hook_injection"], m["A_bm25_turns_top5"]
    tone = {"D_mc_hook_injection": "good", "A_bm25_turns_top5": "bad"}
    findings = ["The full memory_query payload (claims + raw source messages) carries old values by design; "
                "the hook and the claim list do not."]
    if drift:
        dm = drift["metrics"]["D_mc_hook_injection"]
        findings.insert(0, f"Logic: when the final change is saved under a reworded subject, the hook serves an "
                           f"outdated value {round(dm['stale_exposure_pct'] * n / 100)}/{n} times; nothing links "
                           f"the two subjects (memcontext/conflicts.py:51).")
    current = _frac(cnt("D_mc_hook_injection", "current_hit_pct"), n)
    return {
        "eval": "stale_exposure", "title": "Stale exposure", "n": n,
        "question": "After a decision changes, how often does what memory hands the agent still contain the old value?",
        "method": (f"{s['n_sessions']} stored turns from 44 hand-written decision histories (34 change 2-3 times), "
                   f"stored through the public API, one session per change. The same {n} questions about changed "
                   f"decisions go to every condition; a value counts as served if it appears word-bounded in the "
                   f"served text. Deterministic, no LLM."),
        "commit": s["repo_sha"][:10], "timestamp_utc": _mtime_iso(path), "deterministic": True,
        "headline": {"label": "prompts where the hook served an outdated value",
                     "value": f"{cnt('D_mc_hook_injection', 'stale_exposure_pct')}/{n}",
                     "pct": hook["stale_exposure_pct"], "better": "lower",
                     "baseline": {"label": "plain search over the history",
                                  "value": f"{cnt('A_bm25_turns_top5', 'stale_exposure_pct')}/{n}",
                                  "pct": base["stale_exposure_pct"]}},
        "metrics": [
            {"name": "Hook: current value present", **current, "better": "higher", "gate": True,
             "definition": "the current value appears in the injected context"},
            {"name": "Hook: median injected size", "value": f"{hook['median_chars']:.0f} chars",
             "definition": f"plain search serves {base['median_chars']:.0f} chars"},
            {"name": "Hook: median build time", "value": f"{hook['median_query_seconds'] * 1000:.0f} ms",
             "definition": "time to build the injected context, in process"},
        ],
        "charts": [{"type": "hbar", "title": "Outdated value served, by condition", "unit": "%",
                    "items": [{"label": CONDITION_NAMES.get(c, c), "value": v["stale_exposure_pct"],
                               "display": f"{cnt(c, 'stale_exposure_pct')}/{n}", "tone": tone.get(c, "base")}
                              for c, v in m.items()]},
                   {"type": "hbar", "title": "Current value present, by condition", "unit": "%",
                    "items": [{"label": CONDITION_NAMES.get(c, c), "value": v["current_hit_pct"],
                               "display": f"{cnt(c, 'current_hit_pct')}/{n}", "tone": tone.get(c, "base")}
                              for c, v in m.items()]}],
        "tables": [{"title": "All conditions",
                    "columns": ["condition", "outdated served", "current present", "current only", "median chars"],
                    "rows": [[CONDITION_NAMES.get(c, c), f"{cnt(c, 'stale_exposure_pct')}/{n}",
                              f"{cnt(c, 'current_hit_pct')}/{n}", f"{cnt(c, 'current_only_pct')}/{n}",
                              f"{v['median_chars']:.0f}"] for c, v in m.items()]}],
        "limits": ["Synthetic histories written by the author; measures what is served, not the model's answer.",
                   "Lexical mode; the semantic-mode run needs the local embedding model."],
        "findings": findings, "link": f"stale_exposure/{path.parent.name}/report.html",
    }


def adapt_supersession() -> dict | None:
    det_p, sem_p = RESULTS / "supersession_matrix_deterministic_dev.json", RESULTS / "supersession_matrix_semantic_dev.json"
    if not det_p.exists():
        return None
    det = _load(det_p)
    sem = _load(sem_p) if sem_p.exists() else None
    o, cats = det["overall"], {c: v for c, v in det["per_category"].items() if v["n"]}
    causes: dict[str, list[str]] = {}
    for f in det["failures"]:
        for rc in f.get("root_causes", [])[:1]:
            causes.setdefault(f"{rc['layer']} {rc['where']}", []).append(f["id"])
    metrics = [{"name": "Deterministic mode", **_frac(o["correct"], o["n"]), "better": "higher",
                "definition": f"{o['excluded']} ambiguous / known-limitation cases reported, not scored"}]
    series = [{"name": "deterministic", "tone": "accent",
               "values": [round(100 * v["correct"] / v["n"], 1) for v in cats.values()]}]
    if sem:
        so = sem["overall"]
        metrics.append({"name": "Semantic mode (BGE-M3)", **_frac(so["correct"], so["n"]), "better": "higher",
                        "definition": "same cases with embedding-based identity on"})
        series.append({"name": "semantic", "tone": "base",
                       "values": [round(100 * sem["per_category"][c]["correct"] / sem["per_category"][c]["n"], 1)
                                  if sem["per_category"].get(c, {}).get("n") else None for c in cats]})
    worst = sorted(cats, key=lambda c: cats[c]["correct"] / cats[c]["n"])[:3]
    findings = [f"{k.split(' ', 1)[0]}: {len(v)} failing cases trace to {k.split(' ', 1)[1]} ({', '.join(v[:4])}"
                f"{', ...' if len(v) > 4 else ''})." for k, v in sorted(causes.items(), key=lambda x: -len(x[1]))]
    findings.append("Weakest categories: " + ", ".join(
        f"{c.replace('_', ' ')} {cats[c]['correct']}/{cats[c]['n']}" for c in worst) + ".")
    return {
        "eval": "supersession_matrix", "title": "Change handling", "n": o["n"],
        "question": "When a new fact arrives, does memory replace, keep both, or flag a conflict correctly, "
                    "within the right scope and trust rules?",
        "method": "Hand-labelled cases across 14 categories (update, additive, negation, duplicates, cross-session, "
                  "cross-namespace, trust direction...), labelled from documented behaviour before running. Dev split "
                  "shown; the held-out split is never used for tuning. Every failure is diagnosed to a code location.",
        "commit": det["meta"]["git_commit"][:10], "timestamp_utc": det["meta"]["timestamp_utc"],
        "deterministic": True, "split": "dev",
        "headline": {"label": "cases handled correctly (deterministic)", "value": f"{o['correct']}/{o['n']}",
                     "pct": round(100 * o["accuracy"], 1), "better": "higher"},
        "metrics": metrics,
        "charts": [{"type": "grouped", "title": "Correct by category", "unit": "%",
                    "categories": [c.replace("_", " ") for c in cats], "series": series}],
        "tables": [{"title": "Failures by diagnosed root cause", "columns": ["layer and location", "failing cases"],
                    "rows": [[k, ", ".join(v)] for k, v in sorted(causes.items(), key=lambda x: -len(x[1]))]}],
        "limits": ["Small hand-written set (5-9 cases per category); it diagnoses, it does not rank systems."],
        "findings": findings,
    }


def adapt_recall() -> dict | None:
    files = sorted(RESULTS.glob("claude_code_recall_*_summary.json"))
    if not files:
        return None
    runs = [_load(f) for f in files]
    keys = ("current_recall", "stale_answer", "action_correct", "stale_action", "step_fully_correct")
    names = {"current_recall": ("Current decision recalled", "higher"), "stale_answer": ("Outdated decision stated", "lower"),
             "action_correct": ("Code used current decision", "higher"), "stale_action": ("Code used outdated decision", "lower"),
             "step_fully_correct": ("Steps fully correct", "higher")}
    pooled = {k: (sum(r["summary"][k]["hits"] for r in runs), sum(r["summary"][k]["n"] for r in runs)) for k in keys}
    label = [r["meta"]["timestamp"][4:6] + "/" + r["meta"]["timestamp"][6:8] + " " + r["meta"]["timestamp"][9:11] + ":"
             + r["meta"]["timestamp"][11:13] for r in runs]
    last = runs[-1]
    return {
        "eval": "claude_code_recall", "title": "Real Claude Code sessions", "n": pooled["stale_answer"][1],
        "question": "Across separate, fresh Claude Code sessions, does the agent use the current decision, in its "
                    "answers and in the code it writes?",
        "method": (f"{len(runs)} runs x {last['summary']['sessions']} real `claude -p` sessions "
                   f"({last['meta']['model']}, {last['meta']['claude_version']}), 6 scenarios: decision changes, "
                   f"changed twice, two topics, user vs assistant suggestion, 'actually' correction, stable control. "
                   f"Scored from the final answer line and the file written. Pooled over runs."),
        "commit": last["meta"]["commit"][:10], "timestamp_utc": last["meta"]["timestamp"], "deterministic": False,
        "headline": {"label": "fresh-session answers that used an outdated decision", **_frac(*pooled["stale_answer"]),
                     "better": "lower"},
        "metrics": [{"name": names[k][0], **_frac(*pooled[k]), "better": names[k][1], "gate": k == "stale_action"}
                    for k in keys],
        "charts": [{"type": "line", "title": "By run", "unit": "%", "x": label,
                    "series": [{"name": names[k][0], "tone": t,
                                "values": [round(100 * r["summary"][k]["rate"], 1) for r in runs]}
                               for k, t in (("current_recall", "good"), ("stale_answer", "bad"), ("stale_action", "warn"))]}],
        "tables": [{"title": "Runs", "columns": ["run (UTC)", "code", "recalled", "outdated answers", "correct code",
                                                 "outdated code", "fully correct", "cost"],
                    "rows": [[lb, r["meta"]["commit"][:8], *(f"{r['summary'][k]['hits']}/{r['summary'][k]['n']}" for k in keys),
                              f"${r['summary']['total_cost_usd']:.2f}"] for lb, r in zip(label, runs, strict=True)]}],
        "limits": ["Small n per run and LLM variance: read the runs together, not one run alone.",
                   "MemContext arm only; the with-vs-without comparison is the A/B eval."],
        "findings": [],
    }


def _decorate_scale(e: dict) -> None:
    s = e.get("series") or {}
    if not s.get("sizes"):
        return
    x = [f"{v:,}" for v in s["sizes"]]
    e["charts"] = [{"type": "line", "title": "Stale and current, as unrelated decisions are added", "unit": "%",
                    "x": x, "xlabel": "unrelated decisions added",
                    "series": [{"name": "hook: current present", "tone": "good", "values": s["hook_current_pct"]},
                               {"name": "hook: outdated served", "tone": "accent", "values": s["hook_stale_pct"]},
                               {"name": "plain search: outdated", "tone": "bad", "values": s["bm25_stale_pct"]}]},
                   {"type": "line", "title": "Hook latency p95 (ms)", "unit": " ms", "x": x,
                    "xlabel": "unrelated decisions added",
                    "series": [{"name": "hook p95 ms", "tone": "warn", "values": s["hook_p95_ms"]},
                               {"name": "ingest ms per stored turn", "tone": "base", "values": s.get("ingest_ms_per_turn", [])}]}]
    for m in e.get("metrics", []):
        if m["name"].lower().startswith("hook current"):
            m["gate"] = True
    # the guarantee that holds (0 outdated) stays a metric; the headline is what degrades with size
    cur, top = s["hook_current_pct"], s["sizes"][-1]
    n = int(e["headline"]["value"].split("/")[1])
    e["short"] = f"current decisions found, {top:,} stored"
    e["headline"] = {"label": f"current decision present with {top:,} unrelated decisions in memory",
                     "value": f"{round(cur[-1] * n / 100)}/{n}", "pct": cur[-1], "better": "higher",
                     "baseline": {"label": "with no unrelated decisions", "value": f"{round(cur[0] * n / 100)}/{n}",
                                  "pct": cur[0]}}


def _auto_chart(e: dict) -> None:
    items = [{"label": m["name"], "value": _pct_of(m["value"]), "display": m["value"],
              "tone": {"pass": "good", "watch": "warn", "fail": "bad"}.get(m.get("status"), "base")}
             for m in e.get("metrics", []) if _pct_of(m["value"]) is not None]
    if items and not e.get("charts"):
        e["charts"] = [{"type": "hbar", "title": "Metrics", "unit": "%", "items": items}]


ARM_NAMES = {"memcontext": "MemContext", "notes": "Decisions file", "self_notes": "Claude's own notes",
             "none": "Claude's own notes"}


def _split(value: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(value))
    return (int(m[1]), int(m[2])) if m else None


def _normalize_ab(e: dict) -> None:
    """A/B headline: the MemContext arm's number, with the best other arm as the baseline."""
    by = e["headline"].get("by_arm")
    if not by or "memcontext" not in by:
        return
    others = {k: v for k, v in by.items() if k != "memcontext"}
    best = min(others, key=lambda k: (_split(others[k]) or (0, 1))[0] / max(1, (_split(others[k]) or (0, 1))[1]))
    e["headline"].update({"value": by["memcontext"], "pct": _pct_of(by["memcontext"]),
                          "baseline": {"label": ARM_NAMES.get(best, best), "value": others[best],
                                       "pct": _pct_of(others[best])}})
    e["vs_label"] = ARM_NAMES.get(best, best)
    e["card"] = {"type": "arms", "rows": [{"label": ARM_NAMES.get(k, k), "value": v, "pct": _pct_of(v) or 0,
                                           "tone": "accent" if k == "memcontext" else "base"} for k, v in by.items()]}


def card_spec(e: dict) -> dict | None:
    """A small visual per eval card, chosen to fit what that eval measures."""
    h, name = e["headline"], e["eval"]
    if e.get("card"):
        return e["card"]
    if name == "stale_exposure" and h.get("baseline"):
        return {"type": "arms", "rows": [{"label": "MemContext", "value": h["value"], "pct": h["pct"], "tone": "good"},
                                         {"label": "Plain search", "value": h["baseline"]["value"],
                                          "pct": h["baseline"]["pct"], "tone": "bad"}]}
    if name == "scale" and e.get("series", {}).get("hook_current_pct"):
        return {"type": "spark", "values": e["series"]["hook_current_pct"], "tone": "bad",
                "from": f"{e['series']['sizes'][0]:,}", "to": f"{e['series']['sizes'][-1]:,} stored"}
    if name == "claude_code_recall":
        chart = (e.get("charts") or [{}])[0]
        stale = next((s for s in chart.get("series", []) if "Outdated" in s["name"]), None)
        if stale:
            return {"type": "spark", "values": stale["values"], "tone": "warn", "from": "run 1",
                    "to": f"run {len(stale['values'])}"}
    if name == "provenance_trace":
        return {"type": "checks", "items": [{"label": m["name"], "ok": m.get("status") == "pass"}
                                            for m in e.get("metrics", [])[:4]]}
    split = _split(h["value"])
    if not split:
        return None
    hits, n = split
    if n <= 40:
        bad = h.get("better") == "lower"
        return {"type": "dots", "n": n, "hits": hits, "tone": "bad" if bad else "good"}
    return {"type": "stack", "parts": [{"value": hits, "tone": "good" if h.get("better") == "higher" else "bad"},
                                       {"value": n - hits, "tone": "base"}]}


def finalize(e: dict) -> dict:
    """Status per metric and per eval; the eval is the worst of its headline and any gate metric."""
    if e["eval"] == "scale":
        _decorate_scale(e)
    for m in e.get("metrics", []):
        m.setdefault("pct", _pct_of(m.get("value", "")))
        m["status"] = grade(m.get("pct"), m.get("better")) if m.get("pct") is not None else ""
    h = e["headline"]
    h.setdefault("pct", _pct_of(h["value"]))
    status = grade(h.get("pct"), h.get("better"))
    e["status_reason"] = ""
    for m in e.get("metrics", []):
        if m.get("gate") and m["status"] and RANK[m["status"]] > RANK[status]:
            status, e["status_reason"] = m["status"], f"{m['name']}: {m['value']}"
    e["status"] = status
    e["promise_short"] = PROMISE.get(e["eval"], e.get("promise", ""))
    e["title"] = TITLE.get(e["eval"], e["title"])
    e.setdefault("short", SHORT.get(e["eval"], ""))
    e.setdefault("vs_label", VS.get(e["eval"], "Baseline"))
    e.setdefault("n", None)
    if e["eval"] == "recall_ab":
        _normalize_ab(e)
        e["headline"]["better"] = "lower"
        e["status"] = grade(e["headline"]["pct"], "lower")
    e["card"] = card_spec(e)
    e["commit"] = (e.get("commit") or "")[:10]
    ts = str(e.get("timestamp_utc") or "")
    if re.fullmatch(r"\d{8}T\d{6}Z", ts):  # compact run ids, e.g. 20261001T012926Z
        e["timestamp_utc"] = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}+00:00"
    _auto_chart(e)
    return e


def issues(evals: list[dict]) -> list[dict]:
    """Every finding that names a LIPI layer or a code location: the open-defect list."""
    out = []
    for e in evals:
        for f in e.get("findings") or []:
            layer, where = LAYER_RE.search(f), WHERE_RE.search(f)
            if layer or where:
                out.append({"eval": e["eval"], "eval_title": e["title"], "text": f, "status": e.get("status", "info"),
                            "layer": layer[1] if layer else "", "where": where[1] if where else ""})
    return out


def collect() -> list[dict]:
    found = {}
    for name, fn in (("stale_exposure", adapt_stale_exposure), ("supersession_matrix", adapt_supersession),
                     ("claude_code_recall", adapt_recall)):
        data = fn()
        if data:
            found[name] = data
    for p in RESULTS.glob("*/*.json"):
        if p.name in ("latest.json", "dev.json"):
            d = _load(p)
            if isinstance(d, dict) and d.get("eval") and d.get("headline"):
                found.setdefault(d["eval"], d)
    ordered = [found[k] for k in ORDER if k in found] + [v for k, v in found.items() if k not in ORDER]
    return [finalize(e) for e in ordered]


def key_result(evals: list[dict]) -> dict | None:
    """The one comparison the page leads with: outdated context, plain search vs MemContext."""
    se = next((e for e in evals if e["eval"] == "stale_exposure"), None)
    if not se or not se["headline"].get("baseline"):
        return None
    h, b = se["headline"], se["headline"]["baseline"]
    n = h["value"].split("/")[1]
    return {"title": "Outdated context handed to the agent",
            "subtitle": f"{n} questions about decisions that changed. Lower is better.",
            "rows": [{"label": "Plain search over history", "value": b["value"], "pct": b["pct"], "tone": "bad"},
                     {"label": "MemContext", "value": h["value"], "pct": h["pct"], "tone": "good"}]}


def render(evals: list[dict], standalone: bool = True) -> str:
    commits = sorted({e.get("commit") for e in evals if e.get("commit")}, key=lambda c: c or "")
    data = {"generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "evals": evals,
            "issues": issues(evals), "commit": commits[-1] if len(commits) == 1 else f"{len(commits)} commits",
            "key": key_result(evals)}
    return render_page(data, standalone)


def run_fast_evals() -> None:
    cmds = [[sys.executable, str(HERE / "stale_exposure.py"), "run", "--mode", "lexical", "--brief", "--tag", "dashboard"]]
    for name, args in (("injection_noise.py", ["run", "--split", "dev"]), ("provenance_trace.py", ["run"])):
        if (HERE / name).exists():
            cmds.append([sys.executable, str(HERE / name), *args])
    for c in cmds:
        print("running", Path(c[1]).name, flush=True)
        subprocess.run(c, cwd=REPO, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", help="re-run the fast deterministic evals first")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "dashboard.html"))
    a = ap.parse_args()
    if a.run:
        run_fast_evals()
    evals = collect()
    out = Path(a.out)
    out.write_text(render(evals), encoding="utf-8")
    print(f"MemContext product evals: {out}")
    for e in evals:
        h = e["headline"]
        base = f"   ({e.get('vs_label') or 'baseline'}: {h['baseline']['value']})" if h.get("baseline") else ""
        print(f"  {e['status'].upper():<5}  {e['title']:<28} {h['value']:>6}  {e.get('short') or h['label']}{base}")
    print(f"  {len(issues(evals))} open findings with code locations")
    if not a.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
