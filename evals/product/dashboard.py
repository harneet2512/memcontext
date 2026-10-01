"""Product eval dashboard: every eval's metrics, method, n, baseline and limits on one page.

Eval harness, NOT product code. Reads committed/latest results; computes nothing new.

  python evals/product/dashboard.py                 build results/dashboard.html and open it
  python evals/product/dashboard.py --run           re-run the fast deterministic evals first
  python evals/product/dashboard.py --no-open       build only

Evals that write the common schema (results/<eval>/latest.json or <split>.json) are picked
up as-is; the three older evals (stale_exposure, supersession_matrix, claude_code_recall)
are adapted here from their own result files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = HERE.parents[1]
# order on the page: value in real sessions first, then the guarantees, then costs
ORDER = ("recall_ab", "stale_exposure", "claude_code_recall", "injection_noise", "provenance_trace",
         "supersession_matrix", "scale")
CONDITION_NAMES = {
    "A_bm25_turns_top5": "Plain search: top 5 messages (baseline)",
    "A_bm25_turns_top1": "Plain search: top message",
    "A_dense_turns_top5": "Vector search: top 5 messages",
    "B_mc_claims": "memory_query: ranked claims",
    "B5_mc_claims_first5": "memory_query: first 5 claims",
    "C_mc_claims_plus_episodes": "memory_query: claims + source messages",
    "D_mc_hook_injection": "MemContext prompt hook (what Claude Code sees)",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mtime_iso(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat(timespec="seconds")


def _frac(hits: int, n: int) -> dict:
    return {"value": f"{hits}/{n}", "pct": round(100.0 * hits / n, 1) if n else None}


def _ensure_question_view(run_dir: Path, summary: dict) -> None:
    """The per-question page is written by `stale_exposure run --html`; build it if that flag was not used."""
    if (run_dir / "report.html").exists() or not (run_dir / "per_question.json").exists():
        return
    sys.path.insert(0, str(HERE))
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
    m = s["metrics"]
    n = m["D_mc_hook_injection"]["n_changed_q"]

    def cnt(cond: str, key: str) -> int:
        return round(m[cond][key] * n / 100)

    rows = [[CONDITION_NAMES.get(c, c), f"{cnt(c, 'stale_exposure_pct')}/{n}", f"{cnt(c, 'current_hit_pct')}/{n}",
             f"{cnt(c, 'current_only_pct')}/{n}", f"{v['median_chars']:.0f}"] for c, v in m.items()]
    bars = [{"label": CONDITION_NAMES.get(c, c), "pct": v["stale_exposure_pct"],
             "good": c == "D_mc_hook_injection"} for c, v in m.items()]
    findings = []
    if drift:
        dm = drift["metrics"]["D_mc_hook_injection"]
        findings.append(f"Stress test: when the final change is saved under a reworded subject, the hook serves an "
                        f"outdated value {round(dm['stale_exposure_pct'] * n / 100)}/{n} times "
                        f"({dm['stale_exposure_pct']}%), because nothing links the two subjects (commit "
                        f"{drift['repo_sha'][:8]}). This is the top known weakness.")
    _ensure_question_view(path.parent, s)
    findings.append("The full memory_query payload (claims + raw source messages) carries old values by design: "
                    "source messages are history. The hook and the claim list do not.")
    hook, base = m["D_mc_hook_injection"], m["A_bm25_turns_top5"]
    return {
        "eval": "stale_exposure", "title": "Stale exposure", "promise": "Current, not stale",
        "question": "After a decision changes, how often does what memory hands the agent still contain the old value?",
        "method": (f"{s['n_sessions']} stored turns from 44 hand-written decision histories (34 change 2-3 times), "
                   f"stored through the public API, one session per change. The same {n} questions about changed "
                   f"decisions go to every condition; a value counts as served if it appears word-bounded in "
                   f"the served text. Deterministic, no LLM."),
        "commit": s["repo_sha"][:10], "timestamp_utc": _mtime_iso(path), "deterministic": True,
        "headline": {"label": "prompts where the hook served an outdated value",
                     "value": f"{cnt('D_mc_hook_injection', 'stale_exposure_pct')}/{n}",
                     "pct": hook["stale_exposure_pct"], "better": "lower",
                     "baseline": {"label": "plain search over the history",
                                  "value": f"{cnt('A_bm25_turns_top5', 'stale_exposure_pct')}/{n}",
                                  "pct": base["stale_exposure_pct"]}},
        "metrics": [
            {"name": "hook: current value present", **_frac(cnt("D_mc_hook_injection", "current_hit_pct"), n),
             "n": n, "better": "higher", "definition": "the current value appears in the injected context"},
            {"name": "hook: median injected size", "value": f"{hook['median_chars']:.0f} chars", "pct": None,
             "n": n, "better": "lower", "definition": f"plain search serves {base['median_chars']:.0f} chars"},
            {"name": "hook: median latency", "value": f"{hook['median_query_seconds'] * 1000:.0f} ms", "pct": None,
             "n": n, "better": "lower", "definition": "time to build the injected context, in process"},
        ],
        "bars": {"title": "Outdated value served, by condition (lower is better)", "items": bars},
        "tables": [{"title": "All conditions",
                    "columns": ["condition", "outdated served", "current present", "current only", "median chars"],
                    "rows": rows}],
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
    o = det["overall"]
    cats = det["per_category"]
    rows, bars = [], []
    for c, v in cats.items():
        if not v["n"]:
            continue
        sv = sem["per_category"].get(c) if sem else None
        rows.append([c.replace("_", " "), f"{v['correct']}/{v['n']}",
                     f"{sv['correct']}/{sv['n']}" if sv and sv["n"] else "-"])
        bars.append({"label": c.replace("_", " "), "pct": round(100 * v["correct"] / v["n"], 1),
                     "good": v["correct"] == v["n"]})
    causes: dict[str, int] = {}
    for f in det["failures"]:
        for rc in f.get("root_causes", [])[:1]:
            key = f"{rc['layer']} - {rc['where']}"
            causes[key] = causes.get(key, 0) + 1
    metrics = [{"name": "deterministic mode", **_frac(o["correct"], o["n"]), "n": o["n"], "better": "higher",
                "definition": f"{o['excluded']} ambiguous / known-limitation cases reported but not scored"}]
    if sem:
        so = sem["overall"]
        metrics.append({"name": "semantic mode (BGE-M3)", **_frac(so["correct"], so["n"]), "n": so["n"],
                        "better": "higher", "definition": "same cases with embedding-based identity on"})
    worst = sorted(((v["correct"] / v["n"], c) for c, v in cats.items() if v["n"]))[:3]
    return {
        "eval": "supersession_matrix", "title": "Change handling", "promise": "Right state decisions",
        "question": "When a new fact arrives, does memory replace, keep both, or flag a conflict correctly, "
                    "within the right scope and trust rules?",
        "method": "Hand-labelled cases across 14 categories (update, additive, negation, duplicates, "
                  "cross-session, cross-namespace, trust direction...), labelled from documented behaviour "
                  "before running. Dev split shown; the held-out split is never used for tuning.",
        "commit": det["meta"]["git_commit"][:10], "timestamp_utc": det["meta"]["timestamp_utc"],
        "deterministic": True,
        "headline": {"label": "cases handled correctly (deterministic)", "value": f"{o['correct']}/{o['n']}",
                     "pct": round(100 * o["accuracy"], 1), "better": "higher"},
        "metrics": metrics,
        "bars": {"title": "Correct by category, deterministic mode (higher is better)", "items": bars},
        "tables": [{"title": "By category", "columns": ["category", "deterministic", "semantic"], "rows": rows},
                   {"title": "Failures by diagnosed root cause (LIPI layer - code location)",
                    "columns": ["root cause", "failing cases"],
                    "rows": [[k, str(v)] for k, v in sorted(causes.items(), key=lambda x: -x[1])]}],
        "limits": ["Small hand-written set (5-9 cases per category); it diagnoses, it does not rank systems."],
        "findings": ["Weakest categories: " + ", ".join(
            f"{c.replace('_', ' ')} {cats[c]['correct']}/{cats[c]['n']}" for _, c in worst) + ".",
            "Most failures trace to one rule (memcontext/supersession.py:304): low word overlap means no "
            "update is detected, and high overlap collapses facts that should both stay."],
    }


def adapt_recall() -> dict | None:
    files = sorted(RESULTS.glob("claude_code_recall_*_summary.json"))
    if not files:
        return None
    runs = [_load(f) for f in files]
    keys = ("current_recall", "stale_answer", "action_correct", "stale_action", "step_fully_correct")
    pooled = {k: [sum(r["summary"][k]["hits"] for r in runs), sum(r["summary"][k]["n"] for r in runs)] for k in keys}
    rows = [[r["meta"]["timestamp"][:8] + " " + r["meta"]["timestamp"][9:13] + "Z", r["meta"]["commit"][:8],
             *(f"{r['summary'][k]['hits']}/{r['summary'][k]['n']}" for k in keys),
             f"${r['summary']['total_cost_usd']:.2f}"] for r in runs]
    last = runs[-1]
    defs = {"current_recall": ("fresh session states the current decision", "higher"),
            "stale_answer": ("fresh session states an outdated decision", "lower"),
            "action_correct": ("code written with the current decision", "higher"),
            "stale_action": ("code written with an outdated decision", "lower"),
            "step_fully_correct": ("recall steps right end to end", "higher")}
    return {
        "eval": "claude_code_recall", "title": "Real Claude Code sessions", "promise": "Works where the user is",
        "question": "Across separate, fresh Claude Code sessions, does the agent use the current decision, "
                    "in its answers and in the code it writes?",
        "method": (f"{len(runs)} runs x {last['summary']['sessions']} real `claude -p` sessions "
                   f"({last['meta']['model']}, {last['meta']['claude_version']}), 6 scenarios: decision changes, "
                   f"changed twice, two topics, user vs assistant suggestion, 'actually' correction, stable "
                   f"control. Scored from the final answer line and the written file. Pooled over runs."),
        "commit": last["meta"]["commit"][:10], "timestamp_utc": last["meta"]["timestamp"], "deterministic": False,
        "headline": {"label": "fresh-session answers that used an outdated decision (pooled)",
                     **_frac(*pooled["stale_answer"]), "better": "lower"},
        "metrics": [{"name": k.replace("_", " "), **_frac(*pooled[k]), "n": pooled[k][1],
                     "definition": defs[k][0], "better": defs[k][1]} for k in keys],
        "tables": [{"title": "Runs", "columns": ["run (UTC)", "commit", "current recall", "stale answers",
                                                 "correct actions", "stale actions", "fully correct", "cost"],
                    "rows": rows}],
        "limits": ["Small n per run and LLM variance: read the runs together, not one run alone.",
                   "Only the MemContext arm; the with-vs-without comparison is the A/B eval."],
        "findings": [],
    }


def collect() -> list[dict]:
    found = {}
    for name, fn in (("stale_exposure", adapt_stale_exposure), ("supersession_matrix", adapt_supersession),
                     ("claude_code_recall", adapt_recall)):
        with_data = fn()
        if with_data:
            found[name] = with_data
    for p in RESULTS.glob("*/*.json"):
        if p.name in ("latest.json", "dev.json"):
            d = _load(p)
            if isinstance(d, dict) and d.get("eval") and d.get("headline"):
                found.setdefault(d["eval"], d)
    return [found[k] for k in ORDER if k in found] + [v for k, v in found.items() if k not in ORDER]


# ----------------------------------------------------------------- render ---
def render(evals: list[dict], standalone: bool = True) -> str:
    data = {"generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "evals": evals}
    page = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    return STANDALONE_HEAD + page + "</body>\n</html>\n" if standalone else page


def run_fast_evals() -> None:
    cmds = [[sys.executable, str(HERE / "stale_exposure.py"), "run", "--mode", "lexical", "--brief", "--tag", "dashboard"]]
    for name, args in (("injection_noise.py", ["run", "--split", "dev"]), ("provenance_trace.py", ["run"])):
        if (HERE / name).exists():
            cmds.append([sys.executable, str(HERE / name), *args])
    for c in cmds:
        print("running", Path(c[1]).name, flush=True)
        subprocess.run(c, cwd=REPO, check=True)


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
    print(f"dashboard: {out}  ({len(evals)} evals: {', '.join(e['eval'] for e in evals)})")
    if not a.no_open:
        webbrowser.open(out.as_uri())


STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
</head>
<body>
"""

PAGE = r"""<title>MemContext Product Evals</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap">
<style>
/* Layout: one scoreboard row (each eval's headline), then one section per eval: method, metrics, chart, tables, limits. */
:root {
  color-scheme: dark;
  --bg: #0e1116; --panel: #161b23; --panel2: #1b212b; --line: #263040; --fg: #e6ebf2; --muted: #8f9bb0;
  --good: #43d18a; --good-bg: rgba(67, 209, 138, .14); --bad: #ff6b5e; --bad-bg: rgba(255, 107, 94, .14);
  --base: #7d8aa3; --focus: #8fb3ff;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    color-scheme: light;
    --bg: #f5f7fa; --panel: #ffffff; --panel2: #f0f3f7; --line: #dde3ec; --fg: #141a23; --muted: #5a667a;
    --good: #13804b; --good-bg: rgba(19, 128, 75, .1); --bad: #c9372c; --bad-bg: rgba(201, 55, 44, .1);
    --base: #6b778c; --focus: #2f5bd3;
  }
}
:root[data-theme="light"] {
  color-scheme: light;
  --bg: #f5f7fa; --panel: #ffffff; --panel2: #f0f3f7; --line: #dde3ec; --fg: #141a23; --muted: #5a667a;
  --good: #13804b; --good-bg: rgba(19, 128, 75, .1); --bad: #c9372c; --bad-bg: rgba(201, 55, 44, .1);
  --base: #6b778c; --focus: #2f5bd3;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 var(--sans); }
main { max-width: 1280px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 48px; display: grid; gap: 22px; }
h1 { font-size: clamp(24px, 3vw, 32px); line-height: 1.2; margin: 0; text-wrap: balance; }
h2 { font-size: 20px; margin: 0; text-wrap: balance; }
h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 0 0 8px; font-weight: 600; }
.sub { color: var(--muted); margin: 6px 0 0; max-width: 90ch; }
.board { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
  min-width: 0; text-decoration: none; color: inherit; display: grid; gap: 4px; }
.tile:hover { border-color: var(--muted); }
.tile:focus-visible, a:focus-visible, button:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.tile .promise { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.tile .t { font-weight: 600; }
.tile .v { font-size: 30px; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.tile .l { font-size: 13px; color: var(--muted); }
.tile .b { font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }
.good { color: var(--good); } .bad { color: var(--bad); } .neutral { color: var(--fg); }
.eval { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 22px; min-width: 0;
  display: grid; gap: 18px; scroll-margin-top: 12px; }
.head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; }
.head .q { color: var(--muted); flex-basis: 100%; max-width: 95ch; }
.badge { font: 12px/1 var(--mono); border: 1px solid var(--line); border-radius: 999px; padding: 4px 8px; color: var(--muted); }
.hero { display: flex; flex-wrap: wrap; gap: 14px 40px; align-items: flex-end; }
.hero .big { font-size: 52px; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }
.hero .lbl { color: var(--muted); max-width: 32ch; }
.hero .vs { display: grid; gap: 2px; }
.hero .vs.base .big { font-size: 34px; color: var(--base); }
.method { color: var(--fg); max-width: 100ch; margin: 0; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
.metric { background: var(--panel2); border-radius: 10px; padding: 10px 12px; min-width: 0; }
.metric .n { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.metric .name { font-weight: 600; font-size: 14px; }
.metric .def { color: var(--muted); font-size: 12.5px; }
.bars { display: grid; gap: 7px; }
.bar { display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr) 56px; gap: 10px; align-items: center; font-size: 13.5px; }
.bar .track { height: 14px; background: var(--panel2); border-radius: 4px; overflow: hidden; }
.bar .fill { height: 100%; border-radius: 4px; background: var(--base); }
.bar .fill.g { background: var(--good); }
.bar .pct { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 12.5px; }
td:first-child { font-weight: 600; }
.notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.notes ul { margin: 0; padding-left: 18px; display: grid; gap: 6px; }
.notes li { max-width: 80ch; }
.ex { display: grid; gap: 8px; }
.ex pre { margin: 0; background: var(--panel2); border-radius: 8px; padding: 10px 12px; font: 12.5px/1.45 var(--mono);
  white-space: pre-wrap; overflow-wrap: anywhere; }
a { color: var(--focus); }
summary { cursor: pointer; list-style: none; }
summary h3 { display: inline; }
summary::before { content: "\25B8  "; color: var(--muted); }
details[open] summary::before { content: "\25BE  "; }
footer { color: var(--muted); font-size: 13px; }
@media (max-width: 700px) {
  .bar { grid-template-columns: minmax(0, 1fr) 48px; }
  .bar .track { grid-column: 1 / -1; grid-row: 2; }
  .hero .big { font-size: 40px; }
}
</style>

<main>
  <header>
    <h1>MemContext product evals</h1>
    <p class="sub">MemContext makes three promises: the agent gets the <b>current</b> decision, not a stale one; memory makes the
      <b>right call</b> when a new fact arrives; and it works <b>where the user is</b>, inside real Claude Code sessions.
      Each eval below checks one promise, with its method, sample size, baseline and limits. <span id="gen"></span></p>
  </header>
  <nav class="board" id="board" aria-label="Scoreboard"></nav>
  <div id="evals" style="display:grid;gap:22px"></div>
  <footer>Evals live in <code>evals/product/</code>; each prints and writes its own results. Small hand-written sets:
    they find and diagnose failures, they do not rank systems.</footer>
</main>
<script>
const D = __DATA__;
const PROMISE = {stale_exposure: "Current, not stale", provenance_trace: "Traceable history", injection_noise: "Quiet when irrelevant",
  scale: "Holds as memory grows", supersession_matrix: "Right state decisions", claude_code_recall: "Works where the user is",
  recall_ab: "Value in real sessions"};
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const tone = h => {
  if (h.pct == null) return "neutral";
  if (h.better === "lower") return h.pct <= 5 ? "good" : h.pct >= 25 ? "bad" : "neutral";
  if (h.better === "higher") return h.pct >= 90 ? "good" : h.pct < 60 ? "bad" : "neutral";
  return "neutral";
};
const when = s => { const d = new Date(s); return isNaN(d) ? s : d.toISOString().slice(0, 16).replace("T", " ") + " UTC"; };
document.getElementById("gen").textContent = "Page built " + when(D.generated_utc) + ".";

const board = document.getElementById("board"), host = document.getElementById("evals");
for (const e of D.evals) {
  const h = e.headline;
  const a = el("a", "tile"); a.href = "#" + e.eval;
  a.append(el("span", "promise", PROMISE[e.eval] || e.promise || ""), el("span", "t", e.title), el("span", "v " + tone(h), h.value), el("span", "l", h.label));
  if (h.baseline) a.append(el("span", "b", `baseline: ${h.baseline.value} (${h.baseline.label})`));
  board.append(a);

  const s = el("section", "eval"); s.id = e.eval;
  const head = el("div", "head");
  head.append(el("h2", null, e.title));
  head.append(el("span", "badge", e.deterministic ? "deterministic" : "LLM sessions"));
  if (e.commit) head.append(el("span", "badge", "code " + e.commit + (e.product_code_dirty ? " +local edits" : "")));
  if (e.timestamp_utc) head.append(el("span", "badge", when(e.timestamp_utc)));
  if (e.split) head.append(el("span", "badge", e.split + " split"));
  head.append(el("p", "q", e.question || ""));
  s.append(head);

  const hero = el("div", "hero");
  const main = el("div", "vs");
  main.append(el("span", "big " + tone(h), h.value + (h.pct != null ? `  (${h.pct}%)` : "")), el("span", "lbl", h.label));
  hero.append(main);
  if (h.baseline) {
    const b = el("div", "vs base");
    b.append(el("span", "big", h.baseline.value + (h.baseline.pct != null ? `  (${h.baseline.pct}%)` : "")), el("span", "lbl", h.baseline.label));
    hero.append(b);
  }
  s.append(hero);

  if (e.method) { const m = el("div"); m.append(el("h3", null, "How it is measured"), el("p", "method", e.method)); s.append(m); }

  if (e.metrics && e.metrics.length) {
    const wrap = el("div"); wrap.append(el("h3", null, "Metrics"));
    const g = el("div", "metrics");
    for (const m of e.metrics) {
      const c = el("div", "metric");
      c.append(el("div", "name", m.name), el("div", "n " + tone(m), m.value + (m.pct != null ? ` (${m.pct}%)` : "")));
      if (m.definition) c.append(el("div", "def", m.definition));
      g.append(c);
    }
    wrap.append(g); s.append(wrap);
  }

  if (e.bars) {
    const wrap = el("div"); wrap.append(el("h3", null, e.bars.title));
    const g = el("div", "bars");
    for (const b of e.bars.items) {
      const r = el("div", "bar"), t = el("div", "track"), f = el("div", "fill" + (b.good ? " g" : ""));
      f.style.width = Math.max(0, Math.min(100, b.pct)) + "%";
      t.append(f); r.append(el("span", null, b.label), t, el("span", "pct", b.pct + "%"));
      g.append(r);
    }
    wrap.append(g); s.append(wrap);
  }

  for (const tb of e.tables || []) {
    const long = tb.rows.length > 10;
    const wrap = el(long ? "details" : "div");
    if (long) { const sm = el("summary"); sm.append(el("h3", null, `${tb.title} (${tb.rows.length} rows, show all)`)); wrap.append(sm); }
    else wrap.append(el("h3", null, tb.title));
    const tw = el("div", "tablewrap"), t = el("table"), thead = el("thead"), tr = el("tr");
    for (const c of tb.columns) tr.append(el("th", null, c));
    thead.append(tr); t.append(thead);
    const tbody = el("tbody");
    for (const row of tb.rows) { const r = el("tr"); for (const c of row) r.append(el("td", null, String(c))); tbody.append(r); }
    t.append(tbody); tw.append(t); wrap.append(tw); s.append(wrap);
  }

  if (e.examples && e.examples.length) {
    const wrap = el("div", "ex"); wrap.append(el("h3", null, "Examples"));
    for (const x of e.examples) wrap.append(el("pre", null, typeof x === "string" ? x : JSON.stringify(x, null, 1)));
    s.append(wrap);
  }

  const notes = el("div", "notes");
  for (const [title, items] of [["Findings", e.findings], ["Limits", e.limits]]) {
    if (!items || !items.length) continue;
    const box = el("div"), ul = el("ul");
    box.append(el("h3", null, title));
    for (const i of items) ul.append(el("li", null, i));
    box.append(ul); notes.append(box);
  }
  if (notes.childElementCount) s.append(notes);
  if (e.link) { const p = el("p"); const a2 = el("a", null, "Open the per-question view"); a2.href = e.link; p.append(a2); s.append(p); }
  host.append(s);
}
</script>
"""


if __name__ == "__main__":
    main()
