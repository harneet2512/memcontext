"""HTML view of a stale-exposure run: plain search vs what the Claude Code hook injects.

Eval harness, NOT product code. Shows what stale_exposure.py measured; computes nothing new.
"""
from __future__ import annotations

import json
from pathlib import Path

BASELINE = "A_bm25_turns_top5"
MEMCONTEXT = "D_mc_hook_injection"
DEFAULT_QUESTION = "What database does the orders service use?"


def report_data(summary: dict, perq: dict, questions: list[dict], n_histories: int) -> dict:
    """One record per changed-fact question: its value history and what each side served."""
    history = {q["q"]: q["stale"] + [q["current"]] for q in questions}
    base = {r["q"]: r for r in perq[BASELINE]}
    rows = []
    for r in perq[MEMCONTEXT]:
        if not r["changed"]:
            continue
        b = base[r["q"]]
        rows.append({"q": r["q"], "subject": r["subject"], "history": history[r["q"]],
                     "base": {"served": b["served"], "stale": b["stale_hit"], "cur": b["cur_hit"]},
                     "mc": {"served": r["served"], "stale": r["stale_hit"], "cur": r["cur_hit"]}})
    return {"sha": summary["repo_sha"][:10], "n_histories": n_histories, "rows": rows,
            "default_q": DEFAULT_QUESTION}


def render(summary: dict, perq: dict, questions: list[dict], n_histories: int,
           standalone: bool = True) -> str:
    """The page. standalone=False gives the fragment a host wraps itself (a published artifact)."""
    data = json.dumps(report_data(summary, perq, questions, n_histories)).replace("</", "<\\/")
    page = PAGE.replace("__DATA__", data)
    return STANDALONE_HEAD + page + "</body>\n</html>\n" if standalone else page


def write_report(summary: dict, perq: dict, questions: list[dict], n_histories: int,
                 out: Path, standalone: bool = True) -> Path:
    out.write_text(render(summary, perq, questions, n_histories, standalone), encoding="utf-8")
    return out


STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
</head>
<body>
"""

PAGE = r"""<title>MemContext Stale Exposure</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap">
<style>
/* Layout: verdict first (two scorecards, one square per question), then one question opened up side by side. */
:root {
  color-scheme: dark;
  --bg: #0e1116; --panel: #161b23; --line: #263040; --fg: #e6ebf2; --muted: #8f9bb0;
  --stale: #ff6b5e; --stale-bg: rgba(255, 107, 94, .15);
  --fresh: #43d18a; --fresh-bg: rgba(67, 209, 138, .15);
  --empty: #2f3a4b; --focus: #8fb3ff;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    color-scheme: light;
    --bg: #f5f7fa; --panel: #ffffff; --line: #dde3ec; --fg: #141a23; --muted: #5a667a;
    --stale: #c9372c; --stale-bg: rgba(201, 55, 44, .1);
    --fresh: #13804b; --fresh-bg: rgba(19, 128, 75, .1);
    --empty: #d3dae5; --focus: #2f5bd3;
  }
}
:root[data-theme="light"] {
  color-scheme: light;
  --bg: #f5f7fa; --panel: #ffffff; --line: #dde3ec; --fg: #141a23; --muted: #5a667a;
  --stale: #c9372c; --stale-bg: rgba(201, 55, 44, .1);
  --fresh: #13804b; --fresh-bg: rgba(19, 128, 75, .1);
  --empty: #d3dae5; --focus: #2f5bd3;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 var(--sans); }
main { max-width: 1240px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 40px;
  display: grid; gap: 18px; }
h1 { font-size: clamp(24px, 3vw, 32px); line-height: 1.2; margin: 0; text-wrap: balance; }
.sub { color: var(--muted); margin: 6px 0 0; max-width: 75ch; }
.cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.card, .detail { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 20px 22px; min-width: 0; }
.card h2 { font-size: 16px; font-weight: 600; margin: 0; }
.card .how { color: var(--muted); font-size: 13px; margin: 2px 0 14px; font-family: var(--mono); }
.big { display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 12px; }
.big .n { font-size: 64px; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }
.big .of { font-size: 26px; color: var(--muted); font-weight: 600; font-variant-numeric: tabular-nums; }
.big .what { font-size: 15px; color: var(--muted); }
.bad .n { color: var(--stale); }
.good .n { color: var(--fresh); }
.second { margin: 10px 0 16px; font-size: 14px; color: var(--muted); }
.second b { color: var(--fg); font-variant-numeric: tabular-nums; }
.dots { display: grid; grid-template-columns: repeat(21, minmax(0, 1fr)); gap: 5px; }
.dot { aspect-ratio: 1; max-width: 100%; min-width: 0; border-radius: 4px; border: 0; padding: 0;
  cursor: pointer; background: var(--empty); }
.dot.s { background: var(--stale); }
.dot.c { background: var(--fresh); }
.dot.sel { outline: 2px solid var(--fg); outline-offset: 2px; }
button:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; color: var(--muted); font-size: 13px; margin: -4px 0 0; }
.legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.qbar { display: flex; align-items: center; gap: 10px 12px; flex-wrap: wrap; }
.qbar .q { font-size: 20px; font-weight: 700; flex: 1 1 320px; min-width: 0; text-wrap: balance; }
.qbar button { background: transparent; color: var(--fg); border: 1px solid var(--line); border-radius: 8px;
  padding: 6px 12px; font: inherit; font-size: 14px; cursor: pointer; }
.nav { display: flex; align-items: center; gap: 8px; }
.qbar .count { color: var(--muted); font-size: 14px; font-variant-numeric: tabular-nums; }
.timeline { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin: 14px 0 18px; font-size: 14px; }
.timeline .lbl { color: var(--muted); }
.chip { border-radius: 999px; padding: 3px 12px; font-family: var(--mono); font-size: 13px; }
.chip.old { color: var(--stale); background: var(--stale-bg); text-decoration: line-through; }
.chip.cur { color: var(--fresh); background: var(--fresh-bg); font-weight: 600; }
.arrow { color: var(--muted); }
.cols { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.col { min-width: 0; }
.col h3 { font-size: 14px; margin: 0 0 8px; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 4px 8px; }
.verdict { font-weight: 600; }
.verdict.s { color: var(--stale); }
.verdict.c { color: var(--fresh); }
.verdict.n { color: var(--muted); }
.served { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.served li { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; min-width: 0;
  font: 13.5px/1.45 var(--mono); white-space: pre-wrap; overflow-wrap: anywhere; }
.served li.none { color: var(--muted); font-family: var(--sans); }
mark { border-radius: 4px; padding: 0 3px; font-weight: 600; }
mark.s { background: var(--stale-bg); color: var(--stale); text-decoration: line-through; }
mark.c { background: var(--fresh-bg); color: var(--fresh); }
footer { color: var(--muted); font-size: 13px; max-width: 110ch; }
@media (max-width: 760px) {
  .cards, .cols { grid-template-columns: minmax(0, 1fr); }
  .dots { grid-template-columns: repeat(14, minmax(0, 1fr)); }
  .big .n { font-size: 48px; }
}
</style>

<main>
  <header>
    <h1>Does the agent get handed an outdated decision?</h1>
    <p class="sub" id="sub"></p>
  </header>

  <section class="cards" aria-label="Results">
    <div class="card bad">
      <h2>Plain search over the conversation history</h2>
      <div class="how">BM25 over every stored message, top 5</div>
      <div class="big"><span class="n" id="baseN"></span><span class="of" id="baseOf"></span><span class="what">served an outdated value</span></div>
      <div class="second" id="baseCur"></div>
      <div class="dots" id="baseDots"></div>
    </div>
    <div class="card good">
      <h2>MemContext, injected into Claude Code</h2>
      <div class="how">the prompt hook's context: current decisions only</div>
      <div class="big"><span class="n" id="mcN"></span><span class="of" id="mcOf"></span><span class="what">served an outdated value</span></div>
      <div class="second" id="mcCur"></div>
      <div class="dots" id="mcDots"></div>
    </div>
  </section>
  <div class="legend">
    <span><i style="background:var(--stale)"></i>outdated value served</span>
    <span><i style="background:var(--fresh)"></i>current value only</span>
    <span><i style="background:var(--empty)"></i>neither value served</span>
    <span>One square per question. Click a square or use the arrow keys.</span>
  </div>

  <section class="detail" aria-label="One question side by side">
    <div class="qbar">
      <div class="q" id="q"></div>
      <div class="nav">
        <span class="count" id="count"></span>
        <button id="prev" aria-label="Previous question">&larr;</button>
        <button id="next" aria-label="Next question">&rarr;</button>
      </div>
    </div>
    <div class="timeline" id="timeline"></div>
    <div class="cols">
      <div class="col"><h3><span>Plain search served</span><span class="verdict" id="baseV"></span></h3><ul class="served" id="baseServed"></ul></div>
      <div class="col"><h3><span>MemContext injected</span><span class="verdict" id="mcV"></span></h3><ul class="served" id="mcServed"></ul></div>
    </div>
  </section>

  <footer id="foot"></footer>
</main>
<script>
const D = __DATA__;
const rows = D.rows, n = rows.length;
const $ = id => document.getElementById(id);
const kind = side => side.stale.length ? "s" : side.cur ? "c" : "n";
const count = (key, f) => rows.filter(r => f(r[key])).length;

$("sub").textContent = `${D.n_histories} decision histories, ${n} questions about decisions that changed. ` +
  `Both sides get the same questions and are scored on exactly what they served. ` +
  `Deterministic, no LLM judge. Code at ${D.sha}.`;
for (const key of ["base", "mc"]) {
  $(key + "N").textContent = count(key, s => s.stale.length);
  $(key + "Of").textContent = "/ " + n;
  const b = document.createElement("b");
  b.textContent = `${count(key, s => s.cur)} / ${n}`;
  $(key + "Cur").replaceChildren("current value present: ", b);
}
$("foot").textContent = "Limits: the decision histories are a synthetic set, and the score is what each side served, " +
  "not the model's final answer. The hook matches on shared words, so it misses the current value on some questions.";

let sel = Math.max(0, rows.findIndex(r => r.q === D.default_q));
const dots = {base: [], mc: []};
for (const key of ["base", "mc"]) {
  rows.forEach((r, i) => {
    const b = document.createElement("button");
    b.className = "dot " + kind(r[key]);
    b.title = r.q;
    b.setAttribute("aria-label", r.q);
    b.onclick = () => show(i);
    $(key + "Dots").append(b);
    dots[key].push(b);
  });
}

const norm = s => s.toLowerCase().replace(/\s+/g, " ");
function marked(text, row) {
  // whole-value matches, case-insensitive and whitespace-tolerant, like the eval's own scoring
  const last = row.history.length - 1;
  const vals = row.history.map((v, i) => ({v: norm(v), cls: i === last ? "c" : "s"}));
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/ /g, "\\s+");
  const alts = vals.map(x => esc(x.v)).sort((a, b) => b.length - a.length).join("|");
  const re = new RegExp("(?<![a-z0-9])(" + alts + ")(?![a-z0-9])", "gi");
  const frag = document.createDocumentFragment();
  let at = 0;
  for (const m of text.matchAll(re)) {
    frag.append(text.slice(at, m.index));
    const el = document.createElement("mark");
    el.className = (vals.find(x => x.v === norm(m[0])) || {cls: "s"}).cls;
    el.textContent = m[0];
    frag.append(el);
    at = m.index + m[0].length;
  }
  frag.append(text.slice(at));
  return frag;
}

function fill(listId, items, row) {
  const ul = $(listId);
  ul.replaceChildren();
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "none";
    li.textContent = "Nothing served for this question.";
    ul.append(li);
  }
  for (const t of items) {
    const li = document.createElement("li");
    li.append(marked(t, row));
    ul.append(li);
  }
}

function verdict(id, side) {
  const k = kind(side), el = $(id);
  el.className = "verdict " + k;
  el.textContent = k === "s" ? "outdated value served" : k === "c" ? "current value only" : "neither value served";
}

function show(i) {
  sel = (i + n) % n;
  const r = rows[sel], last = r.history.length - 1;
  $("q").textContent = r.q;
  $("count").textContent = `question ${sel + 1} of ${n}`;
  const tl = $("timeline");
  const lbl = document.createElement("span");
  lbl.className = "lbl";
  lbl.textContent = "How the decision changed:";
  tl.replaceChildren(lbl);
  r.history.forEach((v, j) => {
    if (j) {
      const a = document.createElement("span");
      a.className = "arrow";
      a.textContent = "\u2192";
      tl.append(a);
    }
    const c = document.createElement("span");
    c.className = "chip " + (j === last ? "cur" : "old");
    c.textContent = j === last ? v + " (current)" : v;
    tl.append(c);
  });
  fill("baseServed", r.base.served, r);
  fill("mcServed", r.mc.served, r);
  verdict("baseV", r.base);
  verdict("mcV", r.mc);
  for (const key of ["base", "mc"]) dots[key].forEach((d, j) => d.classList.toggle("sel", j === sel));
}

$("prev").onclick = () => show(sel - 1);
$("next").onclick = () => show(sel + 1);
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") show(sel + 1);
  if (e.key === "ArrowLeft") show(sel - 1);
});
show(sel);
</script>
"""
