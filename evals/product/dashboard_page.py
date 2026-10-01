"""Page template for evals/product/dashboard.py. Eval harness, NOT product code.

render(data) embeds the collected eval data as JSON; everything visual is drawn in the page.
"""
from __future__ import annotations

import json

STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
</head>
<body>
"""


def render(data: dict, standalone: bool = True) -> str:
    page = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    return STANDALONE_HEAD + page + "</body>\n</html>\n" if standalone else page


PAGE = r"""<title>MemContext Product Evals</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
/* Layout: fixed sidebar (overview, findings, one link per eval with status) + main column:
   KPI row, eval health table, open findings, then one detail panel per eval. */
:root {
  color-scheme: light;
  --bg: #f4f6f9; --panel: #ffffff; --sunken: #f7f9fb; --line: #e3e8ef; --line2: #d4dbe5;
  --fg: #111827; --fg2: #374151; --muted: #6b7686; --side: #0f1729; --side-fg: #c9d3e3; --side-muted: #7f8ba1;
  --accent: #2f5bd3; --accent-bg: rgba(47, 91, 211, .08);
  --pass: #16794a; --pass-bg: #e6f4ec; --watch: #a15c07; --watch-bg: #fdf1dc; --fail: #c0362c; --fail-bg: #fbe7e5;
  --info: #556274; --info-bg: #eef1f5; --base: #94a3b8;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #0b0f17; --panel: #121826; --sunken: #0f1520; --line: #1f2838; --line2: #2a3547;
    --fg: #e7ecf3; --fg2: #c3ccd8; --muted: #8592a6; --side: #080c13; --side-fg: #c9d3e3; --side-muted: #6f7b90;
    --accent: #7ea2ff; --accent-bg: rgba(126, 162, 255, .12);
    --pass: #46c98a; --pass-bg: rgba(70, 201, 138, .13); --watch: #f0b450; --watch-bg: rgba(240, 180, 80, .13);
    --fail: #ff7a6e; --fail-bg: rgba(255, 122, 110, .13); --info: #9aa6b8; --info-bg: rgba(154, 166, 184, .12); --base: #5c6a80;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0b0f17; --panel: #121826; --sunken: #0f1520; --line: #1f2838; --line2: #2a3547;
  --fg: #e7ecf3; --fg2: #c3ccd8; --muted: #8592a6; --side: #080c13; --side-fg: #c9d3e3; --side-muted: #6f7b90;
  --accent: #7ea2ff; --accent-bg: rgba(126, 162, 255, .12);
  --pass: #46c98a; --pass-bg: rgba(70, 201, 138, .13); --watch: #f0b450; --watch-bg: rgba(240, 180, 80, .13);
  --fail: #ff7a6e; --fail-bg: rgba(255, 122, 110, .13); --info: #9aa6b8; --info-bg: rgba(154, 166, 184, .12); --base: #5c6a80;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.5 var(--sans); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
code, .mono { font-family: var(--mono); font-size: 12.5px; }

.app { display: grid; grid-template-columns: 248px minmax(0, 1fr); min-height: 100vh; }
.side { background: var(--side); color: var(--side-fg); position: sticky; top: env(safe-area-inset-top, 0px);
  height: 100vh; overflow-y: auto; padding: 18px 14px; display: flex; flex-direction: column; gap: 18px; }
.brand { display: flex; align-items: center; gap: 10px; padding: 2px 6px; }
.mark { width: 30px; height: 30px; border-radius: 7px; background: linear-gradient(135deg, #3b6cf0, #19b394);
  display: grid; place-items: center; color: #fff; font: 700 13px/1 var(--mono); letter-spacing: -.02em; }
.brand b { display: block; color: #fff; font-size: 14.5px; }
.brand span { display: block; color: var(--side-muted); font-size: 12px; }
.nav { display: grid; gap: 2px; }
.nav .grp { color: var(--side-muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; padding: 12px 8px 4px; }
.nav a { color: var(--side-fg); display: flex; align-items: center; gap: 9px; padding: 7px 8px; border-radius: 6px; font-size: 13.5px; }
.nav a:hover { background: rgba(255, 255, 255, .06); text-decoration: none; }
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; background: var(--info); }
.dot.pass { background: #46c98a; } .dot.watch { background: #f0b450; } .dot.fail { background: #ff7a6e; }
.side .foot { margin-top: auto; color: var(--side-muted); font-size: 12px; padding: 0 8px; }

.main { min-width: 0; padding-inline: 28px; padding-block: 22px 56px; display: grid; gap: 22px; align-content: start; }
.top { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px 24px; }
.crumb { color: var(--muted); font-size: 12.5px; }
h1 { font-size: 24px; line-height: 1.25; margin: 2px 0 6px; letter-spacing: -.01em; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { border: 1px solid var(--line); background: var(--panel); border-radius: 6px; padding: 3px 8px; font-size: 12px; color: var(--fg2); }
.chip b { font-weight: 600; color: var(--fg); }
.summary { display: flex; gap: 8px; flex-wrap: wrap; }
.pill { display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 2px 9px; font-size: 12px; font-weight: 600; white-space: nowrap; }
.pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.pill.pass { color: var(--pass); background: var(--pass-bg); } .pill.watch { color: var(--watch); background: var(--watch-bg); }
.pill.fail { color: var(--fail); background: var(--fail-bg); } .pill.info { color: var(--info); background: var(--info-bg); }

.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; min-width: 0; }
.panel > .ph { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
  padding: 12px 16px; border-bottom: 1px solid var(--line); }
.panel > .ph h2 { font-size: 14.5px; margin: 0; }
.panel > .ph .sub { color: var(--muted); font-size: 12.5px; }
.pb { padding: 16px; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 12px; }
.kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 13px 14px 12px; display: grid; gap: 6px;
  color: inherit; min-width: 0; border-top: 3px solid var(--info); }
.kpi:hover { border-color: var(--line2); text-decoration: none; }
.kpi.pass { border-top-color: var(--pass); } .kpi.watch { border-top-color: var(--watch); } .kpi.fail { border-top-color: var(--fail); }
.kpi .row { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.kpi .pill { flex: none; }
.kpi .name { flex: 1; min-width: 0; font-weight: 600; font-size: 13px; color: var(--fg2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kpi .val { font-size: 28px; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; letter-spacing: -.01em; }
.kpi .lbl { color: var(--muted); font-size: 12.5px; line-height: 1.35; }
.kpi .why { font-size: 12px; font-weight: 600; }
.kpi .why.watch { color: var(--watch); } .kpi .why.fail { color: var(--fail); }
.kpi .ft { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; border-top: 1px dashed var(--line); padding-top: 6px; }

.tw { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); vertical-align: top; font-size: 13px; }
th { color: var(--muted); font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; background: var(--sunken); white-space: nowrap; }
tbody tr:hover td { background: var(--accent-bg); }
tr:last-child td { border-bottom: 0; }
td.num { text-align: right; white-space: nowrap; }
#health td:nth-child(6) { min-width: 150px; }
.layer { font: 500 11.5px/1 var(--mono); border: 1px solid var(--line2); border-radius: 4px; padding: 3px 6px; color: var(--fg2); white-space: nowrap; }

.eval { scroll-margin-top: 16px; }
.eh { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; padding: 14px 16px; border-bottom: 1px solid var(--line); }
.eh h2 { font-size: 17px; margin: 0; margin-right: 4px; }
.eh .q { flex-basis: 100%; color: var(--fg2); margin: 2px 0 0; max-width: 110ch; }
.tag { font: 500 11.5px/1 var(--mono); color: var(--muted); border: 1px solid var(--line); border-radius: 4px; padding: 4px 6px; }
.egrid { display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr); }
.hero { padding: 16px; border-right: 1px solid var(--line); display: grid; gap: 14px; align-content: start; }
.hero .big { font-size: 40px; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.hero .big.pass { color: var(--pass); } .hero .big.watch { color: var(--watch); } .hero .big.fail { color: var(--fail); }
.hero .hl { color: var(--fg2); font-size: 13px; }
.hero .bl { color: var(--muted); font-size: 12.5px; border-left: 3px solid var(--base); padding-left: 8px; }
.hero .bl b { color: var(--fg); font-size: 18px; display: block; font-variant-numeric: tabular-nums; }
.mlist { display: grid; gap: 0; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.mrow { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; padding: 7px 10px; border-bottom: 1px solid var(--line); font-size: 12.5px; }
.mrow:last-child { border-bottom: 0; }
.mrow .mv { font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.mrow .mv.pass { color: var(--pass); } .mrow .mv.watch { color: var(--watch); } .mrow .mv.fail { color: var(--fail); }
.body { padding: 16px; display: grid; gap: 18px; min-width: 0; }
.ct { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin: 0 0 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px; color: var(--fg2); margin-bottom: 8px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.hbar { display: grid; gap: 6px; }
.hbar .r { display: grid; grid-template-columns: minmax(0, 260px) minmax(0, 1fr) 64px; gap: 10px; align-items: center; font-size: 12.5px; }
.hbar .lb { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--fg2); }
.track { height: 12px; background: var(--sunken); border: 1px solid var(--line); border-radius: 3px; overflow: hidden; }
.fill { height: 100%; }
.hbar .v { text-align: right; font-variant-numeric: tabular-nums; color: var(--fg2); }
.grp { display: grid; gap: 9px; }
.grp .r { display: grid; grid-template-columns: minmax(0, 200px) minmax(0, 1fr); gap: 10px; align-items: center; font-size: 12.5px; }
.grp .bars { display: grid; gap: 3px; }
.grp .b { display: grid; grid-template-columns: minmax(0, 1fr) 44px; gap: 8px; align-items: center; }
.grp .b .track { height: 8px; }
.grp .b span { text-align: right; color: var(--muted); font-size: 11.5px; font-variant-numeric: tabular-nums; }
svg.line { width: 100%; height: auto; display: block; }
svg.line text { fill: var(--muted); font: 11px var(--sans); }
svg.line .grid { stroke: var(--line); }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
details > summary { cursor: pointer; list-style: none; color: var(--fg2); font-weight: 600; font-size: 13px; }
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "\25B8"; display: inline-block; width: 16px; color: var(--muted); }
details[open] > summary::before { content: "\25BE"; }
details[open] > summary { margin-bottom: 8px; }
.notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; border-top: 1px solid var(--line); padding: 14px 16px; background: var(--sunken); border-radius: 0 0 10px 10px; }
.notes h4 { margin: 0 0 6px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.notes p, .notes li { font-size: 12.5px; color: var(--fg2); margin: 0; }
.notes ul { margin: 0; padding-left: 16px; display: grid; gap: 4px; }
pre.ex { margin: 0; background: var(--sunken); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
  font: 12px/1.45 var(--mono); white-space: pre-wrap; overflow-wrap: anywhere; color: var(--fg2); }

@media (max-width: 900px) {
  .app { grid-template-columns: minmax(0, 1fr); }
  .side { position: static; height: auto; padding: 12px 16px; gap: 10px; }
  .nav { display: flex; overflow-x: auto; gap: 4px; }
  .nav .grp, .side .foot { display: none; }
  .nav a { white-space: nowrap; }
  .main { padding-inline: 16px; }
  .egrid { grid-template-columns: minmax(0, 1fr); }
  .hero { border-right: 0; border-bottom: 1px solid var(--line); }
  .hbar .r { grid-template-columns: minmax(0, 1fr) 56px; }
  .hbar .track { grid-column: 1 / -1; grid-row: 2; }
  .grp .r { grid-template-columns: minmax(0, 1fr); }
}
</style>

<div class="app">
  <aside class="side" aria-label="Navigation">
    <div class="brand"><div class="mark">MC</div><div><b>MemContext</b><span>Product evals</span></div></div>
    <nav class="nav" id="nav"></nav>
    <div class="foot" id="sidefoot"></div>
  </aside>
  <main class="main">
    <header class="top" id="overview">
      <div>
        <div class="crumb">MemContext / Quality / Product evals</div>
        <h1>Product evals</h1>
        <div class="chips" id="chips"></div>
      </div>
      <div class="summary" id="summary"></div>
    </header>
    <section class="kpis" id="kpis" aria-label="Key metrics"></section>
    <section class="panel">
      <div class="ph"><h2>Eval health</h2><span class="sub">Status: pass / watch / fail against each eval's threshold</span></div>
      <div class="tw"><table id="health"></table></div>
    </section>
    <section class="panel" id="findings">
      <div class="ph"><h2>Open findings</h2><span class="sub" id="fsub"></span></div>
      <div class="tw"><table id="ftable"></table></div>
    </section>
    <div id="evals" style="display:grid;gap:22px"></div>
  </main>
</div>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const when = s => { const d = new Date(s); return isNaN(d) ? String(s) : d.toISOString().slice(0, 16).replace("T", " ") + " UTC"; };
const STATUS = {pass: "Pass", watch: "Watch", fail: "Fail", info: "Info"};
const pill = s => el("span", "pill " + s, STATUS[s] || s);
const TONE = {good: "var(--pass)", bad: "var(--fail)", warn: "var(--watch)", base: "var(--base)", accent: "var(--accent)", muted: "var(--info)"};
const color = t => TONE[t] || t || "var(--accent)";

function table(t, columns, rows, numeric) {
  const thead = el("thead"), tr = el("tr");
  columns.forEach((c, i) => { const th = el("th", numeric && numeric.includes(i) ? "num" : null, c); tr.append(th); });
  thead.append(tr);
  const tb = el("tbody");
  for (const r of rows) {
    const row = el("tr");
    r.forEach((c, i) => { const td = el("td", numeric && numeric.includes(i) ? "num" : null); if (c instanceof Node) td.append(c); else td.textContent = c == null ? "" : String(c); row.append(td); });
    tb.append(row);
  }
  t.replaceChildren(thead, tb);
  return t;
}

/* ---- header, sidebar, KPIs, health ---- */
const counts = {pass: 0, watch: 0, fail: 0, info: 0};
D.evals.forEach(e => counts[e.status] = (counts[e.status] || 0) + 1);
for (const [k, v] of [["Evals", D.evals.length], ["Code", D.commit], ["Built", when(D.generated_utc)], ["Findings", D.issues.length]]) {
  const c = el("span", "chip"); c.append(`${k} `); c.append(el("b", null, v)); $("chips").append(c);
}
for (const s of ["pass", "watch", "fail"]) if (counts[s]) { const p = pill(s); p.textContent = `${counts[s]} ${STATUS[s]}`; $("summary").append(p); }

const nav = $("nav");
const navLink = (href, label, s) => { const a = el("a"); a.href = href; if (s) a.append(el("span", "dot " + s)); a.append(el("span", null, label)); return a; };
nav.append(navLink("#overview", "Overview"), navLink("#findings", "Open findings"), el("div", "grp", "Evals"));
D.evals.forEach(e => nav.append(navLink("#" + e.eval, e.title, e.status)));
$("sidefoot").textContent = "Built " + when(D.generated_utc);

for (const e of D.evals) {
  const h = e.headline, a = el("a", "kpi " + e.status); a.href = "#" + e.eval;
  const row = el("div", "row"); row.append(el("span", "name", e.title), pill(e.status));
  a.append(row, el("div", "val", h.value), el("div", "lbl", h.label));
  if (e.status_reason) a.append(el("div", "why " + e.status, "Status set by " + e.status_reason));
  const foot = [];
  if (h.baseline) foot.push(`baseline ${h.baseline.value}`);
  if (e.n) foot.push(`n = ${e.n}`);
  foot.push(e.deterministic ? "deterministic" : "live sessions");
  a.append(el("div", "ft", foot.join("  ·  ")));
  $("kpis").append(a);
}

function statusCell(e) {
  const d = el("div"); d.append(pill(e.status));
  if (e.status_reason) { const w = el("div", null, e.status_reason); w.style.cssText = "color:var(--muted);font-size:12px;margin-top:4px"; d.append(w); }
  return d;
}
table($("health"), ["Eval", "Promise", "Primary metric", "Result", "Baseline", "Status", "Type", "Run", "Code"],
  D.evals.map(e => {
    const link = el("a", null, e.title); link.href = "#" + e.eval;
    return [link, e.promise_short, e.headline.label, e.headline.value, e.headline.baseline ? e.headline.baseline.value : "—",
      statusCell(e), e.deterministic ? "Deterministic" : "LLM sessions", when(e.timestamp_utc), e.commit || ""];
  }), [3, 4]);

$("fsub").textContent = `${D.issues.length} diagnosed weaknesses, each with its code location. Not yet fixed.`;
table($("ftable"), ["Eval", "Finding", "Layer", "Location"],
  D.issues.map(i => {
    const a = el("a", null, i.eval_title); a.href = "#" + i.eval;
    return [a, i.text, i.layer ? el("span", "layer", i.layer) : "—", i.where ? el("code", null, i.where) : "—"];
  }));

/* ---- charts ---- */
function hbar(spec) {
  const box = el("div");
  box.append(el("div", "ct", spec.title));
  const g = el("div", "hbar"), max = spec.max || 100;
  for (const it of spec.items) {
    const r = el("div", "r"), t = el("div", "track"), f = el("div", "fill");
    f.style.width = Math.max(0, Math.min(100, 100 * it.value / max)) + "%";
    f.style.background = color(it.tone);
    t.append(f);
    r.append(el("span", "lb", it.label), t, el("span", "v", it.display ?? (it.value + (spec.unit || ""))));
    r.firstChild.title = it.label;
    g.append(r);
  }
  box.append(g);
  return box;
}

function grouped(spec) {
  const box = el("div");
  box.append(el("div", "ct", spec.title));
  const lg = el("div", "legend");
  spec.series.forEach(s => { const x = el("span"); const i = el("i"); i.style.background = color(s.tone); x.append(i, s.name); lg.append(x); });
  box.append(lg);
  const g = el("div", "grp");
  spec.categories.forEach((c, ci) => {
    const r = el("div", "r"), bars = el("div", "bars");
    spec.series.forEach(s => {
      const v = s.values[ci], b = el("div", "b"), t = el("div", "track"), f = el("div", "fill");
      f.style.width = (v == null ? 0 : Math.max(0, Math.min(100, v))) + "%"; f.style.background = color(s.tone);
      t.append(f); b.append(t, el("span", null, v == null ? "—" : v + (spec.unit || ""))); bars.append(b);
    });
    r.append(el("span", null, c), bars); g.append(r);
  });
  box.append(g);
  return box;
}

function line(spec) {
  const NS = "http://www.w3.org/2000/svg", W = 600, H = 230, L = 46, R = 44, T = 12, B = 34;
  const box = el("div");
  box.append(el("div", "ct", spec.title));
  const lg = el("div", "legend");
  spec.series.forEach(s => { const x = el("span"); const i = el("i"); i.style.background = color(s.tone); x.append(i, s.name); lg.append(x); });
  box.append(lg);
  const all = spec.series.flatMap(s => s.values).filter(v => v != null);
  const raw = spec.ymax || Math.max(1, ...all);
  const step = raw <= 100 && spec.unit === "%" ? 25 : Math.pow(10, Math.floor(Math.log10(raw))) * (raw / Math.pow(10, Math.floor(Math.log10(raw))) > 5 ? 2 : 1);
  const ymax = spec.unit === "%" ? 100 : Math.ceil(raw / step) * step;
  const xs = i => L + (spec.x.length === 1 ? (W - L - R) / 2 : i * (W - L - R) / (spec.x.length - 1));
  const ys = v => T + (H - T - B) * (1 - v / ymax);
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("class", "line"); svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", spec.title);
  const mk = (tag, attrs, text) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (text != null) n.textContent = text; svg.append(n); return n; };
  for (let v = 0; v <= ymax + 1e-9; v += (spec.unit === "%" ? 25 : step)) {
    mk("line", {x1: L, x2: W - R, y1: ys(v), y2: ys(v), class: "grid"});
    mk("text", {x: L - 8, y: ys(v) + 4, "text-anchor": "end"}, v + (spec.unit === "%" ? "%" : ""));
  }
  spec.x.forEach((x, i) => mk("text", {x: xs(i), y: H - 12, "text-anchor": "middle"}, String(x)));
  if (spec.xlabel) mk("text", {x: (L + W - R) / 2, y: H - 0, "text-anchor": "middle"}, spec.xlabel);
  for (const s of spec.series) {
    const pts = s.values.map((v, i) => v == null ? null : [xs(i), ys(v)]).filter(Boolean);
    mk("polyline", {points: pts.map(p => p.join(",")).join(" "), fill: "none", stroke: color(s.tone), "stroke-width": 2.5, "stroke-linejoin": "round"});
    pts.forEach(p => mk("circle", {cx: p[0], cy: p[1], r: 3.5, fill: color(s.tone)}));
    const lastV = s.values[s.values.length - 1];
    if (lastV != null) mk("text", {x: xs(s.values.length - 1) - 6, y: ys(lastV) - 8, "text-anchor": "end", style: `fill:${color(s.tone)};font-weight:600`}, lastV + (spec.unit || ""));
  }
  box.append(svg);
  return box;
}
const CHART = {hbar, grouped, line};

/* ---- eval panels ---- */
for (const e of D.evals) {
  const s = el("section", "panel eval"); s.id = e.eval;
  const eh = el("div", "eh");
  eh.append(el("h2", null, e.title), pill(e.status), el("span", "tag", e.deterministic ? "deterministic" : "LLM sessions"));
  if (e.commit) eh.append(el("span", "tag", "code " + e.commit + (e.product_code_dirty ? " + local edits" : "")));
  if (e.split) eh.append(el("span", "tag", e.split + " split"));
  eh.append(el("span", "tag", when(e.timestamp_utc)));
  eh.append(el("p", "q", e.question || ""));
  s.append(eh);

  const grid = el("div", "egrid"), hero = el("div", "hero");
  const h = e.headline;
  const big = el("div", "big " + e.status, h.value);
  hero.append(el("div", null), big, el("div", "hl", h.label));
  hero.firstChild.remove();
  if (h.baseline) { const b = el("div", "bl"); b.append(el("b", null, h.baseline.value), h.baseline.label); hero.append(b); }
  if (e.metrics && e.metrics.length) {
    const ml = el("div", "mlist");
    for (const m of e.metrics) {
      const r = el("div", "mrow"); r.title = m.definition || "";
      r.append(el("span", null, m.name), el("span", "mv " + (m.status || ""), m.value));
      ml.append(r);
    }
    hero.append(ml);
  }
  const body = el("div", "body");
  if (e.charts && e.charts.length) {
    const cg = el("div", "charts");
    e.charts.forEach(c => cg.append(CHART[c.type](c)));
    body.append(cg);
  }
  for (const tb of e.tables || []) {
    const d = el("details"); if (tb.rows.length <= 6) d.open = true;
    d.append(el("summary", null, `${tb.title} (${tb.rows.length})`));
    const tw = el("div", "tw"); tw.append(table(el("table"), tb.columns, tb.rows)); d.append(tw);
    body.append(d);
  }
  if (e.examples && e.examples.length) {
    const d = el("details"); d.append(el("summary", null, `Examples (${e.examples.length})`));
    const box = el("div"); box.style.display = "grid"; box.style.gap = "8px";
    e.examples.forEach(x => box.append(el("pre", "ex", typeof x === "string" ? x : JSON.stringify(x, null, 1))));
    d.append(box); body.append(d);
  }
  if (e.link) { const p = el("p"); p.style.margin = "0"; const a = el("a", null, "Open the per-question view →"); a.href = e.link; p.append(a); body.append(p); }
  grid.append(hero, body);
  s.append(grid);

  const notes = el("div", "notes");
  if (e.method) { const m = el("div"); m.append(el("h4", null, "Method"), el("p", null, e.method)); notes.append(m); }
  for (const [title, items] of [["Findings", e.findings], ["Limits", e.limits]]) {
    if (!items || !items.length) continue;
    const box = el("div"), ul = el("ul");
    box.append(el("h4", null, title)); items.forEach(i => ul.append(el("li", null, i))); box.append(ul); notes.append(box);
  }
  s.append(notes);
  $("evals").append(s);
}
</script>
"""
