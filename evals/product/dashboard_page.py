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
/* Layout: light sidebar (Overview + one item per eval) and a single view at a time.
   Overview = key result, one card per eval, issues list. Eval view = 4 tiles, chart, collapsed detail.
   Deliberately single-theme (light), every color set explicitly. */
:root {
  color-scheme: light;
  --bg: #f6f7f9; --panel: #ffffff; --line: #e6e8ec; --line2: #d9dde3; --hover: #f2f4f7;
  --fg: #101828; --fg2: #344054; --muted: #667085; --faint: #98a2b3;
  --accent: #2b59c3; --accent-bg: #eef3fd;
  --good: #12805c; --good-bg: #ecf7f2; --warn: #b54708; --warn-bg: #fef4e6; --bad: #c4320a; --bad-bg: #fdf0ec;
  --base: #c3c9d3;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.5 var(--sans); }
a { color: var(--accent); text-decoration: none; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
[hidden] { display: none !important; }

.app { display: grid; grid-template-columns: 232px minmax(0, 1fr); min-height: 100vh; }
.side { background: var(--panel); border-right: 1px solid var(--line); padding: 20px 12px; position: sticky;
  top: env(safe-area-inset-top, 0px); height: 100vh; display: flex; flex-direction: column; gap: 22px; }
.brand { display: flex; align-items: center; gap: 10px; padding: 0 8px; }
.mark { width: 28px; height: 28px; border-radius: 7px; background: var(--accent); color: #fff;
  display: grid; place-items: center; font: 600 12px/1 var(--mono); }
.brand b { font-size: 14px; display: block; }
.brand span { font-size: 12px; color: var(--muted); display: block; }
.nav { display: grid; gap: 2px; }
.nav .grp { font-size: 11px; font-weight: 600; color: var(--faint); text-transform: uppercase; letter-spacing: .08em; padding: 14px 10px 6px; }
.nav button { all: unset; box-sizing: border-box; display: flex; align-items: center; gap: 10px; padding: 7px 10px; border-radius: 6px;
  color: var(--fg2); font-size: 13.5px; cursor: pointer; }
.nav button:hover { background: var(--hover); }
.nav button.on { background: var(--accent-bg); color: var(--accent); font-weight: 600; }
.nav button:focus-visible { outline: 2px solid var(--accent); }
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; background: var(--faint); }
.dot.pass { background: var(--good); } .dot.watch { background: #f79009; } .dot.fail { background: var(--bad); }
.side .foot { margin-top: auto; font-size: 12px; color: var(--faint); padding: 0 10px; }

.main { min-width: 0; padding-inline: 32px; padding-block: 28px 56px; }
.view { max-width: 1160px; display: grid; gap: 20px; }
.vh h1 { font-size: 22px; font-weight: 600; margin: 0; letter-spacing: -.01em; }
.vh .meta { color: var(--muted); font-size: 13px; margin-top: 4px; display: flex; flex-wrap: wrap; gap: 4px 14px; }
.vh .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }

.pill { display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 600; }
.pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.pill.pass { color: var(--good); background: var(--good-bg); } .pill.watch { color: var(--warn); background: var(--warn-bg); }
.pill.fail { color: var(--bad); background: var(--bad-bg); } .pill.info { color: var(--muted); background: var(--hover); }

.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; min-width: 0; }
.card > h2 { font-size: 14px; font-weight: 600; margin: 0; padding: 14px 18px; border-bottom: 1px solid var(--line); }
.cb { padding: 18px; }

/* key result */
.key { display: grid; gap: 14px; }
.key .t { font-size: 13px; color: var(--muted); }
.cmp { display: grid; gap: 10px; }
.cmp .r { display: grid; grid-template-columns: 200px minmax(0, 1fr) 72px; gap: 14px; align-items: center; }
.cmp .n { font-size: 13.5px; color: var(--fg2); }
.cmp .bar { height: 26px; background: var(--hover); border-radius: 5px; overflow: hidden; }
.cmp .bar i { display: block; height: 100%; border-radius: 5px; min-width: 3px; }
.cmp .v { font-size: 20px; font-weight: 700; text-align: right; font-variant-numeric: tabular-nums; }

/* eval cards */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 14px; }
.tile { all: unset; box-sizing: border-box; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px;
  display: grid; gap: 10px; cursor: pointer; min-width: 0; }
.tile:hover { border-color: var(--line2); box-shadow: 0 1px 3px rgba(16, 24, 40, .06); }
.tile:focus-visible { outline: 2px solid var(--accent); }
.tile .top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.tile .nm { font-size: 13.5px; font-weight: 600; color: var(--fg2); min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tile .pill { flex: none; }
.tile .big { font-size: 30px; font-weight: 700; line-height: 1; letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
.tile .lb { font-size: 13px; color: var(--muted); }
.tile .vs { font-size: 12.5px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 9px; }
.tile .vs b { color: var(--fg2); font-variant-numeric: tabular-nums; }

/* card visuals */
.viz { min-height: 34px; display: grid; align-content: center; }
.arms { display: grid; gap: 5px; }
.arms div { display: grid; grid-template-columns: 112px minmax(0, 1fr) 40px; gap: 8px; align-items: center; font-size: 11.5px; color: var(--muted); }
.arms .tr { height: 6px; background: var(--hover); border-radius: 3px; overflow: hidden; }
.arms .tr i { display: block; height: 100%; border-radius: 3px; min-width: 2px; }
.arms span:first-child { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.arms b { font-weight: 600; color: var(--fg2); text-align: right; font-variant-numeric: tabular-nums; }
.dots { display: grid; grid-template-columns: repeat(15, 1fr); gap: 3px; max-width: 220px; }
.dots i { aspect-ratio: 1; border-radius: 2px; background: var(--hover); border: 1px solid var(--line); }
.checks { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 12px; font-size: 12px; color: var(--fg2); }
.checks span::before { content: "\2713"; color: var(--good); font-weight: 700; margin-right: 6px; }
.checks span.no::before { content: "\2715"; color: var(--bad); }
.stack { display: flex; height: 8px; border-radius: 4px; overflow: hidden; gap: 2px; }
.spark { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 8px; align-items: center; font-size: 11.5px; color: var(--muted); }
.spark svg { width: 100%; height: 30px; display: block; }

/* issues */
.issues { display: grid; }
.iss { border-bottom: 1px solid var(--line); }
.iss:last-child { border-bottom: 0; }
.iss summary { list-style: none; cursor: pointer; display: grid; grid-template-columns: 10px 150px minmax(0, 1fr) auto; gap: 12px;
  align-items: center; padding: 11px 18px; }
.iss summary::-webkit-details-marker { display: none; }
.iss summary:hover { background: var(--hover); }
.iss .area { font-size: 13px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.iss .txt { font-size: 13.5px; color: var(--fg); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.loc { font: 12px/1 var(--mono); color: var(--fg2); background: var(--hover); border-radius: 4px; padding: 4px 7px; white-space: nowrap; }
.iss .full { padding: 0 18px 14px 190px; color: var(--fg2); font-size: 13px; max-width: 110ch; }

/* eval view */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }
.m { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; display: grid; gap: 6px; min-width: 0; }
.m .k { font-size: 12.5px; color: var(--muted); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.m .v { font-size: 24px; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; letter-spacing: -.01em; }
.m .v.pass { color: var(--good); } .m .v.watch { color: var(--warn); } .m .v.fail { color: var(--bad); }
.m .s { font-size: 12px; color: var(--muted); }
.m.hero { border-left: 3px solid var(--accent); }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px; }
.ct { font-size: 13px; font-weight: 600; color: var(--fg2); margin: 0 0 14px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--muted); margin: -6px 0 12px; }
.legend i { display: inline-block; width: 10px; height: 3px; border-radius: 2px; margin-right: 6px; vertical-align: 3px; }
.hb { display: grid; gap: 9px; }
.hb .r { display: grid; grid-template-columns: minmax(0, 210px) minmax(0, 1fr) 52px; gap: 12px; align-items: center; font-size: 12.5px; }
.hb .lb { color: var(--fg2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hb .tr { height: 10px; background: var(--hover); border-radius: 3px; overflow: hidden; }
.hb .tr i { display: block; height: 100%; border-radius: 3px; }
.hb .v { text-align: right; font-variant-numeric: tabular-nums; color: var(--fg2); }
.gb .r { grid-template-columns: minmax(0, 170px) minmax(0, 1fr); align-items: start; }
.gb .pair { display: grid; gap: 4px; }
.gb .pair div { display: grid; grid-template-columns: minmax(0, 1fr) 44px; gap: 10px; align-items: center; }
.gb .pair .tr { height: 7px; }
.gb .pair span { text-align: right; font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
svg.ln { width: 100%; height: auto; display: block; }
svg.ln text { fill: var(--muted); font: 11px var(--sans); }
svg.ln .g { stroke: var(--line); }
.flist { display: grid; }
.flist div { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 12px; align-items: start; padding: 10px 18px; border-bottom: 1px solid var(--line); font-size: 13px; color: var(--fg2); }
.flist div:last-child { border-bottom: 0; }
details.more { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
details.more > summary { list-style: none; cursor: pointer; padding: 13px 18px; font-size: 13.5px; font-weight: 600; color: var(--fg2);
  display: flex; justify-content: space-between; }
details.more > summary::-webkit-details-marker { display: none; }
details.more > summary::after { content: "Show"; font-weight: 500; color: var(--accent); font-size: 13px; }
details.more[open] > summary::after { content: "Hide"; }
details.more[open] > summary { border-bottom: 1px solid var(--line); }
.dbody { padding: 16px 18px; display: grid; gap: 18px; }
.dbody h3 { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin: 0 0 8px; }
.dbody p, .dbody li { font-size: 13px; color: var(--fg2); margin: 0; max-width: 110ch; }
.dbody ul { margin: 0; padding-left: 18px; display: grid; gap: 4px; }
.tw { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { background: #fafbfc; color: var(--muted); font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }

@media (max-width: 900px) {
  .app { grid-template-columns: minmax(0, 1fr); }
  .side { position: static; height: auto; padding: 12px 16px; gap: 10px; border-right: 0; border-bottom: 1px solid var(--line); }
  .nav { display: flex; overflow-x: auto; }
  .nav .grp, .side .foot { display: none; }
  .nav button { white-space: nowrap; }
  .main { padding-inline: 16px; }
  .cmp .r { grid-template-columns: minmax(0, 1fr) 60px; }
  .cmp .bar { grid-column: 1 / -1; grid-row: 2; }
  .iss summary { grid-template-columns: 10px minmax(0, 1fr); }
  .iss .area, .iss .loc { display: none; }
  .iss .full { padding-left: 18px; }
  .charts { grid-template-columns: minmax(0, 1fr); }
  .hb .r { grid-template-columns: minmax(0, 1fr) 52px; }
  .hb .tr { grid-column: 1 / -1; grid-row: 2; }
}
</style>

<div class="app">
  <aside class="side">
    <div class="brand"><div class="mark">MC</div><div><b>MemContext</b><span>Product evals</span></div></div>
    <nav class="nav" id="nav" aria-label="Views"></nav>
    <div class="foot" id="foot"></div>
  </aside>
  <main class="main" id="main"></main>
</div>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const when = s => { const d = new Date(s); return isNaN(d) ? String(s || "") : d.toISOString().slice(0, 16).replace("T", " ") + " UTC"; };
const NAME = {pass: "Pass", watch: "Watch", fail: "Fail", info: "Info"};
const pill = s => el("span", "pill " + s, NAME[s] || s);
const TONE = {good: "var(--good)", bad: "var(--bad)", warn: "#f79009", base: "var(--base)", accent: "var(--accent)"};
const col = t => TONE[t] || "var(--accent)";
const firstSentence = t => { const m = String(t).match(/^(.+?[.;])(\s|$)/); return (m ? m[1] : t).replace(/[.;]$/, ""); };
const counts = {pass: 0, watch: 0, fail: 0};
D.evals.forEach(e => { if (e.status in counts) counts[e.status]++; });

/* ---------- views ---------- */
const views = {};
function show(id) {
  if (!views[id]) id = "overview";
  for (const k in views) views[k].hidden = k !== id;
  document.querySelectorAll(".nav button").forEach(b => b.classList.toggle("on", b.dataset.id === id));
  try { history.replaceState(null, "", "#" + id); } catch (_) {}
  window.scrollTo(0, 0);
}

function navButton(id, label, status) {
  const b = el("button"); b.dataset.id = id; b.type = "button";
  if (status) b.append(el("span", "dot " + status));
  b.append(el("span", null, label)); b.onclick = () => show(id);
  return b;
}
$("nav").append(navButton("overview", "Overview"), el("div", "grp", "Evals"));
D.evals.forEach(e => $("nav").append(navButton(e.eval, e.title, e.status)));
$("foot").textContent = "Built " + when(D.generated_utc);

/* ---------- card visuals ---------- */
function vizEl(c) {
  const box = el("div", "viz");
  if (!c) return box;
  if (c.type === "arms") {
    const g = el("div", "arms");
    c.rows.forEach(r => { const d = el("div"), tr = el("span", "tr"), i = el("i");
      i.style.width = Math.max(0, Math.min(100, r.pct || 0)) + "%"; i.style.background = col(r.tone); tr.append(i);
      d.append(el("span", null, r.label), tr, el("b", null, r.value)); g.append(d); });
    box.append(g);
  } else if (c.type === "dots") {
    const g = el("div", "dots");
    for (let k = 0; k < c.n; k++) { const i = el("i"); if (k < c.hits) { i.style.background = col(c.tone); i.style.borderColor = "transparent"; } g.append(i); }
    box.append(g);
  } else if (c.type === "checks") {
    const g = el("div", "checks"); c.items.forEach(it => g.append(el("span", it.ok ? "" : "no", it.label))); box.append(g);
  } else if (c.type === "stack") {
    const g = el("div", "stack"), tot = c.parts.reduce((a, p) => a + p.value, 0) || 1;
    c.parts.forEach(p => { const i = el("i"); i.style.flex = String(p.value / tot); i.style.background = col(p.tone); g.append(i); });
    box.append(g);
  } else if (c.type === "spark") {
    const NS = "http://www.w3.org/2000/svg", W = 160, H = 30, v = c.values, mx = Math.max(...v, 1), mn = Math.min(...v, 0);
    const x = k => 3 + k * (W - 6) / Math.max(1, v.length - 1), y = val => 3 + (H - 6) * (1 - (val - mn) / ((mx - mn) || 1));
    const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("preserveAspectRatio", "none");
    const pl = document.createElementNS(NS, "polyline");
    pl.setAttribute("points", v.map((val, k) => x(k) + "," + y(val)).join(" ")); pl.setAttribute("fill", "none");
    pl.setAttribute("stroke", col(c.tone)); pl.setAttribute("stroke-width", "2"); pl.setAttribute("vector-effect", "non-scaling-stroke");
    svg.append(pl);
    const g = el("div", "spark"); g.append(el("span", null, c.from || ""), svg, el("span", null, c.to || "")); box.append(g);
  }
  return box;
}

/* ---------- overview ---------- */
const ov = el("section", "view"); views.overview = ov;
{
  const vh = el("div", "vh");
  vh.append(el("h1", null, "Product evals"));
  const meta = el("div", "meta");
  meta.append(el("span", null, `${D.evals.length} evals`), el("span", null, `${counts.pass} pass · ${counts.watch} watch · ${counts.fail} fail`),
              el("span", null, `${D.issues.length} open issues`), el("span", null, "Updated " + when(D.generated_utc)));
  vh.append(meta); ov.append(vh);

  if (D.key) {
    const c = el("div", "card"), h = el("h2", null, D.key.title), b = el("div", "cb key");
    b.append(el("div", "t", D.key.subtitle));
    const cmp = el("div", "cmp");
    for (const r of D.key.rows) {
      const row = el("div", "r"), bar = el("div", "bar"), i = el("i");
      i.style.width = r.pct + "%"; i.style.background = col(r.tone); bar.append(i);
      const v = el("span", "v", r.value); v.style.color = col(r.tone);
      row.append(el("span", "n", r.label), bar, v); cmp.append(row);
    }
    b.append(cmp); c.append(h, b); ov.append(c);
  }

  const grid = el("div", "grid");
  for (const e of D.evals) {
    const t = el("button", "tile"); t.type = "button"; t.onclick = () => show(e.eval);
    const top = el("div", "top"); top.append(el("span", "nm", e.title), pill(e.status));
    t.append(top, el("div", "big", e.headline.value), el("div", "lb", e.short || e.headline.label), vizEl(e.card));
    grid.append(t);
  }
  ov.append(grid);

  const ic = el("div", "card"); ic.append(el("h2", null, `Open issues (${D.issues.length})`));
  const list = el("div", "issues");
  for (const i of D.issues) {
    const d = el("details", "iss"), s = el("summary");
    s.append(el("span", "dot " + i.status), el("span", "area", i.eval_title), el("span", "txt", firstSentence(i.text)));
    s.append(i.where ? el("span", "loc", i.where) : el("span"));
    d.append(s, el("div", "full", i.text)); list.append(d);
  }
  ic.append(list); ov.append(ic);
}
$("main").append(ov);

/* ---------- charts ---------- */
function hbar(spec) {
  const box = el("div", "card cb"); box.append(el("div", "ct", spec.title));
  const g = el("div", "hb"), max = spec.max || 100;
  for (const it of spec.items) {
    const r = el("div", "r"), tr = el("div", "tr"), i = el("i");
    i.style.width = Math.max(0, Math.min(100, 100 * it.value / max)) + "%"; i.style.background = col(it.tone);
    tr.append(i); const lb = el("span", "lb", it.label); lb.title = it.label;
    r.append(lb, tr, el("span", "v", it.display ?? it.value)); g.append(r);
  }
  box.append(g); return box;
}
function grouped(spec) {
  const box = el("div", "card cb"); box.append(el("div", "ct", spec.title));
  const lg = el("div", "legend");
  spec.series.forEach(s => { const x = el("span"), i = el("i"); i.style.background = col(s.tone); x.append(i, s.name); lg.append(x); });
  box.append(lg);
  const g = el("div", "hb gb");
  spec.categories.forEach((c, ci) => {
    const r = el("div", "r"), pair = el("div", "pair");
    spec.series.forEach(s => {
      const v = s.values[ci], d = el("div"), tr = el("div", "tr"), i = el("i");
      i.style.width = (v == null ? 0 : v) + "%"; i.style.background = col(s.tone); tr.append(i);
      d.append(tr, el("span", null, v == null ? "—" : v + "%")); pair.append(d);
    });
    r.append(el("span", "lb", c), pair); g.append(r);
  });
  box.append(g); return box;
}
function line(spec) {
  const NS = "http://www.w3.org/2000/svg", W = 560, H = 220, L = 44, R = 46, T = 10, B = 30;
  const box = el("div", "card cb"); box.append(el("div", "ct", spec.title));
  const lg = el("div", "legend");
  spec.series.forEach(s => { const x = el("span"), i = el("i"); i.style.background = col(s.tone); x.append(i, s.name); lg.append(x); });
  box.append(lg);
  const pctAxis = spec.unit === "%";
  const raw = Math.max(1, ...spec.series.flatMap(s => s.values).filter(v => v != null));
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = pctAxis ? 25 : (raw / mag > 5 ? 2 * mag : raw / mag > 2 ? mag : mag / 2);
  const ymax = pctAxis ? 100 : Math.ceil(raw / step) * step;
  const xs = i => L + (spec.x.length < 2 ? (W - L - R) / 2 : i * (W - L - R) / (spec.x.length - 1));
  const ys = v => T + (H - T - B) * (1 - v / ymax);
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("class", "ln"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", spec.title);
  const mk = (tag, a, text) => { const n = document.createElementNS(NS, tag); for (const k in a) n.setAttribute(k, a[k]); if (text != null) n.textContent = text; svg.append(n); return n; };
  for (let v = 0; v <= ymax + 1e-9; v += step) {
    mk("line", {x1: L, x2: W - R, y1: ys(v), y2: ys(v), class: "g"});
    mk("text", {x: L - 8, y: ys(v) + 4, "text-anchor": "end"}, pctAxis ? v + "%" : String(+v.toFixed(2)));
  }
  spec.x.forEach((x, i) => mk("text", {x: xs(i), y: H - 8, "text-anchor": "middle"}, String(x)));
  for (const s of spec.series) {
    const pts = s.values.map((v, i) => v == null ? null : [xs(i), ys(v)]).filter(Boolean);
    mk("polyline", {points: pts.map(p => p.join(",")).join(" "), fill: "none", stroke: col(s.tone), "stroke-width": 2.25, "stroke-linejoin": "round"});
    pts.forEach(p => mk("circle", {cx: p[0], cy: p[1], r: 3, fill: "#fff", stroke: col(s.tone), "stroke-width": 2}));
    const lv = s.values[s.values.length - 1];
    if (lv != null) mk("text", {x: xs(s.values.length - 1) + 8, y: ys(lv) + 4, style: `fill:${col(s.tone)};font-weight:600`}, (pctAxis ? lv + "%" : String(Math.round(lv))));
  }
  box.append(svg); return box;
}
const CHART = {hbar, grouped, line};

/* ---------- one view per eval ---------- */
function tableEl(tb) {
  const w = el("div", "tw"), t = el("table"), th = el("thead"), tr = el("tr");
  tb.columns.forEach(c => tr.append(el("th", null, c))); th.append(tr);
  const body = el("tbody");
  tb.rows.forEach(r => { const row = el("tr"); r.forEach(c => row.append(el("td", null, c == null ? "" : String(c)))); body.append(row); });
  t.append(th, body); w.append(t); return w;
}

for (const e of D.evals) {
  const v = el("section", "view"); v.hidden = true; views[e.eval] = v;
  const vh = el("div", "vh"), row = el("div", "row");
  row.append(el("h1", null, e.title), pill(e.status)); vh.append(row);
  const meta = el("div", "meta");
  meta.append(el("span", null, e.question || ""));
  vh.append(meta);
  const meta2 = el("div", "meta");
  meta2.append(el("span", null, e.deterministic ? "Deterministic" : "Live Claude Code sessions"));
  if (e.n) meta2.append(el("span", null, "n = " + e.n));
  if (e.commit) meta2.append(el("span", null, "Code " + e.commit));
  meta2.append(el("span", null, "Run " + when(e.timestamp_utc)));
  vh.append(meta2); v.append(vh);

  const tiles = el("div", "tiles");
  const hero = el("div", "m hero");
  hero.append(el("div", "k", e.short || e.headline.label), el("div", "v " + e.status, e.headline.value));
  if (e.headline.baseline) hero.append(el("div", "s", `${e.vs_label || "Baseline"}: ${e.headline.baseline.value}`));
  tiles.append(hero);
  (e.metrics || []).filter(m => m.value !== e.headline.value).slice(0, 3).forEach(m => {
    const t = el("div", "m"); t.title = m.definition || "";
    t.append(el("div", "k", m.name), el("div", "v " + (m.status || ""), m.value));
    tiles.append(t);
  });
  v.append(tiles);
  if (e.status_reason) {
    const why = el("div", "meta"); why.style.marginTop = "-6px";
    why.append(el("span", null, "Status is set by " + e.status_reason)); v.append(why);
  }

  if (e.charts && e.charts.length) {
    const cg = el("div", "charts"); e.charts.slice(0, 2).forEach(c => cg.append(CHART[c.type](c))); v.append(cg);
  }

  const mine = D.issues.filter(i => i.eval === e.eval);
  if (mine.length) {
    const c = el("div", "card"); c.append(el("h2", null, `Issues (${mine.length})`));
    const l = el("div", "flist");
    mine.forEach(i => { const d = el("div"); d.append(el("span", null, i.text), i.where ? el("span", "loc", i.where) : el("span")); l.append(d); });
    c.append(l); v.append(c);
  }

  const more = el("details", "more");
  more.append(el("summary", null, "Data, method and limits"));
  const db = el("div", "dbody");
  (e.tables || []).forEach(tb => { const s = el("div"); s.append(el("h3", null, tb.title), tableEl(tb)); db.append(s); });
  if (e.method) { const s = el("div"); s.append(el("h3", null, "Method"), el("p", null, e.method)); db.append(s); }
  const notes = (e.findings || []).filter(f => !D.issues.some(i => i.eval === e.eval && i.text === f));
  for (const [title, items] of [["Notes", notes], ["Limits", e.limits || []]]) {
    if (!items.length) continue;
    const s = el("div"), ul = el("ul"); items.forEach(i => ul.append(el("li", null, i))); s.append(el("h3", null, title), ul); db.append(s);
  }
  if (e.link) { const a = el("a", null, "Open the per-question view"); a.href = e.link; db.append(a); }
  more.append(db); v.append(more);
  $("main").append(v);
}

show((location.hash || "").slice(1) || "overview");
</script>
"""
