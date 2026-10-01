"""Provenance-integrity eval: when a decision has changed, can you trust its history?

Eval harness, NOT product code. Deterministic, lexical, no LLM, no judge.

  run  ingest the stale-exposure decision histories (datasets/stale_exposure.json)
       through the public MemContext API (stale_exposure.build_turns + ingest), then
       call the real memory_trace tool (mcp_tools.handle_memory_trace, what Claude
       Code's memory_trace calls) once per history and score the returned lineage.

Per changed history (2-3 values), all checks against DOCUMENTED behaviour:
  complete  the lineage holds exactly the history's values, one version each,
            nothing from another subject
  ordered   oldest-first lineage == the order the values were decided, exactly one
            live version, and it is the newest (current) value
  grounded  every version's source turn text contains that version's value
  edges     every older version is `superseded`, linked to the next by the edge the
            Pass-1 contract documents for a user restating a user value
            (memcontext/supersession.py _classify_edge): REFINES when the new value's
            tokens are a strict subset of the old one's, else USER_CORRECTION
A changed history is "fully correct" when all four hold. Unchanged facts (1 value)
must trace to exactly one live version with no supersession.

Layouts: per-change (one session per change; the headline) and epoch (one session per
time step). Stress row, expected_limitation, never in the headline: subject drift, the
final change stored under a paraphrased subject (alt_subject).

  python evals/product/provenance_trace.py run [--repo PATH]
Result: evals/product/results/provenance_trace/latest.json (common eval schema).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stale_exposure as se  # noqa: E402

OUT = HERE / "results" / "provenance_trace" / "latest.json"
PREDICATE = "decision_made"
TRACE_SESSION = "default"  # a fresh session with no claims: the documented any-session fallback
LIVE = {"active", "confirmed", "audited"}
CHECKS = ("complete", "ordered", "grounded", "edges")
EXAMPLE_SUBJECT = "orders-service database"


# ----------------------------------------------------------------- scoring ---
def _tokens(value: str) -> set[str]:
    # the same tokenisation the documented REFINES rule uses (supersession._tokens)
    return {t for t in re.split(r"[\s,;/]+", value.lower().strip()) if t}


def expected_edge(old: str, new: str) -> str:
    """The Pass-1 edge documented for a user restating a user value."""
    o, n = _tokens(old), _tokens(new)
    return "refines" if o and n and n < o else "user_correction"


def oldest_first(trace: dict) -> list[dict]:
    return list(reversed(trace.get("lineage") or []))


def score_changed(values: list[str], trace: dict) -> dict:
    """Score one changed history's trace payload. Returns {check: bool, ...}."""
    steps = oldest_first(trace)
    got = [se._norm(s.get("value") or "") for s in steps]
    want = [se._norm(v) for v in values]
    live = [s for s in steps if s.get("status") in LIVE]
    complete = sorted(got) == sorted(want)
    ordered = (got == want and len(live) == 1 and bool(steps)
               and steps[-1].get("status") in LIVE)
    grounded = bool(steps) and all(se.mentions(s.get("text") or "", s.get("value") or "")
                                   for s in steps)
    edges = ordered and all(
        s.get("status") == "superseded"
        and s.get("edge_type") == expected_edge(values[i], values[i + 1])
        for i, s in enumerate(steps[:-1]))
    r = {"complete": complete, "ordered": ordered, "grounded": grounded, "edges": edges}
    r["fully_correct"] = all(r[c] for c in CHECKS)
    r["edge_types"] = [s.get("edge_type") for s in steps[:-1]]
    r["error"] = trace.get("error")
    return r


def score_unchanged(value: str, trace: dict) -> bool:
    steps = oldest_first(trace)
    return (len(steps) == 1 and steps[0].get("status") in LIVE
            and se._norm(steps[0].get("value") or "") == se._norm(value)
            and steps[0].get("edge_type") == "active"
            and se.mentions(steps[0].get("text") or "", value))


def score_drift(values: list[str], trace_old: dict, trace_alt: dict) -> dict:
    """Subject drift: the last value was stored under alt_subject."""
    def head(t: dict) -> str | None:
        steps = oldest_first(t)
        return se._norm(steps[-1]["value"]) if steps else None
    cur = se._norm(values[-1])
    conflicts = [se._norm(c.get("value") or "") for c in trace_old.get("conflicts") or []]
    return {
        "old_subject_current": head(trace_old) == cur,
        "alt_subject_full_history": score_changed(values, trace_alt)["fully_correct"],
        "drift_flagged": cur in conflicts,  # trace of the old subject names the new value
    }


def render_steps(trace: dict) -> list[dict]:
    return [{"value": s.get("value"), "status": s.get("status"),
             "source_text": s.get("text"), "edge": s.get("edge_type")}
            for s in oldest_first(trace)]


# --------------------------------------------------------------------- run ---
def trace(mc, conn, subject: str) -> dict:
    return mc.mcp_tools.handle_memory_trace(conn, session_id=TRACE_SESSION,
                                            subject=subject, predicate=PREDICATE)


def run_layout(mc, hist: list[dict], layout: str, drift: bool, workdir: Path) -> dict:
    db = workdir / f"{layout}{'_drift' if drift else ''}.db"
    se.ingest(mc, db, se.build_turns(hist, layout, drift), hist)
    conn = mc.open_database(db)
    changed, unchanged, drifted, examples = [], [], [], []
    for h in hist:
        vals = h["values"]
        if len(vals) == 1:
            unchanged.append(score_unchanged(vals[0], trace(mc, conn, h["subject"])))
            continue
        if drift:
            drifted.append(score_drift(vals, trace(mc, conn, h["subject"]),
                                       trace(mc, conn, h["alt_subject"])))
            continue
        t = trace(mc, conn, h["subject"])
        s = score_changed(vals, t)
        changed.append(s)
        examples.append({"subject": h["subject"], "values": vals, "score": s,
                         "versions": render_steps(t)})
    conn.close()
    return {"changed": changed, "unchanged": unchanged, "drift": drifted, "examples": examples}


def _count(rows: list[dict], key: str) -> int:
    return sum(1 for r in rows if r[key])


def _pct(k: int, n: int) -> float | None:
    return round(100.0 * k / n, 1) if n else None


def product_dirty(repo: Path) -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", "memcontext", "predicate_packs"],
                             cwd=repo, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.strip())


def build_result(repo: Path, layouts: dict, n_hist: int) -> dict:
    main = layouts["per-change"]
    ch, un = main["changed"], main["unchanged"]
    n = len(ch)
    full = _count(ch, "fully_correct")
    defs = {
        "complete": "the trace holds exactly the history's values, one version each",
        "ordered": "versions in decision order; exactly one live version, the current one",
        "grounded": "each version's source message contains that version's value",
        "edges": "each older version superseded by the next with the documented edge type",
    }
    metrics = [{"name": c, "value": f"{_count(ch, c)}/{n}", "pct": _pct(_count(ch, c), n), "n": n,
                "definition": defs[c], "better": "higher"} for c in CHECKS]
    metrics.append({"name": "unchanged-correct", "value": f"{sum(un)}/{len(un)}",
                    "pct": _pct(sum(un), len(un)), "n": len(un),
                    "definition": "a never-changed decision traces to one live version, no supersession",
                    "better": "higher"})

    rows = []
    for name in ("per-change", "epoch"):
        lc, lu = layouts[name]["changed"], layouts[name]["unchanged"]
        rows.append([name, len(lc), *[_count(lc, c) for c in CHECKS], _count(lc, "fully_correct"),
                     f"{sum(lu)}/{len(lu)}", "normal"])
    dr = layouts["drift"]["drift"]
    drift_rows = [["subject drift (stress)", len(dr), _count(dr, "old_subject_current"),
                   _count(dr, "alt_subject_full_history"), _count(dr, "drift_flagged"),
                   "expected_limitation"]]

    edge_seen: dict[str, int] = {}
    for r in ch:
        for e in r["edge_types"]:
            edge_seen[str(e)] = edge_seen.get(str(e), 0) + 1

    exs = main["examples"]
    good = next((e for e in exs if e["subject"] == EXAMPLE_SUBJECT and e["score"]["fully_correct"]),
                next((e for e in exs if e["score"]["fully_correct"] and len(e["values"]) == 3), None))
    bad = [e for e in exs if not e["score"]["fully_correct"]][:3]
    examples = ([{"kind": "correct", "subject": good["subject"], "versions": good["versions"]}] if good else [])
    examples += [{"kind": "failure", "subject": e["subject"], "expected": e["values"],
                  "failed_checks": [c for c in CHECKS if not e["score"][c]],
                  "versions": e["versions"]} for e in bad]

    findings = []
    for e in bad:
        findings.append(f"{e['subject']}: failed {', '.join(c for c in CHECKS if not e['score'][c])}; "
                        f"trace values {[v['value'] for v in e['versions']]} vs history {e['values']}")
    d = dr
    if d:
        findings.append(f"subject drift: trace of the original subject still names the outdated value as "
                        f"current in {len(d) - _count(d, 'old_subject_current')}/{len(d)} histories; "
                        f"the new subject's trace has no history in "
                        f"{len(d) - _count(d, 'alt_subject_full_history')}/{len(d)}; the old subject's trace "
                        f"lists the newer value under conflicts in {_count(d, 'drift_flagged')}/{len(d)} "
                        f"(LIPI Logic: decisions are linked by subject words, memcontext/conflicts.py:51 "
                        f"live_same_kind; reworded subjects share none)")
    if "refines" not in edge_seen:
        findings.append("coverage gap: no history changes to a token-subset of its old value, so the "
                        "REFINES edge (memcontext/supersession.py:52) is never exercised here")

    return {
        "eval": "provenance_trace",
        "title": "Provenance integrity",
        "promise": "Current, not stale: when a decision changes, the old value is kept as traceable "
                   "history, never lost and never served as current.",
        "question": "When a decision has changed, does memory_trace give back its complete, correctly "
                    "ordered history, each version tied to the message that said it?",
        "method": f"{n_hist} hand-written decision histories ({n} changed 1-2 times, {len(un)} never "
                  f"changed) stored through the public API, one structured decision_made claim per "
                  f"message; then the real memory_trace tool (handle_memory_trace, session "
                  f"'{TRACE_SESSION}') per history. Scored against documented behaviour, not current "
                  f"output. Deterministic, lexical mode, no LLM.",
        "commit": se.repo_sha(repo),
        "product_code_dirty": product_dirty(repo),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "deterministic": True,
        "headline": {"label": "changed decisions with a fully correct trace", "value": f"{full}/{n}",
                     "pct": _pct(full, n), "better": "higher"},
        "metrics": metrics,
        "tables": [
            {"title": "Checks passed by layout (changed decisions)",
             "columns": ["layout", "n", *CHECKS, "fully correct", "unchanged correct", "label"],
             "rows": rows},
            {"title": "Stress: subject drift (final change saved under a reworded subject)",
             "columns": ["condition", "n", "old subject traces to current value",
                         "new subject has full history", "old subject's trace flags the new value",
                         "label"],
             "rows": drift_rows},
            {"title": "Edge types on the per-change traces",
             "columns": ["edge type", "count"],
             "rows": [[k, v] for k, v in sorted(edge_seen.items())]},
        ],
        "examples": examples,
        "limits": [
            "Hand-written synthetic histories, one author; every value is stored as a structured "
            "claim, so extraction quality is not tested here.",
            "All changes come from the user; assistant-sourced changes and the trust guard are "
            "covered by the supersession matrix, not this eval.",
            "Subject drift is an expected limitation: decisions are linked by subject name, not "
            "by meaning.",
        ],
        "findings": findings,
    }


def cmd_run(a: argparse.Namespace) -> None:
    repo = Path(a.repo)
    hist = json.loads(Path(a.dataset).read_text(encoding="utf-8"))["histories"]
    mc = se.load_memcontext(repo, "lexical")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        layouts = {"per-change": run_layout(mc, hist, "per-change", False, work),
                   "epoch": run_layout(mc, hist, "epoch", False, work),
                   "drift": run_layout(mc, hist, "per-change", True, work)}
    result = build_result(repo, layouts, len(hist))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    h = result["headline"]
    print(f"Provenance integrity at {result['commit'][:10]}"
          f"{' (product code dirty)' if result['product_code_dirty'] else ''}")
    print(f"  {h['label']}: {h['value']} ({h['pct']}%)")
    for m in result["metrics"]:
        print(f"  {m['name']:<18} {m['value']:>6}  {m['definition']}")
    for t in result["tables"][:2]:
        print(f"  {t['title']}:")
        for r in t["rows"]:
            print("    " + " | ".join(str(x) for x in r))
    for f in result["findings"]:
        print(f"  finding: {f}")
    print(f"wrote {OUT.relative_to(se.REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ingest histories, trace each, score")
    r.add_argument("--repo", default=str(se.REPO), help="MemContext checkout under test")
    r.add_argument("--dataset", default=str(HERE / "datasets" / "stale_exposure.json"))
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
