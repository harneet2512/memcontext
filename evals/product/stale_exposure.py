"""Stale-exposure eval: does memory serve the CURRENT value of a fact after it changes?

Eval harness, NOT product code. Deterministic, retrieval-level, no LLM, no judge.

  run      ingest hand-written decision histories through the public MemContext API
           (handle_memory_store, structured decision_made claims, one session per
           change by default), then serve the same questions through several
           conditions and score what each condition SERVED.
  compare  diff two finished runs (determinism check) and print both tables.

Examples (from the repo root; TRANSFORMERS_NO_TF=1 USE_TF=0 recommended):
  python -m evals.product.stale_exposure run --mode lexical --tag lex_r1 --examples
  python -m evals.product.stale_exposure run --mode lexical --tag lex_r2
  python -m evals.product.stale_exposure compare lex_r1 lex_r2
  python -m evals.product.stale_exposure run --mode semantic --tag sem_r1   # loads BGE-M3
  python -m evals.product.stale_exposure run --mode lexical --layout epoch --tag lex_epoch
  python -m evals.product.stale_exposure run --mode lexical --subject-drift --tag lex_drift
Results go to evals/product/results/stale_exposure/<tag>/.

Conditions (same questions, TOP_K=5 requested everywhere):
  A_bm25_turns_top5 / _top1  naive memory: BM25 over every stored turn text
  A_dense_turns_top5         naive vector memory: BGE-M3 cosine over turn texts (semantic mode)
  B_mc_claims                handle_memory_query(session_id=None, top_k=5)["claims"]
  B5_mc_claims_first5        the first 5 of those claims (what a top-5 consumer reads)
  C_mc_claims_plus_episodes  claims + episodes, the full served payload
  D_mc_hook_injection        http_server._prompt_context(q, None): what Claude Code sees

Value matching: case-insensitive, whitespace-normalised substring of the FULL value
string, bounded by non-alphanumerics on both sides, so "npm" never matches inside
"pnpm", "weekly" never inside "biweekly", "Python 3.1" never inside "Python 3.11".
The run aborts-with-report (sanity_issues) if any value matches a sibling value, its
own subject/question text, or a turn belonging to another history.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results" / "stale_exposure"
TOP_K = 5
# Only one BGE-M3 process at a time (each holds several GB of RAM).
LOCK_DIR = Path(os.environ.get("MEMCONTEXT_EVAL_MODEL_LOCK",
                               str(Path(tempfile.gettempdir()) / "memcontext-bge.lock")))
LOCK_RETRY_S = 30
EXAMPLE_QS = (
    "What database does the orders service use?",         # old/new values share no tokens
    "Which Python version does the backend run on?",      # old/new values share most tokens
    "Which package manager do we use for the frontend?",  # "npm" vs "pnpm" substring trap
)
EXAMPLE_MAX_CHARS = 700
CHANGE_TEMPLATES = (
    "Change of plan: the {s} is now {v}.",
    "Update: going forward, the {s} is {v}.",
    "Heads up, we switched the {s} to {v}.",
)


# ---------------------------------------------------------------- utilities ---
def repo_sha(repo: Path) -> str:
    """Resolve HEAD by reading git files (no git process), incl. worktrees."""
    try:
        git = repo / ".git"
        gitdir = Path(git.read_text().split(":", 1)[1].strip()) if git.is_file() else git
        head = (gitdir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1]
        common = gitdir.parent.parent if gitdir.parent.name == "worktrees" else gitdir
        for base in (gitdir, common):
            if (base / ref).is_file():
                return (base / ref).read_text().strip()
        for line in (common / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "unknown"


@contextlib.contextmanager
def model_lock(enabled: bool):
    """Exclusive lock so only one BGE-M3 process runs at a time (memory pressure)."""
    if not enabled:
        yield
        return
    while True:
        try:
            LOCK_DIR.mkdir()  # atomic: fails if it already exists
            break
        except FileExistsError:
            print(f"[lock] {LOCK_DIR} held; retrying in {LOCK_RETRY_S}s", flush=True)
            time.sleep(LOCK_RETRY_S)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            LOCK_DIR.rmdir()


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def mentions(text: str, value: str) -> bool:
    pat = r"(?<![a-z0-9])" + re.escape(_norm(value)) + r"(?![a-z0-9])"
    return re.search(pat, _norm(text)) is not None


# ------------------------------------------------------------------ dataset ---
def build_turns(hist: list[dict], layout: str, drift: bool) -> list[dict]:
    """Time-ordered turns: epoch e holds every history's e-th value."""
    turns = []
    epochs = max(len(h["values"]) for h in hist)
    for epoch in range(epochs):
        for i, h in enumerate(hist):
            vals = h["values"]
            if len(vals) == 1:
                if epoch != i % epochs:  # spread unchanged facts over time
                    continue
                v, subj = vals[0], h["subject"]
                text = f"Decision: we will use {v} for the {subj}."
            elif epoch < len(vals):
                v = vals[epoch]
                last = epoch == len(vals) - 1
                subj = h["alt_subject"] if (drift and last and epoch > 0) else h["subject"]
                text = (f"Decision: we will use {v} for the {subj}." if epoch == 0
                        else CHANGE_TEMPLATES[(i + epoch) % 3].format(s=subj, v=v))
            else:
                continue
            sid = f"s{i:02d}_e{epoch}" if layout == "per-change" else f"epoch{epoch}"
            turns.append({"hist": i, "epoch": epoch, "session_id": sid, "text": text,
                          "claim": {"subject": subj, "predicate": "decision_made",
                                    "value": v, "confidence": 0.9}})
    return turns


def sanity_issues(hist: list[dict], turns: list[dict]) -> list[str]:
    issues = []
    for i, h in enumerate(hist):
        for a in h["values"]:
            issues += [f"{a!r} matches inside sibling {b!r}" for b in h["values"]
                       if a != b and mentions(b, a)]
            issues += [f"{a!r} appears in {q!r}" for q in
                       h["questions"] + [h["subject"], h.get("alt_subject", "")] if mentions(q, a)]
            issues += [f"{a!r} (hist {i}) matches foreign turn {t['text']!r}"
                       for t in turns if t["hist"] != i and mentions(t["text"], a)]
    return issues


def build_questions(hist: list[dict]) -> list[dict]:
    return [{"q": q, "subject": h["subject"], "current": h["values"][-1],
             "stale": h["values"][:-1], "changed": len(h["values"]) > 1}
            for h in hist for q in h["questions"]]


# ------------------------------------------------------------------- ingest ---
def ingest(mc, db_path: Path, turns: list[dict], hist: list[dict], backfill: bool = False) -> dict:
    conn = mc.open_database(db_path)
    stats = {"stored": 0, "not_admitted": [], "claims_created": 0, "supersessions": 0}
    t0 = time.perf_counter()
    for t in turns:
        r = mc.mcp_tools.handle_memory_store(conn, text=t["text"], speaker="user",
                                             session_id=t["session_id"], claims=[t["claim"]],
                                             namespace="default")
        stats["stored"] += 1
        stats["claims_created"] += r["claims_created"]
        stats["supersessions"] += r["supersessions"]
        if not r["admitted"]:
            stats["not_admitted"].append(t["text"])
    stats["ingest_seconds"] = round(time.perf_counter() - t0, 1)
    if backfill:
        # The product never embeds claims on the ingest path; only the CLI backfill
        # does. This opt-in variant runs that same backfill to show what fact-level
        # semantic ranking would give if claims were embedded.
        sids = [r[0] for r in conn.execute("SELECT DISTINCT session_id FROM turns").fetchall()]
        stats["claims_backfilled"] = sum(mc.backfill_embeddings(conn, sid, client=mc.episode_embedder())
                                         for sid in sids)
        conn.commit()
    stats["claim_embeddings_stored"] = conn.execute("SELECT COUNT(*) FROM claim_embeddings").fetchone()[0]
    stats["turn_embeddings_stored"] = conn.execute("SELECT COUNT(*) FROM turn_embeddings").fetchone()[0]
    stats.update(slot_audit(mc, conn, hist))
    conn.close()
    return stats


def slot_audit(mc, conn, hist: list[dict]) -> dict:
    """Per history: is exactly the current value live (ingest-level correctness)?"""
    live = {"active", "confirmed", "audited"}
    rows = conn.execute("SELECT subject, value, status FROM claims").fetchall()
    ok, problems, n_changing = 0, [], 0
    for h in hist:
        keys = {mc.norm_subject(h["subject"]), mc.norm_subject(h.get("alt_subject") or "")}
        live_vals = sorted({r["value"] for r in rows if (r["subject"] or "") in keys
                            and r["value"] in h["values"] and r["status"] in live})
        if len(h["values"]) > 1:
            n_changing += 1
            ok += live_vals == [h["values"][-1]]
        if live_vals != [h["values"][-1]]:
            problems.append({"subject": h["subject"], "live": live_vals})
    counts = dict(conn.execute("SELECT status, COUNT(*) FROM claims GROUP BY status").fetchall())
    return {"slot_resolved": f"{ok}/{n_changing}", "slot_problems": problems,
            "status_counts": counts}


# --------------------------------------------------------------- conditions ---
def copy_db(base: Path, dst: Path) -> Path:
    src, d = sqlite3.connect(base), sqlite3.connect(dst)
    src.backup(d)
    src.close()
    d.close()
    return dst


def cond_bm25(mc, db: Path, k: int):
    rows = mc.open_database(db).execute("SELECT text FROM turns ORDER BY ts ASC").fetchall()
    texts = [r["text"] for r in rows]
    docs = [mc.tokenize(t) for t in texts]

    def serve(q: str) -> list[str]:
        s = mc.bm25(mc.tokenize(q), docs)
        order = sorted(range(len(texts)), key=lambda j: (-s[j], j))
        return [texts[j] for j in order[:k] if s[j] > 0]
    return serve


def cond_dense(mc, db: Path, k: int):
    texts = [r["text"] for r in
             mc.open_database(db).execute("SELECT text FROM turns ORDER BY ts ASC").fetchall()]
    emb = mc.episode_embedder()
    vecs = emb.embed(texts)

    def cos(a, b):
        den = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
        return sum(x * y for x, y in zip(a, b, strict=True)) / (den or 1.0)

    def serve(q: str) -> list[str]:
        qv = emb.embed([mc.apply_query_prefix(q)])[0]
        s = [cos(qv, v) for v in vecs]
        return [texts[j] for j in sorted(range(len(texts)), key=lambda j: (-s[j], j))[:k]]
    return serve


def cond_mc(mc, db: Path, part: str):
    conn = mc.open_database(db)

    def serve(q: str) -> list[str]:
        r = mc.mcp_tools.handle_memory_query(conn, query=q, session_id=None, top_k=TOP_K)
        claims = [f"{c['subject']} / {c['predicate']}: {c['value']}" for c in r["claims"]]
        if part == "claims":
            return claims
        if part == "claims5":
            return claims[:TOP_K]
        return claims + [e["text"] for e in r["episodes"]]
    return serve


def cond_hook(mc, db: Path):
    mc.http_server.init_db(str(db))

    def serve(q: str) -> list[str]:
        ctx = mc.http_server._prompt_context(q, None)
        return [ctx] if ctx else []
    return serve


# ------------------------------------------------------------------ scoring ---
def score(serve, questions: list[dict], history_intent) -> tuple[dict, list[dict]]:
    rows = []
    for qq in questions:
        t0 = time.perf_counter()
        items = serve(qq["q"])
        secs = time.perf_counter() - t0
        text = "\n".join(items)
        rows.append({"q": qq["q"], "subject": qq["subject"], "changed": qq["changed"],
                     "current": qq["current"], "cur_hit": mentions(text, qq["current"]),
                     "stale_hit": [v for v in qq["stale"] if mentions(text, v)],
                     "n_items": len(items), "chars": len(text), "seconds": round(secs, 3),
                     "history_intent": history_intent(qq["q"]), "served": items,
                     "sha": hashlib.sha256(text.encode()).hexdigest()[:12]})
    return metrics(rows), rows


def metrics(rows: list[dict]) -> dict:
    ch = [r for r in rows if r["changed"]]
    un = [r for r in rows if not r["changed"]]

    def pct(xs, f):
        return round(100.0 * sum(1 for r in xs if f(r)) / len(xs), 1) if xs else None
    return {
        "n_changed_q": len(ch), "n_unchanged_q": len(un),
        "stale_exposure_pct": pct(ch, lambda r: bool(r["stale_hit"])),
        "current_hit_pct": pct(ch, lambda r: r["cur_hit"]),
        "conflict_pct": pct(ch, lambda r: r["cur_hit"] and bool(r["stale_hit"])),
        "current_only_pct": pct(ch, lambda r: r["cur_hit"] and not r["stale_hit"]),
        "median_chars": statistics.median(r["chars"] for r in ch),
        "median_items": statistics.median(r["n_items"] for r in ch),
        "distractor_current_hit_pct": pct(un, lambda r: r["cur_hit"]),
        "median_query_seconds": round(statistics.median(r["seconds"] for r in rows), 3),
    }


# ---------------------------------------------------------------- reporting ---
def print_table(s: dict) -> None:
    print(f"\n### {s['tag']}  sha={s['repo_sha'][:10]} mode={s['mode']} layout={s['layout']} "
          f"drift={s['subject_drift']} backfill={s.get('backfill_claim_embeddings', False)} sessions={s['n_sessions']} turns={s['n_turns']} "
          f"slots_resolved={s['ingest']['slot_resolved']}")
    print("| condition | stale-exposure | current-hit | conflict | current-only "
          "| median chars | median items | distractor current-hit | median s/query |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n, m in s["metrics"].items():
        print(f"| {n} | {m['stale_exposure_pct']}% | {m['current_hit_pct']}% | {m['conflict_pct']}% "
              f"| {m['current_only_pct']}% | {m['median_chars']:.0f} | {m['median_items']:.0f} "
              f"| {m['distractor_current_hit_pct']}% | {m['median_query_seconds']} |")
    m = next(iter(s["metrics"].values()))
    print(f"n = {m['n_changed_q']} changed-fact questions, {m['n_unchanged_q']} unchanged-fact questions")


def print_hook_precision(perq: dict) -> None:
    rows = perq.get("D_mc_hook_injection", [])
    injected = [r for r in rows if r["served"]]
    lines = own = 0
    for r in injected:
        ls = r["served"][0].split("\n- ")[1:]
        key = "_".join(r["subject"].lower().split()) + " /"
        lines += len(ls)
        own += sum(1 for ln in ls if ln.lower().startswith(key))
    print(f"hook: {len(injected)}/{len(rows)} prompts got an injection; "
          f"{own}/{lines} injected lines were about the asked subject")


def print_examples(perq: dict) -> None:
    for cond, rows in perq.items():
        print(f"\n#### {cond}")
        for q in EXAMPLE_QS:
            r = next(x for x in rows if x["q"] == q)
            text = "\n".join(r["served"]) or "<nothing served>"
            if len(text) > EXAMPLE_MAX_CHARS:
                text = text[:EXAMPLE_MAX_CHARS] + f"\n...[+{len(text) - EXAMPLE_MAX_CHARS} chars, {r['n_items']} items]"
            print(f"Q: {q}  current={r['current']!r} cur_hit={r['cur_hit']} stale_hit={r['stale_hit']}")
            print("\n".join("   | " + ln for ln in text.split("\n")))


# ---------------------------------------------------------------- commands ---
class _MC:
    """memcontext handles, imported only after the environment is configured."""


def load_memcontext(repo: Path, mode: str) -> _MC:
    os.environ["ACTIVE_PACK"] = "general,developer"
    os.environ["SUBSTRATE_PACKS_DIR"] = str(repo / "predicate_packs")
    os.environ["MEMCONTEXT_EMBED_EPISODES"] = "1" if mode == "semantic" else "0"
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    sys.path.insert(0, str(repo))
    import structlog
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))
    from memcontext import http_server, mcp_tools, retrieval
    from memcontext.claims import _normalise_subject
    from memcontext.predicate_packs import active_pack
    from memcontext.schema import open_database

    active_pack.cache_clear()
    if "decision_made" not in active_pack().single_valued:
        raise SystemExit("developer pack not active: decision_made is not single-valued")
    if (retrieval.episode_embedder() is not None) != (mode == "semantic"):
        raise SystemExit(f"embedder state does not match --mode {mode}")
    mc = _MC()
    mc.http_server, mc.mcp_tools, mc.open_database = http_server, mcp_tools, open_database
    mc.norm_subject, mc.episode_embedder = _normalise_subject, retrieval.episode_embedder
    mc.tokenize, mc.bm25 = retrieval._tokenize_for_bm25, retrieval._bm25_over_docs
    mc.backfill_embeddings = retrieval.backfill_embeddings
    mc.apply_query_prefix, mc.history_intent = retrieval.apply_query_prefix, retrieval.detect_history_intent
    return mc


def cmd_run(a) -> None:
    repo = Path(a.repo)
    hist = json.loads(Path(a.dataset).read_text(encoding="utf-8"))["histories"]
    out = RESULTS / a.tag
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    if a.backfill_claim_embeddings and a.mode != "semantic":
        raise SystemExit("--backfill-claim-embeddings needs --mode semantic")
    turns = build_turns(hist, a.layout, a.subject_drift)
    questions = build_questions(hist)
    issues = sanity_issues(hist, turns)
    if issues:
        print("\n".join(issues))
        raise SystemExit("value-matching sanity check failed; fix datasets/stale_exposure.json")
    (out / "turns.json").write_text(json.dumps(turns, indent=1), encoding="utf-8")

    with model_lock(a.mode == "semantic"):
        sha_at_import = repo_sha(repo)  # the code actually imported and tested
        mc = load_memcontext(repo, a.mode)
        base = out / "base.db"
        ingest_stats = ingest(mc, base, turns, hist, a.backfill_claim_embeddings)
        conds = {
            "A_bm25_turns_top5": lambda: cond_bm25(mc, copy_db(base, out / "a5.db"), TOP_K),
            "A_bm25_turns_top1": lambda: cond_bm25(mc, copy_db(base, out / "a1.db"), 1),
            "B_mc_claims": lambda: cond_mc(mc, copy_db(base, out / "b.db"), "claims"),
            "B5_mc_claims_first5": lambda: cond_mc(mc, copy_db(base, out / "b5.db"), "claims5"),
            "C_mc_claims_plus_episodes": lambda: cond_mc(mc, copy_db(base, out / "c.db"), "full"),
            "D_mc_hook_injection": lambda: cond_hook(mc, copy_db(base, out / "d.db")),
        }
        if a.mode == "semantic":
            conds["A_dense_turns_top5"] = lambda: cond_dense(mc, copy_db(base, out / "ad.db"), TOP_K)
        results, perq = {}, {}
        for name, make in conds.items():
            results[name], perq[name] = score(make(), questions, mc.history_intent)
            print(f"[done] {name}", flush=True)

    summary = {"tag": a.tag, "repo": str(repo), "repo_sha": sha_at_import,
               "repo_sha_at_end": repo_sha(repo), "mode": a.mode,
               "layout": a.layout, "subject_drift": a.subject_drift,
               "backfill_claim_embeddings": a.backfill_claim_embeddings, "top_k": TOP_K,
               "n_turns": len(turns), "n_sessions": len({t["session_id"] for t in turns}),
               "ingest": ingest_stats,
               "history_intent_questions": [q["q"] for q in questions if mc.history_intent(q["q"])],
               "metrics": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "per_question.json").write_text(json.dumps(perq, indent=1), encoding="utf-8")
    if summary["repo_sha_at_end"] != sha_at_import:
        print(f"WARNING: checkout moved during the run ({sha_at_import[:10]} -> "
              f"{summary['repo_sha_at_end'][:10]}); modules imported lazily may mix commits")
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=1))
    print_table(summary)
    print_hook_precision(perq)
    if a.examples:
        print_examples(perq)


def cmd_compare(a) -> None:
    def load(tag):
        d = RESULTS / tag
        return (json.loads((d / "summary.json").read_text(encoding="utf-8")),
                json.loads((d / "per_question.json").read_text(encoding="utf-8")))
    (s1, p1), (s2, p2) = load(a.tag1), load(a.tag2)
    print_table(s1)
    print_table(s2)
    print(f"\n### determinism {a.tag1} vs {a.tag2}")
    lines = lambda r: sorted("\n".join(r["served"]).split("\n"))  # noqa: E731
    for c in p1:
        pairs = list(zip(p1[c], p2[c], strict=False))
        text = sum(x["sha"] != y["sha"] for x, y in pairs)
        sets = sum(lines(x) != lines(y) for x, y in pairs)
        flips = sum((x["cur_hit"], x["stale_hit"]) != (y["cur_hit"], y["stale_hit"]) for x, y in pairs)
        print(f"{c}: served text differs {text}/{len(pairs)}, served line-set differs {sets}, "
              f"score flips {flips}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ingest + serve + score one configuration")
    r.add_argument("--mode", choices=["lexical", "semantic"], required=True,
                   help="lexical: MEMCONTEXT_EMBED_EPISODES=0; semantic: BGE-M3 (product default)")
    r.add_argument("--layout", choices=["per-change", "epoch"], default="per-change",
                   help="per-change: one session per change; epoch: one session per time step")
    r.add_argument("--subject-drift", action="store_true",
                   help="final change is stored under a paraphrased subject key (alt_subject)")
    r.add_argument("--backfill-claim-embeddings", action="store_true",
                   help="semantic only: run the product's claim-embedding backfill after ingest")
    r.add_argument("--tag", required=True, help="run name; output in runs/<tag>/")
    r.add_argument("--repo", default=str(REPO), help="MemContext checkout under test")
    r.add_argument("--dataset", default=str(HERE / "datasets" / "stale_exposure.json"))
    r.add_argument("--examples", action="store_true", help="print 3 verbatim examples per condition")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare", help="diff two finished runs")
    c.add_argument("tag1")
    c.add_argument("tag2")
    c.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
