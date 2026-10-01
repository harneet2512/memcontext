"""Scale eval: do MemContext's guarantees and costs hold as a project's memory grows?

Eval harness, NOT product code. Lexical mode, deterministic scoring, no LLM, no judge.

  run  For each size N, ingest the 44 stale-exposure decision histories (the scored set)
       interleaved in time with N unrelated distractor decisions, all through the public
       API (handle_memory_store, one decision_made claim per turn, one session each).
       Then ask the same questions through the Claude Code prompt hook and a BM25
       baseline and record stale exposure, current-value hits, injected size, hook
       latency, ingest cost and database size.

Example (repo root):  python evals/product/scale.py run --sizes 0,500,2000,5000
Results: evals/product/results/scale/latest.json (common eval schema, read by the dashboard).

Distractors are generated deterministically (fixed seed) as "<service> <aspect>" subjects
with aspect-appropriate values. Some aspects deliberately share words with scored
questions (e.g. "database", "message queue"): many services each having a database is
exactly what a growing memory looks like. The run aborts if any scored value matches
inside a distractor turn, so a distractor can never count as a stale or current hit.
Latency is measured wall time on this machine; everything else is deterministic.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stale_exposure as se  # noqa: E402

OUT = HERE / "results" / "scale"
SEED = 20261001
DEFAULT_SIZES = (0, 250, 500, 1000, 2000)  # ingest cost grows per store; 5000 takes ~10 min

DOMAINS = ("inventory", "shipping", "catalog", "notifications", "reporting", "ledger",
           "identity", "pricing", "recommendations", "analytics", "media", "geo", "audit",
           "onboarding", "support", "scheduling", "fulfillment", "returns", "loyalty", "partners")
KINDS = ("service", "worker", "api", "gateway", "sync job", "scheduler")
ASPECTS: dict[str, tuple[str, ...]] = {
    "database": ("MySQL 8", "CockroachDB", "DynamoDB", "MongoDB 7", "ClickHouse"),
    "message queue": ("NATS JetStream", "Google Pub/Sub", "Azure Service Bus", "Redpanda"),
    "retry policy": ("exponential backoff with 3 retries", "exponential backoff with 6 retries",
                     "fixed 2-second retry", "no automatic retries"),
    "request timeout": ("2 seconds", "5 seconds", "12 seconds", "45 seconds"),
    "owner team": ("team Atlas", "team Borealis", "team Cobalt", "team Dune", "team Ember"),
    "on-call rotation": ("weekly rotation", "follow-the-sun rotation", "two-person rotation"),
    "log retention": ("7 days", "30 days", "90 days", "1 year"),
    "metrics namespace": ("svc_core", "svc_edge", "svc_batch", "svc_ext"),
    "health check path": ("/healthz", "/livez", "/status", "/ping"),
    "deployment strategy": ("blue-green deploys", "canary deploys", "rolling deploys"),
    "replica count": ("2 replicas", "3 replicas", "6 replicas", "12 replicas"),
    "cpu limit": ("500 millicores", "1 core", "2 cores", "4 cores"),
    "memory limit": ("512 MiB", "1 GiB", "2 GiB", "8 GiB"),
    "schema migration tool": ("Alembic", "Flyway", "Liquibase", "goose"),
    "API versioning": ("URL path versioning", "header versioning", "date-based versioning"),
    "auth scope": ("read-only scope", "service-to-service mTLS", "admin scope"),
    "data classification": ("internal data", "confidential data", "public data"),
    "backup schedule": ("nightly snapshots", "hourly snapshots", "continuous WAL archiving"),
    "SLO target": ("99.5 percent availability", "99.9 percent availability",
                   "99.95 percent availability"),
    "alert channel": ("#alerts-core", "#alerts-edge", "#alerts-batch"),
    "queue consumer concurrency": ("4 consumers", "8 consumers", "32 consumers"),
    "batch size": ("50 items", "250 items", "1000 items"),
    "serialization format": ("Protocol Buffers", "Avro", "MessagePack", "CBOR"),
    "HTTP framework": ("FastAPI", "Flask", "Gin", "Fastify", "Axum"),
    "load balancer": ("NGINX", "Envoy", "HAProxy", "Traefik"),
    "CDN": ("Fastly", "Akamai", "Bunny CDN"),
    "config source": ("Consul KV", "etcd", "AWS AppConfig"),
    "service mesh": ("Istio", "Linkerd", "no service mesh"),
    "tracing backend": ("Jaeger", "Tempo", "Honeycomb", "Zipkin"),
    "load test tool": ("k6", "Locust", "Gatling"),
    "code owner": ("Priya", "Mateo", "Hana", "Olu", "Ines"),
    "language": ("Kotlin", "Rust", "Elixir", "Java 21", "C#"),
    "repo location": ("platform monorepo", "standalone repo", "infra repo"),
    "cron schedule": ("every 5 minutes", "every hour", "nightly at 02:00"),
    "feature freeze day": ("Monday", "Wednesday", "Friday"),
    "pagination limit": ("50 per page", "200 per page", "500 per page"),
    "idempotency key": ("Idempotency-Key header", "client request id", "no idempotency key"),
    "PII handling": ("tokenized PII", "encrypted PII columns", "no PII stored"),
    "dependency update bot": ("Renovate", "Dependabot"),
    "docs location": ("Backstage", "Confluence space", "README only"),
    "error budget policy": ("freeze on budget burn", "warn only", "page on budget burn"),
    "circuit breaker": ("opens after 5 failures", "opens after 20 failures", "no circuit breaker"),
    "TLS termination": ("at the edge", "at the pod", "at the mesh sidecar"),
    "image registry": ("GHCR", "ECR", "Artifact Registry", "Harbor"),
    "audit log sink": ("BigQuery", "Splunk", "Loki"),
}
TEMPLATES = (
    "Decision: for the {s}, we chose {v}.",
    "We agreed the {s} will be {v}.",
    "Decision recorded: {s} is {v}.",
)


# ---------------------------------------------------------------- distractors ---
def distractor_subjects() -> list[str]:
    """Every '<domain> <kind> <aspect>' subject, in one fixed seeded order (nested prefixes)."""
    subjects = [f"{d} {k} {a}" for d in DOMAINS for k in KINDS for a in ASPECTS]
    random.Random(SEED).shuffle(subjects)
    return subjects


def make_distractors(n: int) -> list[dict]:
    """The first n distractor decisions; deterministic, so size N's set contains size M's for M < N."""
    subjects = distractor_subjects()
    if n > len(subjects):
        raise SystemExit(f"at most {len(subjects)} distractors available, asked for {n}")
    rng = random.Random(SEED + 1)
    out = []
    for j, s in enumerate(subjects[:n]):
        v = rng.choice(ASPECTS[_aspect_of(s)])
        out.append({"hist": -1, "epoch": -1, "session_id": f"d{j:05d}",
                    "text": TEMPLATES[j % len(TEMPLATES)].format(s=s, v=v),
                    "claim": {"subject": s, "predicate": "decision_made", "value": v, "confidence": 0.9}})
    return out


def _aspect_of(subject: str) -> str:
    rest = subject.split(" ", 1)[1]
    for k in sorted(KINDS, key=len, reverse=True):
        if rest.startswith(k + " "):
            return rest[len(k) + 1:]
    raise ValueError(subject)


def scored_value_pattern(hist: list[dict]) -> re.Pattern[str]:
    """One regex for se.mentions() over every scored value (old and current)."""
    vals = sorted({se._norm(v) for h in hist for v in h["values"]}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(v) for v in vals) + r")(?![a-z0-9])")


def collision_issues(hist: list[dict], distractors: list[dict]) -> list[str]:
    """A distractor must never be scorable: no scored value inside it, no scored subject reused."""
    pat = scored_value_pattern(hist)
    scored_subjects = {se._norm(s) for h in hist for s in (h["subject"], h.get("alt_subject") or "") if s}
    issues = []
    for d in distractors:
        m = pat.search(se._norm(d["text"]))
        if m:
            issues.append(f"scored value {m.group(1)!r} inside distractor {d['text']!r}")
        if se._norm(d["claim"]["subject"]) in scored_subjects:
            issues.append(f"distractor reuses scored subject {d['claim']['subject']!r}")
    return issues


def interleave(scored: list[dict], distractors: list[dict]) -> list[dict]:
    """Spread distractors evenly through the scored timeline (not all before or after)."""
    if not distractors:
        return list(scored)
    out, per = [], len(distractors) / (len(scored) + 1)
    di = 0
    for i, t in enumerate(scored):
        upto = round(per * (i + 1))
        out += distractors[di:upto]
        di = upto
        out.append(t)
    return out + distractors[di:]


# ---------------------------------------------------------------- measuring ---
def pctl(xs: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100) of a non-empty list."""
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, round(p / 100 * len(s) + 0.5) - 1))]


def timed(serve):
    times: list[float] = []

    def wrapped(q: str) -> list[str]:
        t0 = time.perf_counter()
        out = serve(q)
        times.append((time.perf_counter() - t0) * 1000)
        return out
    return wrapped, times


def db_bytes(db: Path) -> int:
    return sum(p.stat().st_size for p in (db, Path(str(db) + "-wal")) if p.exists())


def run_size(mc, n: int, hist: list[dict], questions: list[dict], work: Path) -> dict:
    scored = se.build_turns(hist, "per-change", False)
    turns = interleave(scored, make_distractors(n))
    db = work / f"scale_{n}.db"
    t0 = time.perf_counter()
    ingest_stats = se.ingest(mc, db, turns, hist)
    ingest_ms = (time.perf_counter() - t0) * 1000 / len(turns)
    size = db_bytes(db)

    hook, times = timed(se.cond_hook(mc, db))
    hook(questions[0]["q"])  # warm-up, not counted
    times.clear()
    hm, hrows = se.score(hook, questions, mc.history_intent)
    bm, _ = se.score(se.cond_bm25(mc, db, se.TOP_K), questions, mc.history_intent)
    chars = [r["chars"] for r in hrows]
    # every scored decision has one current value and distractors are distinct decisions,
    # so any CONFLICT warning the hook injects here is a false alarm
    false_conflicts = sum(1 for r in hrows if any("CONFLICT" in x for x in r["served"]))
    return {"false_conflicts": false_conflicts, "n_questions": len(hrows),"n": n, "turns": len(turns), "hook": hm, "bm25": bm,
            "median_chars": statistics.median(chars), "p95_chars": pctl(chars, 95),
            "hook_p50_ms": round(pctl(times, 50), 2), "hook_p95_ms": round(pctl(times, 95), 2),
            "ingest_ms_per_turn": round(ingest_ms, 2), "db_mb": round(size / 1e6, 2),
            "slot_resolved": ingest_stats["slot_resolved"], "supersessions": ingest_stats["supersessions"],
            "not_admitted": len(ingest_stats["not_admitted"])}


def product_dirty(repo: Path) -> list[str]:
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--", "memcontext", "predicate_packs"],
                           cwd=repo, capture_output=True, text=True, timeout=30)
        return [ln[3:] for ln in r.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        return ["<git unavailable>"]


# ---------------------------------------------------------------- reporting ---
def frac(m: dict, key: str) -> str:
    return f"{round(m[key] * m['n_changed_q'] / 100)}/{m['n_changed_q']}"


def diagnose(res: list[dict], sha: str) -> list[str]:
    """Measured symptoms, each with the root cause diagnosed at this commit (LIPI layer + file:line)."""
    base, big, out = res[0], res[-1], []
    if big["hook"]["current_hit_pct"] < base["hook"]["current_hit_pct"]:
        out.append(f"Current-value hit falls from {frac(base['hook'], 'current_hit_pct')} to "
                   f"{frac(big['hook'], 'current_hit_pct')} at {big['n']:,} unrelated decisions while storage "
                   f"stays correct ({big['slot_resolved']} slots resolved). Diagnosed at {sha}, Logic: the hook "
                   "scores only the 500 newest live claims (memcontext/http_server.py:768 "
                   "_TOOL_CONTEXT_ROW_LIMIT, applied at :777), so an older current decision is never a candidate.")
    if big["hook"]["stale_exposure_pct"] > base["hook"]["stale_exposure_pct"]:
        out.append(f"Hook stale exposure rises from {frac(base['hook'], 'stale_exposure_pct')} to "
                   f"{frac(big['hook'], 'stale_exposure_pct')}.")
    if big["false_conflicts"] > base["false_conflicts"]:
        out.append(f"{big['false_conflicts']}/{big['n_questions']} prompts get a false CONFLICT warning at "
                   f"{big['n']:,} decisions. Diagnosed at {sha}, Logic: unprefixed subjects all share prefix '' "
                   "and a subset of topic words counts as the same decision, so 'message queue' matches "
                   "'inventory worker message queue' (memcontext/conflicts.py:44).")
    if big["hook_p95_ms"] > 10 * max(base["hook_p95_ms"], 1):
        out.append(f"Hook p95 latency grows from {base['hook_p95_ms']} ms to {big['hook_p95_ms']} ms. Diagnosed "
                   f"at {sha}, Implementation: each injected candidate re-reads every live claim of its "
                   "predicate and regex-compares subjects (memcontext/conflicts.py:51 live_same_kind, called "
                   "per candidate from memcontext/http_server.py:550 _with_conflicts).")
    if big["ingest_ms_per_turn"] > 3 * max(base["ingest_ms_per_turn"], 0.1):
        out.append(f"Ingest cost per stored turn grows from {base['ingest_ms_per_turn']} ms to "
                   f"{big['ingest_ms_per_turn']} ms (linear per store, quadratic in total). Diagnosed at {sha}, "
                   "Implementation: the subject-drift warning rescans every live subject and recomputes corpus "
                   "word statistics on every store (memcontext/mcp_tools.py:117 _subject_drift -> "
                   "memcontext/conflicts.py:227 similar_subjects, :207 _corpus_common_words); "
                   "importance.recompute_all_importance (memcontext/on_new_turn.py:326) rescans all claims "
                   "periodically.")
    return out


def build_result(res: list[dict], repo: Path, dirty: list[str], n_hist: int, n_q: int) -> dict:
    big = res[-1]
    hk, bm = big["hook"], big["bm25"]
    table = [[f"{r['n'] + n_hist:,}", frac(r["hook"], "stale_exposure_pct"), frac(r["hook"], "current_hit_pct"),
              frac(r["bm25"], "stale_exposure_pct"), r["median_chars"], r["p95_chars"],
              r["hook_p50_ms"], r["hook_p95_ms"], r["ingest_ms_per_turn"],
              f"{r['false_conflicts']}/{r['n_questions']}"] for r in res]
    findings = diagnose(res, se.repo_sha(repo)[:7])
    return {
        "eval": "scale", "title": "Scale",
        "promise": "Current, not stale, at any memory size, without making every prompt slower or longer.",
        "question": "As unrelated decisions pile up, does the Claude Code hook still inject only current "
                    "values, still find them, and stay small and fast?",
        "method": f"{n_hist} scored decision histories (41 changed-fact questions) interleaved in time with "
                  f"N unrelated decisions (N = {', '.join(str(r['n']) for r in res)}), all stored through "
                  "handle_memory_store; questions asked through http_server._prompt_context (the hook) and "
                  "BM25 over every stored turn (top 5). Lexical mode.",
        "commit": se.repo_sha(repo), "product_code_dirty": bool(dirty), "product_dirty_files": dirty,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "deterministic": True,
        "deterministic_note": "Scores are deterministic; latency and ingest time are wall-clock on this machine.",
        "headline": {"label": f"Outdated values injected with {big['n']:,} unrelated decisions in memory",
                     "value": frac(hk, "stale_exposure_pct"), "pct": hk["stale_exposure_pct"], "better": "lower"},
        "metrics": [
            {"name": "Hook stale exposure", "value": frac(hk, "stale_exposure_pct"), "pct": hk["stale_exposure_pct"],
             "n": hk["n_changed_q"], "better": "lower",
             "definition": f"Changed-fact questions whose injected context contains an outdated value, at N={big['n']:,}."},
            {"name": "Hook current-value hit", "value": frac(hk, "current_hit_pct"), "pct": hk["current_hit_pct"],
             "n": hk["n_changed_q"], "better": "higher",
             "definition": f"Changed-fact questions whose injected context contains the current value, at N={big['n']:,}."},
            {"name": "BM25 stale exposure (baseline)", "value": frac(bm, "stale_exposure_pct"),
             "pct": bm["stale_exposure_pct"], "n": bm["n_changed_q"], "better": "lower",
             "definition": f"Same questions, BM25 top 5 over every stored turn, at N={big['n']:,}."},
            {"name": "Hook latency p95", "value": f"{big['hook_p95_ms']} ms", "pct": None, "n": n_q,
             "better": "lower", "definition": "95th percentile wall time of one prompt-hook call (warm)."},
            {"name": "Median injected context", "value": f"{big['median_chars']:.0f} chars", "pct": None, "n": n_q,
             "better": "lower", "definition": "Median characters the hook adds to a prompt."},
            {"name": "Ingest cost", "value": f"{big['ingest_ms_per_turn']} ms/turn", "pct": None, "n": big["turns"],
             "better": "lower", "definition": "Mean wall time of handle_memory_store per stored turn."},
        ],
        "tables": [{"title": "By memory size",
                    "columns": ["decisions in memory", "hook stale", "hook current", "BM25 stale",
                                "median injected chars", "p95 injected chars", "hook p50 ms", "hook p95 ms",
                                "ingest ms/turn", "false CONFLICT warnings"],
                    "rows": table}],
        "series": {"sizes": [r["n"] for r in res],
                   "hook_stale_pct": [r["hook"]["stale_exposure_pct"] for r in res],
                   "hook_current_pct": [r["hook"]["current_hit_pct"] for r in res],
                   "bm25_stale_pct": [r["bm25"]["stale_exposure_pct"] for r in res],
                   "hook_p95_ms": [r["hook_p95_ms"] for r in res],
                   "median_chars": [r["median_chars"] for r in res],
                   "db_mb": [r["db_mb"] for r in res],
                   "slot_resolved": [r["slot_resolved"] for r in res],
                   "false_conflicts": [r["false_conflicts"] for r in res],
                   "ingest_ms_per_turn": [r["ingest_ms_per_turn"] for r in res]},
        "limits": ["Distractors are template-generated, single-valued decisions; real memories also change and "
                   "carry longer free text.",
                   "Lexical mode only; semantic mode would add embedding cost per turn.",
                   "Measures what the hook injects, not the model's final answer.",
                   "Latency is one machine, one process, warm cache."],
        "findings": findings,
    }


def cmd_run(a) -> None:
    repo = Path(a.repo)
    sizes = sorted({int(x) for x in a.sizes.split(",")})
    hist = json.loads((HERE / "datasets" / "stale_exposure.json").read_text(encoding="utf-8"))["histories"]
    issues = collision_issues(hist, make_distractors(max(sizes)))
    if issues:
        print("\n".join(issues[:20]))
        raise SystemExit(f"{len(issues)} distractor collisions; fix ASPECTS/DOMAINS")
    questions = se.build_questions(hist)
    dirty = product_dirty(repo)
    mc = se.load_memcontext(repo, "lexical")
    res = []
    tmp = Path(tempfile.mkdtemp(prefix="mc-scale-"))
    try:
        for n in sizes:
            r = run_size(mc, n, hist, questions, tmp)
            res.append(r)
            print(f"[done] N={n:,} turns={r['turns']} ingest={r['ingest_ms_per_turn']} ms/turn", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # Windows: files still open by the serving paths stay behind
    result = build_result(res, repo, dirty, len(hist), len(questions))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    t = result["tables"][0]
    print("\n" + " | ".join(t["columns"]))
    for row in t["rows"]:
        print(" | ".join(str(x) for x in row))
    if dirty:
        print(f"NOTE: uncommitted product changes present: {', '.join(dirty)}")
    for f in result["findings"]:
        print("finding:", f)
    print(f"wrote {OUT / 'latest.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ingest at each size, serve, score, write results/scale/latest.json")
    r.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    r.add_argument("--repo", default=str(se.REPO), help="MemContext checkout under test")
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
