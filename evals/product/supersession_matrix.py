"""Supersession decision matrix -- a diagnostic product eval (NOT a benchmark).

Question: when a new fact arrives for an existing (subject, predicate), does MemContext
make the right state decision -- retire the old value, keep both, or surface a conflict?

Each hand-labelled case (datasets/supersession_{dev,heldout}.json) is a sequence of 2-3
stores. Every case runs in a fresh ``:memory:`` DB through the PUBLIC write path
(``mcp_tools.handle_memory_store`` with pre-structured claims) and the final state is read
back from SQL (``claims.status`` + ``supersession_edges``) plus the public
``handle_memory_contradictions``. NL-only facts go through the same public path with an
out-of-pack predicate, which the product demotes to an NL-only fact (asserted per store).

Modes:
  deterministic  MEMCONTEXT_EMBED_EPISODES=0 -> Pass-2 off (Pass-1 only)
  semantic       production embedder (BAAI/bge-m3) + Pass-2 on; aborts loudly unless
                 embeddings are really produced (probe + turn_embeddings rows per case +
                 no swallowed Pass-2/episode-embed failures).

Every failing case carries a root-cause guess (Logic / Implementation / Integration /
Plumbing) with a file:line resolved from the CURRENT source, so it stays accurate as the
product changes. Cases labelled ``ambiguous`` / ``expected_limitation`` are reported but
excluded from accuracy.

    python -m evals.product.supersession_matrix --split dev --mode both
    python -m evals.product.supersession_matrix --split heldout --mode both   # once per change
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
RESULTS_DIR = HERE / "results"
PACK = "general,developer"
LIVE = frozenset({"active", "confirmed", "audited"})
EXCLUDED_LABELS = frozenset({"ambiguous", "expected_limitation"})
CATEGORIES = (
    "update", "update_no_overlap", "additive", "attribute_slot", "history_window",
    "quantity", "negation", "duplicate", "change_after_duplicates", "cross_session",
    "cross_namespace", "trust_direction", "out_of_order", "nl_only",
)
TRUST = {"user": 1.0, "assistant": 0.5}  # source_trust.py tiers for conversation turns
# Swallowed-failure log events: in semantic mode any of these means "semantic is not
# really on" and the run aborts instead of reporting numbers.
FATAL_SEMANTIC_EVENTS = frozenset({
    "substrate.semantic_supersession_failed", "substrate.embed_episode_failed",
    "substrate.episode_embed_failed", "substrate.semantic_embed_length_mismatch",
})


class HarnessError(RuntimeError):
    """The harness could not exercise the product as intended (not a product verdict)."""


# ---------------------------------------------------------------- env / logging ---

_captured_events: list[str] = []


def _configure(mode: str) -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ["ACTIVE_PACK"] = PACK
    os.environ.setdefault("SUBSTRATE_PACKS_DIR", str(REPO_ROOT / "predicate_packs"))
    os.environ["MEMCONTEXT_EMBED_EPISODES"] = "0" if mode == "deterministic" else "1"
    from memcontext.predicate_packs import active_pack

    active_pack.cache_clear()


def _install_log_capture() -> None:
    """Quiet structlog (it prints every insert) but record swallowed-failure events."""
    import logging

    import structlog

    def _capture(_logger: Any, _name: str, event_dict: dict) -> dict:
        _captured_events.append(str(event_dict.get("event", "")))
        return event_dict

    structlog.configure(
        processors=[_capture, structlog.processors.KeyValueRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(file=open(os.devnull, "w")),  # noqa: SIM115
        cache_logger_on_first_use=False,
    )


# ------------------------------------------------------------ source locations ---

_ANCHORS: dict[str, tuple[str, str]] = {
    "P1_SCOPE": ("memcontext/supersession.py", '" AND c.subject = ? AND c.predicate = ?"'),
    "P1_NEEDS_TRIPLE": ("memcontext/supersession.py", "if not new_claim.subject or not new_claim.predicate"),
    "P1_CARD": ("memcontext/supersession.py", "if new_claim.predicate in active_pack().single_valued:"),
    "P1_ATTR": ("memcontext/supersession.py", "_SINGLE_VALUED_ATTRIBUTES: dict"),
    "P1_WINDOW": ("memcontext/supersession.py", "_CLOSED_WINDOW_RE = "),
    "P1_SAMEVAL": ("memcontext/supersession.py", "new_value_norm = new_claim.value.strip().lower()"),
    "P1_QTY": ("memcontext/supersession.py", "if new_nn and new_nn == old_nn and new_content != old_content"),
    "P1_JACCARD": ("memcontext/supersession.py", "if len(shared) >= 2 and jaccard >= 0.3"),
    "P1_TRUST": ("memcontext/supersession.py", "if _claim_trust(conn, new_claim.claim_id) + 0.2 <"),
    "P1_COPY": ("memcontext/supersession.py", "copy = row_to_claim(row)"),
    "P2_THRESHOLD": ("memcontext/supersession_semantic.py", "DEFAULT_COSINE_THRESHOLD = "),
    "P2_NL_SCOPE": ("memcontext/supersession_semantic.py", "NL-only identity is fuzzy, so it stays session-scoped"),
    "P2_CAND_PRED": ("memcontext/supersession_semantic.py", '" AND c.predicate = ?"'),
    "P2_IDENTITY": ("memcontext/supersession_semantic.py", "def identity_text"),
    "P2_NO_TRUST": ("memcontext/supersession_semantic.py", "edge_type=EdgeType.SEMANTIC_REPLACE"),
    "P2_CALLSITE": ("memcontext/on_new_turn.py", "if semantic is not None and edge1 is None"),
    "API_NO_EVENT_TIME": ("memcontext/mcp_tools.py", "def handle_memory_store("),
}


def _loc(name: str) -> str:
    rel, anchor = _ANCHORS[name]
    try:
        for i, line in enumerate((REPO_ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
            if anchor in line:
                return f"{rel}:{i}"
    except OSError:
        pass
    return f"{rel}:? (anchor not found: {anchor!r})"


# ------------------------------------------------------------------- dataset ---


def load_cases(split: str) -> tuple[list[dict], dict[str, str]]:
    files = ["dev", "heldout"] if split == "all" else [split]
    cases: list[dict] = []
    digests: dict[str, str] = {}
    for f in files:
        path = DATASETS / f"supersession_{f}.json"
        raw = path.read_bytes()
        digests[path.name] = hashlib.sha256(raw).hexdigest()
        for c in json.loads(raw)["cases"]:
            _validate_case(c)
            cases.append({**c, "split": f})
    return cases, digests


def _validate_case(c: dict) -> None:
    keys = [s["key"] for s in c["stores"]]
    exp = c["expected"]
    if len(set(keys)) != len(keys):
        raise HarnessError(f"{c['id']}: duplicate store keys")
    if set(exp["active"]) | set(exp["superseded"]) != set(keys) or set(exp["active"]) & set(exp["superseded"]):
        raise HarnessError(f"{c['id']}: expected active/superseded must partition the store keys")
    if c["category"] not in CATEGORIES:
        raise HarnessError(f"{c['id']}: unknown category {c['category']!r}")


def label_for(case: dict, mode: str) -> str:
    return case.get("label_by_mode", {}).get(mode, case.get("label", "normal"))


# ------------------------------------------------------------------ run case ---


def run_case(case: dict, mode: str) -> dict:
    from memcontext.mcp_tools import handle_memory_contradictions, handle_memory_store
    from memcontext.schema import open_database

    conn = open_database(":memory:")
    started = time.perf_counter()
    n_events = len(_captured_events)
    key_of: dict[str, str] = {}
    for st in case["stores"]:
        claim = {"subject": st["subject"], "predicate": st["predicate"], "value": st["value"]}
        out = handle_memory_store(
            conn, text=st["text"], speaker=st["speaker"], session_id=st["session"],
            namespace=st["namespace"], claims=[claim],
        )
        if not out["admitted"] or out["claims_created"] != 1:
            raise HarnessError(f"{case['id']}/{st['key']}: store not ingested as one claim: {out}")
        cid = out["claim_ids"][0]
        pred = conn.execute("SELECT predicate FROM claims WHERE claim_id = ?", (cid,)).fetchone()[0]
        if bool(st.get("nl")) != (pred is None):
            raise HarnessError(f"{case['id']}/{st['key']}: nl={st.get('nl')} but stored predicate={pred!r}")
        key_of[cid] = st["key"]
    _check_embedding_integrity(conn, case, mode, n_events)

    status = {
        key_of[r["claim_id"]]: r["status"]
        for r in conn.execute("SELECT claim_id, status FROM claims").fetchall()
    }
    edges = [
        {"old": key_of[r["old_claim_id"]], "new": key_of[r["new_claim_id"]],
         "type": r["edge_type"], "score": r["identity_score"]}
        for r in conn.execute(
            "SELECT old_claim_id, new_claim_id, edge_type, identity_score"
            " FROM supersession_edges ORDER BY created_ts").fetchall()
    ]
    contradiction = handle_memory_contradictions(conn)["count"] > 0
    obs = {
        "active": sorted(k for k, s in status.items() if s in LIVE),
        "superseded": sorted(k for k, s in status.items() if s == "superseded"),
        "other": {k: s for k, s in status.items() if s not in LIVE and s != "superseded"},
        "contradiction": contradiction, "edges": edges, "status": status,
    }
    exp = case["expected"]
    correct = (
        obs["active"] == sorted(exp["active"])
        and obs["superseded"] == sorted(exp["superseded"])
        and not obs["other"]
        and (exp["contradiction"] is None or exp["contradiction"] == contradiction)
    )
    result = {
        "id": case["id"], "category": case["category"], "split": case["split"],
        "label": label_for(case, mode), "description": case["description"],
        "correct": correct, "expected": exp, "observed": obs,
        "seconds": round(time.perf_counter() - started, 3),
    }
    if not correct:
        result["root_causes"] = diagnose(case, obs, mode)
        if mode == "semantic":
            result["pass2_cosines"] = pass2_cosines(conn, case, key_of)
    conn.close()
    return result


def _check_embedding_integrity(conn: Any, case: dict, mode: str, n_events_before: int) -> None:
    n_emb = conn.execute("SELECT COUNT(*) FROM turn_embeddings").fetchone()[0]
    if mode == "deterministic":
        if n_emb:
            raise HarnessError(f"{case['id']}: deterministic mode produced {n_emb} embeddings")
        return
    fatal = [e for e in _captured_events[n_events_before:] if e in FATAL_SEMANTIC_EVENTS]
    if n_emb != len(case["stores"]) or fatal:
        raise SystemExit(
            f"ABORT semantic mode: case {case['id']} produced {n_emb} turn_embeddings for "
            f"{len(case['stores'])} stores; swallowed failures={fatal}. Semantic memory is NOT "
            "actually on -- refusing to report semantic numbers."
        )


def pass2_cosines(conn: Any, case: dict, key_of: dict[str, str]) -> dict[str, float]:
    """Cosine between the exact texts Pass-2 compares, for every earlier->later pair."""
    from memcontext.claims import row_to_claim
    from memcontext.retrieval import episode_embedder
    from memcontext.supersession_semantic import cosine, identity_text

    emb = episode_embedder()
    if emb is None:
        return {}
    rows = conn.execute(
        "SELECT c.*, t.text AS turn_text FROM claims c JOIN turns t ON t.turn_id = c.source_turn_id"
        " ORDER BY c.created_ts").fetchall()
    texts, keys = [], []
    for r in rows:
        c = row_to_claim(r)
        texts.append(c.text if not c.predicate else identity_text(c, r["turn_text"]))
        keys.append(key_of[c.claim_id])
    vecs = emb.embed(texts)
    return {
        f"{keys[i]}->{keys[j]}": round(cosine(vecs[i], vecs[j]), 4)
        for i in range(len(keys)) for j in range(i + 1, len(keys))
    }


# ----------------------------------------------------------------- diagnosis ---

_NOISE = frozenset(["the", "a", "an", "is", "was", "to", "for", "and", "or", "of", "in", "on", "at", "it", "my", "i", "me", "we", "up", "so", "no", "not", "but", "with", "has", "had", "be", "do", "did", "will", "been", "just", "very", "really", "also", "about", "some", "from", "that", "this", "more", "than", "each", "during"])
_NUMWORDS = frozenset(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "couple", "few", "several", "many", "single", "both", "dozen", "hundred", "thousand", "a"])
_HISTORY_MARKERS = re.compile(r"\b(used to|previously|formerly|back then|lived|worked)\b", re.I)
_NEGATION = re.compile(r"\b(no longer|stopped|not|n't|anymore|cancelled|removed|off)\b", re.I)


def _content(v: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", v.lower())) - _NOISE


def _overlap(a: str, b: str) -> str:
    ca, cb = _content(a), _content(b)
    shared = ca & cb
    j = len(shared) / len(ca | cb) if ca | cb else 0.0
    return f"shared content tokens={sorted(shared)} jaccard={j:.2f}"


def _rc(layer: str, anchor: str, why: str) -> dict:
    return {"layer": layer, "where": _loc(anchor), "why": why}


def _norm(v: str) -> str:
    return v.strip().lower()


def _single_valued() -> frozenset[str]:
    from memcontext.predicate_packs import active_pack

    return active_pack().single_valued


def diagnose(case: dict, obs: dict, mode: str) -> list[dict]:
    """Heuristic root-cause guess per wrong decision (read the edges, not just the states)."""
    st = {s["key"]: s for s in case["stores"]}
    exp = case["expected"]
    causes: list[dict] = []
    for k in exp["active"]:
        if obs["status"][k] not in LIVE:
            retiring = [e for e in obs["edges"] if e["old"] == k and e["type"] != "contradicts"]
            causes.append(_why_over(st, k, retiring[0] if retiring else None))
    for k in exp["superseded"]:
        if obs["status"][k] in LIVE:
            causes.append(_why_under(case, st, obs, k, mode))
    if exp["contradiction"] is not None and exp["contradiction"] != obs["contradiction"]:
        causes.append(_why_contradiction(exp, obs))
    return causes


def _why_over(st: dict, k: str, edge: dict | None) -> dict:
    if edge is None:
        return _rc("Integration", "P1_COPY", f"{k} left the active set with no supersession edge")
    old, new = st[edge["old"]], st[edge["new"]]
    tag = f"{edge['old']}->{edge['new']} ({edge['type']})"
    if edge["type"] == "semantic_replace":
        if old["namespace"] != new["namespace"]:
            return _rc("Logic", "P2_NL_SCOPE", f"{tag}: Pass-2 NL-mode candidates are selected by session_id only, so a same-named session in ANOTHER namespace is in scope -- tenant isolation leak")
        if _norm(old["value"]) == _norm(new["value"]):
            return _rc("Integration", "P2_CALLSITE", f"{tag}: Pass-1 deliberately skips an identical value (duplicates are recurrence evidence), but the pipeline then runs Pass-2 because edge1 is None; value-free identity scores ~1.0 and retires the duplicate")
        if TRUST[old["speaker"]] > TRUST[new["speaker"]] + 0.2:
            return _rc("Logic", "P2_NO_TRUST", f"{tag}: Pass-2 has no source-trust guard (Pass-1's lives at {_loc('P1_TRUST')}); an assistant fact retired a user fact")
        if not (old.get("nl") or new.get("nl")) and _norm(old["subject"]) != _norm(new["subject"]):
            return _rc("Logic", "P2_CAND_PRED", f"{tag}: Pass-2 structured candidates match on predicate only (subject-blind); identity is dominated by turn phrasing, so another subject's fact was retired")
        if old.get("nl") or new.get("nl"):
            return _rc("Logic", "P2_THRESHOLD", f"{tag}: NL identity is the whole fact text; related-but-distinct facts clear cosine 0.88")
        return _rc("Logic", "P2_IDENTITY", f"{tag}: identity text excludes the value by design, so distinct values under the same subject+predicate with similar phrasing clear 0.88 -- Pass-2 cannot tell additive from update")
    if new["predicate"] in _single_valued():
        return _rc("Logic", "P1_CARD", f"{tag}: declared single_valued predicate -- the cardinality rule retires any different value, so distinct concurrent facts on one subject collapse")
    from memcontext.supersession import _has_closed_window

    if _has_closed_window(old["value"]) or _has_closed_window(new["value"]):
        return _rc("Logic", "P1_JACCARD", f"{tag}: the closed-window history guard ({_loc('P1_WINDOW')}) is applied only in the attribute-slot branch; the Jaccard branch ignores it, so shared verb/year tokens retire one historical record with another ({_overlap(old['value'], new['value'])})")
    if _HISTORY_MARKERS.search(old["value"]) or _HISTORY_MARKERS.search(new["value"]):
        return _rc("Logic", "P1_WINDOW", f"{tag}: history guard only recognises closed ranges/years; past-tense markers ('used to', 'previously') are read as a current value of the same attribute slot")
    if re.sub(r"\W", "", old["value"].lower()) == re.sub(r"\W", "", new["value"].lower()):
        return _rc("Logic", "P1_SAMEVAL", f"{tag}: duplicate test is exact strip().lower(); a punctuation-only variant counts as a new value and then Jaccard=1.0 supersedes it")
    nn_old = {t for t in _content(old["value"]) if not t.isdigit() and t not in _NUMWORDS}
    nn_new = {t for t in _content(new["value"]) if not t.isdigit() and t not in _NUMWORDS}
    if nn_old == nn_new:
        return _rc("Logic", "P1_QTY", f"{tag}: quantity rule fired although the numbers describe different things")
    return _rc("Logic", "P1_JACCARD", f"{tag}: >=2 shared content tokens and J>=0.3 read as an update; additive facts sharing template words collapse ({_overlap(old['value'], new['value'])})")


def _why_under(case: dict, st: dict, obs: dict, k: str, mode: str) -> dict:
    old = st[k]
    order = [s["key"] for s in case["stores"]]
    later = [x for x in order[order.index(k) + 1:] if x in case["expected"]["active"]]
    new = st[later[-1]] if later else st[order[-1]]
    tag = f"{k} should be retired by {new['key']}"
    contra = [e for e in obs["edges"] if e["type"] == "contradicts" and k in (e["old"], e["new"])]
    if contra:
        other = contra[0]["new"] if contra[0]["old"] == k else contra[0]["old"]
        if obs["status"][other] not in LIVE:
            return _rc("Logic", "P1_COPY", f"{tag}: the correction retires only the newest prior value plus exact copies of it; an older divergent value kept alive by the trust guard's CONTRADICTS edge ({k}<->{other}) survives, leaving two current values")
        return _rc("Logic", "P1_TRUST", f"{tag}: source-trust guard downgraded the edge to CONTRADICTS")
    if old.get("nl") or new.get("nl"):
        if mode == "deterministic":
            return _rc("Integration", "P1_NEEDS_TRIPLE", f"{tag}: NL-only facts have no triple, Pass-1 returns early and Pass-2 is off in this mode")
        return _rc("Logic", "P2_THRESHOLD", f"{tag}: Pass-2 NL cosine below 0.88 for a genuine update (see pass2_cosines)")
    if old["predicate"] != new["predicate"]:
        return _rc("Logic", "P1_SCOPE", f"{tag}: Pass-1/Pass-2 candidates require the SAME predicate ({old['predicate']} vs {new['predicate']}); one slot under two predicates is never linked")
    if _norm(old["subject"]).replace(" ", "_") != _norm(new["subject"]).replace(" ", "_"):
        return _rc("Logic", "P1_SCOPE", f"{tag}: subjects differ after normalisation ({old['subject']!r} vs {new['subject']!r})")
    extra = " Pass-2 also missed it (cosine < 0.88, see pass2_cosines)." if mode == "semantic" else ""
    if _NEGATION.search(new["value"]):
        return _rc("Logic", "P1_JACCARD", f"{tag}: no polarity/negation rule -- the negated restatement shares too few content tokens with the positive fact ({_overlap(old['value'], new['value'])}).{extra}")
    return _rc("Logic", "P1_JACCARD", f"{tag}: no Pass-1 rule fired -- not single_valued, no attribute slot ({_loc('P1_ATTR')}), no pure-quantity change, and {_overlap(old['value'], new['value'])} is below >=2 tokens / J>=0.3.{extra}")


def _why_contradiction(exp: dict, obs: dict) -> dict:
    if exp["contradiction"] and not obs["contradiction"]:
        if any(e["type"] == "contradicts" for e in obs["edges"]):
            return _rc("Logic", "P1_COPY", "conflict was recorded as CONTRADICTS against a value that was later replaced; nothing re-links the user fact to the NEW conflicting value, so memory_contradictions goes silent")
        if any(e["type"] == "semantic_replace" for e in obs["edges"]):
            return _rc("Logic", "P2_NO_TRUST", "Pass-2 retired a fact instead of surfacing the user-vs-assistant conflict")
        return _rc("Logic", "P1_TRUST", "trust guard only runs after a Pass-1 match; no Pass-1 rule matched, so the conflict is neither recorded nor surfaced")
    return _rc("Logic", "P1_TRUST", "a conflict was surfaced where none was intended")


# ------------------------------------------------------------------ reporting ---


def summarize(results: list[dict]) -> dict:
    per_cat: dict[str, dict] = {}
    for r in results:
        d = per_cat.setdefault(r["category"], {"n": 0, "correct": 0, "excluded": 0, "excluded_correct": 0})
        if r["label"] in EXCLUDED_LABELS:
            d["excluded"] += 1
            d["excluded_correct"] += int(r["correct"])
            continue
        d["n"] += 1
        d["correct"] += int(r["correct"])
    for d in per_cat.values():
        d["accuracy"] = round(d["correct"] / d["n"], 3) if d["n"] else None
    n = sum(d["n"] for d in per_cat.values())
    ok = sum(d["correct"] for d in per_cat.values())
    return {"per_category": {c: per_cat[c] for c in CATEGORIES if c in per_cat},
            "overall": {"n": n, "correct": ok, "accuracy": round(ok / n, 3) if n else None,
                        "excluded": sum(d["excluded"] for d in per_cat.values())}}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Files whose content decides every verdict here; hashed so a result is tied to the exact
# code that produced it even when the worktree is dirty.
DECISION_PATH = (
    "memcontext/supersession.py", "memcontext/supersession_semantic.py",
    "memcontext/on_new_turn.py", "memcontext/claims.py", "memcontext/mcp_tools.py",
    "memcontext/retrieval.py", "predicate_packs/general/predicates.json",
    "predicate_packs/developer/predicates.json",
)


def _meta(mode: str, split: str, digests: dict[str, str], extra: dict) -> dict:
    dirty = _git("status", "--porcelain", "--", "memcontext", "predicate_packs")
    return {
        "eval": "supersession_matrix", "mode": mode, "split": split,
        "git_commit": _git("rev-parse", "HEAD"),
        "product_code_dirty": dirty not in ("", "unknown"),
        "product_dirty_files": [ln[3:] for ln in dirty.splitlines()] if dirty != "unknown" else [],
        "decision_path_sha256": {
            p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest()[:16] for p in DECISION_PATH
        },
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "active_pack": PACK, "datasets_sha256": digests, **extra,
    }


def _semantic_preflight() -> dict:
    from memcontext.retrieval import (
        BGE_M3_VERSION_TAG,
        probe_embedder,
        semantic_supersession,
    )

    t0 = time.perf_counter()
    probe = probe_embedder()
    if probe is None or not probe.ok:
        raise SystemExit(f"ABORT semantic mode: embedder probe failed ({probe}). Not reporting numbers.")
    if semantic_supersession() is None:
        raise SystemExit("ABORT semantic mode: semantic_supersession() is None although the probe passed.")
    return {"embedder": BGE_M3_VERSION_TAG, "embed_dim": probe.dim,
            "model_load_seconds": round(time.perf_counter() - t0, 1)}


def run_mode(mode: str, split: str, cases: list[dict], digests: dict[str, str]) -> dict:
    _configure(mode)
    extra: dict = {}
    if mode == "semantic":
        extra = _semantic_preflight()
    else:
        from memcontext.retrieval import semantic_supersession

        if semantic_supersession() is not None:
            raise HarnessError("deterministic mode but Pass-2 is wired")
    results = []
    for i, c in enumerate(cases, 1):
        results.append(run_case(c, mode))
        if mode == "semantic" and i % 20 == 0:
            print(f"  [{mode}] {i}/{len(cases)} cases", flush=True)
    summary = summarize(results)
    failures = [r for r in results if not r["correct"] and r["label"] not in EXCLUDED_LABELS]
    excluded = [r for r in results if r["label"] in EXCLUDED_LABELS]
    report = {"meta": _meta(mode, split, digests, extra), **summary,
              "failures": failures, "excluded_cases": excluded, "cases": results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"supersession_matrix_{mode}_{split}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return report


def disagreements(det: dict, sem: dict) -> list[dict]:
    by_id = {r["id"]: r for r in sem["cases"]}
    rows = []
    for d in det["cases"]:
        s = by_id[d["id"]]
        if d["correct"] != s["correct"] or d["observed"]["active"] != s["observed"]["active"] \
                or d["observed"]["contradiction"] != s["observed"]["contradiction"]:
            rows.append({
                "id": d["id"], "category": d["category"],
                "label_det": d["label"], "label_sem": s["label"],
                "deterministic": {"correct": d["correct"], "active": d["observed"]["active"],
                                  "edges": [f"{e['old']}->{e['new']}:{e['type']}" for e in d["observed"]["edges"]]},
                "semantic": {"correct": s["correct"], "active": s["observed"]["active"],
                             "edges": [f"{e['old']}->{e['new']}:{e['type']}" for e in s["observed"]["edges"]]},
            })
    return rows


def print_table(reports: dict[str, dict]) -> None:
    modes = list(reports)
    head = f"{'category':26s}" + "".join(f"{m:>18s}" for m in modes)
    print("\n" + head + "\n" + "-" * len(head))
    for cat in CATEGORIES:
        cells = []
        for m in modes:
            d = reports[m]["per_category"].get(cat)
            if d is None:
                cells.append(f"{'-':>18s}")
                continue
            acc = f"{d['accuracy']:.0%}" if d["accuracy"] is not None else "n/a"
            ex = f"+{d['excluded']}x" if d["excluded"] else ""
            cells.append(f"{d['correct']:>3d}/{d['n']:<3d}{acc:>5s} {ex:>4s}".rjust(18))
        print(f"{cat:26s}" + "".join(cells))
    print("-" * len(head))
    print(f"{'OVERALL (scored)':26s}" + "".join(
        f"{reports[m]['overall']['correct']:>3d}/{reports[m]['overall']['n']:<3d}"
        f"{reports[m]['overall']['accuracy']:>6.1%}".rjust(18) for m in modes))
    print("(+Nx = N ambiguous/expected_limitation cases, excluded)\n")
    for m in modes:
        print(f"FAILURES [{m}]")
        for r in reports[m]["failures"]:
            rc = r["root_causes"][0] if r["root_causes"] else {"layer": "?", "where": "?", "why": ""}
            print(f"  {r['id']:8s} exp active={r['expected']['active']} con={r['expected']['contradiction']}"
                  f" | obs active={r['observed']['active']} con={r['observed']['contradiction']}"
                  f" | {rc['layer']} {rc['where']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    ap.add_argument("--mode", choices=["deterministic", "semantic", "both"], default="deterministic")
    args = ap.parse_args(argv)
    _configure("deterministic")
    _install_log_capture()
    cases, digests = load_cases(args.split)
    modes = ["deterministic", "semantic"] if args.mode == "both" else [args.mode]
    reports = {m: run_mode(m, args.split, cases, digests) for m in modes}
    print_table(reports)
    if len(reports) == 2:
        rows = disagreements(reports["deterministic"], reports["semantic"])
        out = RESULTS_DIR / f"supersession_matrix_disagreements_{args.split}.json"
        out.write_text(json.dumps({"meta": _meta("both", args.split, digests, {}), "rows": rows},
                                  indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDISAGREEMENTS deterministic vs semantic ({len(rows)}) -> {out.relative_to(REPO_ROOT)}")
        for r in rows:
            print(f"  {r['id']:8s} det ok={r['deterministic']['correct']!s:5s} {r['deterministic']['edges']}"
                  f" | sem ok={r['semantic']['correct']!s:5s} {r['semantic']['edges']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
