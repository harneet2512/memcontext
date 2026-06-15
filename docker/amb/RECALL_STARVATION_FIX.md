# Recall-starvation in cross-session retrieval — diagnosis & fix

**Date:** 2026-06-15
**Trial branch:** `codex/repro-raw-amb-bridge` (this branch — reproducible benchmark artifact)
**Product fix lands on:** `master` (`memcontext/retrieval.py`, `tests/test_two_tier_wiring.py`)
**Status:** root cause proven and fixed; net benchmark accuracy **not yet re-measured**.

---

## Verdict

The bad AMB/LongMemEval numbers were a **recall** failure in the **product's
cross-session retrieval**, not the answer layer and not the bridge wiring.
`retrieve_memory_across` applied a single **global** `fused[:top_k]` cap after
per-session RRF. Whenever the haystack held **≥ `top_k` sessions**, that cap
admitted only each session's **rank-1** turn — so a gold turn that was rank-2+
*within its own session* was **structurally unreachable**, no matter how good
ingest, embeddings, or extraction were.

Measured effect of removing the starvation (same embeddings, only the retrieval
strategy changed): **gold-turn recall 33% → 72%.**

This is **one design decision inside one function**, not architecture-wide rot.
The substrate model (claims + episodes + provenance + supersession + RRF), the
ingest/embedding path, and the bridge are sound.

---

## The defect

`memcontext/retrieval.py`, `retrieve_memory_across` (pre-fix):

```python
fused = []
for sid in session_ids:
    fused.extend(retrieve_memory(conn, session_id=sid, query=query, top_k=top_k, ...))
fused.sort(key=lambda h: (-h[1], h[0].kind != "fact", h[0].id))
return fused[:top_k]            # <-- global cap
```

Per-session RRF gives every session's **rank-1** the score `w/(RRF_K+1)`, which
sorts **above any rank-2** (`w/(RRF_K+2)`) from *any* session. So a global
`fused[:top_k]` fills entirely with rank-1s once `n_sessions ≥ top_k`. A
session never contributes a second turn. The function's own docstring stated the
intended contract — *"every queried session is represented (up to `top_k`)"* —
which is exactly the breadth-only behavior that starves depth.

Compounding it: the real product's `classify_query_depth` (`retrieval.py:768`)
returns `top_k = 15` for a plain **factual** query (30 temporal, 50 aggregation).
Over a 53-session LongMemEval haystack that serves ~15 of 53 sessions, one turn
each. The AMB bridge hardcoded `top_k=50`, which **masked** the worst of it —
"the bridge is built wrong" was half-right: it was hiding the product defect, not
causing it.

### Why this only bites at benchmark scale

The cap is harmless when `n_sessions < top_k` (budget exceeds session count, so
rank-2/3 survive — normal product use: a user's handful of sessions). It only
starves when sessions **outnumber** the budget — the LongMemEval haystack shape
(1–2 gold sessions buried among ~53).

---

## Measurement (reproducible)

Gold-turn recall = the `has_answer` turn's exact text present in the served
context, using the dataset's real `answer_session_ids` ground truth. The
effective bridge provider ingests the 53-session haystack, retrieves, and we
check presence.

**Recall at `top_k=50` (the bridge's current cap), n=2/category:**

| category | recall |
|---|---|
| single-session-user | 2/2 |
| temporal-reasoning | 2/2 |
| multi-session | 1/2 |
| knowledge-update | **0/2** |
| single-session-preference | **0/2** |
| single-session-assistant | **0/2** |
| **TOTAL** | **5/12 = 42%** |

**Mutation check — global cap vs per-session keep, same embedded data, n=3/cat:**

| category | global top-50 | **per-session keep-3** |
|---|---|---|
| knowledge-update | 0/3 | **2/3** |
| multi-session | 2/3 | **3/3** |
| single-session-assistant | 0/3 | 1/3 |
| single-session-preference | 0/3 | 1/3 |
| single-session-user | 2/3 | **3/3** |
| temporal-reasoning | 2/3 | **3/3** |
| **TOTAL** | **6/18 = 33%** | **13/18 = 72%** |

**Disproof of the cheap fix — you cannot concentrate depth on "relevant"
sessions:**

| strategy | recall | avg turns |
|---|---|---|
| global top-50 | 33% | 50 |
| top-8 sessions × 3 (by score) | **17%** | 23 |
| top-12 sessions × 3 (by score) | **17%** | 35 |
| **all sessions × top-3** | **72%** | 138 |

Score-ranking sessions does **worse** than the baseline, because raw
cross-session scores are not comparable (a long off-topic session out-scores a
terse session that holds the answer) — the exact reason RRF exists. The answer
session gets dropped. **Breadth is load-bearing; depth is layered on top of every
session.** The cost is real: recovering recall means serving ~138 turns over a
53-session haystack, because the needle's session cannot be pre-identified.

> Caveat: the harness ran `SimpleExtractor` (no extractor key), so absolute
> recall is a **lower bound**; the real run uses MiniMax extraction. The
> *relative* result (global cap starves, per-session keep recovers) is a
> retrieval-structure effect and is extractor-independent.

---

## The fix

`memcontext/retrieval.py`, `retrieve_memory_across` — two-pass reserve/overflow:

```python
# Each session RESERVES its top-`per_session_k`; the remainder share the budget.
per_session_k = max(1, per_session_k)
reserved, overflow = [], []
for sid in session_ids:
    hits = retrieve_memory(conn, session_id=sid, query=query, top_k=top_k, ...)
    reserved.extend(hits[:per_session_k])
    overflow.extend(hits[per_session_k:])
reserved.sort(key=tie); overflow.sort(key=tie)
budget = min(max(top_k, len(reserved)), MAX_ACROSS_HITS)   # never below the guarantee
return (reserved + overflow)[:budget]
```

New module constants: `DEFAULT_PER_SESSION_K = 3`, `MAX_ACROSS_HITS = 300`.
`per_session_k` is a defaulted kwarg, so all callers (`mcp_tools.handle_memory_query`,
`tool_retrieval`, the AMB bridge) inherit the fix without signature churn.

**Properties:**
- `n_sessions < top_k`: `budget = top_k`, behavior ≈ prior (depth already
  allowed) — no regression to normal product use.
- `n_sessions ≥ top_k`: `budget` grows to the per-session guarantee, so each
  session keeps its top-`per_session_k` — starvation removed.
- `MAX_ACROSS_HITS` caps a pathological session count.

`per_session_k` is the **recall ↔ context knob.** At 3 over 53 sessions the
bridge serves ~138 turns instead of 50; drop to 2 to cap context before a run.

---

## Test (mutation-verified)

`tests/test_two_tier_wiring.py::test_cross_session_keeps_per_session_depth_when_sessions_exceed_top_k`:
8 sessions, `top_k=3`, 3 retrievable turns each. Asserts `len(hits) > top_k` and
that at least one session keeps ≥2 hits.

- Fix in place → **GREEN**.
- Mutation (revert budget to `fused[:top_k]`) → **RED** (`len == top_k`, one hit
  per session). Restored byte-identical.
- Full product suite: **291 passed, 1 skipped.**

Without this test the starvation is invisible to CI (existing tests use few
sessions), so the defect could silently return — the test is the tripwire.

---

## LIPI

- **Logic** — root cause: global cap defeats multi-source RRF when sources ≥
  budget. Fixed at source. Clean.
- **Implementation** — two-pass reserve/overflow; suite green; test bites under
  mutation. Clean.
- **Integration** — defaulted kwarg; callers + bridge inherit; small-n path
  preserved. Clean.
- **Plumbing** — no embedding/schema change; NullEmbedder tests pass. Clean.

---

## Owed (do not over-trust)

1. **Net benchmark accuracy is unmeasured.** Recall is up; the reader (gpt-oss)
   now sees ~138 turns instead of 50 — recall↑ / precision↓ is a real tradeoff
   that only an honest run resolves. **No benchmark-gain claim until then.**
   Consider `per_session_k=2` first to bound context.
2. **Second defect, untouched:** single-session-assistant / single-session-
   preference stay ~33% *even* with per-session keep — the gold turn isn't its
   session's top-3 by **within-session** ranking (the query matches the user's
   question, not the assistant's answer / the preference statement). This is a
   within-session ranking problem (query→answer-turn similarity), separate from
   the cross-session cap. Next target.
3. **Frozen answer layer is the real ceiling** — gpt-oss reader + gemini judge vs
   the native 88% harness's GPT-5-mini + GPT-4o-2024-11-20. Outside the product;
   caps how high any retrieval fix can push the score.

---

## Reproduce

```bash
# dataset (real LongMemEval-S, with answer_session_ids / has_answer ground truth)
curl -fsSL --retry-all-errors --retry 6 -o /tmp/lme.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"

# recall harness (effective bridge provider; SUBSTRATE_PACKS_DIR + ACTIVE_PACK set)
SUBSTRATE_PACKS_DIR=<repo>/predicate_packs ACTIVE_PACK=personal_assistant \
PYTHONPATH=<effective-provider> python recall2.py 3      # global vs per-session-keep
```

The harness scripts (`recall.py`, `recall2.py`, `recall3.py`) ingest the
haystack, retrieve with each strategy, and score gold-turn presence against the
dataset's own `has_answer` flags — no answer-substring proxy.
