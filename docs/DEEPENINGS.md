# Substrate deepenings

General, deterministic, zero-LLM strengthenings of the memory/context substrate.
Each helps **any** user of the product (storing and querying their own
conversations, pages, tools, documents) and is independent of any benchmark — the
benchmark only *measures* these, it never shapes them. Every decision is
data-driven (embedding distance, score distribution, timestamps), never a
hardcoded predicate/answer list.

---

## 1. Tie-aware rank fusion (`_rrf_ranks`)

**Capability.** Multi-signal retrieval fuses channels (semantic, entity, temporal,
BM25, …) by Reciprocal Rank Fusion. A channel with no opinion on a query (flat /
degenerate scores) must not distort the ranking.

**The bug it fixes.** The old `_rrf_ranks` broke ties by **index** (`(-score, i)`),
so a flat channel emitted ranks `1..N` in claim order — injecting index-ordered
noise that buries an item another channel ranked #1. Now it uses **competition
ranking**: equal scores share the same (lowest) rank, so a flat channel assigns
every item the same rank and contributes a *constant* — staying neutral. Distinct
scores are unaffected (identical to the old strict ordering), so the only behaviour
that changes is genuine ties — exactly the degenerate-channel case.

**Why general.** Pure property of score distributions; helps any user whose
high-precision lexical/entity hit would otherwise be diluted by an abstaining
channel. It's the standard competition-ranking definition, not a tuned constant.

- **Class:** PROVEN (correctness fix; competition ranking is the textbook RRF input).
- **Change:** `memcontext/retrieval.py` `_rrf_ranks`.
- **Proof:** `results/proof_fusion_master.py` — a flat channel buried an entity-needle
  to **#4** under the old ranking; tie-aware ranking restores it to **#1**.
- **Test:** `tests/test_retrieval.py::test_rrf_ranks_ties` — property-based over 100
  random inputs (equal→equal rank, higher→lower rank); not hardcoded examples.
- **LIPI.** *Logic:* root cause was the index tie-break; competition ranking makes a
  flat channel neutral. *Implementation:* single function, signature unchanged.
  *Integration:* all channels in `retrieve_hybrid` benefit; distinct-score paths
  unchanged (no blast radius). *Plumbing:* deterministic; final sort still tie-breaks
  on `claim_id` for stable order.

---

## 2. Distinct-instance enumeration (`enumeration.py`)

**Capability.** Answer "how many X / list all X" (purchases, meetings, errors,
contacts) — which top-k retrieval plus a newest-wins projection structurally cannot,
because they converge to one value per `(subject, predicate)`.

**Mechanism.** `count_distinct_instances` groups the instances of a slot and
de-duplicates by embedding cluster — using the **max-gap valley** of the sorted
pairwise-cosine distribution as the deterministic cluster separator (not a magic
threshold). Counts retired instances too (the whole point: a superseded instance
still happened). A temporal guard keeps same-value/different-`event_ts` instances
distinct.

**Why general.** Every memory user eventually asks the system to tally or enumerate
what it has observed; grouping by entity/embedding is how a human assistant answers
"how many times did I…".

- **Class:** PROVEN (real-embedder: 5 distinct kits × 3 paraphrases each + 100 noise
  → `distinct == 5`). Deterministic aggregation over structured facts is the standard
  alternative to asking an LLM to count over a retrieved blob.
- **Change:** new `memcontext/enumeration.py`.
- **Test:** `tests/test_enumeration.py` (NullEmbedder-safe cases always run; real-
  embedder cases skip honestly if the model can't load — no fake pass).
- **LIPI.** *Logic:* max-gap valley separator (a 90th-percentile knee was rejected as
  the wrong statistic). *Implementation:* deterministic; dead `_percentile` removed.
  *Integration:* additive — exposes an enumeration result, never alters existing
  retrieval. *Plumbing:* skips under a degenerate embedder rather than miscount.

---

## 3. `event_ts` supersession guard (`detect_pass1`)

**Capability.** Update a single-valued *state* attribute (current address, employer,
mortgage balance) while never collapsing distinct *dated events* (two 5K runs, two
deploys) — which interval/counting questions need kept apart.

**Mechanism.** Before superseding a same-`(subject, predicate)` candidate, if **both**
claims carry an explicit `event_ts` and they **differ**, treat them as distinct
occurrences and keep both. No/equal `event_ts` ⇒ existing state-supersession
unchanged. Applied to all three Pass-1 cardinality paths plus a final guard.

**Why this, not embeddings.** Embedding distance was measured **unusable** as the
update-vs-event discriminator on the production embedder (numeric/ID updates like
`$350k`→`$400k`, cosine 0.989, are *more* similar than a genuine distinct event,
0.955 — negative class margin). The guard keys on `event_ts` presence/inequality —
deterministic, zero-LLM, no predicate list.

- **Class:** PROVEN (representation: bi-temporal validity, Zep/Graphiti, SQL:2011);
  the `event_ts` lever is deterministic and conservative (fires only when both sides
  are explicitly dated).
- **Change:** `memcontext/supersession.py` `_event_blocks` + guards in `detect_pass1`.
- **Test:** `tests/test_event_ts_guard.py` (12 cases). Existing `test_supersession.py`
  stays fully green — no regression.
- **LIPI.** *Logic:* distinct dated events are not contradictions. *Implementation:*
  one helper, guards all paths + final check. *Integration:* state attributes
  (`event_ts is None`) supersede as before. *Plumbing:* `event_ts` column already in
  `schema.py`; zero-LLM.

---

## Deliberately NOT done

- **Convex-combination fuser** — the tie-aware `_rrf_ranks` fix addresses the same
  flat-channel dilution at its root, generally and safely. A separate score-normalised
  convex fuser was prototyped but **regressed below RRF** in its naive form and only
  worked behind a fragile guard; the root-cause tie fix makes it unnecessary here.
- **Embedding-dimension fix** — master already resolves embedder dims via
  `_embedding_dim_for` (arctic-embed-s → 384 correctly); no bug to fix.
- **Embedding-distance supersession discriminator** — disproven on the production
  embedder (see §3); not shipped.

## Benchmark isolation

These are product capabilities measured by, but never tuned to, any benchmark. When
quoting LongMemEval/AMB numbers, note the disclosed judge-model deviation
(`gemini-3-flash` vs the official `GPT-4o`) — instrument configuration, not the
product. No change here references benchmark data, answers, or categories.
