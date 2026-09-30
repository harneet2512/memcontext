# HAR-95 baseline — what the claim-centric store actually does

Measured on `master` @ `c79d380` before any HAR-95 product change.
Reproduce:

```bash
python -m pytest tests/test_har95_baseline.py -q -rxX   # at c79d380: 8 passed, 9 xfailed (strict)
                                                         # after slice 1: GAP-1a, 2-4 promoted (SLICE1.md)
python -m tests.test_har95_baseline                      # prints the slot/token matrix below
```

Full suite before any change: **382 passed, 1 skipped** (`python -m pytest tests -q`).

## Scenario

One session. All writes go through the production handlers (`handle_memory_store`,
`handle_memory_query`, `handle_memory_trace`); the browser turn goes through the real
extraction tail (`run_extraction`), because no product door ingests browser content yet.

| Turn | Source | Text | Claims (caller-supplied) |
|---|---|---|---|
| T1 | user | Sarah said Acme will probably renew, but only if security approves SOC2 before Friday. | (Acme, renewal_status, probable), (Acme, renewal_blocker, security_approval) |
| T2 | user | Globex asked for a quote … no renewal decision yet. | (Globex, renewal_status, evaluating) |
| D0–D19 | user | 20 unrelated account notes, so `top_k` actually binds | (acct, observation, note) |
| T3 | user | Update: security approved SOC2 on Wednesday and Acme confirmed they are renewing. | (Acme, renewal_status, confirmed) |
| T4 | user | Acme is renewing, confirmed again on the call today. | (Acme, renewal_status, confirmed) |
| TB | browser | Forum post: Acme is churning to a competitor next quarter. | (Acme, renewal_status, churning) |

There are two vocabularies:
- **general**: the default pack. `renewal_*` is out of vocabulary.
- **renewal/single_valued**: a test-local pack that adds `renewal_status` and
  `renewal_blocker`, with `renewal_status` declared `single_valued`.

The embedder is off (lexical BM25 + entity + recency), as in CI. A real-embedder run is follow-up (plan step 4).

## Premise check: the ticket's hypothesis is partly refuted

The ticket says every useful piece of information has to survive conversion into
`(subject, predicate, value)`. **As stated, that is false on current master:**
- Each turn's full text is stored verbatim (`turns.text`).
- Each claim keeps `source_turn_id`, and `memory_trace` returns the source text.
- `memory_query` serves raw turns ("episodes") alongside claims.

The nuance (Sarah, the condition, SOC2, Friday) is **preserved**. What is missing is the
**relationship between served evidence and the state derived from it**: grouping,
historical marking, trust on evidence, and evidence ↔ claim links in the response.
This changes what the first slice has to build (see "Storage model" below).

## Five questions

The verdict is shown for each pack (general / renewal-single_valued). Slot counts are items
served at the auto-selected `top_k=15`.

| # | Question | Verdict | What happens |
|---|---|---|---|
| 1 | What is Acme's current renewal status? | **Fragmented** / **Reconstructable** | general: `probable`, `confirmed`×2 and `churning` are all active, text-only, with no supersession edge. renewal: `confirmed` is active and `probable` is superseded; `churning` is active but quarantined, with a `contradicts` edge. In lexical mode 10 distractor claims rank above `confirmed`, and the slot trace names `churning` as current (GAP-5). |
| 2 | What is blocking renewal? | **Wrong (stale)** in both | `security_approval` stays active after T3 says SOC2 was approved (GAP-7). The correction is only in T3's text, and T3's episode is not served for this question. |
| 3 | Why does Sarah think Acme will renew? | **Preserved (evidence)** in both | T1 is served verbatim. The claims alone carry none of the nuance. T1 is not marked as outdated even though its status claim was superseded (GAP-4). |
| 4 | What exactly did Sarah say? | **Preserved** in both | T1 is served verbatim. |
| 5 | How did Acme's renewal situation change? | **Lost** via query / **Reconstructable** via trace | `memory_query` serves no superseded claim ("change" is not a history cue, GAP-6). `memory_trace(claim_id=<T3 claim>)` returns `confirmed ← probable` with T1's text. `memory_trace(subject, predicate)` starts at `churning` and shows no history (GAP-5). |

### Slot / token matrix (`python -m tests.test_har95_baseline`)

| Pack | Question | served | tokens | T1 slots | T1 episode served |
|---|---|---|---|---|---|
| general | 1 status | 15 | 199 | 2 | no |
| general | 2 blocker | 15 | 183 | 2 | no |
| general | 3 why | 15 | 202 | **3** | yes |
| general | 4 quote | 15 | 236 | **3** | yes |
| general | 5 change | 15 | 199 | 2 | no |
| renewal | 1 status | 15 | 230 | 1 | no |
| renewal | 2 blocker | 15 | 214 | 1 | no |
| renewal | 3 why | 15 | 225 | **2** | yes |
| renewal | 4 quote | 15 | 248 | **2** | yes |
| renewal | 5 change | 15 | 230 | 1 | no |

Every question gets exactly 11 claim slots and 4 episode slots. That comes from
`_fuse_memory`'s fixed 1.0 / 0.9 weighting over rank position, not from relevance.
Evidence is therefore capped at about a quarter of the context even for "why" questions.

## Confirmed gaps (strict-xfail tests)

| Gap | Test | Root cause (layer, location) |
|---|---|---|
| GAP-1 | evidence + derived claims take separate slots | Integration: `_fuse_memory` (retrieval.py) ranks facts and episodes independently. `MemoryHit.source_turn_id` exists but nothing groups by it. |
| GAP-2 | served episodes carry no trust / quarantine | Implementation: `handle_memory_query` sets trust only on `claims_out` (mcp_tools.py). |
| GAP-3 | served claims do not name their source evidence | Implementation: `claims_out` omits `source_turn_id`. |
| GAP-4 | evidence whose state was superseded is served as if current | Logic: an episode has no view of what happened to the claims derived from it. |
| GAP-5 | slot trace head is the newest active claim, even a quarantined one | Logic: `find_same_identity_claim` ignores trust. When the trust guard blocks an override, both claims stay active and the newer low-trust one wins. |
| GAP-6 | "how did X change" is not a history query | Logic: `_HISTORY_INTENT` (retrieval.py) has no evolution phrasing. |
| GAP-7 | a resolved blocker stays active | Logic: there is no retraction primitive. Supersession only replaces a value, so nothing can close a slot. |
| GAP-8 | a restated value is served once per copy | Integration: restated copies stay active by design (recurrence evidence, supersession.py) and are served separately. |

## Plan suspicions: confirmed, refuted, revised

1. Out-of-vocab predicates lose Pass-1 supersession: **confirmed**. They are demoted to text-only, and `memory_store` warns.
2. No grouping: **confirmed** (GAP-1). The size is smaller than feared: 2–3 of 15 slots.
3. Raw turns served without trust flags: **confirmed** (GAP-2). Under `general`, the browser claim is also text-only, so there is no `contradicts` edge either. Its quarantine flag is the only protection.
4. Evidence not marked historical: **confirmed** (GAP-4).
5. Nothing retracts a claim: **confirmed** (GAP-7).

Not in the plan:
- Pass-1 needs declared cardinality. Without `single_valued`, a categorical correction (`probable` → `confirmed`, zero shared tokens) is treated as an additive fact and **both stay active**. This is intended over-supersession protection (supersession.py, multi-valued branch), but the module docstring ("same subject+predicate, different value") no longer says so. Pinned by `test_undeclared_cardinality_treats_categorical_update_as_additive`.
- GAP-5, GAP-6 and GAP-8 above.
- Slot trace vs claim trace: tracing a superseded claim shows only its predecessors (`build_chain` walks backwards). The successor is only in the legacy `supersession_chain` field.

## Bug found while building the baseline (fixed separately)

The first `memory_query` in the renewal scenario took **401 s**, and each later one about 4 s.
`retrieve_event_frames` embedded the query with the process-default client before checking
whether any frame had a stored embedding. That loaded the full BGE-M3 model on the query path
even with `MEMCONTEXT_EMBED_EPISODES=0`, which ingest honours, so no frame is ever embedded
in lexical mode. With no backend installed, `embed()` raises instead, and `serve_event_frames`
swallowed the error and served **no** frames instead of its documented unranked fallback.
The fix and its regression test are `tests/test_event_frame_serving_degradation.py`.
Afterwards the baseline file runs in about 4 s.

## LIPI pass

- **Logic:** the evidence model is sound; turns already are the evidence objects. The broken logic sits at the evidence ↔ state seam: GAP-4, GAP-5, GAP-6 and GAP-7. There is also the cardinality requirement above.
- **Implementation:** the query response omits trust on episodes (GAP-2) and source links on claims (GAP-3). The event-frame query embed ignored the policy and paid for an unusable vector (fixed).
- **Integration:** fact and episode rankings are fused without regard to shared provenance (GAP-1, GAP-8). The 11:4 slot split is structural, not driven by relevance.
- **Plumbing:** the tests use the production handlers, no mocks. The only plumbing fault found was the model load on the query path (fixed). The pack override via `SUBSTRATE_PACKS_DIR` + `ACTIVE_PACK` + `active_pack.cache_clear()` works as documented.

## Storage model: decided from this evidence

Criteria, applied in this order:
1. Does the evidence already survive, verbatim and with provenance? **Yes** (`turns`).
2. Does the agreed granularity (one Memory per turn) need anything a turn row lacks? **No.** A turn has content, source type and metadata, timestamp, speaker, namespace, and an embedding sidecar. Its claims link to it by `source_turn_id`. Entities attach through its claims.
3. Do the gaps need new stored state? **GAP-1 to GAP-4 and GAP-8 do not.** They are read-side joins over `turns`, `claims` and `claim_metadata`. GAP-7 (retraction) needs a new lifecycle transition, not a new table.
4. Would a separate `memories` table add anything now? **No.** It would mirror `turns` 1:1 and create a second source of truth to keep consistent.

**Decision:** the first slice adds no table and no migration. A Memory is the existing
`turns` row, exposed through a first-class `Memory` read model (content, source, trust,
derived claims with their current status). Revisit a dedicated table only when a Memory
can outgrow a turn: sub-turn spans, a memory merged from several sources, or
memory → memory `updates` / `extends` edges.
