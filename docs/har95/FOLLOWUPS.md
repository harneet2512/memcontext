# HAR-95 follow-ups: slot budget, hook conflicts, trust-aware trace, evolution queries

Builds on [SLICE1.md](SLICE1.md). Each step is its own commit. Eval numbers are dev split
only, lexical mode, taken before and after (see "Evals" at the end).

## Step 1: GAP-1b, one evidence object = one retrieval slot

**Problem:** `_fuse_memory` cut the fused ranking at `top_k` items. A fact and the episode it
came from used two items, so at `top_k=15` the scenario served only **10 distinct memories**.

**Change (API migration of the `memory_query` response):**
- `retrieval.select_by_memory` keeps the best `top_k` hits as before. Each fact whose
  episode is also in that window shares the episode's slot. Each freed slot goes to the
  next-ranked hit from a memory not yet represented. Hits past the window from memories
  already represented are skipped. `retrieve_memory_across` uses the same rule for its
  per-session guarantee and for the final cut.
- Serving: a ranked claim whose episode is served is listed **only** under
  `episodes[i]["claims"]`, in full. Top-level `claims` holds only ranked claims whose
  evidence is not served. `linked_claim_ids` (from slice 1, never pushed) is gone.
- **Migration:** read claims with `serving.iter_served_claims(result)`. It returns the ranked
  claims best first, wherever they sit. Historical context claims are returned only with
  `include_historical=True`, so a consumer cannot mistake them for state. The prompt hook
  and 12 existing test files now read claims this way (see below).

**A bug in my first version (caught by measurement, not by tests).** I first let an episode
join a memory already represented by its fact for free. All tests passed, but the scenario's
composition went from 11 claims + 4 episodes to 0 claims + 15 episodes, each carrying its
claim, and content tokens roughly doubled (202 → 442 on the "why" question). Root cause
(Logic): slot counting ignored representation cost, so every compact claim was upgraded to
full evidence text for free. Fixed by the window-plus-backfill rule. Regression test:
`test_an_episode_behind_its_own_fact_never_rides_in_past_the_window`.

**A second defect, caught by an existing test.** The first `iter_served_claims` also returned
historical context claims. `test_integration_cycles` failed because a non-history query then
"served" the superseded claim. Historical claims are now opt-in.

**Measured** (baseline scenario, `general` pack, `top_k=15`):

| question | distinct memories before → after | content tokens before → after |
|---|---|---|
| 1 status | 10 → 14 | 199 → 266 |
| 2 blocker | 13 → 14 | 183 → 210 |
| 3 why | 10 → 15 | 202 → 287 |
| 4 quote | 10 → 15 | 236 → 309 |
| 5 change | 10 → 14 | 199 → 270 |

The token growth is the added breadth: tokens per memory stay at about 19–20. The hook's
injected context is capped separately by lines and characters, so prompt size does not grow.
A caller that wants fewer tokens should lower `top_k`. The slot count is unchanged.

**Existing tests affected:**
- 12 files moved from `result["claims"]` to `iter_served_claims(result)`. The assertions are unchanged, and the denial checks now look at every served claim, not only the top-level list: `test_connectedness`, `test_cycle_b_temporal`, `test_integration_cycles`, `test_lexical_mode_never_loads_model`, `test_mcp_tools`, `test_nl_only_serving`, `test_phase1_ranking`, `test_serve_audit_contradictions`, `test_trust_p4_antipoisoning`, `test_trust_p5_isolation`, `test_trust_p7_access_control`, `test_two_tier_wiring`.
- One assertion was restated: `test_cross_session_normal_case_stays_within_top_k…` now checks `slots_used(hits) == top_k` instead of `len(hits) == top_k`. The envelope is `top_k` evidence objects, and a fact riding in its episode's slot is not an extra item.

**Stale-exposure eval** (`evals/product/stale_exposure.py`, lexical, n = 41 changed facts):
unchanged, except that hook current-hit under `--subject-drift` goes from 63.4% to 61.0%.
Two questions flipped in opposite directions:
- Python runtime: stale 3.12 dropped out of the injection.
- Monorepo tool: current Turborepo was replaced by stale Nx.

Both are boundary effects of the hook's ranked set in lexical mode (GAP-9). Both are drift
cases, which step 2 targets.

**Other consumers migrated:** `evals/product/stale_exposure.py` (with a fallback for
checkouts that predate the change), `scripts/demo/core_memory.py` and `scripts/demo/pyright_observe.py`.
