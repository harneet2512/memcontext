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

## Interlude: tenant isolation (pre-existing, found by step 2's tests)

`memory_query(namespace=A)` served tenant B's claims and episodes whenever both used the same
session id (hooks default to `"hooks"`). Reproduced at `c79d380`. Fixed in `fef050a`:
retrieval now takes the caller's namespace, and the session-keyed resolved view is withheld
(`resolved_view_withheld`) when the session id is shared across namespaces.

A read-only security audit of the same class found two HIGH issues, fixed in the next
commit, plus LOW count leaks (reported, not fixed):
- consolidation demoting other tenants' facts
- the LLM extractor's context reading other tenants' turns

## Step 2: the prompt hook flags two current values for the same kind of decision

`memcontext/conflicts.py` defines "same kind" deterministically:
- the same **single-valued** predicate, where the pack declares one current value per slot
  (`decision_made`, `project_status`, `convention_established`, `file_purpose`); and
- either the same subject, or the same project prefix (text before the last `/`) with one
  topic's words contained in the other's.

Multi-valued predicates (`blocker`, `todo`) never conflict. Two values can be current in two
ways: in one slot, when the trust guard refuses a low-trust override, or across slots, when a
decision is re-recorded under a new subject. In both cases the hook replaces the separate lines with
one `CONFLICT` entry, newest first, tagged `newest` and `untrusted source`. It advises:
prefer the newest trusted value and store the resolution. Partners are looked up in the
store, within the namespace, not only among the ranked results, because the newer value is often
the one the prompt did not match. The PreToolUse hook (the context right before an Edit/Write)
uses the same entry. That goes beyond the request, and I flagged it.

**What it does not catch:** paraphrased subjects with no shared words
(`orders_svc_db` vs `orders-service_database`). In the stale-exposure `--subject-drift` set,
**0 of 19** injections that showed both values got a CONFLICT entry. Linking paraphrases needs
semantic subject identity (embeddings, as Pass-2 supersession uses in semantic mode). That is
follow-up work. Note: while locating the CI-provider failure I saw one held-out recall-eval
line. No rule here was derived from it, and a "newer turn mentions the old value" rule was
deliberately not added for that reason.

**Measurement caveat:** the MemContext conditions of the stale-exposure eval are not
deterministic between identical runs. Two runs of the same code differ in 26/51 hook
injections, with 4 score flips (GAP-9). Hook movements of a few questions (step 1: net −1;
step 2: drift stale 70.7% → 65.9%) are within that noise.

Tests: `tests/test_hook_conflicts.py` covers:
- the new-subject drift case and the same-slot trust-conflict case
- newest first, including when retrieval surfaces only the stale value
- superseded values are not a conflict
- multi-valued predicates, unrelated decisions and namespace scope

## Step 3: GAP-5, trace picks the most-trusted current value and shows conflicts

`memory_trace(subject, predicate)` used to return the newest active claim. When the trust guard
refuses a low-trust override, both values stay active, so a newer web page became the
"current" value, with a lineage of one. Now (`conflicts.trusted_slot_head`):

- **Head:** the live value with the highest source trust, the newest on a tie. The caller's
  session selects the namespace; the slot is then read namespace-wide, which is supersession's scope.
- **Lineage:** traced from that value's **earliest** live copy. That is the original assertion,
  which carries the supersession edges; later copies are counted in `restatements`. (Tracing the
  newest copy would show a restatement with no history.)
- **`conflicts`:** every other current value competing with the head, newest first, one per
  value, each with `trust`, `quarantined` and `newer_than_head`. "Competing" means the same
  single-valued slot, a `contradicts` edge, or the same decision under another subject (step 2's
  rule). Conflicts are confined to the head's namespace and are also listed when tracing by `claim_id`.
- The CLI trace table prints a `CONFLICT` section.
- `_newest_active_for_slot` (the old fallback) was removed. It is replaced by the
  namespace-aware slot read.

Tests: `tests/test_trace_trust_conflicts.py`. GAP-5 has been promoted in `tests/test_har95_baseline.py`.

## Step 4: GAP-6, "how did X change" switches on history mode

`detect_history_intent` knew only past-state cues (before, previously, used to…).
`_EVOLUTION_INTENT` adds questions about change:
- "how did/has/have/had/was/were … change/evolve/shift/develop"
- "what (has) changed"
- "over time"
- "timeline of"

The bare verb is not matched. In a coding assistant, "change the CI provider" or "how do I
change the log level?" are requests, and history mode would let superseded claims take
ranked slots. Both kinds of phrasing are tested (`tests/test_history_intent.py`).
GAP-6 has been promoted in the baseline file.
