# HAR-95 slice 1 — evidence with structured state

The slice demonstrates one thing: **MemContext keeps nuanced evidence without giving up
deterministic structured state.** The baseline and its measurements are in [BASELINE.md](BASELINE.md).

## 1. Baseline behaviour and limitation

See BASELINE.md. In short, evidence was already preserved verbatim. What was missing was
any link between served evidence and the state derived from it:
- no grouping
- no trust flag on evidence
- no source link on served claims
- superseded evidence served as though it were current

## 2. Final data model

```
Memory (memcontext/memories.py; one per turn, backed by the `turns` row)
├── memory_id        == turn_id
├── content          verbatim turn text
├── speaker, ts, session_id
├── source_type      conversation | tool_call | browser
├── source_metadata  JSON provenance (url, tool name, …)
├── trust            source_trust.trust_for_source(source_type, speaker)
├── quarantined      trust < QUARANTINE_THRESHOLD
├── claims[]         DerivedClaim(claim_id, fact, status, trust)
└── state            current | historical | mixed | no_claims
```

Claims are unchanged. They still own every state transition (Pass-1/Pass-2 supersession,
`memory_correct`, dismissal). A Memory only *reports* what became of its claims, so claim
supersession semantics are never applied to evidence (ticket risk 5).

## 3. Schema / migration changes

**None.** `SCHEMA_VERSION` stays at 13. The reasoning is in BASELINE.md ("Storage model").

## 4. Provenance chain

```
served claim ──source_turn_id──▶ Memory (turn) ──source_type / source_metadata──▶ Source
```

- `memory_query`: every top-level claim now carries `source_turn_id`, and every served episode is a Memory.
- `memory_trace(claim_id)`: `source_turn` now also returns `source_type`. `source_metadata` is deliberately **not** exposed there, because trace has no namespace gate (a pre-existing gap) and the metadata may hold URLs or tool arguments. It is available from the Python read model. `lineage` returns each superseded predecessor with its own source text.
- Python: `memory_for_claim(conn, claim_id)` → `Memory`; `get_memory(conn, turn_id)`.

## 5. Retrieval behaviour

Ranking is unchanged: the same `retrieve_memory` / `_fuse_memory` RRF with the same `top_k`.
The top-level `claims` list is byte-for-byte what it was, plus `source_turn_id`.
Each served episode gains the following fields (all additive):

| field | meaning |
|---|---|
| `trust`, `quarantined` | source trust of the evidence, the same rule claims already used |
| `state` | whether the state derived from this evidence is still current |
| `linked_claim_ids` | derived claims already served in top-level `claims` (referenced, not repeated) |
| `claims` | **historical** derived claims (superseded / dismissed) not served elsewhere, newest first, each with `status`, `trust` and `quarantined`. At most 8; `historical_claims_omitted` counts the rest |

`token_report` gains `served_memories` (distinct evidence objects behind the served items)
and `nested_claims`.

## 6. Context deduplication / grouping

Invariants, all tested:
- No claim is served twice in one response (top-level vs inline).
- Every served claim carries its own trust and quarantine flag, wherever it appears.
- Current derived claims that did not rank are **not** inlined: they would restate the episode's own text, and `state` already reports them.
- Top-level `claims` are exactly the ranker's fact hits, in order, with the pre-HAR-95 keys plus `source_turn_id`.
- A Memory only lists claims from its own session (a scope join in `get_memories`).
- An episode whose turn vanished mid-request is skipped, not served without trust annotations.

Measured on the baseline scenario (lexical mode, `top_k=15`):
- **Content tokens** are identical to the baseline for 8 of the 10 pack × question cells. The two evidence questions under the renewal pack gain +7 tokens each: the inline superseded `probable` line, which is the historical signal itself.
- **Raw JSON** of `claims` + `episodes` grows by about 820–960 characters (≈200–240 tokens, +18–21%), from the new keys and the `source_turn_id`s. That is +5–6% of the full `memory_query` response. The full response is about 16–17k characters, and most of that is the pre-existing `world_state` / briefing block, which this slice does not touch. `token_report` does not count it (see Known limitations).

**Not done: budget-level dedup (GAP-1b).** An episode and one of its ranked claims still use
two of the `top_k` items. Folding the claim into the episode would empty the top-level
`claims` list in the common case. The first implementation did exactly that; see §10.

## 7. Files changed

- `memcontext/memories.py` (new): Memory / DerivedClaim read model.
- `memcontext/serving.py`: `annotate_served_evidence`.
- `memcontext/mcp_tools.py`: claims get `source_turn_id`, episodes are annotated, the serve ledger covers inline claims (access bumps still count only ranked claims), and trace exposes `source_type`.
- `memcontext/retrieval.py`: event-frame query fix (separate commit, found by the baseline).

## 8. Tests added

- `tests/test_har95_baseline.py`: scenario, characterization tests, gap xfails.
- `tests/test_har95_memory_slice.py`: representation, provenance, serving invariants, compat, namespace, cross-session, claim-free and browser memories.
- `tests/test_event_frame_serving_degradation.py`: regression for the query-path model load.

## 9. Existing tests affected

**None were changed.** The first design (below) broke 17 of them. With the final design,
all 382 pre-existing tests pass unmodified. Full suite: 382 passed, 1 skipped before HAR-95
→ **439 passed, 1 skipped, 6 xfailed** after.

The guards were mutation-checked, and each deliberate breakage turns tests red:
- dropping the session-scope join: 1 test fails
- inlining current claims: 4 fail
- dropping per-claim quarantine: 1 fails
- an oldest-first cap: 1 fails
- a ranked-only serve ledger: 1 fails
- the rejected remove-from-top-level design: 14 fail

## 10. Meaningful bug found during implementation

**Symptom:** the first grouping design nested each ranked claim under its served episode and
removed it from the top-level `claims` list. The HAR-95 tests passed, but 17 existing tests
failed. The prompt hook then injected nothing, for example `use SQLite` was missing from the
injected context. The namespace-isolation tests failed only because they read `claims`;
nothing leaked.

**Root cause (Integration):** top-level `claims` is a consumer contract.
`http_server.py` UserPromptSubmit reads only `result["claims"]`. Whenever a turn and its
claim are both served (the normal case for a fresh fact), the claim moved out of the list
every consumer reads. The design also inverted the ticket's principle: it moved *state*
out of the state list to save slots.

**Fix:** keep `claims` complete and deduplicate on the evidence side (`linked_claim_ids`, and
inline only historical claims). **Regression protection:**
`test_top_level_claims_are_the_complete_ranked_state` compares the served list with the raw
ranker output. The unmodified existing suite also covers it.

Two smaller self-inflicted defects were caught the same way:
- **Nested claims dropped their own quarantine flag.** Only the parent episode had one, so a consumer checking each claim's flag would have seen an untrusted claim with none (ticket risk: "evidence retrieval bypassing source trust"). Regression: `test_inline_browser_claim_keeps_its_own_quarantine_flag` and `test_low_trust_browser_claim_cannot_retire_user_state`.
- **Inlining every unranked derived claim** repeated each episode's own text: +30 content tokens on the blocker question. Fixed by inlining only historical claims.

The query-path model load (401 s cold) is documented in BASELINE.md.

An independent review of the final diff found no invariant violations. It did find tests
that would pass without the feature: a quarantine test that looped over zero inline claims,
and a tautological compat check. It also flagged a missing session scope on `get_memories`,
an oldest-first silent cap, and `source_metadata` exposure on trace. All were fixed before
commit; the mutation checks above cover them.

## 11. Known limitations

> **Status update:** GAP-1b, GAP-5 and GAP-6 were closed after slice 1. Three pre-existing
> tenant-isolation bugs were also found and fixed. See [FOLLOWUPS.md](FOLLOWUPS.md). Still open:
> GAP-7 (retraction), GAP-8 (duplicate serving), GAP-9 (lexical tie bias).

- **GAP-9 (found after slice 1 was committed):** in lexical mode, five or six hybrid
  channels are nearly always tied (confidence, usage, frequency, trust, importance), and
  `_rrf_ranks` breaks ties by insertion order. Older claims win those channels, which can
  outweigh BM25. Claims created in the same millisecond also come back in random-id order.
  As a result, `test_state_question_still_serves_current_state`, reported as passing in
  slice 1, was **flaky (3 of 12 runs failed)**: Acme's current status sits at the `top_k`
  cut. The test now checks state with a non-binding `top_k`. The bias is pinned as a
  strict xfail in `tests/test_lexical_tie_bias.py`. Sharing ranks between ties fixes it
  but re-ranks every query (12 tests move), so it needs a ranking decision and a
  benchmark re-run.

- **GAP-1b:** budget-level slot dedup needs a documented migration of `claims` consumers (the hooks).
- **GAP-5:** `memory_trace(subject, predicate)` picks the newest active claim, even a quarantined one.
- **GAP-6:** "how did X change" does not switch on history mode, so evolution questions need `memory_trace`.
- **GAP-7:** there is no retraction. A resolved blocker stays active; the demo shows this.
- **GAP-8:** restated identical values are served once per copy.
- The CLI `memcontext query` prints raw `retrieve_memory` hits without evidence annotations.
- `memory_store` without claims falls back to the regex `SimpleExtractor`, which derives low-quality claims from any text (e.g. `we observation We talked through…`). This is the admission-vs-extraction risk; it is not addressed here.
- `token_report` counts content text only. It excludes JSON keys and the `world_state` / briefing block, which is the larger share of the response.
- All measurements are lexical-only (no embedder). A real-embedder run is still to do.

## 12–15. Follow-up work

- **Full two-path retrieval:** give the evidence path its own relevance-driven share instead of `_fuse_memory`'s fixed 1.0 / 0.9 rank weighting (measured 11 claims : 4 episodes regardless of question), and route by question type: state / evidence / evolution (GAP-6).
- **Heterogeneous graph:** memory → memory `updates` / `extends` edges, distinct from claim supersession. Entity → memory `mentioned_in` (today entities attach only to claims). A dedicated `memories` table becomes justified once spans or multi-source memories exist.
- **Task-aware context compiler:** consume `state`, `linked_claim_ids` and `served_memories` to select "N current claims + M supporting memories + 1 historical change". Includes GAP-1b once `claims` consumers can read grouped output, and budgeting the `world_state` block.
- **Lifecycle:** a retraction / resolution transition (GAP-7), and a trust-aware slot head (GAP-5).

## 16. Reproducing the Claude Code demo

The live server uses the `general,developer` pack. `renewal_status` is not in it, so the demo
uses `project_status` (single-valued) and `blocker`.

**Reset.** Use a fresh session id; a session is never shared with earlier runs.
To reset the database too, point `.mcp.json`'s memcontext server at a new `--db` path.

**Prompts** (paste into Claude Code, in order):

1. `Store in memory (session_id "demo-acme"): "Sarah said Acme will probably renew, but only if security approves SOC2 before Friday." with claims {"subject": "Acme renewal", "predicate": "project_status", "value": "probable, conditional on security approving SOC2 before Friday"} and {"subject": "Acme renewal", "predicate": "blocker", "value": "security approval of SOC2 before Friday"}.`
2. `Store in memory (session_id "demo-acme"): "Correction: security approved SOC2 on Wednesday; Acme confirmed the renewal." with claim {"subject": "Acme renewal", "predicate": "project_status", "value": "confirmed"}.` The result shows `supersessions: 1`.
3. `memory_query "What is the current status of the Acme renewal?" (session_id "demo-acme")`. The top-level claims show `project_status confirmed`, with `source_turn_id` pointing at the correction.
4. `memory_query "What exactly did Sarah say about Acme?" (session_id "demo-acme")`. The episode holds Sarah's verbatim sentence with `state: "mixed"` and the superseded `probable…` claim inline.
5. `memory_trace subject "Acme renewal" predicate "project_status" (session_id "demo-acme")`. Lineage is `confirmed ← probable`, each step with its source text and `source_type`.

What the demo also shows, honestly: `blocker` stays active after the correction (GAP-7).

Scripted equivalent (no Claude Code): run `python -m pytest tests/test_har95_memory_slice.py -q`.
