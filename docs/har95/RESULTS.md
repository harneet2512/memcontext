# HAR-95 results: final `memory_query` response shape

This page describes `memory_query` as it is on `master` after HAR-95. It is the same for
MCP, HTTP `/api/memory/query` and the Python handler `mcp_tools.handle_memory_query`.
Every field from before HAR-95 keeps its meaning. Everything HAR-95 adds is a new key.

## Contract (unchanged)

- **`claims`** lists every served claim, in rank order, exactly as before HAR-95. A client
  that reads `result["claims"]` (for example `claims[:3]`) sees the same claims in the same
  order. The scripts in `scripts/smoke/` stand in for external clients and run unmodified.
- **`episodes`** lists the served raw turns, as before.
- **`token_report.served_items`** is still `len(claims) + len(episodes)`.

An earlier HAR-95 commit (`d005da8`) moved claims under `episodes[i]["claims"]`. That broke
the smokes (`memory_loop_smoke` 18/20, `mcp_smoke` 21/22) and so every external client. It
was reverted to the shape above, and the 12 test files that commit migrated are back to
their pre-HAR-95 contents, byte for byte.

## What HAR-95 adds

Every claim in `claims` gains:

| key | meaning |
|---|---|
| `source_turn_id` | the Memory (stored turn) the claim was derived from |

Every episode in `episodes` gains:

| key | meaning |
|---|---|
| `trust`, `quarantined` | source trust of the evidence (`source_trust.trust_for_source`), the same rule claims use |
| `state` | `current` / `historical` / `mixed` / `no_claims`: whether the claims derived from this evidence are still current |
| `claim_ids` | ids of this episode's claims that are served in `claims` |
| `claims` | this episode's **historical** claims (superseded / dismissed) not served elsewhere, newest first, max 8, each with `claim_id`, `fact`, `status`, `trust`, `quarantined` |
| `historical_claims_omitted` | present only if more than 8 historical claims were left out |

`token_report` gains:

| key | meaning |
|---|---|
| `served_slots` | retrieval slots used: one per episode, plus one per claim whose episode is not served |
| `served_memories` | distinct evidence objects behind the served items |
| `nested_claims` | historical claims listed inline under episodes |

A tenant-scoped query whose session id is also used in another namespace omits the
session-keyed resolved view and sets `resolved_view_withheld`.

## Slot accounting

`retrieval.select_by_memory` keeps the best `top_k` hits, the same window as before. A
claim whose episode is also in that window is the same evidence object, so the pair counts
as **one slot**. Each freed slot goes to the next-ranked hit from a memory not yet represented.
Hits past the window from memories already represented are skipped, so an episode never
rides in for free behind its own top-ranked claim. The order inside the window is unchanged,
and backfilled hits come after it.

The effect is that `top_k` now buys `top_k` distinct memories instead of `top_k` items. The
payload can therefore hold more than `top_k` items.

**Measured** (HAR-95 baseline scenario, `general` pack, lexical, `top_k` = 15; `fec53a4` → `master`):

| question | items | distinct memories | content tokens | claims+episodes JSON chars |
|---|---|---|---|---|
| status | 15 → 19 | 10 → 14 | 194 → 263 | 4,912 → 6,369 |
| blocker | 15 → 16 | 12 → 14 | 178 → 195 | 4,820 → 5,037 |
| why | 15 → 20 | 10 → 15 | 202 → 287 | 4,978 → 6,830 |
| quote | 15 → 20 | 10 → 15 | 236 → 309 | 5,323 → 7,060 |
| change | 15 → 19 | 10 → 14 | 199 → 270 | 4,946 → 6,562 |

Content per memory stays about the same; the growth is the extra breadth. The hook's
injected context is capped by lines and characters, so prompt size does not grow.

## Reading claims

`serving.iter_served_claims(result)` returns `result["claims"]`. With
`include_historical=True` it also appends the inline historical claims. Existing clients
do not need it: `result["claims"]` is the ranked state.

## Verification (local replay of `.github/workflows/ci.yml`)

| check | result |
|---|---|
| `ruff check` (tracked files) | clean |
| `python -m pytest tests -q` | 503 passed, 1 skipped, 3 xfailed (GAP-7, GAP-8, GAP-9) |
| `scripts/smoke/cli_smoke.py` | 10/10 (with `MEMCONTEXT_EMBED_EPISODES=0`, see note) |
| `scripts/smoke/memory_loop_smoke.py` | 20/20 |
| `scripts/smoke/mcp_smoke.py` | 22/22 |
| `python -m pyright` | 0 errors |

**Note on `cli_smoke`:** on this machine it times out (30 s) in `memcontext query` both on
`master` and on `fec53a4`, before HAR-95. The local install has the embedding backend, so the
CLI loads BGE-M3 on first query. CI installs only `.[dev]`, so it has no backend and loads
no model. With embeddings off it passes 10/10 on both commits.

## Capture-time subject-drift warning (`895e0b8`)

When `memory_store` writes a single-valued claim (`decision_made`, `project_status`,
`convention_established`, `file_purpose`) into a **new** slot (no supersession and no
restatement of that subject), it looks for live claims of the same predicate in the same
namespace whose topic resembles the new subject. The topic is the part after the last `/`;
the project prefix is ignored. Resembling means sharing a word that is not a container word
such as "service", "system" or "project". Matches are returned as:

```json
"similar_subjects": [{"subject": "project/authentication", "predicate": "decision_made",
                      "value": "JWT", "claim_id": "cl_..."}],
"warnings": ["decision subject 'auth-system/authentication-method' is new, but a similar live
              decision exists under 'project/authentication' (= 'JWT'). If this updates it, store
              it again with the same subject 'project/authentication' so it supersedes the old
              value; otherwise ignore."]
```

It only warns. Nothing is merged or superseded. The `memory_store` tool description tells the
agent to re-store under the suggested subject when the new claim is an update. Both keys are
new and are absent when there is no match.

**Measured:**

- **Coverage probe.** The stale-exposure drift histories (34 decisions) were replayed through
  `memory_store`, with the final change stored under the paraphrased subject.
  - **17/34** of those writes return the original subject.
  - **17/34** miss. These are paraphrases with no shared word ("CI provider" vs "continuous
    integration service") or with plural or derived forms ("payment" vs "payments",
    "cache" vs "caching"), since there is no stemming.
  - **9** suggestions point at a different decision.
  - The word rule was **not** adjusted after seeing these misses. This dataset has no
    held-out split, so any tweak would be fitted to it. Paraphrases need semantic subject
    similarity, which is follow-up work.
- **Stale-exposure `--subject-drift`.** Hook: 73.2% stale, 65.9% current. That is within the
  measured noise of the earlier runs (70.7% / 61.0–65.9%), as expected. The harness ingests
  directly and never reacts to a warning, so it cannot show this feature.
- **Recall eval, dev** (`claude_code_recall_20261001T012926Z_summary.json`):
  - current 10/11, stale answer 1/8, correct action 5/6, stale action 1/4.
  - Both failures are the MCP-connect plumbing flake: the stdio server took 7.1 s to
    connect, and Claude Code then found no memcontext tools. The same flake hit 2 of the 3
    recall runs today, at 9.3 s and 7.1 s.
  - In this run Claude reused the subject (`project/authentication`), so no drift occurred
    and the warning was not exercised.
  - This run overlapped the `total`-scoping edit (`65f9711`). That edit is a no-op for this
    eval, which uses a shared token and so has no namespace.

## Tenant-scoped `total` (`65f9711`)

For a namespaced caller, `total` counts only that namespace's active claims, on the
cross-session and the single-session paths. With `namespace=None`, the single-tenant
behaviour is unchanged.

Not done yet: building the derived caches (profiles, session digests, life events) per tenant.
That needs a namespace in their keys, which is a schema migration, so it is pending approval.
