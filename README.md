# MemContext

**Domain-agnostic memory and context substrate for AI agents.**

MemContext observes information from conversations, browser pages, tools, and documents; converts it into provenance-backed structured claims; tracks changes and supersession over time; and serves clean, queryable context to agents through the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP).

> Extracted from [RobbyMD](https://github.com/harneet2512/RobbyMD), a clinical diagnostic agent where the memory layer was first built to give physician-steering agents persistent, auditable recall across sessions.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │           Input Sources              │
                        │  Conversation  Browser  Documents    │
                        └──────────────┬──────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────┐
                        │    Admission Filter       │
                        │  reject noise, fillers,   │
                        │  sub-threshold turns      │
                        └──────────┬───────────────┘
                                   │
                                   ▼
                        ┌──────────────────────────┐
                        │    Claim Extraction       │
                        │  LLMExtractor (Ollama/OR) │
                        │  PassthroughExtractor     │
                        │  SimpleExtractor (regex)  │
                        └──────────┬───────────────┘
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
                ┌────────────────┐ ┌────────────────────┐
                │   Pass 1       │ │   Pass 2            │
                │  Deterministic │ │  Semantic Identity   │
                │  Supersession  │ │  (embedding cosine)  │
                └───────┬────────┘ └──────────┬──────────┘
                        │                     │
                        └────────┬────────────┘
                                 ▼
                        ┌──────────────────────────┐
                        │   Active Projection       │
                        │  current world-state from │
                        │  non-superseded claims    │
                        └──────────┬───────────────┘
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
                ┌────────────────┐ ┌────────────────────┐
                │  MCP Server    │ │  Hybrid Retrieval   │
                │  5 tools over  │ │  semantic + BM25 +  │
                │  stdio         │ │  entity + temporal   │
                └────────────────┘ └────────────────────┘
```

Every claim carries a **provenance chain**: the source turn, character span, extraction confidence, and full supersession history. Nothing is silently overwritten — old facts are marked `SUPERSEDED` with a typed edge explaining *why* (user correction, refinement, contradiction, semantic replacement).

---

## Key Concepts

### Claims

The atomic unit of memory. A claim is a `(subject, predicate, value)` triple extracted from a turn, with confidence, temporal validity window, and a pointer back to the exact source text.

```
Claim: subject="user", predicate="user_preference", value="prefers dark mode"
       confidence=0.85, source_turn="tu_3a8f...", status=ACTIVE
```

### Supersession

When new information conflicts with old, MemContext doesn't delete — it supersedes. Two passes:

| Pass | Method | Edge Types |
|------|--------|------------|
| **Pass 1** — Deterministic | Same `(session, subject, predicate)` + different value | `USER_CORRECTION`, `REFINES`, `CONTRADICTS`, `ASSISTANT_CONFIRM` |
| **Pass 2** — Semantic | Embedding cosine similarity > 0.88 on identity text (excluding value) | `SEMANTIC_REPLACE` |

### Predicate Packs

Closed vocabularies that define what a domain cares about. Packs compose — `general,developer` merges both.

- **General** (10 families): `user_fact`, `user_preference`, `user_event`, `user_relationship`, `user_goal`, `user_constraint`, `context`, `action`, `observation`, `metadata`
- **Developer** (10 families): `decision_made`, `bug_fixed`, `convention_established`, `file_purpose`, `dependency_reason`, `api_contract`, `todo`, `blocker`, `user_preference`, `project_status`
- **Personal Assistant** (6 families): `user_fact`, `user_preference`, `user_event`, `user_relationship`, `user_goal`, `user_constraint`

### Projections

A **projection** is the current world-state: all claims with status `ACTIVE`, `CONFIRMED`, or `AUDITED`, grouped by subject and predicate. Projections rebuild after every turn, giving agents a clean snapshot without stale facts.

---

## Browser Observation

MemContext includes a Playwright-based browser observation system that lets agents watch web pages and remember what they see.

**Pipeline:**
1. **Capture** — `capture_snapshot(page)` grabs the URL, title, and full accessibility tree from a live Playwright page
2. **Extract** — `AccessibilityTreeExtractor` walks the a11y tree depth-first, pulling structured claims from headings, form fields, links, and text content
3. **Store** — Claims flow through the standard pipeline (admission, extraction, supersession)
4. **Revisit** — `diff_snapshots()` compares old vs. new observations, classifying changes as added/removed/changed. `apply_changes()` writes the delta back, triggering supersession automatically

Each observation gets a deterministic `snapshot_id` (SHA-256 of URL + timestamp), and every extracted claim carries its accessibility role and a stable `obs_key` for cross-visit matching.

<!-- Demo walkthrough coming soon -->

---

## MCP Integration

MemContext exposes its memory tools over the Model Context Protocol (stdio transport):

| Tool | Purpose |
|------|---------|
| `memory_store` | Ingest a turn + optional pre-structured claims |
| `memory_query` | Retrieve ranked claims by relevance |
| `brain` | Deterministic world-state by subject — value, status, confidence, provenance span, and per-subject gaps (no LLM) |
| `memory_payload` | The same question as a `summary` / `vector` / `memcontext` payload (see the demo below) |
| `memory_trace` | Walk the typed supersession lineage for a `claim_id` or a `(subject, predicate)` slot |
| `memory_correct` | Dismiss a claim or replace it with a corrected value |
| `memory_observe` | Ingest a browser page snapshot as structured claims |

The MCP tools are pure functions in `mcp_tools.py` — no protocol dependency. The thin `mcp_server.py` wrapper handles stdio transport. You can import and test the tools without the MCP package installed.

```bash
# Start the MCP server (PATH-independent launch)
python -m memcontext.mcp_server --db memory.db
```

---

## Install & attach (Claude Code + Codex)

One command installs an isolated environment, verifies the stdio server launches, and
prints ready-to-paste MCP config for both clients:

```bash
./install.sh [DB_PATH]          # default DB: ~/.memcontext/memcontext.db
```

Or emit the config yourself (PATH-independent launch — no console script required):

```bash
memcontext mcp-config --client both --db memcontext.db
```

- **Claude Code** — paste the JSON into `.mcp.json` (project) or `~/.claude.json` (user).
- **Codex** — paste the TOML block into `~/.codex/config.toml`.

Both launch the server as `<python> -m memcontext.mcp_server --db <path>`. Restart the
client and the `memory_*` tools appear.

> The HTTP transport (`memcontext serve-http`) is loopback-only and bearer-authenticated by
> default: every `/api/*` route requires `Authorization: Bearer $MEMCONTEXT_HTTP_TOKEN`
> (auto-generated and printed if unset), CORS is locked to `MEMCONTEXT_HTTP_ORIGINS`, and
> `--share` (public tunnel) refuses to start unless a token is set.

## Validation experiment: does memory change the answer?

A two-block experiment measures whether the **host model** answers with the **current**
value of a fact (vs a stale/superseded one) across a multi-session task with planted
correction points. The reader is the host model in both blocks; the only variable is
MemContext's presence.

- **Block A** — host tool with native memory only, MemContext not attached.
- **Block B** — same task, MemContext attached over MCP.

```bash
# Block B: seed the task into MemContext, capture the projection at each probe
python -m validation.run --block B --db validation.db
# Block A: same probes, native memory only (no seeding)
python -m validation.run --block A --db validation.db
```

Each command writes `validation_answers_<block>.jsonl`. The host model answers each probe
(for Block B it may call `memory_query`/`brain`), filling the `answer` field. Then score:

```bash
python -m validation.score validation_answers_B.jsonl
```

The forgetting-aware scorer rewards using the current value and penalizes reliance on
invalidated facts, reporting `accuracy`, `stale_rate`, and `traceable`. To verify the
harness end-to-end without the host model, run Block B with `--self-check` (fills answers
from the deterministic projection and scores — a pipeline smoke, not the experiment).

> Multi-tenancy (a tenant boundary above `session_id`) is a deferred platform-tier item,
> out of scope for this build.

---

## Demo: one corrected fact, three memories

A self-contained, model-free showcase of MemContext's three differentiators —
**deterministic projection**, **span-level provenance**, and **typed
supersession** — in a single scenario. The reader is held constant (the host
model attached to the MCP server) and only the *memory payload* varies. The demo
makes no external API call, uses no API key, and introduces no second model. (The
vector payload uses the repo's existing *local* embedder; everything else runs
fully offline.)

**Scenario** — a team states their database, then corrects it four turns later:

```
Turn 1: We use Postgres for the main database.
Turn 5: Actually we migrated off Postgres, we're on DynamoDB now.
Question: What database do they use?
```

**Run it:**

```bash
memcontext demo                 # developer pack (default)
memcontext demo --pack general
```

It seeds the transcript, then prints the `brain()` world-state, the supersession
trace, and the same question answered from three payloads:

```
[ vector payload ]  top-k raw statements by cosine similarity
  sim 0.647  "We use Postgres for the main database."            <- stale, ranked first
  sim 0.446  "Actually we migrated off Postgres, ... DynamoDB now."

[ memcontext payload ]  structured projection
  current_value: DynamoDB  (ACTIVE, conf 0.95)
  source: Turn 5 span [44:52] "DynamoDB"
  superseded: Postgres  <-- user_correction  (Turn 1)
```

**Why the same reader can only answer from the projection:**

- **Summary payload** is the raw transcript — both values appear as prose with no current-value field and no source, so the reader can paraphrase but cannot say which is current or cite where it came from.
- **Vector payload** is top-k statements by similarity — both values surface and similarity is not recency or truth (here the *stale* Postgres line even outranks the migration), so there is no signal for which one holds.
- **memcontext payload** is the deterministic projection — one current value (DynamoDB, ACTIVE) with an exact source span, the prior value retained beneath it via a typed `user_correction` edge, so the reader states the answer and cites Turn 5.

**Run it live inside an MCP client** (same reader, three payloads): after
`memcontext demo`, serve the seeded database and ask the host model to call the
`memory_payload` tool with `mode=summary`, `mode=vector`, and `mode=memcontext`
(session `demo`), then compare the three answers side by side.

```bash
memcontext serve --db memcontext_demo.db --transport stdio
```

> Individual views are also available as CLI commands: `memcontext brain --db
> memcontext_demo.db --session demo --pack developer` and `memcontext trace --db
> memcontext_demo.db --session demo --subject main_database --predicate
> decision_made`. (Pass the same `--pack` the db was seeded with — the gaps
> report is computed against that vocabulary.)

See [`demo/`](demo/) for the hardcoded transcript and the one-command runner.

---

## Quick Start

```bash
# Install core
pip install -e .

# With MCP server support
pip install -e ".[mcp]"

# With embedding models (for semantic supersession + retrieval)
pip install -e ".[embeddings]"

# Initialize a database
memcontext init --db memory.db

# Ingest a turn
memcontext ingest "I prefer dark mode for all my editors" --db memory.db

# Query memory
memcontext query "what are the user's preferences?" --db memory.db --top-k 5

# Check status
memcontext status --db memory.db
```

---

## Retrieval

Four-signal hybrid retrieval fused via Reciprocal Rank Fusion (k=60):

| Signal | Method |
|--------|--------|
| **Semantic** | Cosine similarity on all-MiniLM-L6-v2 embeddings (384-dim, local) |
| **BM25** | Token-level scoring for exact matches |
| **Entity** | Binary match on normalized subject keys |
| **Temporal** | Recency ranking via `valid_from_ts` |

Falls back gracefully: without embeddings installed, retrieval uses token-overlap scoring. No external API calls required for the default configuration.

---

## Data Model

```
turns ──┐
        ├──→ claims ──→ claim_metadata
        │       │
        │       ├──→ supersession_edges
        │       ├──→ claim_embeddings
        │       └──→ output_sentences (provenance)
        │
        ├──→ event_frames ──→ event_frame_claims
        │                  └──→ event_frame_embeddings
        │
        └──→ decisions (audit trail)
```

11 SQLite tables. WAL mode, foreign keys enforced, nanosecond timestamps for strict ordering.

---

## Benchmark: LongMemEval-S

**Predecessor system (RobbyMD):** 442/500 (88.4%)
Reader: GPT-5-mini | Judge: GPT-4o | Scoring: official LongMemEval protocol

| Category | Score | Accuracy | Status |
|----------|-------|----------|--------|
| single-session-user | 69/70 | 98.6% | Solved |
| single-session-assistant | 55/56 | 98.2% | Solved |
| knowledge-update | 73/78 | 93.6% | Strong |
| abstention | 27/30 | 90.0% | Strong |
| temporal-reasoning | 117/133 | 88.0% | Good |
| multi-session | 106/133 | 79.7% | Active work |
| single-session-preference | 22/30 | 73.3% | Active work |
| **Overall** | **442/500** | **88.4%** | |

**MemContext (generalized substrate):** Diagnostic only — full 500 not yet run.

| Category | Diagnostic (30q) | Direction |
|----------|-------------------|-----------|
| preference | 5/10 (50%) | Improving — source excerpts approach working |
| multi-session | 5/10 (50%) | Improving — retrieval recall with top-50 |
| temporal | 8/10 (80%) | Strong — per-excerpt temporal offsets working |

Methodology: same reader (GPT-5-mini), same official judge protocol, same dataset. MemContext adds generalized LLM extraction, per-excerpt temporal offsets with gap markers, and source-turn context to reader. Full 500-question run pending.

**Scoring protocol:** Two-tier system matching official LongMemEval evaluation — strict normalized boundary match for short answers (≤3 tokens), LLM-as-judge (GPT-4o) with task-specific rubrics for everything else.

---

## Project Structure

```
memcontext/
  schema.py              # SQLite schema, data model, enums
  claims.py              # Claim CRUD, validation, active-state queries
  supersession.py        # Pass 1: deterministic structural supersession
  supersession_semantic.py  # Pass 2: semantic identity via embeddings
  retrieval.py           # Multi-signal retrieval (semantic, hybrid RRF, BM25)
  on_new_turn.py         # Pipeline orchestrator
  projections.py         # Active-claims projections
  provenance.py          # Forward/back-link provenance utilities
  extractors.py          # LLMExtractor + PassthroughExtractor + SimpleExtractor
  predicate_packs.py     # Domain vocabulary management
  mcp_tools.py           # MCP tool handlers (no protocol dependency)
  mcp_server.py          # MCP server over stdio transport
  cli.py                 # CLI: init, status, ingest, query, serve
  observe/
    browser.py           # PageSnapshot, capture_snapshot, observe_page
    extractors.py        # AccessibilityTreeExtractor, DOMExtractor
    revisit.py           # diff_snapshots, apply_changes
evals/
  metrics.py             # Scoring functions
  runner.py              # Suite runner
  longmemeval.py         # LongMemEval benchmark integration
  longmemeval_prompts.py # Category-specific answer prompts
  ceiling.py             # Failure classification
predicate_packs/
  general/               # General-purpose vocabulary (10 families)
  developer/             # Developer-context vocabulary (10 families)
  personal_assistant/    # Conversational memory (6 families)
scripts/
  demo/                  # Pyright observation demo
  smoke/                 # CLI, MCP, browser, memory loop smoke tests
```

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

All tests use `:memory:` SQLite and `NullEmbedder`. Zero model downloads in CI. 189 tests, strict pyright, ruff linting.

```bash
# External smoke tests (run from outside the repo)
python scripts/smoke/cli_smoke.py            # 10 checks
python scripts/smoke/mcp_smoke.py            # 22 checks (handlers + stdio protocol)
python scripts/smoke/observe_smoke.py        # 15 checks
python scripts/smoke/memory_loop_smoke.py    # 20 checks (5 core behaviors)
python scripts/smoke/browser_agent_smoke.py  # 17 checks (real Playwright)
```

---

## License

MIT
