# MemContext

**Domain-agnostic memory and context substrate for AI agents.**

MemContext gives AI agents persistent, auditable memory. It observes information from conversations, browser pages, tools, and documents; converts it into provenance-backed structured claims; tracks changes and supersession over time; and serves clean, queryable context through the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) and a REST API.

> Extracted from [RobbyMD](https://github.com/harneet2512/RobbyMD), a clinical diagnostic agent where the memory layer was first built to give physician-steering agents persistent, auditable recall across sessions.

---

## Demo

<div align="center">

https://github.com/user-attachments/assets/cbd05bcb-d272-4051-b2a7-210f4bcc5777

</div>

---

## The Agent

MemContext ships with an autonomous browser agent that watches what you do, remembers what it sees, and serves that context back to any AI that asks.

### How it works

```
                  Chrome Extension
                  (export sessions)
                        |
                        v
 +--------------------------------------------------+
 |           Agent Browser (Patchright)              |
 |  - Autonomous Chromium with dedicated profile     |
 |  - Injected overlay UI (mode indicator + toggle)  |
 |  - Session propagation from user's Chrome         |
 +--------------------------------------------------+
        |                              |
        v                              v
 +------------------+     +-----------------------+
 | Observe Pages    |     | Overlay UI            |
 | (a11y tree →     |     | - Agent/Human toggle  |
 |  structured      |     | - Orange/green border |
 |  claims)         |     | - Click to take over  |
 +------------------+     +-----------------------+
        |
        v
 +--------------------------------------------------+
 |              MemContext Memory                     |
 |  SQLite + claims + supersession + provenance      |
 +--------------------------------------------------+
        |                    |                |
        v                    v                v
 +-------------+   +-----------------+   +----------+
 | MCP Server  |   | REST API (:8100)|   | Hooks    |
 | (stdio)     |   | (FastAPI)       |   | (silent  |
 |             |   |                 |   |  context |
 | Claude Code |   | ChatGPT, Gemini|   |  inject) |
 | Cursor      |   | Custom agents  |   |          |
 +-------------+   +-----------------+   +----------+
```

### Overlay UI

The agent browser injects a floating overlay on every page:

- **Orange border + pill** = agent mode (autonomous, user input blocked)
- **Green border + pill** = human mode (you're in control, agent watches)
- Click the toggle button to switch between modes
- Dark mode detection — overlay adapts to page background

### Session propagation

1. User installs the MemContext Connector extension in their Chrome
2. Clicks "Export Sessions to Agent" — cookies POST to `localhost:8100`
3. Agent browser picks up exported cookies and injects them
4. Sessions persist in a dedicated agent profile across restarts

### Silent context injection (hooks)

The HTTP server exposes hook endpoints that Claude Code calls automatically:

| Hook | What it does |
|------|-------------|
| `pre_tool_use` | Injects relevant memory claims as context before tool calls — the AI "just knows" what it saw |
| `post_tool_use` | Captures meaningful actions (edits, writes, commands) as memory claims |
| `user_prompt_submit` | Stores user decisions and intent silently |

The agent never interrupts. Context appears in the AI's prompt without the user seeing any API calls.

### Two doors into the same memory

| Interface | Transport | Clients |
|-----------|-----------|---------|
| **MCP Server** | stdio / Streamable HTTP | Claude Code, Cursor, any MCP client |
| **REST API** | HTTP `:8100` | ChatGPT (via GPT Actions), Gemini, browser extensions, custom agents |

Both hit the same SQLite database. An observation made through the REST API is queryable via MCP and vice versa.

---

## Why MemContext

- **Provenance-backed claims** — every fact traces back to exact source text, turn, and extraction confidence. Nothing is silently overwritten.
- **Two-pass supersession** — deterministic structural matching (Pass 1) + semantic embedding similarity (Pass 2). Typed edges explain *why* a fact changed: `USER_CORRECTION`, `REFINES`, `CONTRADICTS`, `SEMANTIC_REPLACE`.
- **Zero cloud API calls required** — local Ollama inference or regex extractors. Embeddings via sentence-transformers. No mandatory external dependencies.
- **Composable domain vocabularies** — predicate packs define what a domain cares about. Swap `general` for `developer` or `personal_assistant`, or compose them.
- **Four-signal hybrid retrieval** — semantic + BM25 + entity + temporal, fused via Reciprocal Rank Fusion.
- **Browser observation** — accessibility tree extraction from live pages. Diff-based revisits trigger automatic supersession.
- **Deterministic core** — projections, profiles, digests, chains, provenance, importance scoring all run without LLMs. Reduces cost and variance.
- **Audit-first design** — full claim lifecycle, supersession history, decision tracking. Built for medical audit in RobbyMD; works for any high-stakes memory.

---

## Quick Start

```bash
# Install core
pip install -e .

# With MCP server + embeddings
pip install -e ".[mcp,embeddings]"

# Initialize a database
memcontext init --db memory.db

# Store a turn
memcontext ingest "I prefer dark mode for all my editors" --db memory.db

# Query memory
memcontext query "what are the user's preferences?" --db memory.db --top-k 5

# Start MCP server (for Claude Code / Cursor)
memcontext serve --db memory.db --transport stdio

# Start REST API (for ChatGPT / browser agent / hooks)
memcontext serve-http --db memory.db --port 8100
```

---

## Architecture

```
                     +--------------------------------------+
                     |           Input Sources               |
                     |  Conversation  Browser  Documents     |
                     +-----------------+--------------------+
                                       |
                                       v
                     +--------------------------+
                     |    Admission Filter       |
                     |  reject noise, fillers,   |
                     |  sub-threshold turns      |
                     +------------+-------------+
                                  |
                                  v
                     +--------------------------+
                     |    Claim Extraction       |
                     |  LLMExtractor (Ollama)    |
                     |  PassthroughExtractor     |
                     |  SimpleExtractor (regex)  |
                     +------------+-------------+
                                  |
                       +----------+----------+
                       v                     v
             +----------------+   +--------------------+
             |   Pass 1       |   |   Pass 2            |
             |  Deterministic |   |  Semantic Identity   |
             |  Supersession  |   |  (embedding cosine)  |
             +-------+--------+   +----------+----------+
                     |                       |
                     +----------+------------+
                                v
                     +--------------------------+
                     |   Active Projection       |
                     |  current world-state from |
                     |  non-superseded claims    |
                     +------------+-------------+
                                  |
            +----------+----------+----------+
            v                     v          v
  +----------------+   +----------+   +----------+
  |  MCP Server    |   |  REST    |   |  Hybrid  |
  |  8 tools over  |   |  API     |   |  Retrieval|
  |  stdio / HTTP  |   |  :8100   |   |  RRF     |
  +----------------+   +----------+   +----------+
```

Every claim carries a **provenance chain**: source turn, character span, extraction confidence, and full supersession history.

---

## Key Concepts

### Claims

The atomic unit of memory. A claim is a `(subject, predicate, value)` triple with confidence, temporal validity window, and provenance back to the exact source text.

```
Claim: subject="user", predicate="user_preference", value="prefers dark mode"
       confidence=0.85, source_turn="tu_3a8f...", status=ACTIVE
```

### Supersession

When new information conflicts with old, MemContext doesn't delete — it supersedes with typed edges:

| Pass | Method | Edge Types |
|------|--------|------------|
| **Pass 1** — Deterministic | Same `(session, subject, predicate)` + different value | `USER_CORRECTION`, `REFINES`, `CONTRADICTS`, `ASSISTANT_CONFIRM` |
| **Pass 2** — Semantic | Embedding cosine > 0.88 on identity text (excluding value) | `SEMANTIC_REPLACE` |

### Predicate Packs

Closed, composable vocabularies that define what a domain cares about:

- **General** (10 families): `user_fact`, `user_preference`, `user_event`, `user_relationship`, `user_goal`, `user_constraint`, `context`, `action`, `observation`, `metadata`
- **Developer** (10 families): `decision_made`, `bug_fixed`, `convention_established`, `file_purpose`, `dependency_reason`, `api_contract`, `todo`, `blocker`, `user_preference`, `project_status`
- **Personal Assistant** (6 families): `user_fact`, `user_preference`, `user_event`, `user_relationship`, `user_goal`, `user_constraint`

### Projections

Current world-state: all claims with status `ACTIVE`, `CONFIRMED`, or `AUDITED`, grouped by subject and predicate. Rebuilds after every turn.

---

## Browser Observation

Playwright-based observation pipeline that lets agents watch web pages and remember what they see:

1. **Capture** — `capture_snapshot(page)` grabs URL, title, and full accessibility tree
2. **Extract** — `AccessibilityTreeExtractor` walks the a11y tree depth-first, pulling structured claims from headings, form fields, links, and text content
3. **Store** — Claims flow through the standard pipeline (admission, extraction, supersession)
4. **Revisit** — `diff_snapshots()` compares old vs. new, classifying changes as added/removed/changed. `apply_changes()` writes the delta, triggering supersession automatically

Each observation gets a deterministic `snapshot_id` (SHA-256 of URL + timestamp). Every extracted claim carries its accessibility role and a stable `obs_key` for cross-visit matching.

---

## MCP Tools

8 tools over the Model Context Protocol:

| Tool | Purpose |
|------|---------|
| `memory_store` | Ingest a turn + optional pre-structured claims |
| `memory_query` | Retrieve ranked claims by relevance |
| `memory_trace` | Walk the full provenance and supersession chain for a claim |
| `memory_correct` | Dismiss a claim or replace it with a corrected value |
| `memory_observe` | Ingest a browser page snapshot as structured claims |
| `memory_observe_url` | Observe a URL directly (launches Playwright) |
| `memory_profile` | Build a deterministic smart profile of a subject |
| `memory_stats` | Summary stats: sessions, turns, claims, active/superseded counts |

Tool logic lives in `mcp_tools.py` — pure functions, no protocol dependency. Testable without MCP installed.

---

## REST API

The same tools are available over HTTP for non-MCP clients:

```
POST /api/memory/store       — ingest a turn
POST /api/memory/query       — retrieve claims
POST /api/memory/trace       — walk provenance chain
POST /api/memory/observe     — observe a URL
GET  /api/memory/status      — database stats
POST /api/sessions/export    — receive cookies from Chrome extension
POST /api/hooks/pre_tool_use — silent context injection
POST /api/hooks/post_tool_use — capture tool actions
POST /api/hooks/user_prompt_submit — capture user intent
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

Falls back gracefully: without embeddings installed, retrieval uses token-overlap scoring.

---

## Data Model

```
turns ────┐
          ├──> claims ──> claim_metadata
          |       |
          |       ├──> supersession_edges
          |       ├──> claim_embeddings
          |       ├──> claim_entities
          |       └──> output_sentences (provenance)
          |
          ├──> event_frames ──> event_frame_claims
          |                  └──> event_frame_embeddings
          |
          └──> decisions (audit trail)
```

11 SQLite tables. WAL mode, foreign keys enforced, nanosecond timestamps.

---

## Benchmark: LongMemEval-S

**442/500 (88.4%)** on the predecessor system (RobbyMD).

Reader: GPT-5-mini | Judge: GPT-4o | Scoring: official [LongMemEval](https://github.com/xiaowu0162/LongMemEval) protocol

| Category | Score | Accuracy |
|----------|-------|----------|
| single-session-user | 69/70 | 98.6% |
| single-session-assistant | 55/56 | 98.2% |
| knowledge-update | 73/78 | 93.6% |
| abstention | 27/30 | 90.0% |
| temporal-reasoning | 117/133 | 88.0% |
| multi-session | 106/133 | 79.7% |
| single-session-preference | 22/30 | 73.3% |
| **Overall** | **442/500** | **88.4%** |

---

## Project Structure

```
memcontext/
  schema.py                 # SQLite schema, data model, enums
  claims.py                 # Claim CRUD, validation, active-state queries
  admission.py              # Noise filtering
  extractors.py             # LLMExtractor + PassthroughExtractor + SimpleExtractor
  on_new_turn.py            # Pipeline orchestrator
  supersession.py           # Pass 1: deterministic structural supersession
  supersession_semantic.py  # Pass 2: semantic identity via embeddings
  retrieval.py              # Multi-signal retrieval (semantic, hybrid RRF, BM25)
  projections.py            # Active-claims projections
  provenance.py             # Forward/back-link provenance utilities
  profiles.py               # Deterministic smart profiles (zero LLM)
  digests.py                # Per-session summaries with importance scoring
  chains.py                 # Full supersession chain traversal
  life_events.py            # Temporal event tuples and point-in-time queries
  importance.py             # Multi-signal importance scoring
  volatility.py             # Change-frequency tracking
  entities.py               # Entity extraction and linking
  entity_graph.py           # Entity relationship graph
  event_bus.py              # Internal event system
  predicate_packs.py        # Domain vocabulary management
  mcp_tools.py              # MCP tool handlers (no protocol dependency)
  mcp_server.py             # MCP server (stdio + Streamable HTTP)
  http_server.py            # REST API (FastAPI) + hook endpoints
  cli.py                    # CLI: init, status, ingest, query, serve
  observe/
    browser.py              # PageSnapshot, capture_snapshot, observe_page
    extractors.py           # AccessibilityTreeExtractor, DOMExtractor
    revisit.py              # diff_snapshots, apply_changes
evals/
  longmemeval.py            # LongMemEval benchmark integration
  longmemeval_prompts.py    # Category-specific answer prompts
  runner.py                 # Suite runner
  metrics.py                # Scoring functions
  ceiling.py                # Failure classification
predicate_packs/
  general/                  # General-purpose vocabulary (10 families)
  developer/                # Developer-context vocabulary (10 families)
  personal_assistant/       # Conversational memory (6 families)
scripts/
  smoke/                    # CLI, MCP, browser, memory loop smoke tests
```

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

All tests use `:memory:` SQLite and `NullEmbedder`. Zero model downloads in CI.

---

## License

MIT
