# MemContext

**An auditable memory and context substrate for AI agents.**

MemContext represents remembered information as structured claims with provenance and typed update history, then serves the current relevant state through MCP, HTTP, and Python interfaces. It is not a vector store for chat logs: when a fact changes, the old claim is superseded by a typed edge — `USER_CORRECTION`, `REFINES`, `CONTRADICTS`, `SEMANTIC_REPLACE` — not silently overwritten, and every claim traces back to the exact source turn and character span it came from.

> Originated inside [RobbyMD](https://github.com/harneet2512/RobbyMD), a clinical diagnostic agent where memory had to be auditable and correct. Extracted as a standalone, domain-agnostic substrate. AGPL-3.0.

---

## The problem

Agents forget, and naive memory layers make it worse. A vector database over conversation chunks can recall *that something was said* but cannot represent *that a fact changed, why it changed, or which version is current*. It returns stale and current text side by side and lets the model sort it out.

MemContext treats memory as state, not search: claims are written once, supersession is explicit and typed, and what an agent sees is the **active projection** — the current known state — plus the evidence trail behind it.

## What it does differently

- **Structured claims** — `(subject, predicate, value)` with confidence, temporal validity, and immutable source-turn provenance.
- **Typed supersession, two passes** — Pass 1: deterministic structural matching (same identity slot, new value). Pass 2: semantic identity via embeddings (`SEMANTIC_REPLACE`). Corrections are first-class: `memory_correct` records the override as an auditable edge.
- **Active projection** — the "current world state" is rebuilt from non-superseded claims after every write; history is preserved, not lost.
- **Hybrid retrieval** — Reciprocal Rank Fusion over semantic, BM25, entity, temporal, predicate-alignment, confidence, frequency, importance, usage, and source-trust signals.
- **Governance built in** — namespace isolation, per-principal read/write tokens (HTTP), source-trust tiers that gate low-trust supersession, provenance-preserving hard delete (`forget`), contradiction surfacing, and a `trust_status` report.
- **Deterministic core** — the full lifecycle (claims, supersession, projections, profiles, digests) runs without an LLM. Extraction is injected: bring any LLM backend, pass pre-structured claims, or use the bundled regex extractor for development.
- **Honest degradation** — without embedding dependencies, MemContext runs lexical-only: `status` reports `Semantic memory: OFF`, semantic supersession is skipped, nothing crashes.

## Architecture

```
   conversation / agent activity / pre-structured claims
                          |
                          v
                +--------------------+
                |  Admission filter  |   rejects noise, sub-threshold turns
                +---------+----------+
                          v
                +--------------------+
                |   Extraction       |   injected LLMExtractor · Passthrough
                |  (subject, pred,   |   · SimpleExtractor (dev fallback)
                |   value, conf)     |
                +---------+----------+
                          v
             +---------------------------+----------------------------+
             |  Pass 1 deterministic      |  Pass 2 semantic (optional) |
             |  same slot, new value      |  embedding cosine identity  |
             +---------------------------+----------------------------+
                          v
                +--------------------+
                |  Active projection |   current state; superseded kept for audit
                +---------+----------+
                          v
          +---------------+----------------+
          |  Hybrid retrieval (RRF)        |
          |  + episode floor               |
          +---------------+----------------+
                          |
        +-----------------+------------------+
        v                 v                  v
   +---------+      +-----------+      +-----------+
   | MCP     |      | REST API  |      | Python /  |
   | 18 tools|      | /api/*    |      | CLI       |
   | stdio + |      | bearer +  |      |           |
   | HTTP/OAuth|    | principals|      |           |
   +---------+      +-----------+      +-----------+

   Storage: SQLite, WAL mode, 25 tables, nanosecond timestamps.
```

## Quick start

```bash
pip install -e .
```

```console
$ memcontext init --db memory.db
Initialized MemContext database at memory.db
Active pack: general (12 predicates)

$ memcontext ingest "I prefer dark mode for all my editors" --db memory.db
Turn ingested: tu_5136db1574db
Claims created: 1
  [user_preference] user: dark mode for all my editors (confidence=0.5)

$ memcontext ingest "I prefer light mode for all my editors" --db memory.db
Turn ingested: tu_883dcf941d30
Claims created: 1
  [user_preference] user: light mode for all my editors (confidence=0.5)
Supersessions: 1

$ memcontext query "what editor theme does the user prefer?" --db memory.db
Found 3 memory item(s):
{"kind": "fact", "id": "cl_f33def454b7a", "text": "user user_preference light mode ...", "source_turn_id": "tu_883dcf941d30", ...}
{"kind": "episode", "id": "tu_5136db1574db", "text": "I prefer dark mode for all my editors", ...}
```

The dark-mode claim is not deleted — it is `superseded`, still reachable through `memory_trace`, and excluded from the active projection. Every served item carries its `source_turn_id`.

### Installation modes

| Extra | Contents | Needed for |
|-------|----------|-----------|
| *(none)* | core: claims, supersession, lexical retrieval, CLI | local use |
| `embeddings` | FlagEmbedding / sentence-transformers / requests | semantic retrieval + Pass-2 supersession (local BGE-M3, or remote via `MODAL_BGE_M3_URL`) |
| `mcp` | `mcp`, starlette, uvicorn | `memcontext serve` (stdio + Streamable HTTP + optional OAuth) |
| `http` | fastapi, pydantic, uvicorn | `memcontext serve-http` (REST + hooks) |
| `share` | pycloudflared | `memcontext share` / `serve-http --share` outbound tunnel |
| `dev` | pytest, ruff, pyright + all mcp/http deps | development |
| `all` | everything above | batteries included |

```bash
pip install -e ".[mcp,http]"            # both server surfaces
pip install -e ".[all]"                 # + embeddings + share
```

Extraction: no LLM is bundled. Inject your own extractor (Ollama/OpenRouter/Gemini backends via `MEMCONTEXT_EXTRACTOR_BACKEND`), pass `claims=[...]` to `memory_store`, or let the regex `SimpleExtractor` handle dev traffic.

## Interfaces

### MCP — 18 tools, stdio + Streamable HTTP (+ optional OAuth)

| Group | Tools |
|-------|-------|
| Core | `memory_store`, `memory_query`, `memory_trace`, `memory_correct`, `memory_stats` |
| Serving | `memory_working_context`, `memory_digest`, `memory_profile`, `memory_life_events`, `memory_events`, `memory_output_provenance`, `memory_verify` |
| Governance | `memory_forget`, `memory_trust_status`, `memory_contradictions`, `memory_entity_graph` |
| Projection | `brain` |
| Tool activation | `tool_discover` |

```bash
memcontext serve --db memory.db --transport stdio                # local (Claude Code, Cursor)
memcontext serve --db memory.db --transport http --port 8000     # Streamable HTTP
```

Tool handlers are plain functions in `mcp_tools.py` — no protocol dependency, usable without `mcp` installed.

### HTTP API

```bash
memcontext serve-http --db memory.db --port 8100
```

- `POST /api/memory/store`, `POST /api/memory/query`, `POST /api/memory/trace`, `GET /api/memory/status`
- `POST /api/hooks/{pre_tool_use,post_tool_use,user_prompt_submit,stop}` — ambient context injection / capture for hook-capable coding agents (`memcontext hooks install` writes the wiring into `.claude/settings.json`)
- `GET /health`; the MCP Streamable HTTP app mounts at `/mcp` when the `mcp` extra is installed

All `/api/*` routes require a bearer token — generated and printed on first run, or pinned via `MEMCONTEXT_HTTP_TOKEN`. Tokens can be per-principal with namespace + read/write scoping (`memcontext grant`). CORS is default-deny unless `MEMCONTEXT_HTTP_ORIGINS` is set.

### Python

Everything above calls the same handler functions — `handle_memory_store(conn, ...)`, `handle_memory_query`, `handle_memory_trace`, etc. — usable directly from any Python process with `open_database(path)`.

## Trust and governance

- Every claim has an immutable `source_turn_id`; every fact change is a typed, provenance-linked edge.
- `memory_forget` / `memcontext forget` performs cascade-consistent hard deletion recorded in an audit log.
- Namespace isolation bounds retrieval; per-principal tokens enforce read/write scope on the HTTP transport.
- Source-trust tiers influence ranking and block low-trust content from superseding trusted claims; `memory_verify` checks cited claim IDs against a serve-events ledger.

The stdio MCP transport and CLI are local-operator surfaces. `SECURITY.md` covers the deployment posture; `GOVERNANCE_AUDIT.md` has the graded trust matrix.

## Benchmarks

Results recorded on the predecessor system, [RobbyMD](https://github.com/harneet2512/RobbyMD) — the clinical agent this substrate was extracted from:

Reader: GPT-5-mini · Judge: GPT-4o · Scoring: official [LongMemEval](https://github.com/xiaowu0162/LongMemEval) protocol

| Category | Score | Accuracy |
|----------|-------|----------|
| single-session-user | 69/70 | 98.6% |
| single-session-assistant | 55/56 | 98.2% |
| knowledge-update | 73/78 | 93.6% |
| abstention | 27/30 | 90.0% |
| temporal-reasoning | 117/133 | 88.0% |
| multi-session | 106/133 | 79.7% |
| single-session-preference | 22/30 | 73.3% |
| **Overall (RobbyMD)** | **442/500** | **88.4%** |

These numbers belong to the predecessor system, not the standalone package in this repo. The `evals/` directory contains the benchmark adapter and a LongMemEval-S smoke workflow for exercising the harness end to end.

## Predicate packs

Closed, composable vocabularies that define what a domain cares about — swap or compose via `ACTIVE_PACK`:

- `general` — 12 predicate families
- `developer` — 10 families (`decision_made`, `bug_fixed`, `convention_established`, `file_purpose`, `dependency_reason`, `api_contract`, `todo`, `blocker`, `user_preference`, `project_status`)
- `personal_assistant` — 8 families

## Project structure

```
memcontext/
  schema.py                 # SQLite schema, data model, enums
  claims.py                 # claim CRUD, validation, active-state queries
  admission.py              # noise filtering
  extractors.py             # Passthrough / Simple / LLM extractor backends
  on_new_turn.py            # pipeline orchestrator
  supersession.py           # Pass 1: deterministic structural supersession
  supersession_semantic.py  # Pass 2: semantic identity via embeddings
  retrieval.py              # multi-signal retrieval, hybrid RRF, BM25
  projections.py            # active-claims projections
  provenance.py             # forward/back-link provenance
  profiles.py               # deterministic subject profiles
  digests.py                # per-session summaries with importance scoring
  chains.py                 # supersession chain traversal
  life_events.py            # temporal event tuples, point-in-time queries
  importance.py             # multi-signal importance scoring
  volatility.py             # change-frequency tracking
  entities.py / entity_graph.py
  forgetting.py             # cascade-consistent deletion + audit log
  source_trust.py           # source-trust tiers
  authz.py                  # per-principal tokens, namespaces, read/write scope
  trust_report.py           # trust_status observability
  tool_registry.py / tool_activation.py   # MCP tool curation
  mcp_tools.py              # tool handlers (no protocol dependency)
  mcp_server.py             # MCP server (stdio + Streamable HTTP + OAuth)
  mcp_oauth.py              # password-gated OAuth 2.1 provider
  http_server.py            # REST API (FastAPI) + hook endpoints
  serving.py                # remote/local serving plumbing
  relay.py                  # self-hostable share relay (Ed25519 identity)
  cli.py                    # init, ingest, query, serve, hooks, grants, ...
predicate_packs/            # domain vocabularies (general / developer / personal_assistant)
evals/                      # benchmark adapter + LongMemEval-S smoke workflow
scripts/smoke/              # CLI, MCP, memory-loop smoke scripts
sdks/typescript/            # experimental TypeScript client
```

## License

[AGPL-3.0-or-later](LICENSE)
