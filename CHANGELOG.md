# Changelog

All notable changes to MemContext are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic
versioning.

## [Unreleased]

### Added
- Install extras for every interface: `embeddings`, `mcp`, `http`, `share`,
  `dev`, `all`. `pip install -e .` now gives a working lexical core with no
  optional dependencies.
- `EmbeddingClient.backend_available()` capability check — `episode_embedder()`
  returns `None` when no embedding backend is importable/reachable, so semantic
  features degrade to lexical mode instead of crashing.
- `tests/test_no_embeddings_degradation.py` — regression coverage for the
  embedding-free path (repeated writes, honest status, broken-client survival).

### Fixed
- Second write on a fresh install no longer crashes with
  `RuntimeError: FlagEmbedding is not installed` — Pass-2 semantic supersession
  is now best-effort and can never break ingestion mid-write.
- `memcontext serve --transport stdio` no longer corrupts the MCP JSON-RPC
  stream: all status lines and structlog output go to stderr on stdio.
- MCP dependency pinned to `mcp>=1.0,<2` — the server uses the 1.x decorator
  API, which `mcp>=1.0` had been resolving to the incompatible 2.x line.
- Missing `fastapi`/`pydantic`/`uvicorn`/`mcp`/`httpx` declarations that broke
  `serve-http`, test collection, and MCP extras on a clean install.
- `tool_discover` now appears in the HTTP tool list (stdio/HTTP parity: 18
  tools each).
- Undefined `log` in `mcp_tools` (2 sites), unresolved `EventFrame` annotation
  and optional-iterable in `retrieval.py`, `uninstall` redeclaration in
  `cli.py`, nullable client-id paths in `mcp_oauth.py`, `relay.py` key typing.
- `memcontext demo` (imported an unshipped package) removed; verified demo path
  is `scripts/smoke/memory_loop_smoke.py`.

### Removed
- `memcontext_extension/` and `assets/demo.mp4` — the browser-observation
  feature was deleted earlier; the extension posted browser cookies to a dead
  endpoint and had no working consumer.
- Dead `/api/sessions/*` cookie-export endpoints from `http_server.py`.
- Obsolete `correct`/`observe` methods in the TypeScript SDK; the SDK now sends
  `Authorization: Bearer` and matches the real `/api/memory/*` surface.

### Changed
- `pyright` runs in basic mode (strict had ~900 findings; basic is the honest
  gate CI enforces). Ruff excludes `E501` and `E402`-in-scripts deliberately.
- Documentation reconciled with the code: README rewritten, ARCHITECTURE module
  map and counts corrected, ChatGPT custom-GPT doc updated for bearer auth.

## [0.2.0] — 2026-06-07

The **Trust & Governance layer.** MemContext goes from a recall substrate to one
with first-class trust, deletion, isolation, and access control. Grounded in a
code-level posture audit (`GOVERNANCE_AUDIT.md`) and built per
`TRUST_GOVERNANCE_PLAN.md`. Schema migrates v7 → v11 (additive; fresh and legacy
databases upgrade automatically).

### Added
- **Provenance completeness** — `session_digests` carry queryable `source_claim_ids`;
  every served summary is traceable to its source claims.
- **Cascade-consistent deletion** (`forget()`, `memory_forget`, `cli forget`) — hard
  deletes a claim/subject/session/predicate and everything derived from it (embeddings,
  metadata, entities, supersession edges, summaries, output sentences, orphaned turns,
  profile cache), redacts forgotten content from shared surviving turns, and audits
  every erasure to the `decisions` log (right-to-be-forgotten).
- **Source-trust tiering** — `claim_metadata.source_trust`, a source-trust retrieval
  channel, and a supersession guard so low-trust content can't override trusted facts.
- **Anti-poisoning + drift** — served low-trust memory is quarantine-flagged; the
  serving path writes no memory (MINJA loop closed); blocked overrides recorded.
- **Namespace isolation** — a tenant scope above session, enforced in retrieval.
- **Per-principal access control** — sha256-hashed scoped bearer tokens
  (`cli grant`) bind a caller to a namespace + read/write permission on HTTP.
- **Trust observability** — `trust-status` / `memory_trust_status`: source-trust
  distribution, contradiction rate, forgetting + drift audit, tenant counts, and
  per-slot volatility-window staleness.
- **Optional hardening** — configurable staleness windows
  (`MEMCONTEXT_STALE_*_DAYS`); CLI `ingest`/`query` namespace binding; flag-gated
  embedding anomaly detection (`MEMCONTEXT_EXPERIMENTAL_ANOMALY`).

### Changed
- **License: MIT → AGPL-3.0-or-later.**
- 18 MCP tools (16 `memory_*` + `brain` + `tool_discover`) on stdio + HTTP.

### Security
- Closes the GOVERNANCE_AUDIT gaps: all seven dimensions (A–G) PRESENT. See
  `SECURITY.md` for the threat model and deployment guidance.

## [0.1.0]

Initial substrate: two-tier memory (episodes + NL-first facts), deterministic +
semantic supersession, multi-signal retrieval (RRF), active-claim projections,
provenance, predicate packs, and the MCP server (stdio + HTTP).
