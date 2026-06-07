# Changelog

All notable changes to MemContext are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semantic
versioning.

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
- 21 MCP tools (1:1 with handlers) across stdio + HTTP.

### Security
- Closes the GOVERNANCE_AUDIT gaps: all seven dimensions (A–G) PRESENT. See
  `SECURITY.md` for the threat model and deployment guidance.

## [0.1.0]

Initial substrate: two-tier memory (episodes + NL-first facts), deterministic +
semantic supersession, multi-signal retrieval (RRF), active-claim projections,
provenance, predicate packs, and the MCP server (stdio + HTTP).
