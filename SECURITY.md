# Security Policy

MemContext is a memory and context substrate for AI agents. Because it *stores and
serves* an agent's long-term memory, its security surface is trust, provenance,
deletion, and isolation — not just availability. This document states what the
substrate guarantees, how it is deployed safely, and how to report an issue.

The internal, code-grounded posture audit lives in [`GOVERNANCE_AUDIT.md`](GOVERNANCE_AUDIT.md);
the build that closed it is in [`TRUST_GOVERNANCE_PLAN.md`](TRUST_GOVERNANCE_PLAN.md).

## Reporting a vulnerability

Please report security issues **privately**, not via public issues or PRs:

- GitHub → *Security* → *Report a vulnerability* (private advisory), or
- email **baliharneet7@gmail.com** with subject `MemContext security`.

Include a description, affected version/commit, and a reproduction if possible. We
aim to acknowledge within 72 hours and to coordinate a fix and disclosure timeline
with you.

## Trust & governance guarantees (what the substrate enforces)

All of the following are enforced in code and covered by tests:

| Area | Guarantee |
|------|-----------|
| **Provenance** | Every served claim/episode is traceable to an immutable source utterance (`source_turn_id`, NOT NULL + FK, write-once); every served summary links to its source claims. |
| **Deletion** | `forget()` performs cascade-consistent hard deletion — claims, embeddings, metadata, entities, supersession edges, derived summaries, citing output sentences, orphaned source turns, and the profile cache — and **redacts** forgotten content from shared surviving turns (text + stale episode embedding). Every deletion is audited to the `decisions` log and is verifiable. |
| **Source trust** | Each claim carries a `source_trust` tier (user > tool > inferred > web); retrieval ranks by it and a markedly lower-trust source cannot silently supersede a higher-trust fact. |
| **Anti-poisoning** | Low-trust/external content served to the agent is **quarantine-flagged** (citable, not authoritative); the serving path writes no memory (the MINJA query-only loop is closed); blocked overrides are recorded as drift. |
| **Isolation** | Memory is partitioned by `namespace` (tenant scope above session); cross-session retrieval is bounded to the caller's namespace and foreign-session reads are denied. |
| **Access control** | On the HTTP transport, a bearer token resolves to a principal scoped to a namespace + read/write permission; tokens are stored **sha256-hashed**, never in plaintext. |
| **Observability** | `trust-status` exposes source-trust distribution, contradiction rate, forgetting + drift audit, tenant distribution, and per-slot staleness. |

## Threat model

**Defended:** memory-poisoning / injection via query-only or external content
([MINJA](https://arxiv.org/abs/2503.03704), OWASP ASI06) — such content is admitted
at low trust, cannot outrank or silently supersede trusted memory, and is never
written back from the serving path; cross-tenant leakage; un-erasable data
(right-to-be-forgotten); stale-belief drift.

**Out of scope (by design):** the local stdio/CLI surfaces are trusted-operator
surfaces; per-principal authorization is enforced on the **remote HTTP** surface.
Transport encryption (TLS) and network exposure are the deployer's responsibility.

## Deployment guidance

- **Single-tenant / local:** the default. The shared HTTP bearer token applies until
  principals are registered; stdio/CLI are local-trust.
- **Multi-tenant / regulated:** register per-principal scoped tokens
  (`memcontext grant --principal … --namespace … [--read-only]`) so each caller is
  bound to its namespace and write permission. Always run the HTTP server behind TLS;
  set `MEMCONTEXT_HTTP_TOKEN` and `MEMCONTEXT_HTTP_ORIGINS` explicitly (CORS is
  default-deny).
- **Optional hardening:** enable embedding anomaly detection with
  `MEMCONTEXT_EXPERIMENTAL_ANOMALY=1`; tune staleness windows with
  `MEMCONTEXT_STALE_{STABLE,EVOLVING,VOLATILE}_DAYS`.

## Known limitations (honest)

- Anomaly detection is embedding-based and EXPERIMENTAL (flag-gated, records rather
  than blocks); it is a no-op without an embedder.
- Staleness windows are configurable but not learned.
- Turn-text redaction over-redacts (replaces forgotten substrings wholesale) — the
  privacy-safe direction.

## Supported versions

The latest release on `master` receives security fixes.
