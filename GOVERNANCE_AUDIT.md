# MemContext — Trust / Governance / Security Posture Audit

Scope: **memory/substrate only.** Ground-truth from **live code + SQLite schema** (worktree @ `b575125`), not the README. **`AUDIT.md` is absent** in the repo, so posture is traced directly from write/retrieve/delete paths and reachability. Read-only; no fixes.

> One-line verdict: MemContext is a **recall substrate with strong, immutable fact-level provenance and typed supersession — and essentially no trust, deletion, or access-control layer.** Most governance defenses are *structurally enable-able* (the primitives exist) but *not built*.

## Posture matrix (A–G)

| # | Dimension | Posture | Evidence (file:line) | Gap |
|---|---|---|---|---|
| **A** | Write integrity / poisoning | **ABSENT** | `admission.py:36–49` `admit()` gates only empty / silence / filler / `<MIN_CONTENT_WORDS`. No trust/source/identity check. No drift/anomaly path (grep empty). | Any admitted content — incl. via query-only interaction — is stored, retrieved, and actable. MINJA-style injection undefended. |
| **B** | Provenance completeness | **PARTIAL** | `claims.source_turn_id` **NOT NULL + FK → turns** (`schema.py:231`); never mutated (no `UPDATE … source_turn_id`, grep empty) → immutable. Derived links: `life_events.claim_ids` (`schema.py:383`), `event_frame_claims` (join), `output_sentences.source_claim_ids`, `claim_metadata.consolidated_sources`. | **Blind spot:** `session_digests` (`schema.py:370–376`) has `digest_text`/`digest_data` but **no claim link** → lossy summary, no provenance, residual-content risk. |
| **C** | Forgetting / deletion / unlearning | **ABSENT** | No `delete`/`forget`/`unlearn` op. The only `DELETE` (`claims.py:480`, inside `set_claim_status`) drops `claim_embeddings` when a claim is set `SUPERSEDED`; the claim + everything else is **retained**. | No GDPR/HIPAA erase; no dependency-consistent cascade-delete; residual content lingers in `claims`, `claim_metadata`, `claim_entities`, `turns`, `session_digests`, `life_events`, `output_sentences` (only the embedding is dropped). Not verifiable. |
| **D** | Governance / access control | **PARTIAL (transport-only)** | `http_server.py:74–81` `_require_bearer` — single shared bearer token on `/api/` (`secrets.compare_digest`). stdio MCP + CLI + direct SQLite: **no gate**. | One key, all-or-nothing. No identity, no per-user/tenant/record authz, no read-vs-write distinction. |
| **E** | Confidentiality / cross-context leakage | **PARTIAL → leaky** | Only partition is `session_id` (no `user_id`/`tenant`/`project`/`owner` column, grep empty). Single-session retrieval is scoped, **but** the cross-session door (`mcp_tools.py:125–129`: no `session_id` → `SELECT DISTINCT session_id FROM turns` → `retrieve_memory_across`) reads **all sessions** in the DB. | No work/personal/project boundary. One DB = one trust domain; the across-door crosses every session with no restriction. |
| **F** | Confidence / source-trust tiering | **PARTIAL** | `confidence` (extraction certainty) **is** a retrieval signal (`retrieval.py:1045,1066,1075`, `w_conf=0.1`). `source_type` (`SourceType` conversation/tool_call/browser, `schema.py:215`) exists but is **never weighted** (grep empty in retrieval/supersession). | hard-fact vs inferred vs external not distinguished as trust tiers; `confidence ≠ source trust`. |
| **G** | Trust observability | **ABSENT** | No staleness/drift/contradiction-rate/forgetting-quality/governance metric anywhere (grep empty). Only recall accuracy is measured (`evals/longmemeval`). | No way to measure whether trust/governance is working — only whether recall is. |

## Structurally positioned (enable-able *because* a primitive exists)
These are not "done" — they are **cheap to build because the substrate already carries the needed primitive**:
1. **Poisoning attribution + dependency-consistent deletion** ← provenance is immutable and enforced (`source_turn_id` NOT NULL/FK, write-once) and changes are a typed graph (`supersession_edges`). You can already *trace* any served claim to its origin and *walk* the change graph; a real forget could cascade along it.
2. **Source-trust tiering** ← `source_type` already records where an episode came from (conversation/tool_call/browser). The where-from primitive exists; nothing weights it yet.
3. **Contradiction surfacing** ← `EdgeType.CONTRADICTS` / `DISMISSED_BY_USER` already exist in the supersession vocabulary; the signal is recorded, just not surfaced as a trust metric.
4. **Multi-tenant isolation** ← `session_id` is a real partition; a scope *above* session (user/tenant) + enforcement in the across-door is an additive layer, not a rewrite.
5. **Trust-weighted retrieval** ← the RRF fusion already has a `confidence` channel; adding a `source_trust` channel is the same mechanism.

## Genuine gaps (simply absent — no primitive to lean on)
1. **Write-trust gate / poisoning defense** — admission is a noise filter, full stop.
2. **Belief-drift / anomaly detection** on writes — none.
3. **Real deletion / forgetting + residual-artifact cleanup** — supersession retains; digests/summaries keep deleted content.
4. **Per-user / tenant / record access control** — one HTTP key; stdio + CLI ungated.
5. **Cross-context boundary above session** — the across-door reads all sessions.
6. **Source-trust tiering** — `source_type` is recorded but inert.
7. **Trust observability** — no governance/staleness/contradiction metrics.
8. **`session_digests` provenance** — the one served artifact with no traceback.

## Smallest credible trust claim we could make today
Verified-true from code, and **nothing beyond it**:

> **"Every served *claim* and *episode* is traceable to an immutable source utterance** (`claims.source_turn_id`, NOT NULL + FK to `turns`, never mutated)**, and every fact change is a typed, provenance-linked supersession edge."**

Explicitly **NOT** claimable today:
- *"every served **memory** is traceable"* — `session_digests` (a served summary) has no claim link.
- anything about **deletion/forgetting**, **access control**, **source-trust weighting**, **tenant isolation**, or **poisoning resistance** — none exist.

## Couldn't verify
- Whether `digest_data` (opaque JSON) internally carries `claim_ids` (no schema column; the digest builder body was not fully read).
- Whether **every** sidecar/cache (`event_frame_embeddings`, `profiles`, the `brain` projection) is cleaned on supersession — only `claim_embeddings` cleanup was traced (`claims.py:480`).
- **stdio** MCP transport auth — the HTTP bearer gate is confirmed; stdio is assumed local-unauthed but the transport was not traced.
- Whether the **browser/observe** write path passes through `admission` — the HTTP ingest does (`http_server.py:282`); the observe path was not fully traced.

---

## Post-implementation status (Trust & Governance layer, `68b84cb..cf3ccdd`)
The 9-phase build in `TRUST_GOVERNANCE_PLAN.md` (6 phases + 3 remainder-closers) was implemented; this baseline (graded @ `b575125`) has since changed. Re-graded against live code — **all seven dimensions PRESENT:**

| # | Dimension | Was | Now | What landed |
|---|---|---|---|---|
| A | Write integrity / poisoning | ABSENT | **PRESENT** | P4 — served low-trust memory is quarantine-flagged; serving writes no content (MINJA loop closed); blocked overrides recorded as drift |
| B | Provenance completeness | PARTIAL | **PRESENT** | P1 — `session_digests.source_claim_ids`; provenance invariant across served-summary tables |
| C | Forgetting / deletion | ABSENT | **PRESENT** | P2 — `forget()` cascade-consistent hard delete + verifiable `decisions` audit; **P8** — turn-text redaction on shared surviving turns |
| D | Governance / access control | PARTIAL | **PRESENT** | **P7** — per-principal scoped tokens (sha256-hashed) → namespace + read/write binding, enforced in the HTTP transport; `cli grant` |
| E | Confidentiality / isolation | PARTIAL (leaky) | **PRESENT** | P5 — `namespace` tenant scope; cross-session sweep bounded; foreign-session reads denied |
| F | Source-trust tiering | PARTIAL | **PRESENT** | P3 — `claim_metadata.source_trust`, a source-trust RRF channel, and a supersession guard |
| G | Trust observability | ABSENT | **PRESENT** | P6 — `trust_status` / `memory_trust_status` / `cli trust-status`; **P9** — real per-slot volatility-window staleness |

**Smallest credible trust claim now:** every served claim/episode is traceable to an immutable source; memory can be provably, completely, auditably erased (with raw-text redaction on shared turns); memory is ranked and superseded by source trust; low-trust/injected content cannot silently become acted-on fact; memory is tenant-isolated and access-controlled per principal; and the whole posture (incl. staleness) is measured.

**Genuinely still future (honest, by-design or minor):** embedding-based anomaly detection on inbound writes (P4 uses contradiction-based drift, not embedding-anomaly); learned/configurable staleness windows (currently heuristic 365/90/14-day defaults); per-principal binding on the local stdio/CLI paths (local-operator surfaces, trusted by design — authz is enforced on the remote HTTP surface).
