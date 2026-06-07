# MemContext — Trust & Governance Layer: Build Plan

Grounded in `GOVERNANCE_AUDIT.md`. Ordered by **dependency + leverage**, not by the A–G letters. Each phase extends a **structurally-positioned primitive** where one exists (cheap) or is flagged **net-new**. Memory/substrate only. Every phase ships with a migration (if schema), a real production caller, and a test (red-before/green-after).

**Sequencing logic:** complete provenance is the foundation for deletion + audit; source-trust is the foundation for poisoning defense; observability measures everything else. So: **B → C → F → A → D/E → G.**

---

## Phase 1 — Complete provenance (close the one blind spot) · extends a primitive
**Audit basis:** B = PARTIAL. Claims→turns is immutable+enforced (`schema.py:231`), but `session_digests` (`schema.py:370`) has **no claim link** — a served summary with no traceback, and the thing that breaks cascade-deletion.
**Build:**
- Add `source_claim_ids` (JSON) to `session_digests`; have the digest builder record the claims it summarized. Same shape `life_events.claim_ids` already uses.
- Add a write-path invariant test: **assert no served artifact lacks a provenance link** (claims, episodes, digests, life_events, event_frames, output_sentences).
**Test:** a digest built from N claims carries all N claim_ids; the invariant test fails if any new write path skips provenance.
**Unlocks:** makes the "every served *memory* is traceable" claim true (today it's only "every *claim*"); prerequisite for Phase 2.

## Phase 2 — Real, cascade-consistent deletion / forgetting · NET-NEW (highest leverage)
**Audit basis:** C = ABSENT. Only `DELETE` is embedding-cleanup on supersede (`claims.py:480`); supersession retains everything. No GDPR/HIPAA erase.
**Build:**
- `forget(conn, *, claim_id | subject | session_id | predicate, reason)` — a real delete that **cascades along the provenance + supersession graph**: target claim(s) → `claim_embeddings`, `claim_metadata`, `claim_entities`, `supersession_edges`, `output_sentences` citing it, `life_events`/`event_frames`/`session_digests` that referenced it (rebuild or strip via the Phase-1 links).
- **Tombstone + deletion log**: an append-only `deletions(deletion_id, target, scope, reason, ts, claim_ids_removed)` so erasure is **verifiable/auditable** ("prove it's gone").
- BM25 needs nothing (computed live); embeddings already cascade via FK.
- CLI `forget` + MCP `memory_forget` door.
**Test:** after `forget(subject='user X')`, the content appears in **zero** of: claims, metadata, entities, embeddings, digests, life_events, output_sentences, and a re-query returns nothing — and the deletion log proves it.
**Unlocks:** GDPR/HIPAA "right to be forgotten"; residual-artifact safety; gates the high-stakes vertical (#2).

## Phase 3 — Source + trust tiering at write · extends a primitive
**Audit basis:** F = PARTIAL. `source_type` exists (`schema.py:215`) but is **never weighted**; only extraction `confidence` is (`retrieval.py:1075`). No hard-fact / inferred / external tier.
**Build:**
- Add a `source_trust` tier to the write path (e.g. `trusted_user` > `tool_output` > `external_doc` > `web` > `agent_inferred`), recorded per claim (on `claim_metadata`).
- Add a **`source_trust` RRF channel** to `retrieve_hybrid` (same mechanism as the `confidence` channel) so low-trust memory ranks below corroborated/trusted memory.
- Supersession respects it: low-trust content cannot silently supersede high-trust content.
**Test:** a `web`-sourced claim and a `trusted_user` claim, equal on all else → trusted ranks first; a `web` claim cannot supersede a `trusted_user` claim of the same slot.
**Unlocks:** the foundation for Phase 4; trustworthy retrieval for every use case.

## Phase 4 — Write-integrity / anti-poisoning + drift detection · NET-NEW
**Audit basis:** A = ABSENT. `admission.py` is a **noise filter only**; query-submitted content is stored + actable (MINJA-style undefended); no drift detection.
**Build (on Phase 3's trust tier):**
- **Quarantine gate:** content arriving from `external`/`tool_output`/retrieval-context is admitted at **low trust**, not as first-class memory — it can be cited but not silently acted on as fact.
- **No-auto-write-from-query:** the retrieval/serving path never writes back what it just served (closes the query-only-injection loop).
- **Drift/anomaly flag:** a write that contradicts a high-trust active claim (a `contradicts` candidate) or spikes against history is flagged for review, not silently merged.
**Test:** a poisoned "fact" injected via a query-only/external path is stored at low trust, never outranks a trusted fact, and never auto-supersedes one; a contradicting high-trust write raises a drift flag.
**Unlocks:** safety for any agent ingesting tool/web output; closes A.

## Phase 5 — Access control + cross-context isolation · NET-NEW (scope primitive)
**Audit basis:** D = PARTIAL (one shared HTTP token, `http_server.py:74`); E = leaky (only `session_id`; the cross-session door reads **all** sessions, `mcp_tools.py:125`).
**Build:**
- A **scope above session**: `namespace` / `tenant` / `principal` column on `turns`+`claims` (additive to `session_id`).
- **Enforce in retrieval:** `retrieve_memory_across` and the no-session door are bounded to the caller's permitted scopes — never "all sessions in the DB."
- **Per-principal authz:** who-may-write / who-may-read; scoped tokens on the HTTP transport (replace the single key); the stdio/CLI paths bind to a principal.
**Test:** a retrieval from namespace A cannot return a claim written in namespace B; an unauthorized principal cannot write/read across its scope.
**Unlocks:** multi-tenant / work-vs-personal / regulated deployments.

## Phase 6 — Trust observability · NET-NEW (measures Phases 1–5)
**Audit basis:** G = ABSENT. Only recall accuracy is measured.
**Build:** a trust report (CLI `trust-status` + a door) exposing: **staleness** (% served facts past their volatility window), **contradiction surfacing rate** (`contradicts`/`dismissed_by_user` edges), **forgetting verification** (deletions confirmed gone), **governance** (cross-scope access attempts blocked), **source-trust distribution** of served memory.
**Test:** the report reflects injected staleness/contradictions/deletions/scope-violations; numbers move when the underlying state changes.
**Unlocks:** the ability to *claim* trust posture with evidence, not vibes.

---

## What each use case needs (from the 5)
- **#1 coding-agent, #3 CRM, #5 decision-audit** — shippable on today's substrate; benefit from Phases 1, 3, 6 but not blocked.
- **#2 high-stakes / regulated** — **gated on Phases 2 (deletion), 4 (anti-poisoning), 5 (access control).** Do not ship this vertical until those land.
- **#4 contradiction-surfacing** — Phase 6 (surface the `contradicts` edge that already exists) + Phase 3.

## Branch / discipline
Trust-layer code is product → `master`. Each phase: migration + caller + test; no dormant surface; provenance preserved; write path stays deterministic where the design requires. Phases land one at a time, eval/test-gated, reported between.

## Smallest claim after each phase (what we can honestly say)
- After P1: "every served memory is traceable to a source utterance."
- After P2: "+ a user's memory can be provably and completely erased."
- After P3: "+ memory is ranked by source trust, not just recency/keywords."
- After P4: "+ injected/low-trust content cannot silently become acted-on fact."
- After P5: "+ memory is isolated per tenant; no cross-context leakage."
- After P6: "+ all of the above is measured, not asserted."
