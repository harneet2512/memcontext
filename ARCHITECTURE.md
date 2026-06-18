# MemContext — Architecture

**A memory & context substrate for AI agents.** MemContext observes information,
converts it into provenance-backed structured claims, tracks how truth changes over
time (supersession), and serves clean, current, queryable context to any agent over
MCP. The engine is deterministic and local-first; the calling AI does the
understanding — MemContext **stores, resolves, and serves**.

Schema version: **13** · Storage: a single **SQLite** file · LLM in the core path: **none**.

---

## 1. Design principles

1. **Truth maintenance, not append-only.** When a fact changes, the stale one is
   *retired* (status `superseded`, `valid_until_ts` stamped, typed edge written), not
   left to contradict the new one. This is the differentiator.
2. **Deterministic & zero-LLM in the core.** Admission, supersession, projection,
   enrichment, retrieval fusion, and serving are pure functions of the data — no LLM,
   reproducible, auditable, cheap. (Extraction is the one injected, host-owned step.)
3. **Provenance-first.** Every claim links to its source turn with character offsets;
   every correction is a typed edge; every served fact can answer "why is this current".
4. **Local-first & inspectable.** One SQLite file, no service dependency. The same
   engine serves a personal brain on a laptop and a multi-tenant hosted endpoint.
5. **Not an AI layer.** Extraction (turn → claims) is injected by the host. MemContext
   never calls an LLM to decide what to store or how to resolve it.
6. **No architecture without a proven bottleneck.** Graph-traversal-as-retrieval,
   spreading activation, cross-encoder reranking, etc. are deliberately *not* in the
   ranking path (see §11).

---

## 2. Data model (SQLite, ~16 tables)

```
turns ........... raw episodes: text, speaker, ts, source_type
                  (conversation|tool_call|browser), namespace (tenant), extraction_status
claims .......... the atomic unit: NL text + optional (subject, predicate, value),
                  confidence, status, created_ts, valid_from_ts, valid_until_ts,
                  event_ts, source_turn_id, char_start/char_end (span)
claim_metadata .. entity_key, predicate_family, source_trust, importance_score,
                  consolidated, consolidated_sources, demoted, temporal_bin, access_count
supersession_edges  old_claim_id -> new_claim_id, edge_type, identity_score, created_ts
claim_embeddings / turn_embeddings   vector + model_version (semantic channels)
claim_entities .. (claim_id, entity_text, entity_type)   co-occurrence graph source
decisions ....... audit log: drift_blocked / forget / ... (trust observability)
output_sentences  generated text -> source_claim_ids (output provenance)
serve_events .... answer-time ledger: request session -> served claim_ids
profiles ........ cached tiered profile per subject (derived)
session_digests . cached top-facts + updates per session (derived)
event_frames / event_frame_claims / event_frame_embeddings   episodic events (derived)
life_events ..... life-milestone bursts per subject (derived)
principals ...... namespace + read/write permissions (tenant auth)
```

**Claim status lifecycle:** `active`/`confirmed`/`audited` (served) → `superseded`
(retired by a newer value) / `dismissed` (user removed) / `demoted` (consolidated
duplicate, retained for provenance, out of active retrieval). `CONTRADICTS` is the
exception: both endpoints stay active and are surfaced as unresolved until a later
correction resolves them.

**Temporal model (bi-temporal-ish):** `created_ts` (ingestion), `valid_from_ts` /
`valid_until_ts` (the window a fact was/is true), `event_ts` (when the event happened).
Supersession sets `valid_until_ts` on the retired claim.

The DB is **relational**; it *projects* both a **temporal** model (validity windows +
typed lineage) and a **graph** model (entity adjacency + supersession edges) as views —
it is not a native graph or temporal database.

---

## 3. The memory layer — end-to-end flow

```
OBSERVE      a turn arrives:  on_new_turn(session, speaker, text, extractor, namespace)
   |
ADMIT        admission.admit  -> reject noise/fillers; tag durability
   |                              (instruction | preference | ephemeral)
EMBED        embed_and_store_episode (Tier-1 floor; local model, never an LLM)
   |
EXTRACT      run_extraction -> injected extractor -> ExtractedClaim[]  -> insert_claim
   |
RESOLVE      Pass-1 detect_pass1 (deterministic)  -> Pass-2 semantic.detect (embedder)
(truth)      -> rebuild_active_projection (current world-state)
   |
ENRICH       compute_importance (per claim) ; then on a cadence:
(cadence)      every 10 turns/session:  profile, session digest,
                                        event-frames, life-events
               every 25 turns/global:   consolidation (cross-session),
                                        importance recompute (decay over time)
   |
SERVE        retrieve_memory (two-tier RRF) ; handle_memory_query / build_context_briefing
             -> resolved world-state + briefing + digest + events + life-events
                + per-fact trust/quarantine + provenance + unresolved contradictions
                + serve_events ledger for answer-time verification
   |
DELIVER      MCP stdio (local clients) | MCP Streamable HTTP + OAuth (remote) | relay | CLI | library
```

All enrichment is wrapped so a failure can never break ingest (it logs, never silently
swallows). The cadence work is O(active claims) per tick — fine for a personal brain,
to be bounded by age for very large corpora.

---

## 4. Core subsystems

### Admission (`admission.py`)
Deterministic noise filter (min content words, fillers, silence markers). Also classifies
**durability** — `instruction` / `preference` / `ephemeral` — so the serve path can weight
standing guidance over a passing remark.

### Extraction (`extractors.py`, injected)
The host supplies the extractor (`PassthroughExtractor` for pre-structured claims,
`SimpleExtractor` regex fallback, or an `LLMExtractor`). MemContext depends on no LLM
stack; the extractor sees only the turn (never the question or gold answer).

### Supersession — truth maintenance (`supersession.py`, `supersession_semantic.py`)
Two passes, both writing typed edges and stamping validity windows:

- **Pass-1 (structural, deterministic).** Keys on `(session, subject, predicate)`:
  1. **Cardinality** — declared single-valued predicates (predicate packs) supersede
     regardless of token overlap.
  2. **Attribute-slot** — value-level `_attribute_of` maps surface phrasing to a slot
     (residence, employer) so "lives in NYC" → "moved to Boston" resolves even though
     the coarse predicate is the same. **History-guarded**: a value naming a closed time
     range ("from 2010 to 2015") is historical and neither clobbers nor is clobbered.
  3. **Quantity correction** — "two kids" → "three kids" supersedes (non-numeric content
     identical).
  4. **Jaccard** — otherwise require ≥2 shared content tokens, so additive facts
     ("likes pizza" / "likes sushi") coexist (no silent data loss).
  - **Edge typing**: REFINES / ASSISTANT_CONFIRM / USER_CORRECTION / CONTRADICTS.
  - **Trust guard**: a markedly lower-trust source cannot override a higher-trust fact;
    the blocked attempt is audited to `decisions` (drift-blocked).
- **Pass-2 (semantic, embedder-gated).** Identity = `subject + predicate + context`
  (value **excluded**, so "onset 3 days" / "onset 4 days" still match), cosine ≥ 0.88,
  edge `SEMANTIC_REPLACE`. Active wherever a real embedder is configured (CLI, MCP, queue).

### Projection / world-state (`projections.py`, `brain.py`)
`rebuild_active_projection` maintains the active-claims view. `brain()` is the resolved
world-state: one current value per `(subject, predicate)` grouped by subject, each fact
carrying its source span + quote, plus a per-subject **gaps** report (vocabulary
predicates with no active claim). Deterministic, LLM-free.

### Enrichment
- **Importance** (`importance.py`): six deterministic signals (uniqueness, supersession,
  confidence, recency, stability, cross-session). Computed at insert and **re-evaluated
  on a cadence** so the time-dependent signals (recency falls, stability rises) track age.
- **Entity graph** (`entity_graph.py`): in-memory adjacency over `claim_entities`, a
  read-only connective view (not a ranking channel).
- **Event-frames** (`event_frames.py`): groups co-referent claims into multi-slot event
  records (purchase, travel, appointment, named-artifact...) for slot-fill questions.
- **Life-events** (`life_events.py`): bursts of ≥N distinct predicate changes in a window.
- **Consolidation** (`consolidate.py`): graduates facts recurring across ≥3 sessions into a
  durable consolidated claim; demotes duplicates (provenance retained).
- **Profiles / digests** (`profiles.py`, `digests.py`): tiered profile briefing + session
  summary. All namespace-scopable (see §6).

### Retrieval (`retrieval.py`)
- **`retrieve_hybrid`** — facts ranked by Reciprocal Rank Fusion over multi-signal
  channels: semantic (cosine), BM25, entity, temporal, scope, predicate, confidence,
  frequency, importance, usage, source_trust. Query-type tuning (knowledge-update /
  temporal up-weight recency). Status filter keeps only active/confirmed/audited.
- **`retrieve_episodes`** — Tier-1 turn-level retrieval (semantic + BM25 + entity +
  recency); the floor that works even with no extracted facts.
- **`retrieve_memory`** — fuses facts + episodes with a second-level RRF (fact-biased).
- **`retrieve_memory_across`** — cross-session fusion by **rank** (raw scores aren't
  comparable across sessions).

### Serving (`mcp_tools.py::handle_memory_query`, `serving.py::build_context_briefing`)
One call returns the ranked hits **plus the resolved view**: world-state, session
briefing, session digest, entity links, event-frames, life-events, per-fact
durability, unresolved contradictions, and per-fact **trust + quarantine** flags +
provenance "why". The query door appends a `serve_events` row for every served fact,
and `memory_verify` checks cited claim IDs against that ledger. The MCP door and the
library door (`build_context_briefing`) expose the same safety surface.

---

## 5. Provenance & trust

- **Span provenance** (`provenance.py`): `explain_claim` assembles, for any claim, its
  value + source turn (speaker, text, char span) + the typed correction chain it sits on.
- **Source trust** (`source_trust.py`): every claim carries a trust weight derived from its
  origin (user > assistant > tool > browser). Served facts carry `trust` + a `quarantined`
  flag (below threshold → citable, not authoritative).
- **Drift audit**: blocked low-trust overrides are recorded in `decisions`, countable via
  `memory_trust_status`.

---

## 6. Multi-tenancy & isolation

`turns.namespace` is the tenant scope (the level above session). A namespaced caller
cannot read a session owned by another tenant (`_session_in_namespace`; a denied query
returns early, leaking nothing). The subject-keyed derived views (profile, life-events)
are **namespace-scoped at serve time** — `build_smart_profile`/`detect_life_events` filter
claims to those whose source turn is in the namespace — so one tenant's briefing never
aggregates another tenant's facts. `namespace=None` = single-tenant personal brain (no
scoping overhead).

---

## 7. Delivery surfaces

```
LOCAL (no network)            REMOTE (one stable URL)
  Claude Code / Desktop         claude.ai web / ChatGPT / any MCP client
  Cursor / VS Code                     |
        | stdio                        | HTTPS
  memcontext serve              memcontext serve --transport http [--oauth] | memcontext share
        |                              |
        +---- same engine, same 18 MCP tools, same SQLite brain ----+
```

- **stdio** (`mcp_server.py`): local clients, one config line, data never leaves the machine.
- **Streamable HTTP** (`mcp_server.py::create_http_app`): remote clients; bearer-token or
  OAuth 2.1.
- **OAuth 2.1** (`mcp_oauth.py`): metadata discovery + Dynamic Client Registration + PKCE
  + a password login gate (per-IP exponential brute-force lockout) + SQLite-persisted
  tokens that survive restart. The claude.ai "add connector → log in" flow.
- **Relay** (`relay.py`): the invisible plumbing behind `memcontext share`. The brain dials
  *outbound* to a dumb forwarder; its URL is derived from a self-generated **Ed25519** key
  (`brain_id = sha256(pubkey)`), so the link is stable across restarts and only the
  keyholder can claim it. The relay stores nothing; laptop offline → link dead.
- **CLI / library** (`cli.py`, direct imports): `ingest`, `query`, `brain`, `serve`,
  `share`, etc.

---

## 8. Extension seams

- **Predicate packs** (`predicate_packs/`, `predicate_packs.py`): domain vocabulary
  (families, sub-slots, `single_valued` cardinality), composable, env-overridable.
- **Injected extractor**: any callable `Turn -> ExtractedClaim[]`.
- **Embedders**: product default is `BAAI/bge-m3` via sentence-transformers
  (`NullEmbedder` for tests; explicit overrides are ablations);
  gated by `MEMCONTEXT_EMBED_EPISODES`.
- **Event bus** (`event_bus.py`): synchronous pub/sub; `on_new_turn` publishes lifecycle
  events; **no internal subscriber by design** — a host (UI, async worker, audit) subscribes.

---

## 9. Security model

- Transport TLS end-to-end; brain data lives only in the local SQLite file.
- OAuth 2.1 + PKCE + password gate + brute-force lockout + revocable, restart-surviving
  tokens; auth fails closed.
- Relay identity is self-certifying (signed-nonce Ed25519); a brain-id cannot be claimed
  without its private key.
- Anti-poisoning: low-trust sources cannot override higher-trust facts; quarantine flags
  on served facts.
- **Known gap**: the relay terminates TLS at the relay (can see traffic in flight, never
  the DB). The fix is TLS-passthrough (route by SNI, cert on the user's machine) so the
  pipe is blind by construction.

---

## 10. Module map

```
memcontext/
  schema.py            SQLite schema, data model, migrations, open_database (WAL)
  claims.py            Claim CRUD, active-state queries, supersession-chain walk
  admission.py         noise filter + durability classifier
  extractors.py        Passthrough / Simple / LLM extractors (injected)
  on_new_turn.py       the ingest orchestrator (admit -> extract -> resolve -> enrich)
  supersession.py      Pass-1 deterministic supersession
  supersession_semantic.py   Pass-2 semantic supersession
  projections.py       active-claims projection
  brain.py             resolved world-state + gaps
  importance.py        deterministic importance (6 signals) + decay recompute
  entity_graph.py      in-memory entity adjacency view
  event_frames.py      multi-slot episodic event assembly
  life_events.py       life-milestone burst detection
  consolidate.py       cross-session consolidation
  profiles.py          tiered profile briefing (namespace-scopable)
  digests.py           session digests
  retrieval.py         hybrid + episode + two-tier + cross-session retrieval, embeddings
  serving.py           build_context_briefing (resolved view + trust + events + provenance)
  provenance.py        span provenance, explain_claim, output-sentence provenance
  source_trust.py      trust weights + quarantine
  predicate_packs.py   domain vocabulary management
  mcp_tools.py         MCP tool handlers (no protocol dependency)
  mcp_server.py        MCP server (stdio + Streamable HTTP)
  mcp_oauth.py         OAuth 2.1 provider + login gate
  relay.py             outbound-dial relay + self-certifying brain identity
  cli.py               Click CLI (init, ingest, query, brain, serve, share, ...)
  observe/             browser observation sub-package (tool-driven)
```

---

## 11. Known limitations & non-goals

- **Retrieval is O(n) brute-force cosine** (no ANN). Fine for a personal brain; will not
  scale to very large corpora without a vector index (e.g. `sqlite-vec`).
- **Extraction quality is the host's responsibility** — the built-in regex extractor is a
  fallback, not a product.
- **No verified benchmark number yet.** The headline figure is unsubstantiated until an
  honest LongMemEval / AMB run (deliberately a separate, paid step).
- **Deliberately rejected (no proven bottleneck):** graph traversal as a retrieval channel,
  spreading activation, causal edges, cross-encoder reranking, broad narrative chunking.
- **Importance recompute** runs the full active set on a cadence — bound by age for very
  large brains.
- **Relay TLS-passthrough** (blind pipe) is the open hardening item for the remote path.

---

*This document describes the connected substrate as built on `product/connect-substrate`
(schema v13). It is the canonical architecture reference and travels with the product code.*
