# MemContext

## Product Definition

MemContext is a domain-agnostic memory and context substrate for AI agents. It observes information from conversations, browser pages, tools, documents, apps, and user workflows; converts that information into provenance-backed structured claims; tracks changes and supersession over time; and serves clean, current, queryable context to agents through MCP.

The product is not a benchmark hack. The benchmark is only a diagnostic instrument.

## Branch Model (READ BEFORE COMMITTING)

There is exactly **one product branch: `master` (main)**. Everything that *is*
the product — the `memcontext/` package, its unit tests, `predicate_packs/`,
product docs — lives and ships on `master`.

The other branches are **not** the product:
- a **feature-trial** branch — where new feature trials are explored
- a **release-hardening** branch — pre-release stabilization
- **all other branches are trials** — and trials are **reproducible benchmark
  artifacts**: self-contained so anyone can download the branch and re-run the
  trial to verify the result. Benchmark harnesses, eval scripts, run configs,
  dataset wiring, and trial scratch live here, never on `master`.

**The rule:** product code is released to **`master`**. Do **NOT** make a product
release on a trial/benchmark branch — those exist to be reproduced and checked,
not to ship from. When product work was done on a trial branch, promote *only the
product files* (`memcontext/`, `tests/`, `predicate_packs/`, product docs) to
`master`; leave `evals/`, benchmark harnesses, and trial artifacts on the trial
branch. This is the structural side of the Anti-Overfitting / benchmark-isolation
rule: the product and the instrument that measures it never share a branch.

## Project Structure

```
memcontext/          # Core package (pip install memcontext)
  schema.py          # SQLite schema, data model (Claim, Turn, Speaker, etc.)
  claims.py          # Claim CRUD, validation, active-state queries
  supersession.py    # Pass 1: deterministic structural supersession
  supersession_semantic.py  # Pass 2: semantic identity via embeddings
  retrieval.py       # Multi-signal retrieval (semantic, hybrid RRF, BM25)
  on_new_turn.py     # Pipeline orchestrator
  projections.py     # Active-claims projections
  provenance.py      # Forward/back-link provenance utilities
  extractors.py      # PassthroughExtractor (default) + SimpleExtractor (regex fallback)
  mcp_tools.py       # MCP tool handler functions (no protocol dependency)
  mcp_server.py      # MCP server over stdio transport
  cli.py             # Click CLI: init, status, ingest, query, serve
  predicate_packs.py # Domain vocabulary management, pack composition
  observe/           # Browser observation sub-package
    browser.py       # PageSnapshot, capture_snapshot, observe_page
    extractors.py    # AccessibilityTreeExtractor, DOMExtractor
    revisit.py       # diff_snapshots, apply_changes
evals/               # Evaluation suite (not installed by default)
  metrics.py         # Scoring functions
  runner.py          # Suite runner
  longmemeval.py     # LongMemEval benchmark scaffold
  longmemeval_prompts.py  # Category-specific answer prompts
  ceiling.py         # Failure classification
predicate_packs/     # Domain predicate vocabularies
  general/           # General-purpose (10 families)
  developer/         # Developer-context (10 families)
```

## Key Commands

```bash
pip install -e .                    # Install core
pip install -e ".[mcp]"             # With MCP server
pip install -e ".[dev]"             # With test deps
memcontext init --db memcontext.db  # Create database
memcontext ingest "text" --db ...   # Ingest a turn
memcontext query "question" --db .. # Query memory
memcontext serve --transport stdio  # Start MCP server
python -m pytest tests/ -v          # Run tests
```

## Development Rules

- Use `python -m pip` (not bare `pip`) — the venv Python and system pip may differ.
- All tests use `:memory:` SQLite and NullEmbedder. Zero model downloads in CI.
- `SUBSTRATE_PACKS_DIR` env var overrides predicate pack location; conftest sets it automatically.
- `active_pack().cache_clear()` must be called after changing `ACTIVE_PACK` env var.

## Working Discipline — read this file first, then LIPI (every step)

At the **start of every step / unit of work**, re-read this `CLAUDE.md` and name the rules
that bind the step (especially the Development Rules below, the Research Rule, and the
Anti-Overfitting Rule). At the **end of every step**, diagnose with **LIPI**
(see [`LIPI.md`](LIPI.md)): walk all four avenues — **L**ogic, **I**mplementation,
**I**ntegration, **P**lumbing — and state what you checked and found in each (broken or
clean), even the clean ones. Fix the *deepest* layer that explains a failure, not the first
plausible one, then re-check the other three for regressions. A bugfix commit (or PR note)
must reference the LIPI root cause by layer + `file:line`. Do not ship a symptom patch that
leaves a deeper cause live.

## Anti-Overfitting Rule

Do not optimize MemContext specifically for LongMemEval examples, labels, wording, or answer keys.

**Allowed:**
- Use LongMemEval category failures to identify general behavioral weaknesses.
- Improve category-specific answer behavior when the fix generalizes to real memory use.
- Add prompts/routing for broad memory task types: preference, multi-session, temporal, knowledge-update, abstention.
- Run small subsets to verify behavior and catch regressions.
- Inspect wrong answers only to classify failure modes, not to memorize examples.

**Prohibited:**
- Hardcoding benchmark examples, names, answers, categories, IDs, or dataset-specific patterns.
- Tuning prompts to match answer keys in a way that would not generalize.
- Adding retrieval hacks that only work because of LongMemEval structure.
- Claiming benchmark improvement before an honest full run.
- Comparing to OMEGA, Mastra, Ensue, or others unless reader model, scoring method, prompt setup, and dataset split are aligned.
- Treating internal tests as proof of benchmark readiness.

## Research Rule

Before implementing any technique, classify it:
- **PROVEN:** Used by top systems or measured to improve the exact failure mode.
- **PLAUSIBLE:** Logically connected but not proven for this benchmark.
- **EXPERIMENTAL:** Novel idea that must be behind a flag.
- **REJECTED:** Contradicted by evidence or likely to add complexity without benefit.

Do not add architecture unless failure analysis proves architecture is the bottleneck.

## Current Performance Gaps (LongMemEval-S, 88.4% = 442/500)

Scoring: two-tier — strict exact match for short answers (<=3 tokens),
GPT-4o LLM-as-judge with task-specific rubrics for everything else.
Reader: GPT-5-mini. Judge: GPT-4o-2024-11-20. Ported from official
LongMemEval protocol (xiaowu0162/LongMemEval).

| Category | Accuracy | Wrong | Total |
|----------|----------|-------|-------|
| single-session-preference | 73.3% | 8 | 30 |
| multi-session | 79.7% | 27 | 133 |
| temporal-reasoning | 88.0% | 16 | 133 |
| knowledge-update | 93.6% | 5 | 78 |
| single-session-assistant | ~solved | — | — |
| single-session-user | ~solved | — | — |

## Evidence-Based Improvement Order

1. Category-specific answer prompts (PROVEN — OMEGA uses per-category prompts)
2. Preference prompt fix (PROVEN — current failure is prompt-level)
3. Scoring methodology: LLM-as-judge (not fuzzy F1 — fuzzy F1 fails on correct paraphrased answers)
4. Reader model test / reader-mode clarity
5. Dense observation compression from structured claims (EXPERIMENTAL)
6. Only then consider retrieval architecture changes

## Rejected/Unproven for Now

- Spreading activation
- Causal edges
- Cross-encoder reranking
- Broad narrative chunking
- Graph traversal as a new retrieval channel
