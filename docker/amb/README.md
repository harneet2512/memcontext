# Reproducible AMB trial — master's real product under the frozen AMB harness

This image lets anyone re-run the Agent Memory Benchmark (AMB) against the
**actual MemContext product that ships on `master`** and get a citable score.
It exists so the benchmark measures the *real* product, not the Product-B
working tree of this trial branch.

## What it pins (and why that matters)

| Layer | Source | Pinned ref |
|-------|--------|------------|
| **Product under test** | `master` | `f66b25cce11df068217f0c828222b87765013765` (arctic-embed-s embedder) |
| **Provider bridge + installer** | `benchmark/amb` | `578ce2fca8bed1d5e32ab66bde835810e1f3a09a` |
| **AMB harness** (`omb`) | `vectorize-io/agent-memory-benchmark` | `45fa380523afab9b1acd667a03de51c5ea63f4d2` (2026-05-08) |

The product is `git archive`-d **straight from master inside the build** — the
codex / Product-B working tree never enters the image. Both pinned SHAs are
re-emitted into `outputs/run_manifest.json` on every run, so a score always
carries proof of *which product* and *which instrument* produced it.

The provider bridge (`evals/amb/provider.py` on `benchmark/amb`) has been
verified API-compatible with master's product surface: every symbol it imports
(`on_new_turn`, `retrieve_hybrid`, `backfill_embeddings`, `Turn`, `Claim`,
`auto_extractor`, `PassthroughExtractor`, …) resolves on `master @ c1efbec`
with matching signatures.

## Build

Context **must** be the repo root (the build reads `.git`):

```bash
docker build -f docker/amb/Dockerfile -t memcontext-amb .
```

All three refs are pinned by default (product `master@c1efbec`, instrument
`benchmark/amb@578ce2f`, AMB `45fa380`). To track a different AMB rev or product
commit: `--build-arg AMB_REF=<sha>` / `--build-arg PRODUCT_REF=<sha>`.

## Run

Keys are passed at run time and never baked in. Costs are billed to your keys.
Each of the three roles uses its OWN key on its OWN gateway (answer = free, judge =
paid, extractor = its own). Fill in `ANSWER_KEY`, `JUDGE_KEY`, and
`EXTRACTOR_KEY` in your `.env` first (see the role table below).

### Roles, models, and keys

| Role | Whose | Backend | Model (default) | Key env | Base URL env (default) |
|------|-------|---------|-----------------|---------|------------------------|
| Answer | AMB | `openai` | `openai/gpt-oss-120b` | `ANSWER_KEY` (→ `OMB_ANSWER_OPENAI_API_KEY`) | `OMB_ANSWER_OPENAI_BASE_URL` (**OpenRouter** `…openrouter.ai/api/v1`) |
| Judge | AMB | `openai` | `google/gemini-3-flash-preview` | `JUDGE_KEY` (→ `OMB_JUDGE_OPENAI_API_KEY`) | `OMB_JUDGE_OPENAI_BASE_URL` (**TokenRouter** `…tokenrouter.com/v1`) |
| Extractor | MemContext | `openrouter` | `minimax/minimax-m3` (no-think) | `EXTRACTOR_KEY` (→ `MEMCONTEXT_EXTRACTOR_API_KEY`) | `MEMCONTEXT_EXTRACTOR_ENDPOINT` (**TokenRouter** `…tokenrouter.com/v1/chat/completions`) |

Answer and judge use **distinct keys**; the run manifest records
`routing.answer_judge_distinct_keys` and `modifications.amb_per_role_key_patch`.
Embeddings are local MiniLM (unchanged); extraction parallelism stays at 64.

```bash
# Smoke — 5 questions per category of LongMemEval-S (cheap sanity check)
docker run --rm --env-file docker/amb/.env.example \
  -v "$PWD/results/amb:/opt/amb/outputs" memcontext-amb

# Full LongMemEval-S
docker run --rm --env-file docker/amb/.env.example \
  -v "$PWD/results/amb:/opt/amb/outputs" memcontext-amb longmemeval s

# Whole AMB suite (LongMemEval-S, LoCoMo, PersonaMem, BEAM, LifeBench)
docker run --rm --env-file docker/amb/.env.example \
  -v "$PWD/results/amb:/opt/amb/outputs" memcontext-amb ""
```

Results and `run_manifest.json` land in the mounted `outputs/` directory. The
manifest records the product/instrument SHAs, the routing, both modifications
below, and `semantic_memory_enabled` (a guard against a silent degraded run).

## Cost & time (from the frozen harness README)

| Run | Extraction (DeepSeek) | AMB answer+judge | Total | Wall (32 workers) |
|-----|----------------------|------------------|-------|-------------------|
| LongMemEval-S (500q) | ~$8 | ~$3 | **~$11** | ~3–4 h |
| Whole AMB (~315M tok) | ~$22 | ~$4 | **~$26** | ~1 day |

## Models — official source + the 2.5 retirement

The Agent Memory Benchmark is **Gemini-only**: a Gemini model produces the answer
and a second Gemini call judges it. Upstream `src/memory_bench/llm/gemini.py`
**still pins `gemini-2.5-flash-lite`** for both roles
(`def __init__(self, model: str = "gemini-2.5-flash-lite")`) — so the published
leaderboard is on 2.5-flash-lite.

**But `gemini-2.5-flash-lite` is being retired** (stable build EOL **2026-07-22**;
preview already shut down). This image therefore routes all three roles through
**TokenRouter** (an OpenAI-compatible gateway) instead of a native
`GEMINI_API_KEY`, with a **distinct per-role model and per-role key**: answer =
`openai/gpt-oss-120b`, judge = `google/gemini-3-flash-preview`, extractor =
`minimax/minimax-m3` (no-think). See the role table under **Run** above.

> ⚠️ **Comparability:** because answer and judge are NOT
> `gemini-2.5-flash-lite`, this run is **not** directly comparable to the AMB
> leaderboard's 2.5-flash-lite numbers. The manifest's `leaderboard_comparable`
> flag (true only if BOTH `OMB_ANSWER_MODEL` and `OMB_JUDGE_MODEL` are
> `google/gemini-2.5-flash-lite`) records which you ran.

The answer and judge roles go through AMB's `openai` backend but on **separate
keys** — the answer key is free, the judge key is paid — so they must not share
one `OPENAI_API_KEY`. The `patch_amb_llm.py` transport shim (disclosed in the
table below) lets each role carry its own key + base URL. Extraction (which AMB
does not specify) is MemContext's own LLM extractor on its own key + endpoint.

## Legitimacy — exactly what is and isn't modified

| Component | Whose code | Modified? | What / why |
|-----------|-----------|-----------|------------|
| `provider.py` — wiring | **ours** (the adapter) | yes — `patch_provider.py` | Pass `embedder=` + `semantic=` to `on_new_turn`, matching the product's real paths (`cli.py:89`, `mcp_tools.py:61`). Without it the run silently disables Pass-2 semantic supersession + the episode-embedding floor. |
| `provider.py` — fallback | **ours** (the adapter) | yes — `patch_provider.py` | **Removes** the raw-text fallback (`{user_fact: text[:500]}` on extraction miss) present in `benchmark/amb@578ce2f`. That fallback turns MemContext into a raw-text RAG baseline — a prohibited benchmark hack. Misses now contribute no claim, as in the real pipeline. |
| `llm/openai.py` — JSON mode | **AMB's** | yes — `install.py` | `response_format` `json_schema`→`json_object`. Transport shim #1; required *because* answer/judge route through a gateway. Native `GEMINI_API_KEY` avoids it entirely. |
| `llm/openai.py` + `llm/__init__.py` — per-role key | **AMB's** | yes — `patch_amb_llm.py` | Transport shim #2 (**disclosed**, transport-only). `OpenAILLM.__init__` accepts an optional `api_key`/`base_url`; `get_answer_llm()`/`get_judge_llm()` pass each role's OWN key + base URL (`OMB_ANSWER_OPENAI_*` / `OMB_JUDGE_OPENAI_*`). Needed because the answer role (free key) and judge role (paid key) route through the same OpenAI-compatible gateway and would otherwise collide on one `OPENAI_API_KEY`. **Does NOT touch** model selection, prompts, the judge rubric, scoring, or gold answers. Anchors are disjoint from install.py's `response_format` edit, so build order is safe. |
| scoring / prompts / judge rubric / gold answers | AMB's | **no** | untouched |

For a zero-benchmark-edit run, drop TokenRouter and set a native Gemini key:
`-e GEMINI_API_KEY=<AI-Studio-key> -e OMB_ANSWER_LLM=gemini -e OMB_JUDGE_LLM=gemini`
(then `install.py`'s `openai.py` patch is moot). `GEMINI_API_KEY` from
**AI Studio** is a standalone key — not GCP/Vertex.

## Scope / branch hygiene

Everything here is **instrument**, kept on this trial branch per the repo's
branch model — none of it belongs on `master`, and it adds nothing to the
product. `benchmark/amb` is left **pristine**: the provider correction is a
transparent, asserting patch applied at build time, not an edit to the frozen
instrument. The product reaches the image only through the pinned `master`
archive.
