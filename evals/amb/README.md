# MemContext × Agent Memory Benchmark (AMB)

Reproducible setup for evaluating **MemContext** as a memory provider on the
[Agent Memory Benchmark](https://github.com/vectorize-io/agent-memory-benchmark)
(AMB) — the multi-dataset harness from Vectorize.io that includes LongMemEval-S,
LoCoMo, PersonaMem, BEAM, and LifeBench.

AMB controls answer generation and judging; MemContext supplies `ingest()` and
`retrieve()`. Extraction is the provider's choice (like Hindsight uses
gemini-flash-lite, Mem0 uses gemini-2.0-flash) — MemContext uses **DeepSeek V4
Flash**.

---

## Configuration

| Role | Model | Provider | Billed to |
|------|-------|----------|-----------|
| **Extractor** | `deepseek-chat` (→ deepseek-v4-flash) | DeepSeek API | DeepSeek key |
| **Answer generation** | `openai/gpt-oss-120b` | OpenRouter | OpenRouter key |
| **Judge** | `google/gemini-2.5-flash-lite` | OpenRouter | OpenRouter key |
| **Embeddings** | `all-MiniLM-L6-v2` | local | free |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | local | free |

Answer + judge models match AMB's published defaults (`gpt-oss-120b`,
`gemini-2.5-flash-lite`), routed through OpenRouter so a single OpenRouter key
covers both. This keeps scores comparable to the AMB leaderboard.

### Why DeepSeek V4 Flash for extraction

Measured on a 5,900-turn LongMemEval sample:

| Extractor | Failure rate | Notes |
|-----------|-------------|-------|
| **DeepSeek V4 Flash** | **0.12%** | cleanest; ~290 turns/min |
| qwen3:8b (local) | ~0.3% | needs GPU; slow |
| ling-2.6-flash | 59% | cheap but unusable |
| mistral-nemo | 85% | unusable |

DeepSeek auto-caches the static system prompt at `$0.0028/M` (98% off the
`$0.14/M` miss rate), making large extraction jobs cheap.

---

## Cost & time (full run)

| Dataset | Extraction (DeepSeek) | AMB answer+judge | Total |
|---------|----------------------|------------------|-------|
| LongMemEval-S (500q) | ~$8 | ~$3 | **~$11** |
| Whole AMB (~315M doc tokens) | ~$22 | ~$4 | **~$26** |

Time at 32 workers: LongMemEval-S ≈ 3–4 h; whole AMB ≈ 1 day.

---

## Setup

### Prerequisites
- Python 3.12+, `uv` (`pip install uv`)
- A DeepSeek API key ([platform.deepseek.com](https://platform.deepseek.com))
- An OpenRouter API key ([openrouter.ai](https://openrouter.ai))

### 1. Clone AMB next to this repo
```bash
git clone https://github.com/vectorize-io/agent-memory-benchmark.git
cd agent-memory-benchmark && uv sync
```

### 2. Install MemContext + the provider
```bash
# from the agent-memory-benchmark directory
uv pip install -e ../memcontext
python ../memcontext/evals/amb/install.py    # copies provider, registers it, patches openai.py
```

### 3. Set keys
```bash
cp ../memcontext/evals/amb/.env.example .env
# edit .env — fill in DEEPSEEK_API_KEY and OPENROUTER_API_KEY
```

### 4. Run
```bash
# Smoke (5 questions per category)
bash ../memcontext/evals/amb/run.sh longmemeval s --query-limit 5

# Full LongMemEval-S
bash ../memcontext/evals/amb/run.sh longmemeval s

# Whole AMB (all datasets)
bash ../memcontext/evals/amb/run.sh
```

Results land in `outputs/{dataset}/memcontext/{mode}/{domain}.json`.
Browse with `uv run omb view`.

---

## Protocol compliance

- **Answer + judge** use AMB's published default models — scores are
  leaderboard-comparable.
- **Extractor** is a provider implementation detail (AMB does not specify one);
  disclosed here as DeepSeek V4 Flash.
- For a LongMemEval score comparable to the **original paper** (gpt-4o judge,
  not AMB's gemini judge), re-score the saved answers with
  `evals/metrics.py` (gpt-4o-2024-08-06). See `evals/longmemeval.py`.

---

## Files

```
evals/amb/
  README.md        # this file
  provider.py      # MemContext MemoryProvider (ingest + retrieve)
  install.py       # copies provider into AMB, registers it, patches openai.py
  run.sh           # sets env from .env and runs the AMB harness
  .env.example     # key template
```
