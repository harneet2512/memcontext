# AMB (Agent Memory Benchmark) Setup for MemContext

## Cost

Running 500 LongMemEval-S questions through the AMB harness costs **$0.57 total**.

| Role | Provider | Model | Input Rate | Output Rate | 500 Q Cost |
|------|----------|-------|-----------|-------------|------------|
| Answer generation | Groq | `openai/gpt-oss-120b` | $0.15/M tokens | $0.60/M tokens | $0.49 |
| Answer judging | Gemini | `gemini-2.5-flash-lite` | $0.10/M tokens | $0.40/M tokens | $0.08 |

No minimum spend on either. Credit card required for both.

## API Keys

You need two keys:

1. **Groq** — sign up at https://console.groq.com, go to API Keys, create one
2. **Gemini** — sign up at https://aistudio.google.com, go to API Keys, create one

Set them as environment variables before running:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY = "gsk_your_groq_key_here"
$env:GEMINI_API_KEY = "your_gemini_key_here"

# Linux/Mac
export GROQ_API_KEY="gsk_your_groq_key_here"
export GEMINI_API_KEY="your_gemini_key_here"
```

AMB defaults: `OMB_ANSWER_LLM=groq`, `OMB_JUDGE_LLM=gemini`. These match the leaderboard configuration so scores are directly comparable with Hindsight, Mem0, Mastra, etc.

## Prerequisites

- Python 3.12+
- `memcontext` installed: `pip install -e .`
- `uv` package manager: `pip install uv`
- Git

## Step 1: Clone the AMB repo

```bash
git clone https://github.com/vectorize-io/agent-memory-benchmark.git
cd agent-memory-benchmark
uv sync
```

## Step 2: Copy the MemContext provider into AMB

```bash
# Windows (from project root, assuming AMB repo is alongside memcontext)
copy evals\amb_provider.py ..\agent-memory-benchmark\src\memory_bench\memory\memcontext.py

# Linux/Mac
cp evals/amb_provider.py ../agent-memory-benchmark/src/memory_bench/memory/memcontext.py
```

## Step 3: Register the provider

Edit `agent-memory-benchmark/src/memory_bench/memory/__init__.py` and add:

```python
from .memcontext import MemContextProvider
```

## Step 4: Run LongMemEval-S (primary benchmark)

```bash
cd agent-memory-benchmark

# Datasets auto-download on first run
uv run amb run --dataset longmemeval --domain S --memory memcontext
```

Expected runtime: ~30-60 minutes (extraction + embedding is local, LLM calls are fast).
Expected cost: $0.57.

## Step 5: Run other datasets (optional)

```bash
# PersonaMem 32K — preference tracking
uv run amb run --dataset personamem --domain 32K --memory memcontext

# LoCoMo — multi-session conversation memory
uv run amb run --dataset locomo --memory memcontext

# All datasets
uv run amb run --memory memcontext
```

## Step 6: Compare with competitors

```bash
# BM25 baseline
uv run amb run --dataset longmemeval --domain S --memory bm25

# View results table
uv run amb results
```

## MemContext Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ACTIVE_PACK` | `personal_assistant` | Predicate pack for claim extraction |
| `MEMCONTEXT_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (384-dim, local) |
| `SUBSTRATE_PACKS_DIR` | (auto-detected) | Predicate pack directory |

## AMB Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | (required) | Groq API key for answer generation |
| `GEMINI_API_KEY` | (required) | Gemini API key for answer judging |
| `OMB_ANSWER_LLM` | `groq` | Answer provider: `groq`, `openai`, `gemini` |
| `OMB_ANSWER_MODEL` | `openai/gpt-oss-120b` | Override answer model |
| `OMB_JUDGE_LLM` | `gemini` | Judge provider: `gemini`, `openai`, `groq` |
| `OMB_JUDGE_MODEL` | `gemini-2.5-flash-lite` | Override judge model |

## Alternative: OpenRouter Only (one key, slightly higher cost)

If you prefer a single API key instead of two:

```bash
$env:OMB_ANSWER_LLM = "openai"
$env:OMB_ANSWER_MODEL = "google/gemini-2.5-flash-lite"
$env:OMB_JUDGE_LLM = "openai"
$env:OMB_JUDGE_MODEL = "google/gemini-2.5-flash-lite"
$env:OPENAI_API_KEY = "sk-or-v1-your-openrouter-key"
$env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
```

Cost: ~$0.42 + 5.5% credit fee = $0.44. May need a one-line patch in AMB's `src/memory_bench/llm/openai.py` to pass `base_url` through.

## Standalone Runner (no AMB repo needed)

If you don't want to clone AMB, use our built-in runner:

```bash
python evals/amb_runner.py --dataset path/to/dataset.json --reader configured --limit 50
```

This uses OpenRouter (GPT-5-mini reader + GPT-4o judge) via `MEMCONTEXT_READER_API_KEY`. Scores are NOT directly comparable to the AMB leaderboard because the reader/judge models differ.

## What MemContext Controls in AMB

Our adapter only handles `ingest()` and `retrieve()`. The AMB harness controls answer generation and judging. This means our score depends entirely on:

1. **Extraction quality** — how well we pull structured claims from turns
2. **Embedding quality** — how well claims match queries semantically
3. **Retrieval ranking** — how well multi-signal RRF surfaces the right claims
4. **Supersession** — whether we correctly track current vs outdated facts

All deterministic, all local, no LLM calls in our path.
