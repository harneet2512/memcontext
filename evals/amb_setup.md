# AMB (Agent Memory Benchmark) Setup for MemContext

## Prerequisites

- Python 3.12+
- `memcontext` installed: `pip install -e .`
- `uv` package manager (AMB uses it): `pip install uv`
- Git

## Step 1: Clone the AMB repo

```bash
git clone https://github.com/vectorize-io/agent-memory-benchmark.git
cd agent-memory-benchmark
uv sync
```

## Step 2: Symlink the MemContext provider

```bash
# From inside the AMB repo
# Windows:
mklink /H src\memory_bench\memory\memcontext.py ..\memcontext\evals\amb_provider.py

# Linux/Mac:
ln -s ../../memcontext/evals/amb_provider.py src/memory_bench/memory/memcontext.py
```

Or copy the file directly:
```bash
cp ../memcontext/evals/amb_provider.py src/memory_bench/memory/memcontext.py
```

## Step 3: Register the provider

Add to the AMB catalog or import in `src/memory_bench/memory/__init__.py`:
```python
from .memcontext import MemContextProvider
```

## Step 4: Download datasets

AMB downloads datasets automatically on first run, but you can pre-fetch:
```bash
uv run amb download --dataset longmemeval --domain S
uv run amb download --dataset personamem --domain 32K
```

## Step 5: Run benchmarks

```bash
# LongMemEval-S (our primary benchmark)
uv run amb run --dataset longmemeval --domain S --memory memcontext

# PersonaMem 32K
uv run amb run --dataset personamem --domain 32K --memory memcontext

# All datasets
uv run amb run --memory memcontext
```

## Step 6: Compare with competitors

```bash
# Run BM25 baseline for comparison
uv run amb run --dataset longmemeval --domain S --memory bm25

# View results
uv run amb results
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ACTIVE_PACK` | `personal_assistant` | Predicate pack for extraction |
| `MEMCONTEXT_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `SUBSTRATE_PACKS_DIR` | (auto-detected) | Predicate pack directory |

## Our Standalone Runner (no AMB dependency)

If you don't want to clone the AMB repo, use our built-in runner:

```bash
# Download a dataset manually first
python evals/amb_runner.py --dataset path/to/dataset.json --reader none --limit 50
```

This runs the same pipeline (extract → ingest → embed → retrieve → score) without the AMB harness.

## Notes

- AMB uses Gemini for answer generation and judging by default. Our standalone runner uses GPT-5-mini (reader) + GPT-4o (judge) via OpenRouter.
- Scores from AMB harness vs our standalone runner are NOT directly comparable due to different reader/judge models.
- The adapter is deterministic — no LLM calls in the ingest/retrieve path. The AMB harness handles answer generation and scoring externally.
