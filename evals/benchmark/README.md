# MemContext LongMemEval-S Benchmark

Reproducible evaluation of MemContext on the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark (500 questions, 6 categories). This document specifies the exact configuration, commands, and protocol used to produce reported scores.

---

## Overview

LongMemEval-S tests long-term memory systems across six question categories:

| Category | Questions | Tests |
|---|---|---|
| single-session-user | 26 | Recall of user facts from a single conversation |
| single-session-assistant | 26 | Recall of assistant statements from a single conversation |
| single-session-preference | 30 | User preference inference (often implicit) |
| multi-session | 133 | Cross-session fact recall and aggregation |
| temporal-reasoning | 133 | Temporal ordering, duration, date-based questions |
| knowledge-update | 78 | Superseded facts, most-recent-value retrieval |

Abstention questions (suffix `_abs`) are scored within their parent category per the official protocol.

---

## Model Configuration

All API calls route through a single [OpenRouter](https://openrouter.ai/) API key.

| Component | Model ID | Location | Cost |
|---|---|---|---|
| Extractor | `inclusionai/ling-2.6-flash` | OpenRouter | $0.01/M in, $0.03/M out |
| Reader | `openai/gpt-5-mini` | OpenRouter | $0.25/M in, $2.00/M out |
| Judge | `openai/gpt-4o-2024-08-06` | OpenRouter | $2.50/M in, $10.00/M out |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local | Free |
| Cross-encoder reranker | `ms-marco-MiniLM-L-6-v2` | Local | Free |
| Predicate pack | `personal_assistant` | Local | Free |

---

## Estimated Cost

| Run | Questions | Estimated Cost |
|---|---|---|
| Smoke test | 30 (stratified sample) | ~$0.36 |
| Full benchmark | 500 | ~$7.50 |

All costs are via a single OpenRouter API key (`MEMCONTEXT_READER_API_KEY`). Local components (embeddings, cross-encoder, predicate packs) incur no cost.

---

## Quick Start

### Prerequisites

- Python 3.12+
- The LongMemEval-S dataset cloned into `data/longmemeval-s/` (see below)

### 1. Install

```bash
python -m pip install -e ".[embeddings]"
```

### 2. Get the dataset

```bash
git clone https://github.com/xiaowu0162/LongMemEval data/longmemeval-s
```

The runner expects the dataset at `data/longmemeval-s/data/longmemeval_s_cleaned.json`. If you place it elsewhere, pass `--dataset <path>` to the runner.

### 3. Set environment

```bash
# Required: OpenRouter API key (used for extractor, reader, and judge)
export MEMCONTEXT_READER_API_KEY="sk-or-v1-..."

# Required for OpenRouter extraction:
export MEMCONTEXT_EXTRACTOR_BACKEND="openrouter"
export MEMCONTEXT_EXTRACTOR_API_KEY="$MEMCONTEXT_READER_API_KEY"
```

On Windows (PowerShell):

```powershell
$env:MEMCONTEXT_READER_API_KEY = "sk-or-v1-..."
$env:MEMCONTEXT_EXTRACTOR_BACKEND = "openrouter"
$env:MEMCONTEXT_EXTRACTOR_API_KEY = $env:MEMCONTEXT_READER_API_KEY
```

### 4. Run smoke test (30 questions, ~$0.36)

```bash
python evals/run_smoke30.py --seed 42 --yes --extractor-backend openrouter --extractor-model inclusionai/ling-2.6-flash
```

Results are saved to `results/smoke30_<timestamp>.json`.

### 5. Run full benchmark (500 questions, ~$7.50)

```bash
python evals/benchmark/run_official.py \
    --dataset data/longmemeval-s/data/longmemeval_s_cleaned.json \
    --output results/hypothesis.jsonl \
    --reader openai/gpt-5-mini
```

This produces:
- `results/hypothesis.jsonl` -- one `{"question_id": ..., "hypothesis": ...}` per line
- `results/hypothesis.config.json` -- full run configuration (git hash, models, weights, timestamps)

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MEMCONTEXT_READER_API_KEY` | *(required)* | OpenRouter API key for reader, judge, and (optionally) extractor |
| `MEMCONTEXT_READER_MODEL` | `openai/gpt-5-mini` | Reader model for answer generation |
| `MEMCONTEXT_JUDGE_MODEL` | `openai/gpt-4o-2024-08-06` | Judge model for LLM-as-judge scoring |
| `MEMCONTEXT_READER_ENDPOINT` | `https://openrouter.ai/api/v1/chat/completions` | API endpoint for reader and judge |
| `MEMCONTEXT_EXTRACTOR_BACKEND` | `ollama` | Extraction backend: `ollama` or `openrouter` |
| `MEMCONTEXT_EXTRACTOR_MODEL` | `qwen3:8b` (ollama) / `openai/gpt-4.1-nano` (openrouter) | Extraction model |
| `MEMCONTEXT_EXTRACTOR_API_KEY` | *(none)* | API key for OpenRouter extraction (only needed if backend is `openrouter`) |
| `MEMCONTEXT_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (local) |
| `MEMCONTEXT_RERANKER` | `auto` | Reranker mode: `auto`, `cross-encoder`, or empty to disable |
| `ACTIVE_PACK` | `personal_assistant` (set by eval scripts) | Predicate pack for claim extraction |
| `MEMCONTEXT_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL (only for ollama backend) |

---

## Protocol Compliance

This evaluation follows the official LongMemEval scoring protocol from [`xiaowu0162/LongMemEval/src/evaluation/evaluate_qa.py`](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py):

**Judge model:** `openai/gpt-4o-2024-08-06` -- the exact model specified by the official protocol.

**Judge prompts:** Five category-specific prompts taken verbatim from the official `evaluate_qa.py`:

| Prompt key | Used for |
|---|---|
| `default` | single-session-user, single-session-assistant, multi-session |
| `temporal-reasoning` | Temporal ordering (tolerates off-by-one day errors) |
| `knowledge-update` | Knowledge update (accepts updated answer alongside old info) |
| `single-session-preference` | Preference questions (rubric-based, partial credit) |
| `abstention` | Abstention questions (checks if model identifies unanswerable) |

**Scoring:** Two-tier system:
1. **Tier 1 (exact match):** For gold answers with 3 or fewer tokens, normalized boundary matching is applied. Normalization: NFKC casefold, strip currency symbols, collapse whitespace.
2. **Tier 2 (LLM judge):** For all other answers, the judge model is called with `temperature=0.0`, `max_tokens=10`. The judge response is parsed for "yes"/"no".

**Integrity guarantees:**
- No access to gold answers during retrieval or generation
- No benchmark-specific tuning (enforced by anti-overfitting rules in CLAUDE.md)
- Category-specific reader prompts are general-purpose (work for any memory QA, not tuned to LongMemEval answer keys)
- Judge fallbacks (API failures) are tracked and reported in results

---

## Retrieval Architecture

MemContext uses a 9-channel Reciprocal Rank Fusion (RRF, k=60) pipeline followed by optional cross-encoder reranking.

| Channel | Weight | Description |
|---|---|---|
| Semantic | 0.5 | Cosine similarity on claim embeddings |
| Entity | 0.2 | NER + entity_key match |
| Temporal recency | 0.1 | Preference for recent events (event_ts > valid_from_ts > created_ts) |
| BM25 | 0.2 | Lexical term frequency |
| Temporal scope | 0.3 (conditional) | "last N days" window matching |
| Date-value matching | 0.3 (conditional) | Regex date extraction, proximity scored as 1/(1+days) |
| Predicate targeting | 0.2 (conditional) | Query intent classification to predicate families |
| Confidence | 0.1 | Extraction confidence score |
| Frequency | 0.1 | (subject, predicate) occurrence count |

**Post-RRF reranking:** Top-k results are reranked by `ms-marco-MiniLM-L-6-v2` (22 MB local cross-encoder). Enabled by default when `sentence-transformers` is installed (`MEMCONTEXT_RERANKER=auto`).

**Reader prompts:** 8 category-specific chain-of-thought prompts (Extract/Notes, Reason, Answer) route each question through a tailored reasoning path. See `evals/longmemeval_prompts.py`.

---

## Results Format

### Smoke test (`run_smoke30.py`)

Output: `results/smoke30_<YYYYMMDD_HHMMSS>.json`

```json
{
  "seed": 42,
  "questions_sampled": 18,
  "questions_scored": 18,
  "overall_accuracy_raw": 0.8333,
  "overall_accuracy_task_averaged": 0.85,
  "per_category": {
    "single-session-user": {"accuracy": 1.0, "correct": 3, "total": 3},
    "...": "..."
  },
  "extraction_stats": {
    "total_turns": 1200,
    "turns_with_claims": 1150,
    "turns_empty_fallback": 30,
    "turns_failed": 20
  },
  "questions": ["... per-question detail with claims, excerpts, judge verdicts ..."]
}
```

The smoke test samples 3 questions per category (18 total from 6 categories) using a deterministic seed for reproducibility.

### Full benchmark (`run_official.py`)

Output: `results/hypothesis.jsonl` + `results/hypothesis.config.json`

Each line in the JSONL file:

```json
{"question_id": "q_001", "hypothesis": "The user's favorite color is blue."}
```

The config sidecar records the full run configuration:

```json
{
  "dataset_path": "data/longmemeval-s/data/longmemeval_s_cleaned.json",
  "reader": "openai/gpt-5-mini",
  "top_k": 50,
  "weights": [0.7, 0.0, 0.0, 0.3],
  "git_commit": "abc1234...",
  "embedding_model": "all-MiniLM-L6-v2",
  "started_at": "2026-05-27T00:00:00Z",
  "completed_at": "2026-05-27T01:30:00Z",
  "total_questions": 500
}
```

---

## File Reference

```
evals/
  benchmark/
    run_official.py          # Full 500q benchmark runner (hypothesis JSONL output)
    README.md                # This file
  run_smoke30.py             # 30q smoke test with preflight checks
  quickcheck.py              # Stratified subset runner (used by smoke test)
  longmemeval.py             # Dataset loading, session ingestion, eval pipeline
  longmemeval_prompts.py     # Category-specific reader prompts (8 templates)
  metrics.py                 # Two-tier scoring, judge prompts, fuzzy F1 fallback
  runner.py                  # Generic eval runner infrastructure
  ceiling.py                 # Failure classification
results/                     # Output directory for all eval results
data/longmemeval-s/          # Official LongMemEval-S dataset (git clone)
```
