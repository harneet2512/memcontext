# MemContext AMB Adapter

This package evaluates MemContext against
`agent-memory-benchmark` without editing or committing anything in the upstream
AMB checkout. The adapter registers `memcontext-full` at runtime, then hands the
run to AMB's normal `EvalRunner`.

## Repository Rule

Keep AMB as an external harness. If the AMB checkout remote is not owned by
`harneet2512`, treat it as read-only. The runner inspects the remote and refuses
to run against a dirty non-`harneet2512` AMB checkout.

MemContext-side code lives here:

```text
evals/amb_memcontext/
```

AMB outputs default here:

```text
evals/amb_outputs/
```

The output layout is still AMB-style:

```text
evals/amb_outputs/<dataset>/<memory>/<mode>/<split>.json
```

## Setup

Clone AMB separately:

```bash
git clone https://github.com/vectorize-io/agent-memory-benchmark.git /path/to/agent-memory-benchmark
```

Install MemContext and AMB in the same Python environment:

```bash
cd /path/to/memcontext
pip install -e ".[embeddings,dev]"
pip install -e /path/to/agent-memory-benchmark
```

For full product runs, configure router credentials through environment variables
or GitHub Actions secrets. Do not commit secret values.

```bash
export OPENROUTER_AMB_READER_KEY=...
export TOKENROUTER_AMB_GEMINI_KEY=...
export TOKENROUTER_AMB_JUDGE_KEY=...

export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
export TOKENROUTER_BASE_URL=https://api.tokenrouter.com/v1

export OMB_ANSWER_LLM=openrouter-reader
export OMB_ANSWER_MODEL=openai/gpt-oss-120b:free
export OMB_ANSWER_REASONING_EFFORT=high
export OMB_ANSWER_REASONING_EXCLUDE=1

export MEMCONTEXT_EXTRACTOR_BACKEND=openrouter
export MEMCONTEXT_EXTRACTOR_ENDPOINT=https://api.tokenrouter.com/v1/chat/completions
export MEMCONTEXT_EXTRACTOR_MODEL=google/gemini-3-flash-preview
export MEMCONTEXT_EXTRACTOR_REASONING_EXCLUDE=1

export OMB_JUDGE_LLM=tokenrouter-judge
export OMB_JUDGE_MODEL=google/gemini-3-flash-preview
export OMB_JUDGE_REASONING_EXCLUDE=1

export MEMCONTEXT_EMBED_MODEL=BAAI/bge-m3
export MEMCONTEXT_EMBED_EPISODES=1
```

Embeddings require one of:

```bash
export MODAL_BGE_M3_URL=...
```

or local `sentence-transformers` support from `pip install -e ".[embeddings]"`.

## Smoke Run

GitHub Actions smoke workflow:

```text
.github/workflows/amb-longmemeval-smoke.yml
```

Required GitHub secrets:

```text
OPENROUTER_AMB_READER_KEY
TOKENROUTER_AMB_GEMINI_KEY
TOKENROUTER_AMB_JUDGE_KEY
```

The workflow downloads the LongMemEval-S JSON once, shares it with the category
matrix as an artifact, then runs six category jobs in parallel with
`query_limit=5` by default for a 30-question smoke run. The matrix is capped
with `max-parallel: 20` so larger category/shard matrices do not exceed the GHA
parallelism limit.

```bash
python -m evals.amb_memcontext.run \
  --amb-root /path/to/agent-memory-benchmark \
  --dataset longmemeval \
  --split s \
  --memory memcontext-full \
  --query-limit 5
```

## Full Run

```bash
python -m evals.amb_memcontext.run \
  --amb-root /path/to/agent-memory-benchmark \
  --dataset longmemeval \
  --split s \
  --memory memcontext-full
```

## Adapter Behavior

`MemContextFullProvider.prepare()` creates `store_dir/memcontext.db`, resets it
when AMB asks for reset, and writes `memcontext_run_config.json` beside the DB.

`ingest()` maps AMB documents into real MemContext episodes:

- `Document.user_id` becomes the MemContext namespace.
- `Document.id` becomes the MemContext session id.
- `Document.messages` or JSON conversation turns are ingested one turn at a
  time with the speaker preserved.
- Plain `Document.content` is ingested as one episode.
- `Document.context` and `Document.timestamp` are preserved in source metadata
  and context text.
- Full runs use `LLMExtractor` through TokenRouter Gemini, BGE-M3 embeddings,
  semantic supersession, importance, digests, event frames, life events, and
  consolidation.

`retrieve()` uses unified MemContext retrieval across sessions in the requested
namespace and returns AMB `Document` objects containing a compact briefing plus
top fact and episode hits. `query_timestamp` is passed as `valid_at_ts` so
temporal AMB questions can ask what was true at the question date.

## Result Integrity

Report product scores only from AMB artifacts, not adapter unit tests. If a run
uses a weaker local or regex extractor, label it as an ablation rather than
`memcontext-full`.
