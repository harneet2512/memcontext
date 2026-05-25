# Benchmark Results

This directory stores official benchmark results.

## Reproduction

1. Install: `pip install -e ".[dev,embeddings]"`
2. Run: `python evals/benchmark/run_official.py --dataset data/longmemeval-s/data/longmemeval_s_cleaned.json --output results/hypothesis.jsonl --reader gpt-4o-mini`
3. Score: Clone official eval repo and run `evaluate_qa.py` against `hypothesis.jsonl`

## Legitimacy Rules

- Hypothesis file is generated ONCE per config
- Official eval script is NEVER modified
- Config is committed BEFORE the run
- Git commit hash is recorded in config
