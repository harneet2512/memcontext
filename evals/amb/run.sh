#!/usr/bin/env bash
# Run the AMB harness with MemContext + DeepSeek extraction.
# Loads keys/config from .env in the current (agent-memory-benchmark) dir.
#
#   bash run.sh longmemeval s                 # full LongMemEval-S
#   bash run.sh longmemeval s --query-limit 5 # smoke
#   bash run.sh                               # whole AMB (all datasets)
set -euo pipefail

if [ ! -f .env ]; then
  echo "No .env found. Copy evals/amb/.env.example to .env and fill in keys." >&2
  exit 1
fi
set -a; source .env; set +a

if [ $# -eq 0 ]; then
  echo "Running whole AMB (all datasets) with memcontext..."
  uv run omb run --memory memcontext --mode rag
else
  DATASET="$1"; SPLIT="$2"; shift 2 || true
  echo "Running AMB $DATASET/$SPLIT with memcontext..."
  uv run omb run --dataset "$DATASET" --split "$SPLIT" --memory memcontext --mode rag "$@"
fi
