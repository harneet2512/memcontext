#!/usr/bin/env bash
###############################################################################
# Pull the prebuilt AMB image from GHCR and run it LIVE in a Codespace terminal
# (or any Docker host). The image is built once by the GHA workflow and pushed to
# ghcr.io/<repo>/amb:built — this script just pulls it and runs, so you watch the
# whole ingest -> extract -> answer -> judge stream in real time.
#
# Usage (inside a Codespace terminal on this branch):
#   export ANSWER_KEY=...      # OpenRouter key (answer = gpt-oss-120b)
#   export JUDGE_KEY=...       # TokenRouter key (judge = gemini-3-flash, PAID)
#   export EXTRACTOR_KEY=...   # TokenRouter key (extractor = MiniMax-M3)
#   ./docker/amb/run_codespace.sh                 # 30q LongMemEval trial (default)
#   ./docker/amb/run_codespace.sh longmemeval s   # full LongMemEval-S split
#   ./docker/amb/run_codespace.sh locomo test     # a different dataset/split
#
# Optional: MEMCONTEXT_EXTRACTION_WORKERS (default 24 — the MiniMax key saturates
# around here; higher just burns backoff on 429s).
###############################################################################
set -euo pipefail

IMAGE="${AMB_IMAGE:-ghcr.io/harneet2512/memcontext/amb:built}"
: "${ANSWER_KEY:?export ANSWER_KEY (OpenRouter key for the answer role)}"
: "${JUDGE_KEY:?export JUDGE_KEY (TokenRouter key for the judge role)}"
: "${EXTRACTOR_KEY:?export EXTRACTOR_KEY (TokenRouter key for the extractor)}"

echo "[codespace] logging in to GHCR..."
gh auth token | docker login ghcr.io -u "$(gh api user -q .login)" --password-stdin

echo "[codespace] pulling $IMAGE ..."
docker pull "$IMAGE"

ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(longmemeval s --query-limit 5)
mkdir -p out

echo "[codespace] running: ${ARGS[*]}"
docker run --rm \
  -e ANSWER_KEY -e JUDGE_KEY -e EXTRACTOR_KEY \
  -e MEMCONTEXT_EXTRACTION_WORKERS="${MEMCONTEXT_EXTRACTION_WORKERS:-24}" \
  -v "$PWD/out:/opt/amb/outputs" \
  "$IMAGE" "${ARGS[@]}"

echo "=== run complete — outputs in ./out ==="
ls -la out
