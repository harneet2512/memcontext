# syntax=docker/dockerfile:1.7
###############################################################################
# Reproducible AMB trial — master's REAL MemContext product, measured by the
# frozen Agent Memory Benchmark (AMB) harness through the pinned provider bridge.
#
#   Product under test : master @ ${PRODUCT_REF}        (the shipping product)
#   Frozen instrument  : benchmark/amb @ ${HARNESS_REF}  (provider + install.py)
#   External harness    : vectorize-io/agent-memory-benchmark @ ${AMB_REF}
#
# The product is `git archive`-d directly from master inside the image, so the
# codex / Product-B working tree NEVER enters the build. The eval measures the
# real product on main — nothing added, nothing subtracted.
#
# Build (context MUST be the repo root so .git is available):
#   docker build -f docker/amb/Dockerfile -t memcontext-amb \
#     --build-arg AMB_REF=<pin-a-real-amb-commit-sha> .
###############################################################################

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1 — export the two pinned trees from this repo's own git history.
# We archive master's product and benchmark/amb's instrument by SHA; we never
# check out the codex working set.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS export
ARG PRODUCT_REF=f66b25cce11df068217f0c828222b87765013765
ARG HARNESS_REF=578ce2fca8bed1d5e32ab66bde835810e1f3a09a
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY .git ./.git
# Fail loudly if either pinned commit is missing from the provided .git.
RUN git --git-dir=/repo/.git cat-file -e "${PRODUCT_REF}^{commit}" \
 && git --git-dir=/repo/.git cat-file -e "${HARNESS_REF}^{commit}"
RUN mkdir -p /out/product /out/harness \
 && git --git-dir=/repo/.git archive "${PRODUCT_REF}"        | tar -x -C /out/product \
 && git --git-dir=/repo/.git archive "${HARNESS_REF}" evals  | tar -x -C /out/harness
# Resolve the abbreviated refs to full SHAs for the run manifest.
RUN printf '{"product_ref":"%s","harness_ref":"%s"}\n' \
      "$(git --git-dir=/repo/.git rev-parse "${PRODUCT_REF}")" \
      "$(git --git-dir=/repo/.git rev-parse "${HARNESS_REF}")" > /out/refs.json

# ---------------------------------------------------------------------------
# Stage 2 — runtime: install master's product, clone the external AMB harness,
# inject the MemContext provider, expose the entrypoint.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime
ARG AMB_REPO=https://github.com/vectorize-io/agent-memory-benchmark.git
# Pinned to vectorize-io/agent-memory-benchmark HEAD @ 2026-05-08 for a
# reproducible build. Override --build-arg AMB_REF=<sha> to track a different rev.
ARG AMB_REF=45fa380523afab9b1acd667a03de51c5ea63f4d2
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# 1) master's REAL product + the frozen instrument trees
COPY --from=export /out/product /opt/product
COPY --from=export /out/harness /opt/harness
COPY --from=export /out/refs.json /opt/refs.json
# Our adapter's faithful-wiring patch (Finding 3) — corrects OUR provider only.
COPY docker/amb/patch_provider.py /opt/patch_provider.py
# Per-role-key TRANSPORT patch for AMB's openai backend (answer & judge each get
# their own TokenRouter key + base URL — transport only, no scoring change).
COPY docker/amb/patch_amb_llm.py /opt/patch_amb_llm.py
# Restore AMB's NATIVE strict json_schema (undo install.py's json_object) so the
# answer/judge code is PRISTINE — gateways enforce complete fields, no rag/judge
# touch. And fence-strip OUR extractor (MiniMax-M3 wraps JSON in a markdown fence).
COPY docker/amb/revert_json_schema.py /opt/revert_json_schema.py
COPY docker/amb/fix_extractor_fence.py /opt/fix_extractor_fence.py
# Shard enumerator for the sharded GHA workflow (lists dataset/split/category).
COPY docker/amb/discover_shards.py /opt/discover_shards.py

# 2) external AMB harness, pinned by SHA
WORKDIR /opt
RUN git clone "${AMB_REPO}" amb \
 && git -C amb checkout "${AMB_REF}" \
 && git -C amb rev-parse HEAD > /opt/amb_ref.txt
WORKDIR /opt/amb
RUN uv sync
# Install the real product into AMB's uv venv (this is the env `uv run` uses),
# faithful-wire OUR provider (Finding 3), then inject + register it.
# install.py patches AMB's openai.py json mode (json_schema->json_object) — a
# transport shim, required only when routing the answer/judge Gemini calls
# through an OpenAI-compatible gateway (TokenRouter). It edits ONLY the
# response_format block inside generate().
# Then patch_amb_llm.py applies the SECOND transport shim: per-role keys, so the
# answer role (free key) and judge role (paid key) each authenticate with their
# OWN TokenRouter key + base URL instead of colliding on one OPENAI_API_KEY. It
# edits ONLY OpenAILLM.__init__ + get_answer_llm/get_judge_llm — disjoint from
# install.py's response_format edit, so order is safe. AMB is `uv pip install -e`
# editable, so its source lives at /opt/amb/src/memory_bench/llm/ (where install.py
# also writes); patch that tree. Both shims are disclosed in README.
RUN uv pip install "/opt/product[embeddings]" \
 && python /opt/patch_provider.py /opt/harness/evals/amb/provider.py \
 && python /opt/harness/evals/amb/install.py \
 && python /opt/patch_amb_llm.py /opt/amb/src/memory_bench/llm \
 && python /opt/revert_json_schema.py /opt/amb/src/memory_bench/llm/openai.py \
 && python /opt/fix_extractor_fence.py /opt/amb/.venv/lib/python3.12/site-packages/memcontext/extractors.py

# Prefetch + BAKE the AMB datasets into the image (.datasets/) — a single
# sequential download at build time. Otherwise every shard downloads the dataset
# from HuggingFace at runtime, and N parallel shards hammer HF -> HTTP 429 ->
# shard failure (this is what killed multi-session). With them baked in, runtime
# shards read the local copy and never hit HF. `|| true`: datasets needing an
# external data path (membench/memsim) are skipped, not fatal.
RUN cd /opt/amb && (uv run python /opt/discover_shards.py || true) \
 && (du -sh /opt/amb/.datasets 2>/dev/null || echo "no .datasets baked")

# Fold the resolved AMB SHA into the manifest refs.
RUN python - <<'PY'
import json, pathlib
refs = json.loads(pathlib.Path('/opt/refs.json').read_text())
refs['amb_ref'] = pathlib.Path('/opt/amb_ref.txt').read_text().strip()
pathlib.Path('/opt/refs.json').write_text(json.dumps(refs, indent=2, sort_keys=True))
PY

COPY docker/amb/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /opt/amb
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Default: a 5-question-per-category smoke of LongMemEval-S. Override at runtime,
# e.g. `docker run ... memcontext-amb longmemeval s` for the full split, or no
# args for the whole AMB suite.
CMD ["longmemeval", "s", "--query-limit", "5"]
