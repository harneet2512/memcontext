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
ARG PRODUCT_REF=1c28a92ac75a02901e91adb4773a26c90cea2b99
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
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HF_HOME=/opt/hf_cache
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
COPY docker/amb/patch_openrouter_strict.py /opt/patch_openrouter_strict.py
COPY docker/amb/fix_extractor_fence.py /opt/fix_extractor_fence.py
# CONSOLIDATED: the former product-patches (patch_retrieval, patch_supersession,
# patch_fusion) and the vendored enumeration.py are REMOVED. PRODUCT_REF now points at
# the real integrated product (master lineage), which carries tie-aware fusion, robust
# cross-session + attribute-keyed supersession, the retrieval footgun fixes, and
# enumeration.py NATIVELY. The benchmark measures the real product unmodified; only the
# thin bridge (patch_provider) + transport shims remain. No product is patched at build.
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
 && python /opt/patch_openrouter_strict.py /opt/amb/src/memory_bench/llm/openai.py \
 && python /opt/fix_extractor_fence.py /opt/amb/.venv/lib/python3.12/site-packages/memcontext/extractors.py

# Bake the embedding model into the image (HF_HOME=/opt/hf_cache) so retrieval
# NEVER downloads it from HuggingFace at runtime. N parallel shards each pulling
# the same weights from HF = the OSError that crashed single-session-preference
# and silently degraded the rest (substrate.embed_episode_failed). With the
# weights cached in the image, the runtime loads locally; the entrypoint forces
# HF offline for the HF-free datasets so no network call is even attempted.
# Retry the bake: the build-time HF download is the one remaining network step,
# and a transient huggingface.co blip/rate-limit must not kill a 20-min build
# (it did once: LocalEntryNotFoundError mid-build). 6 attempts w/ backoff + a
# longer HF timeout; only a sustained outage fails the build.
ENV HF_HUB_DOWNLOAD_TIMEOUT=60
RUN cd /opt/amb && for i in 1 2 3 4 5 6; do \
      uv run python -c "from memcontext.retrieval import BGE_M3_MODEL_ID as M; from sentence_transformers import SentenceTransformer; m=SentenceTransformer(M); print('baked', M, 'dim=', m.get_sentence_embedding_dimension())" && exit 0; \
      echo "embedder bake attempt $i failed (HF unreachable) — retrying in 20s"; sleep 20; \
    done; \
    echo "embedder bake failed after 6 attempts" >&2; exit 1

# Self-host the single-file datasets — one sequential curl each at BUILD time, so
# runtime shards read a local copy via *_DATA_PATH and NEVER call the source.
# (Parallel shards downloading the same dataset = HTTP 429, which killed
# multi-session.) longmemeval is on HF; locomo/lifebench are on GitHub raw.
# personamem/beam use the HF datasets library (handled via HF_TOKEN at runtime).
# --retry alone does NOT retry TLS/connection failures (curl exit 35: "unexpected
# eof" killed a build); --retry-all-errors does, and the must-succeed longmemeval
# download is wrapped in an outer retry loop so a transient HF blip can't be fatal.
RUN mkdir -p /opt/datasets \
 && for i in 1 2 3 4 5 6; do \
      curl -fSL --retry-all-errors --retry 6 --retry-delay 8 --connect-timeout 30 \
        -o /opt/datasets/longmemeval_s_cleaned.json \
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" \
        && test -s /opt/datasets/longmemeval_s_cleaned.json && break; \
      echo "longmemeval fetch attempt $i failed (HF blip) — retrying in 15s"; sleep 15; \
    done \
 && test -s /opt/datasets/longmemeval_s_cleaned.json \
 && (curl -fSL --retry-all-errors --retry 6 --retry-delay 8 -o /opt/datasets/locomo10.json \
      "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json" || echo "WARN: locomo not fetched") \
 && (curl -fSL --retry-all-errors --retry 6 --retry-delay 8 -o /opt/datasets/our_en.json \
      "https://raw.githubusercontent.com/1754955896/LifeBench/main/life_bench_data/locomo_format/our_en.json" || echo "WARN: lifebench 404 upstream — skipped") \
 && du -sh /opt/datasets/* 2>/dev/null && echo "datasets self-hosted at /opt/datasets"

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
