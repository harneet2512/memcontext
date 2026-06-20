#!/usr/bin/env bash
###############################################################################
# AMB reproducible-trial entrypoint.
#
# Validates the billing keys, pins the model config, writes a run manifest that
# records WHICH product (master SHA) and WHICH instrument (benchmark/amb SHA +
# AMB SHA) produced the score, then hands off to the real AMB harness.
#
# Keys are injected at `docker run -e ...` and never baked into the image.
###############################################################################
set -euo pipefail

# ---- per-role gateway keys (OpenAI-compatible) -----------------------------
# Each role bills to its OWN key on its OWN gateway:
#   answer    = gpt-oss-120b -> OpenRouter  (free key)
#   judge     = gemini-3     -> TokenRouter (paid key)
#   extractor = minimax-m3   -> TokenRouter (its own key)
# AMB's stock `openai` backend reads a single OPENAI_API_KEY, so the
# patch_amb_llm.py transport shim lets answer + judge each carry their OWN key +
# base URL (OMB_ANSWER_OPENAI_* / OMB_JUDGE_OPENAI_*). We accept clear user-facing
# names (ANSWER_KEY / JUDGE_KEY / EXTRACTOR_KEY) and the explicit OMB_*_OPENAI_API_KEY
# names directly.
OPENROUTER_BASE_URL_DEFAULT="https://openrouter.ai/api/v1"
TOKENROUTER_BASE_URL_DEFAULT="https://api.tokenrouter.com/v1"

ANSWER_KEY="${ANSWER_KEY:-${OMB_ANSWER_OPENAI_API_KEY:-}}"
JUDGE_KEY="${JUDGE_KEY:-${OMB_JUDGE_OPENAI_API_KEY:-}}"
EXTRACTOR_KEY="${EXTRACTOR_KEY:-${MEMCONTEXT_EXTRACTOR_API_KEY:-}}"
: "${ANSWER_KEY:?set -e ANSWER_KEY (or OMB_ANSWER_OPENAI_API_KEY) — OpenRouter key for the ANSWER role}"
: "${JUDGE_KEY:?set -e JUDGE_KEY (or OMB_JUDGE_OPENAI_API_KEY) — TokenRouter key for the JUDGE role}"
: "${EXTRACTOR_KEY:?set -e EXTRACTOR_KEY (or MEMCONTEXT_EXTRACTOR_API_KEY) — TokenRouter key for the EXTRACTOR}"

# ---- answer role: openai backend, own key + base URL -----------------------
export OMB_ANSWER_LLM="${OMB_ANSWER_LLM:-openai}"
export OMB_ANSWER_MODEL="${OMB_ANSWER_MODEL:-openai/gpt-oss-120b}"
export OMB_ANSWER_OPENAI_API_KEY="${ANSWER_KEY}"
export OMB_ANSWER_OPENAI_BASE_URL="${OMB_ANSWER_OPENAI_BASE_URL:-$OPENROUTER_BASE_URL_DEFAULT}"

# ---- judge role: openai backend, own (distinct) key + base URL -------------
export OMB_JUDGE_LLM="${OMB_JUDGE_LLM:-openai}"
export OMB_JUDGE_MODEL="${OMB_JUDGE_MODEL:-google/gemini-3-flash-preview}"
export OMB_JUDGE_OPENAI_API_KEY="${JUDGE_KEY}"
export OMB_JUDGE_OPENAI_BASE_URL="${OMB_JUDGE_OPENAI_BASE_URL:-$TOKENROUTER_BASE_URL_DEFAULT}"

# ---- extractor (provider detail; AMB mandates none) ------------------------
# MemContext's own LLM extractor — its own key + endpoint, independent of the AMB
# answer/judge roles. minimax-m3 with reasoning disabled (NO_THINK=1).
# BACKEND is only the TRANSPORT selector (openai-compatible HTTP). The actual URL
# comes from MEMCONTEXT_EXTRACTOR_ENDPOINT below (extractors.py:498-504), so
# 'openrouter' + a TokenRouter ENDPOINT correctly routes to TokenRouter — the
# string 'openrouter' does NOT force the openrouter.ai URL.
export MEMCONTEXT_EXTRACTOR_BACKEND="${MEMCONTEXT_EXTRACTOR_BACKEND:-openrouter}"
export MEMCONTEXT_EXTRACTOR_ENDPOINT="${MEMCONTEXT_EXTRACTOR_ENDPOINT:-${TOKENROUTER_BASE_URL_DEFAULT}/chat/completions}"
export MEMCONTEXT_EXTRACTOR_MODEL="${MEMCONTEXT_EXTRACTOR_MODEL:-MiniMax-M3}"
export MEMCONTEXT_EXTRACTOR_NO_THINK="${MEMCONTEXT_EXTRACTOR_NO_THINK:-1}"
export MEMCONTEXT_EXTRACTOR_API_KEY="${EXTRACTOR_KEY}"
# Parallelism: NO-OP for the faithful AMB ingest. The patched adapter delegates to
# the product's on_new_turn(queue=) -> InlineQueue.drain(), which extracts SERIALLY,
# so this var no longer affects ingest. Benchmark parallelism comes from GHA
# sharding, not in-process workers. Left exported for any other code path that
# reads it; harmless if unset. (Backoff/pool: extractors.py:660,713 / pool=128.)
export MEMCONTEXT_EXTRACTION_WORKERS="${MEMCONTEXT_EXTRACTION_WORKERS:-128}"

# ---- product config --------------------------------------------------------
# Predicate packs are git-archived to /opt/product but pip-install doesn't bundle
# them (pyproject packages only `memcontext`), so the product must be pointed at
# them — else _build_system_prompt() throws and EVERY extraction silently yields [].
export SUBSTRATE_PACKS_DIR="${SUBSTRATE_PACKS_DIR:-/opt/product/predicate_packs}"
export ACTIVE_PACK="${ACTIVE_PACK:-personal_assistant}"
# AMB probes for this at startup even when routed via OpenRouter; any value works.
export GEMINI_API_KEY="${GEMINI_API_KEY:-not-used-routing-via-openrouter}"

# ---- self-hosted datasets (no HuggingFace/GitHub at runtime) ----------------
# Point omb at the files curl'd into the image at build time. With these set, omb
# SKIPS the download entirely — so N parallel shards can't 429 the source (which
# killed multi-session). longmemeval(HF), locomo+lifebench(GitHub) are baked in.
export LONGMEMEVAL_DATA_PATH="${LONGMEMEVAL_DATA_PATH:-/opt/datasets/longmemeval_s_cleaned.json}"
export LOCOMO_DATA_PATH="${LOCOMO_DATA_PATH:-/opt/datasets/locomo10.json}"
export LIFEBENCH_DATA_PATH="${LIFEBENCH_DATA_PATH:-/opt/datasets/our_en.json}"
# personamem/beam use the HF datasets library; an HF token raises the rate limit
# so parallel downloads don't 429 (optional — pass via the HF_TOKEN secret).
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"

# ---- baked embedding model (no HuggingFace at runtime) ----------------------
# arctic-embed-s is baked into the image at /opt/hf_cache during build. Point the
# runtime at that cache so retrieval loads the model locally. For HF-free datasets
# (everything except personamem/beam, which still pull via the HF datasets lib),
# force offline so N parallel shards never even HEAD-check HF for the model — the
# exact network call that 429'd/refused and crashed a shard. personamem/beam keep
# HF online for their dataset; the embedder there still loads from the baked cache.
export HF_HOME="${HF_HOME:-/opt/hf_cache}"
case "${1:-}" in
  personamem|beam) : ;;
  *)
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    ;;
esac

# ---- reproducibility manifest ----------------------------------------------
OUT_DIR="${OUTPUT_DIR:-/opt/amb/outputs}"
mkdir -p "$OUT_DIR"
uv run python - "$OUT_DIR" <<'PY'
import json, os, sys, pathlib
refs = json.loads(pathlib.Path('/opt/refs.json').read_text())

# Legitimacy guard: the product treats lexical-only (no embeddings) as a DEGRADED
# mode, not normal operation. Surface it loudly and record it in the manifest so a
# degraded run can never be mistaken for a real one.
try:
    from memcontext.retrieval import semantic_enabled
    semantic_on = bool(semantic_enabled())
except Exception as e:  # noqa: BLE001
    semantic_on = None
    print(f"[amb-bridge] WARN: could not probe semantic mode: {e}")
if semantic_on is False:
    print("[amb-bridge] WARN: semantic memory is OFF (lexical-only) — Pass-2 "
          "supersession + semantic retrieval DISABLED. This is a DEGRADED run.")

# Legitimacy guard #2: with the raw-text fallback removed, a non-LLM extractor
# (SimpleExtractor/Passthrough) would yield ~no real claims. The benchmark MUST
# run a real LLM extractor. Surface the selected one and flag if it isn't LLM.
try:
    from memcontext.extractors import auto_extractor
    extractor_cls = type(auto_extractor()).__name__
    extractor_is_llm = "LLM" in extractor_cls
except Exception as e:  # noqa: BLE001
    extractor_cls, extractor_is_llm = f"unknown ({e})", None
if extractor_is_llm is False:
    print(f"[amb-bridge] WARN: extractor is {extractor_cls}, NOT an LLM extractor — "
          "claims will be near-empty (fallback removed). Set a real extractor.")

manifest = {
    "trial": "amb-on-master-product",
    "product_under_test": {"branch": "master", "commit": refs.get("product_ref")},
    "instrument": {
        "branch": "benchmark/amb",
        "commit": refs.get("harness_ref"),
        "amb_repo_commit": refs.get("amb_ref"),
    },
    "routing": {
        # Per-role transport: answer, judge, and extractor each route through
        # TokenRouter's OpenAI-compatible API on their OWN key (answer = free,
        # judge = paid, extractor = its own). Base URLs recorded; keys never are.
        "answer_base_url": os.environ.get("OMB_ANSWER_OPENAI_BASE_URL"),
        "judge_base_url": os.environ.get("OMB_JUDGE_OPENAI_BASE_URL"),
        "extractor_endpoint": os.environ.get("MEMCONTEXT_EXTRACTOR_ENDPOINT"),
        "answer_judge_distinct_keys": (
            os.environ.get("OMB_ANSWER_OPENAI_API_KEY")
            != os.environ.get("OMB_JUDGE_OPENAI_API_KEY")
        ),
    },
    "models": {
        "extractor": os.environ.get("MEMCONTEXT_EXTRACTOR_MODEL"),
        "answer": os.environ.get("OMB_ANSWER_MODEL"),
        "judge": os.environ.get("OMB_JUDGE_MODEL"),
        "active_pack": os.environ.get("ACTIVE_PACK"),
        # AMB upstream's pinned leaderboard model (retires 2026-07-22). This run
        # uses distinct per-role models, so it is NOT leaderboard-comparable.
        "official_amb_leaderboard_model": "gemini-2.5-flash-lite",
        "leaderboard_comparable": (
            os.environ.get("OMB_ANSWER_MODEL") == "google/gemini-2.5-flash-lite"
            and os.environ.get("OMB_JUDGE_MODEL") == "google/gemini-2.5-flash-lite"
        ),
    },
    "modifications": {
        # our adapter, corrected to match the product's real ingest path
        "provider_faithful_wiring": "embedder + semantic passed to on_new_turn",
        # our adapter's retrieve() routed through the product's real query door
        # (retrieve_memory_across: per-session facts+episodes RRF fused by rank, as
        # mcp_tools.handle_memory_query serves a multi-session query), not the
        # deprecated facts-only retrieve_hybrid that returned turn.text per fact
        "provider_retrieve_via_retrieve_memory": True,
        # each AMB Document is its own session (amb_{doc.id}); the provider tracks
        # sessions per user and fans out across them, so the product's real
        # multi-session machinery (cross-session RRF fusion) actually runs
        "provider_session_model": "per-conversation + retrieve_memory_across",
        # prohibited raw-text fallback removed (miss -> no claim, not text[:500])
        "provider_raw_text_fallback_removed": True,
        # transport shim #1 on AMB code — proxy-routed JSON (json_schema->json_object)
        "amb_openai_json_object_patch": True,
        # transport shim #2 on AMB code — per-role keys so answer (free key) and
        # judge (paid key) each authenticate with their OWN TokenRouter key +
        # base URL instead of colliding on one OPENAI_API_KEY. Transport only:
        # no scoring/rubric/model-selection change. Applied by patch_amb_llm.py.
        "amb_per_role_key_patch": True,
    },
    "semantic_memory_enabled": semantic_on,
    "extractor_class": extractor_cls,
    "extractor_is_llm": extractor_is_llm,
    # record presence only — never the secret value
    "keys_present": {
        k: bool(os.environ.get(k))
        for k in (
            "OMB_ANSWER_OPENAI_API_KEY",
            "OMB_JUDGE_OPENAI_API_KEY",
            "MEMCONTEXT_EXTRACTOR_API_KEY",
        )
    },
}
out = pathlib.Path(sys.argv[1]) / "run_manifest.json"
out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
p = manifest["product_under_test"]["commit"] or ""
h = refs.get("harness_ref") or ""
print(f"[amb-bridge] product = master@{p[:12]}  instrument = benchmark/amb@{h[:12]}")
print(f"[amb-bridge] answer = {manifest['models']['answer']} (own key) | "
      f"judge = {manifest['models']['judge']} (own key) via TokenRouter; "
      f"distinct_keys={manifest['routing']['answer_judge_distinct_keys']}; "
      f"semantic_memory={semantic_on}")
print(f"[amb-bridge] manifest -> {out}")
PY

# ---- run the real AMB harness ----------------------------------------------
# No args  -> whole AMB suite. `<dataset> <split> [extra...]` -> one split.
if [ "$#" -eq 0 ]; then
    echo "[amb-bridge] omb run --memory memcontext --mode rag  (whole AMB suite)"
    exec uv run omb run --memory memcontext --mode rag
fi
DATASET="$1"; SPLIT="${2:-}"
shift || true
[ "$#" -gt 0 ] && shift || true
echo "[amb-bridge] omb run --dataset ${DATASET} --split ${SPLIT} --memory memcontext --mode rag $*"
exec uv run omb run --dataset "$DATASET" --split "$SPLIT" --memory memcontext --mode rag "$@"
