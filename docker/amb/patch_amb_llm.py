#!/usr/bin/env python3
"""Per-role-key TRANSPORT patch for AMB's OpenAI-compatible backend.

The user routes BOTH the AMB answer LLM and the AMB judge LLM through
TokenRouter's OpenAI-compatible API (AMB's `openai` backend), but with two
SEPARATE keys: a free key for the answer role and a paid key for the judge
role. AMB's stock `openai` backend reads a single `OPENAI_API_KEY` (and the SDK
reads a single `OPENAI_BASE_URL`), so the two roles would collide on one key.

This patches AMB's INSTALLED source (`src/memory_bench/llm/openai.py` and
`src/memory_bench/llm/__init__.py`) so each role can carry its OWN key + base
URL. It is a TRANSPORT-only change — *which credential / which endpoint* a role
authenticates with — and touches NOTHING about model selection, prompts,
scoring, the judge rubric, or gold answers. It is the second disclosed shim,
peer to install.py's `json_schema`->`json_object` transport shim (also required
only because the Gemini roles are proxied through an OpenAI-compatible gateway).

Order note: install.py's `openai.py` edit only rewrites the `response_format=`
block inside `generate()`. This script only touches the `__init__` signature /
client construction. The two edits are disjoint, so build order is safe whether
this runs before or after install.py.

Mirror of patch_provider.py's design: every anchor MUST match exactly once, and
the script fails loudly (non-zero) on any drift instead of silently no-op'ing.

Usage:
    python patch_amb_llm.py <llm-dir>              # dir holding openai.py + __init__.py
    python patch_amb_llm.py <openai.py> <__init__.py>
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# openai.py — OpenAILLM.__init__ accepts optional per-role key + base URL and
# passes them to the OpenAI SDK client (falling back to the stock single-key
# OPENAI_API_KEY / OPENAI_BASE_URL env when a role supplies neither). Behaviour
# is byte-identical to upstream when api_key/base_url are None.
# ---------------------------------------------------------------------------
OPENAI_EDITS = [
    (
        '    def __init__(self, model: str = "gpt-4o"):\n'
        "        from openai import OpenAI\n"
        '        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))\n'
        "        self._model = model",
        '    def __init__(self, model: str = "gpt-4o", api_key=None, base_url=None):\n'
        "        from openai import OpenAI\n"
        "        # Per-role TRANSPORT: a role (answer / judge) may pass its OWN\n"
        "        # api_key + base_url so two roles routed through the same\n"
        "        # OpenAI-compatible gateway do not collide on one OPENAI_API_KEY.\n"
        "        # When unset, fall back to the stock single-key env exactly as\n"
        "        # upstream did (OPENAI_BASE_URL is what the SDK reads by default).\n"
        "        self._client = OpenAI(\n"
        '            api_key=api_key or os.environ.get("OPENAI_API_KEY"),\n'
        '            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),\n'
        "        )\n"
        "        self._model = model",
        "openai.py: OpenAILLM.__init__ accepts optional api_key/base_url -> OpenAI() (transport only)",
    ),
]

# ---------------------------------------------------------------------------
# __init__.py — get_answer_llm() / get_judge_llm(): when the provider is the
# `openai` backend, construct OpenAILLM with the role's OWN key + base URL from
# OMB_ANSWER_OPENAI_* / OMB_JUDGE_OPENAI_* (falling back to the stock
# OPENAI_API_KEY / OPENAI_BASE_URL). gemini / groq construction is UNCHANGED.
# Anchored on each function's full body (env var names make each block unique;
# the bare `return cls(model) if model else cls()` line is NOT unique on its own).
# ---------------------------------------------------------------------------
INIT_EDITS = [
    (
        '    provider = os.environ.get("OMB_ANSWER_LLM", "groq")\n'
        '    model = os.environ.get("OMB_ANSWER_MODEL")\n'
        "    cls = REGISTRY.get(provider)\n"
        "    if cls is None:\n"
        "        raise ValueError(f\"Unknown OMB_ANSWER_LLM: '{provider}'. Available: {list(REGISTRY)}\")\n"
        "    return cls(model) if model else cls()",
        '    provider = os.environ.get("OMB_ANSWER_LLM", "groq")\n'
        '    model = os.environ.get("OMB_ANSWER_MODEL")\n'
        "    cls = REGISTRY.get(provider)\n"
        "    if cls is None:\n"
        "        raise ValueError(f\"Unknown OMB_ANSWER_LLM: '{provider}'. Available: {list(REGISTRY)}\")\n"
        "    # Per-role TRANSPORT: the answer role carries its own gateway key +\n"
        "    # base URL so it never collides with the judge role's key. Transport\n"
        "    # only — model/provider selection is unchanged; gemini/groq untouched.\n"
        '    if provider == "openai":\n'
        '        api_key = os.environ.get("OMB_ANSWER_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")\n'
        '        base_url = os.environ.get("OMB_ANSWER_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")\n'
        "        return cls(model, api_key=api_key, base_url=base_url) if model else cls(api_key=api_key, base_url=base_url)\n"
        "    return cls(model) if model else cls()",
        "__init__.py: get_answer_llm() openai role uses OMB_ANSWER_OPENAI_API_KEY/BASE_URL (transport only)",
    ),
    (
        '    provider = os.environ.get("OMB_JUDGE_LLM", "gemini")\n'
        '    model = os.environ.get("OMB_JUDGE_MODEL")\n'
        "    cls = REGISTRY.get(provider)\n"
        "    if cls is None:\n"
        "        raise ValueError(f\"Unknown OMB_JUDGE_LLM: '{provider}'. Available: {list(REGISTRY)}\")\n"
        "    return cls(model) if model else cls()",
        '    provider = os.environ.get("OMB_JUDGE_LLM", "gemini")\n'
        '    model = os.environ.get("OMB_JUDGE_MODEL")\n'
        "    cls = REGISTRY.get(provider)\n"
        "    if cls is None:\n"
        "        raise ValueError(f\"Unknown OMB_JUDGE_LLM: '{provider}'. Available: {list(REGISTRY)}\")\n"
        "    # Per-role TRANSPORT: the judge role carries its own (paid) gateway key\n"
        "    # + base URL, distinct from the answer role's (free) key. Transport\n"
        "    # only — judging logic / rubric / model selection are unchanged.\n"
        '    if provider == "openai":\n'
        '        api_key = os.environ.get("OMB_JUDGE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")\n'
        '        base_url = os.environ.get("OMB_JUDGE_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")\n'
        "        return cls(model, api_key=api_key, base_url=base_url) if model else cls(api_key=api_key, base_url=base_url)\n"
        "    return cls(model) if model else cls()",
        "__init__.py: get_judge_llm() openai role uses OMB_JUDGE_OPENAI_API_KEY/BASE_URL (transport only)",
    ),
]


def _apply(path: Path, edits) -> int:
    text = path.read_text(encoding="utf-8")
    for anchor, replacement, label in edits:
        count = text.count(anchor)
        if count != 1:
            print(
                f"ERROR: anchor for [{label}] found {count}x (expected 1) in "
                f"{path}. AMB drifted from the pinned SHA — refusing to patch "
                "silently.",
                file=sys.stderr,
            )
            return 1
        text = text.replace(anchor, replacement, 1)
        print(f"  patched: {label}")
    path.write_text(text, encoding="utf-8")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) == 1:
        d = Path(args[0])
        if not d.is_dir():
            print(f"ERROR: {d} is not a directory", file=sys.stderr)
            return 2
        openai_py = d / "openai.py"
        init_py = d / "__init__.py"
    elif len(args) == 2:
        openai_py = Path(args[0])
        init_py = Path(args[1])
    else:
        print(
            "usage: patch_amb_llm.py <llm-dir> | <openai.py> <__init__.py>",
            file=sys.stderr,
        )
        return 2

    for p in (openai_py, init_py):
        if not p.is_file():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 2

    rc = _apply(openai_py, OPENAI_EDITS)
    if rc:
        return rc
    rc = _apply(init_py, INIT_EDITS)
    if rc:
        return rc

    print(f"[patch_amb_llm] per-role-key transport applied to {openai_py.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
