"""Pluggable reader for the eval / validation path.

A *reader* is whatever answers a question from retrieved context. The DEFAULT
reader is the **host model MemContext is attached to** — i.e. no reader runs in
process: no API, no key, no network. The host model reads the served context and
answers; ``answer_question`` returns ``predicted_answer=None`` to signal that.

An optional OpenRouter adapter is provided for offline benchmark runs. It is
gated behind the ``reader-openrouter`` extra (which declares ``requests``) and is
only used when explicitly selected — never by default.

House style mirrors ``ExtractorFn`` in ``on_new_turn.py``: a plain Callable type
alias, no ABCs.
"""
from __future__ import annotations

import os
from collections.abc import Callable

# Maps a fully-formed prompt to an answer string, or None when no in-process
# reader runs (the host model answers from the served context instead).
Reader = Callable[[str], "str | None"]


def null_reader(prompt: str) -> None:
    """Default reader: no in-process LLM. The host model answers from context."""
    return None


def openrouter_reader(prompt: str) -> str:
    """Optional OpenRouter adapter (single user message).

    Requires the ``reader-openrouter`` extra and ``MEMCONTEXT_READER_API_KEY``.
    Never used by default; selected only via ``--reader configured``.
    """
    import requests

    api_key = os.environ.get("MEMCONTEXT_READER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "MEMCONTEXT_READER_API_KEY not set. "
            "Export it before running with --reader configured."
        )

    model = os.environ.get("MEMCONTEXT_READER_MODEL", "openai/gpt-5-mini")
    endpoint = os.environ.get(
        "MEMCONTEXT_READER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
    )
    resp = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    return (content or "").strip()


def openrouter_reader_with_system(system: str, user: str) -> str:
    """OpenRouter adapter with a system+user split (baseline style)."""
    import requests

    api_key = os.environ.get("MEMCONTEXT_READER_API_KEY", "")
    if not api_key:
        raise ValueError("MEMCONTEXT_READER_API_KEY not set.")

    model = os.environ.get("MEMCONTEXT_READER_MODEL", "openai/gpt-5-mini")
    endpoint = os.environ.get(
        "MEMCONTEXT_READER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
    )
    resp = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "X-Title": "memcontext"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system, "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": user},
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    return (content or "").strip()


def get_reader(mode: str) -> Reader:
    """Resolve a reader by mode name.

    ``none``/``host`` -> the host model (``null_reader``, no network);
    ``configured``/``openrouter`` -> the OpenRouter adapter.
    """
    if mode in ("none", "host"):
        return null_reader
    if mode in ("configured", "openrouter"):
        return openrouter_reader
    raise ValueError(f"Unknown reader mode: {mode!r}")
