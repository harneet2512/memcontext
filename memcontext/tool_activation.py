"""Activation layer — serve a curated, ranked tool set for a query.

This is the product surface that sits on top of the memory substrate. Given a
query (and, optionally, the user's memory sessions) it returns the top-K relevant
tools from the registry instead of dumping the whole toolset at the agent —
reducing prompt bloat and sharpening tool selection. The agent still chooses the
tool; this only *curates the candidate set*.

Two modes:
* **query-only** (default): semantic + BM25 over the registry. Standalone value,
  independent of memory.
* **memory-conditioned** (``use_memory=True``): additionally conditions on the
  user's persistent memory, consumed **only** through the Session-1 public surface
  ``retrieve_memory_across`` (via ``build_memory_conditioning``) — so the substrate
  is wired in *by construction*, never bypassed. This mode is **off by default**
  pending a valid offline validation that it improves retrieval.

Zero LLM. Deterministic.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from memcontext.retrieval import EmbeddingClient
from memcontext.supersession_semantic import Embedder
from memcontext.tool_registry import build_tool_index, load_candidates
from memcontext.tool_retrieval import (
    build_memory_conditioning,
    build_memory_instruction,
    build_structured_conditioning,
    retrieve_tools,
)

EmbedderLike = EmbeddingClient | Embedder


@dataclass(frozen=True, slots=True)
class DiscoveredTool:
    """One curated tool returned by discovery."""

    tool_id: str
    name: str
    score: float
    components: dict[str, float]  # per-channel RRF contributions (observability)
    used_memory: bool


def discover_tools(
    conn: sqlite3.Connection,
    *,
    query: str,
    session_ids: Sequence[str] = (),
    top_k: int = 10,
    use_memory: bool = False,
    memory_mode: str = "instruction",
    embedder: EmbedderLike | None = None,
    memory_top_k: int = 10,
    source: str | None = None,
    source_dataset: str | None = None,
) -> list[DiscoveredTool]:
    """Return the top-K registry tools for ``query``, ranked.

    ``use_memory=True`` conditions ranking on the user's memory. Modes:
    * ``"instruction"`` (default, research-backed) — prepend a **deterministic
      instruction** synthesized from the substrate's structured memory
      (``build_memory_instruction``) to the query, then retrieve query-only over
      the augmented query. This is ToolRet's proven instruction-augmentation lever,
      with a zero-LLM, provenance-backed instruction. Benefits most from an
      instruction-tuned retriever (set ``MEMCONTEXT_EMBED_MODEL``).
    * ``"structured"`` — additive RRF channels from structured claims.
    * ``"text"`` — legacy: query-retrieved memory text channels (weakest).
    ``embedder=None`` runs BM25-only; pass the substrate embedder for semantics.
    """
    candidates = load_candidates(conn, source=source, source_dataset=source_dataset)
    if not candidates:
        return []
    index = build_tool_index(candidates)

    effective_query = query
    conditioning = None
    memory_used = False
    if use_memory and session_ids:
        if memory_mode == "instruction":
            instruction = build_memory_instruction(conn, session_id=session_ids[0])
            if instruction:
                effective_query = f"{instruction}\n{query}"
                memory_used = True
        elif memory_mode == "structured":
            conditioning = build_structured_conditioning(
                conn, session_id=session_ids[0], embedder=embedder
            )
            memory_used = not conditioning.is_empty
        else:
            conditioning = build_memory_conditioning(
                conn, session_ids=list(session_ids), query=query,
                embedder=embedder, top_k=memory_top_k,
            )
            memory_used = not conditioning.is_empty

    query_embedding = embedder.embed([effective_query])[0] if embedder is not None else None
    results = retrieve_tools(
        candidates,
        query=effective_query,
        query_embedding=query_embedding,
        conditioning=conditioning,
        top_k=top_k,
        index=index,
    )
    return [
        DiscoveredTool(r.tool_id, r.name, r.score, r.components, memory_used) for r in results
    ]
