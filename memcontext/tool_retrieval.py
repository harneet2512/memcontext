"""Memory-conditioned tool retrieval — the falsification candidate.

Two conditions, one fusion path (``tool_registry.fuse_channels``):

* **Condition A — query-only** (baseline): semantic + BM25 over the registry.
  This is the established query-only tool-retrieval setting that external work
  such as RAG-MCP (arXiv:2505.03275) also occupies; it is the *baseline* here,
  not MemContext's architecture.
* **Condition B — memory-conditioned** (candidate): the same query channels PLUS
  deterministic features derived from the user's persistent memory, consumed
  *only* through Session-1 public surfaces (``retrieve_memory_across``). Memory
  contributes extra RRF channels — it never overrides the query.

Hard constraint: **zero LLM** in conditioning, ranking, or scoring. Every feature
is a token overlap, a cosine, or an RRF rank. The agent still chooses the tool;
this module only curates/ranks the candidate set.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field

from memcontext.retrieval import (
    EmbeddingClient,
    bm25_over_docs,
    retrieve_memory_across,
    tokenize_for_bm25,
)
from memcontext.supersession_semantic import Embedder
from memcontext.tool_registry import (
    ToolCandidate,
    ToolIndex,
    ToolResult,
    bm25_scores_indexed,
    boost_scores_indexed,
    fuse_channels,
    semantic_scores,
    semantic_scores_indexed,
)

# Default channel weights. Query channels mirror rank_query_only; memory channels
# are deliberately secondary (they augment, never dominate, the query signal).
DEFAULT_QUERY_WEIGHTS: tuple[float, float] = (0.5, 0.5)  # semantic, bm25
DEFAULT_MEMORY_WEIGHTS: tuple[float, float, float] = (0.3, 0.3, 0.2)  # semantic, bm25, boost


@dataclass(frozen=True, slots=True)
class MemoryConditioning:
    """Deterministic, LLM-free conditioning features distilled from memory.

    ``query_terms`` augment BM25; ``boost_terms`` drive an entity/domain overlap
    channel; ``memory_embedding`` drives a second semantic channel. ``provenance``
    records which memory hits contributed (for the leakage audit + debug).
    """

    query_terms: tuple[str, ...]
    boost_terms: tuple[str, ...]
    memory_text: str
    memory_embedding: list[float] | None
    weight: float = 1.0
    provenance: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.memory_text.strip()


# Common words carry ~0 BM25 IDF over a large tool corpus but have huge postings
# lists — dropping them makes memory_bm25 ~10× cheaper with negligible score change.
_STOPWORD_TEXT = (
    "a an and the of to for in on at by with from is are was were be been being "
    "i you he she it we they me my your our their this that these those as or if "
    "then than so do does did done have has had will would can could should may "
    "might must not no yes about into over under out up down off again earlier "
    "task worked working work used use using need needs want wants get got make "
    "please now today yesterday some any all more most"
)
_STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())

_MAX_MEMORY_TERMS: int = 24


def _dedupe(tokens: list[str], *, cap: int) -> tuple[str, ...]:
    """Dedupe, drop stopwords + 1-char tokens, preserve order, cap length."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t in seen or t in _STOPWORDS or len(t) <= 1:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= cap:
            break
    return tuple(out)


def build_memory_conditioning(
    conn: sqlite3.Connection,
    *,
    session_ids: Sequence[str],
    query: str,
    embedder: EmbeddingClient | Embedder | None = None,
    top_k: int = 10,
    max_terms: int = _MAX_MEMORY_TERMS,
    weight: float = 1.0,
) -> MemoryConditioning:
    """Distil conditioning features from memory via the Session-1 public surface.

    Uses ``retrieve_memory_across`` to pull the memory most relevant to the query
    across the user's prior sessions, then converts the returned text into
    deterministic features. Using the query to *select* memory is legitimate;
    leakage concerns are about profile *content*, enforced separately by the
    leakage audit (Phase 3).
    """
    if not session_ids:
        return MemoryConditioning((), (), "", None, weight, ())

    # retrieve_memory_across expects a concrete EmbeddingClient type; Embedder
    # stubs (tests) duck-type the same .embed() contract.
    client = embedder  # type: ignore[assignment]
    hits = retrieve_memory_across(
        conn,
        session_ids=list(session_ids),
        query=query,
        top_k=top_k,
        embedding_client=client,  # type: ignore[arg-type]
    )
    texts = [hit.text for hit, _ in hits]
    memory_text = " ".join(texts)
    terms = _dedupe(tokenize_for_bm25(memory_text), cap=max_terms)
    memory_embedding: list[float] | None = None
    if embedder is not None and memory_text.strip():
        memory_embedding = embedder.embed([memory_text])[0]
    provenance = tuple(hit.id for hit, _ in hits)
    return MemoryConditioning(
        query_terms=terms,
        boost_terms=terms,
        memory_text=memory_text,
        memory_embedding=memory_embedding,
        weight=weight,
        provenance=provenance,
    )


def build_structured_conditioning(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    embedder: EmbeddingClient | Embedder | None = None,
    max_terms: int = _MAX_MEMORY_TERMS,
    weight: float = 1.0,
) -> MemoryConditioning:
    """Condition on the substrate's STRUCTURED world-state — not retrieved text.

    This is the real memory<->tool integration. It reads the user's *current*
    claims (``list_active_claims`` — supersession-aware, deduped), their extracted
    entities (``claim_entities``), and importance weights (``claim_metadata``),
    and distils importance-ranked terms from claim **values + entities** (the
    substrate's curated model of who the user is and what domains/tools they
    work with). It deliberately does NOT use raw conversational/episode text and
    is independent of lexical similarity to the current query — so it biases the
    tool set toward the user's actual domains, not toward query-similar noise.
    Zero LLM.
    """
    from memcontext.claims import list_active_claims

    claims = list_active_claims(conn, session_id)
    if not claims:
        return MemoryConditioning((), (), "", None, weight, ())
    ids = [c.claim_id for c in claims]
    ph = ",".join("?" * len(ids))
    importance: dict[str, float] = {
        r["claim_id"]: float(r["imp"])
        for r in conn.execute(
            f"SELECT claim_id, COALESCE(importance_score, 0.5) AS imp"
            f" FROM claim_metadata WHERE claim_id IN ({ph})",
            ids,
        ).fetchall()
    }
    entities: dict[str, list[str]] = {}
    for r in conn.execute(
        f"SELECT claim_id, entity_text FROM claim_entities WHERE claim_id IN ({ph})", ids
    ).fetchall():
        entities.setdefault(r["claim_id"], []).append(r["entity_text"])

    # term -> max importance across the claims that produced it (current truth only).
    scored: dict[str, float] = {}
    summary: list[str] = []
    for c in claims:
        w = importance.get(c.claim_id, 0.5)
        struct_text = " ".join(
            x for x in (c.value, c.value_normalised, *entities.get(c.claim_id, [])) if x
        )
        summary.append(struct_text or (c.text or ""))
        for tok in tokenize_for_bm25(struct_text):
            if tok in _STOPWORDS or len(tok) <= 1:
                continue
            scored[tok] = max(scored.get(tok, 0.0), w)
    terms = tuple(t for t, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:max_terms])
    memory_text = " ".join(summary)
    memory_embedding = (
        embedder.embed([memory_text])[0] if embedder is not None and memory_text.strip() else None
    )
    return MemoryConditioning(
        query_terms=terms,
        boost_terms=terms,
        memory_text=memory_text,
        memory_embedding=memory_embedding,
        weight=weight,
        provenance=tuple(ids),
    )


def build_memory_instruction(
    conn: sqlite3.Connection, *, session_id: str, max_chars: int = 600
) -> str:
    """Deterministically synthesize an instruction from the user's structured memory.

    The research-backed lever for tool retrieval is the *instruction-augmented
    query* (ToolRet): prepend contextual guidance and let the retriever use it.
    ToolRet generates that instruction with GPT-4o; MemContext generates it
    **deterministically and provenance-backed** from the substrate's current
    claims (``list_active_claims`` — supersession-aware) — zero LLM. Preferences
    are surfaced first, then facts, importance-ordered, deduped.

    Returns "" when there is no memory (caller falls back to query-only).
    """
    from memcontext.claims import list_active_claims

    claims = list_active_claims(conn, session_id)
    if not claims:
        return ""
    ids = [c.claim_id for c in claims]
    ph = ",".join("?" * len(ids))
    importance: dict[str, float] = {
        r["claim_id"]: float(r["imp"])
        for r in conn.execute(
            f"SELECT claim_id, COALESCE(importance_score, 0.5) AS imp"
            f" FROM claim_metadata WHERE claim_id IN ({ph})",
            ids,
        ).fetchall()
    }
    # Preferences before facts; then by importance desc; dedupe values.
    def _key(c: object) -> tuple[int, float]:
        pred = getattr(c, "predicate", "") or ""
        return (0 if "preference" in pred else 1, -importance.get(getattr(c, "claim_id", ""), 0.5))

    seen: set[str] = set()
    parts: list[str] = []
    for c in sorted(claims, key=_key):
        val = (c.value or c.text or "").strip()
        if val and val.lower() not in seen:
            seen.add(val.lower())
            parts.append(val)
    if not parts:
        return ""
    instruction = "User context: " + "; ".join(parts) + "."
    return instruction[:max_chars]


def conditioning_from_facts(
    facts: Sequence[str],
    *,
    memory_embedding: list[float] | None = None,
    weight: float = 1.0,
    max_terms: int = _MAX_MEMORY_TERMS,
) -> MemoryConditioning:
    """Build conditioning directly from in-memory profile facts (no DB).

    Equivalent to ``build_memory_conditioning`` for a small profile session (where
    ``retrieve_memory_across`` returns all facts) but with no SQLite round-trip, so
    it is picklable and safe to call inside worker processes. ``memory_embedding``
    is precomputed by the caller (batched) to keep heavy models out of workers.
    The product path still uses ``build_memory_conditioning`` over the live
    substrate; this is the benchmark's fast/parallel equivalent.
    """
    memory_text = " ".join(facts)
    terms = _dedupe(tokenize_for_bm25(memory_text), cap=max_terms)
    return MemoryConditioning(
        query_terms=terms,
        boost_terms=terms,
        memory_text=memory_text,
        memory_embedding=memory_embedding,
        weight=weight,
        provenance=tuple(facts),
    )


def _boost_scores(
    boost_terms: tuple[str, ...], candidates: Sequence[ToolCandidate]
) -> list[float]:
    """Count distinct boost terms present in each candidate's document tokens."""
    bset = set(boost_terms)
    if not bset:
        return [0.0] * len(candidates)
    return [float(len(bset & set(c.doc_tokens))) for c in candidates]


def retrieve_tools(
    candidates: Sequence[ToolCandidate],
    *,
    query: str,
    query_embedding: list[float] | None,
    conditioning: MemoryConditioning | None = None,
    top_k: int = 10,
    query_weights: tuple[float, float] = DEFAULT_QUERY_WEIGHTS,
    memory_weights: tuple[float, float, float] = DEFAULT_MEMORY_WEIGHTS,
    index: ToolIndex | None = None,
) -> list[ToolResult]:
    """Rank tools. ``conditioning is None`` → condition A; else condition B.

    Condition B adds up to three memory channels (semantic / bm25 / boost) on top
    of the identical query channels, fused through the same RRF arithmetic. Pass a
    prebuilt ``index`` (``build_tool_index``) for the fast inverted-index path —
    scores are identical to the linear path; only the candidate set must match.
    """
    if not candidates:
        return []
    docs = [c.doc_tokens for c in candidates]

    def _bm25(tokens: list[str]) -> list[float]:
        if index is not None:
            return bm25_scores_indexed(index, tokens)
        return bm25_over_docs(tokens, docs)

    def _sem(vec: list[float] | None) -> list[float]:
        if index is not None:
            return semantic_scores_indexed(index, vec)
        return semantic_scores(vec, candidates)

    bm25_q = _bm25(tokenize_for_bm25(query))
    sem_q = _sem(query_embedding)
    w_sem, w_bm25 = query_weights
    if not any(s != 0.0 for s in sem_q):
        w_sem = 0.0
    channels: list[tuple[str, float, list[float]]] = [
        ("semantic", w_sem, sem_q),
        ("bm25", w_bm25, bm25_q),
    ]

    if conditioning is not None and conditioning.weight > 0.0 and not conditioning.is_empty:
        wm_sem, wm_bm25, wm_boost = memory_weights
        scale = conditioning.weight
        if conditioning.query_terms:
            bm25_m = _bm25(list(conditioning.query_terms))
            channels.append(("memory_bm25", wm_bm25 * scale, bm25_m))
        sem_m = _sem(conditioning.memory_embedding)
        if any(s != 0.0 for s in sem_m):
            channels.append(("memory_semantic", wm_sem * scale, sem_m))
        if index is not None:
            boost = boost_scores_indexed(index, conditioning.boost_terms)
        else:
            boost = _boost_scores(conditioning.boost_terms, candidates)
        if any(b != 0.0 for b in boost):
            channels.append(("memory_boost", wm_boost * scale, boost))

    return fuse_channels(candidates, channels, top_k=top_k)
