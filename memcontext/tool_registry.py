"""Tool registry — a schema-independent catalog of tools, parallel to memory.

Session 2 (tool/activation track). This module is *additive* and never touches
claim/turn/memory tables. It stores tool schemas (from ToolRet, an MCP
``tools/list`` fixture, or a local fixture), embeds them with the *same* local
embedder the memory substrate uses, and ranks them query-only.

Reuse, not reinvention: the ranking path reuses the substrate's retrieval
primitives verbatim — ``_rrf_ranks``, ``_bm25_over_docs``, ``_tokenize_for_bm25``,
``RRF_K`` (Reciprocal Rank Fusion, k=60) and the ``_encode_vector`` /
``_decode_vector`` blob codec. Only the per-channel feature computations are
tool-specific (they operate on tool documents instead of claims), so no parallel
retrieval engine is introduced.

Zero LLM anywhere in this module — formatting, embedding text, and ranking are
fully deterministic.
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from memcontext.claims import now_ns
from memcontext.retrieval import (
    BGE_M3_VERSION_TAG,
    RRF_K,
    bm25_over_docs,
    decode_vector,
    encode_vector,
    rrf_ranks,
    tokenize_for_bm25,
)
from memcontext.supersession_semantic import Embedder, cosine

# ----------------------------------------------------------------- data model ---


def _empty_metadata() -> dict[str, Any]:
    return {}


@dataclass(frozen=True, slots=True)
class ToolDoc:
    """One tool in the registry, source-faithful.

    Raw benchmark/MCP fields are preserved (``parameters``, ``returns``,
    ``metadata``) so external relevance labels are never destroyed. ``id`` is a
    deterministic key derived from ``(source, source_dataset, source_tool_id)``
    so re-ingest is an idempotent upsert.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] | None = None
    returns: dict[str, Any] | None = None
    parent_mcp_server: str | None = None
    server_url_or_id: str | None = None
    domain_tags: tuple[str, ...] = ()
    source: str = "local"
    source_dataset: str | None = None
    source_tool_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)

    @property
    def id(self) -> str:
        return make_tool_id(self.source, self.source_dataset, self.source_tool_id, self.name)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A ranked tool with its fused score and per-channel contributions."""

    tool_id: str
    name: str
    score: float
    components: dict[str, float]


def make_tool_id(
    source: str,
    source_dataset: str | None,
    source_tool_id: str | None,
    name: str,
) -> str:
    """Deterministic registry key. Falls back to ``name`` when no source id."""
    parts = [source, source_dataset or "", source_tool_id or name]
    return ":".join(p.replace(":", "_") for p in parts)


# ------------------------------------------------------- deterministic document ---


def format_tool_document(tool: ToolDoc) -> str:
    """Deterministic NL document for embedding + BM25.

    Sections, fixed order: name, description, parameters (sorted by name),
    return shape, domain tags, parent server, source metadata, and example usage
    *only if present in the source*. No example is ever synthesized.
    """
    lines: list[str] = [f"name: {tool.name}"]
    if tool.description:
        lines.append(f"description: {tool.description}")

    params: dict[str, Any] = tool.parameters or {}
    # MCP/JSON-schema style nests under "properties"; otherwise params is flat.
    props_obj: object = params.get("properties", params)
    if isinstance(props_obj, dict) and props_obj:
        props = cast(dict[str, Any], props_obj)
        lines.append("parameters:")
        for pname in sorted(props):
            spec = props[pname]
            if isinstance(spec, dict):
                spec_d = cast(dict[str, Any], spec)
                ptype = str(spec_d.get("type", ""))
                pdesc = str(spec_d.get("description", ""))
                suffix = f" ({ptype})" if ptype else ""
                detail = f" — {pdesc}" if pdesc else ""
                lines.append(f"  - {pname}{suffix}{detail}")
            else:
                lines.append(f"  - {pname}: {spec}")

    if tool.returns:
        lines.append(f"returns: {json.dumps(tool.returns, sort_keys=True)}")
    if tool.domain_tags:
        lines.append(f"domain: {', '.join(tool.domain_tags)}")
    if tool.parent_mcp_server:
        lines.append(f"server: {tool.parent_mcp_server}")

    # Example usage only if the source dataset carried one.
    example = tool.metadata.get("example") or tool.metadata.get("example_code")
    if isinstance(example, str) and example.strip():
        lines.append(f"example: {example.strip()}")

    return "\n".join(lines)


# -------------------------------------------------------------------- ingest ---


def upsert_tools(
    conn: sqlite3.Connection,
    tools: Iterable[ToolDoc],
    *,
    now: int | None = None,
) -> int:
    """Idempotent batch upsert into ``tool_schemas``. Returns rows written.

    Conflict on the deterministic ``id`` updates the mutable fields and bumps
    ``updated_at`` while preserving ``created_at`` — re-ingesting the same corpus
    is a no-op on row count and never duplicates.
    """
    ts = now if now is not None else now_ns()
    count = 0
    for tool in tools:
        conn.execute(
            "INSERT INTO tool_schemas"
            " (id, name, description, parameters_json, returns_json,"
            "  parent_mcp_server, server_url_or_id, domain_tags, source,"
            "  source_dataset, source_tool_id, metadata_json, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET"
            "  name=excluded.name, description=excluded.description,"
            "  parameters_json=excluded.parameters_json,"
            "  returns_json=excluded.returns_json,"
            "  parent_mcp_server=excluded.parent_mcp_server,"
            "  server_url_or_id=excluded.server_url_or_id,"
            "  domain_tags=excluded.domain_tags, source=excluded.source,"
            "  source_dataset=excluded.source_dataset,"
            "  source_tool_id=excluded.source_tool_id,"
            "  metadata_json=excluded.metadata_json, updated_at=excluded.updated_at",
            (
                tool.id,
                tool.name,
                tool.description,
                json.dumps(tool.parameters) if tool.parameters is not None else None,
                json.dumps(tool.returns) if tool.returns is not None else None,
                tool.parent_mcp_server,
                tool.server_url_or_id,
                json.dumps(list(tool.domain_tags)) if tool.domain_tags else None,
                tool.source,
                tool.source_dataset,
                tool.source_tool_id,
                json.dumps(tool.metadata) if tool.metadata else None,
                ts,
                ts,
            ),
        )
        count += 1
    return count


def _row_to_tooldoc(row: sqlite3.Row) -> ToolDoc:
    return ToolDoc(
        name=row["name"],
        description=row["description"] or "",
        parameters=json.loads(row["parameters_json"]) if row["parameters_json"] else None,
        returns=json.loads(row["returns_json"]) if row["returns_json"] else None,
        parent_mcp_server=row["parent_mcp_server"],
        server_url_or_id=row["server_url_or_id"],
        domain_tags=tuple(json.loads(row["domain_tags"])) if row["domain_tags"] else (),
        source=row["source"],
        source_dataset=row["source_dataset"],
        source_tool_id=row["source_tool_id"],
        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
    )


def load_tools(
    conn: sqlite3.Connection,
    *,
    source: str | None = None,
    source_dataset: str | None = None,
) -> list[ToolDoc]:
    """Load registry tools, optionally filtered by source / dataset, id-ordered."""
    sql = "SELECT * FROM tool_schemas"
    clauses: list[str] = []
    params: list[str] = []
    if source is not None:
        clauses.append("source = ?")
        params.append(source)
    if source_dataset is not None:
        clauses.append("source_dataset = ?")
        params.append(source_dataset)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id"
    return [_row_to_tooldoc(r) for r in conn.execute(sql, params).fetchall()]


def count_tools(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM tool_schemas").fetchone()[0])


def count_tool_embeddings(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM tool_embeddings").fetchone()[0])


# ------------------------------------------------------------------ embedding ---


def _embedder_version(embedder: Embedder) -> str:
    return getattr(embedder, "model_version", BGE_M3_VERSION_TAG)


def embed_tools(
    conn: sqlite3.Connection,
    *,
    embedder: Embedder,
    batch_size: int = 64,
    reindex: bool = False,
    now: int | None = None,
) -> int:
    """Embed registry tools with the shared local embedder. Returns rows written.

    Skips tools that already have an embedding unless ``reindex=True``. The
    embedded text is the deterministic ``format_tool_document`` output and is
    stored alongside the vector for reproducibility/debug.
    """
    ts = now if now is not None else now_ns()
    version = _embedder_version(embedder)

    if reindex:
        rows = conn.execute("SELECT * FROM tool_schemas ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT s.* FROM tool_schemas s"
            " LEFT JOIN tool_embeddings e ON e.tool_id = s.id"
            " WHERE e.tool_id IS NULL ORDER BY s.id"
        ).fetchall()

    written = 0
    batch: list[tuple[str, str]] = []  # (tool_id, embedded_text)

    def _flush() -> None:
        nonlocal written
        if not batch:
            return
        vectors = embedder.embed([txt for _, txt in batch])
        for (tool_id, txt), vec in zip(batch, vectors, strict=True):
            conn.execute(
                "INSERT INTO tool_embeddings"
                " (tool_id, embedding_model, embedding, embedded_text, created_at)"
                " VALUES (?,?,?,?,?)"
                " ON CONFLICT(tool_id) DO UPDATE SET"
                "  embedding_model=excluded.embedding_model,"
                "  embedding=excluded.embedding,"
                "  embedded_text=excluded.embedded_text,"
                "  created_at=excluded.created_at",
                (tool_id, version, encode_vector(vec), txt, ts),
            )
            written += 1
        batch.clear()

    for row in rows:
        batch.append((row["id"], format_tool_document(_row_to_tooldoc(row))))
        if len(batch) >= batch_size:
            _flush()
    _flush()
    return written


# ----------------------------------------------------------- query-only rank ---


@dataclass(frozen=True, slots=True)
class ToolCandidate:
    tool_id: str
    name: str
    doc_tokens: list[str]
    embedding: list[float] | None


def load_candidates(
    conn: sqlite3.Connection,
    *,
    source: str | None = None,
    source_dataset: str | None = None,
) -> list[ToolCandidate]:
    """Materialise candidates (tokens + embeddings) once for repeated scoring."""
    sql = (
        "SELECT s.id AS id, s.name AS name, s.description AS description,"
        " s.parameters_json, s.returns_json, s.parent_mcp_server,"
        " s.server_url_or_id, s.domain_tags, s.source, s.source_dataset,"
        " s.source_tool_id, s.metadata_json, e.embedding AS embedding"
        " FROM tool_schemas s LEFT JOIN tool_embeddings e ON e.tool_id = s.id"
    )
    clauses: list[str] = []
    params: list[str] = []
    if source is not None:
        clauses.append("s.source = ?")
        params.append(source)
    if source_dataset is not None:
        clauses.append("s.source_dataset = ?")
        params.append(source_dataset)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY s.id"

    out: list[ToolCandidate] = []
    for row in conn.execute(sql, params).fetchall():
        doc = format_tool_document(_row_to_tooldoc(row))
        emb = decode_vector(row["embedding"]) if row["embedding"] is not None else None
        out.append(
            ToolCandidate(
                tool_id=row["id"],
                name=row["name"],
                doc_tokens=tokenize_for_bm25(doc),
                embedding=emb,
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class ToolIndex:
    """Precomputed retrieval index over a candidate set — product-grade hot path.

    Replaces O(N) BM25 rescans with an inverted index (postings + df) that only
    touches docs containing a query token, and stores an embedding matrix for
    vectorized cosine. BM25 scores are numerically identical to
    ``bm25_over_docs`` (same k1/b/idf, duplicate query tokens counted with
    multiplicity); cosine matches ``semantic_scores`` for L2-normalized vectors.
    """

    candidates: tuple[ToolCandidate, ...]
    doc_len: tuple[int, ...]
    avgdl: float
    df: dict[str, int]
    postings: dict[str, list[tuple[int, int]]]  # token -> [(doc_idx, term_freq)]
    emb_matrix: Any  # numpy float32 (N x dim) or None when no embeddings present
    doc_token_sets: tuple[frozenset[str], ...]  # per-doc unique tokens (boost channel)

    def __len__(self) -> int:
        return len(self.candidates)


def build_tool_index(candidates: Sequence[ToolCandidate]) -> ToolIndex:
    """Build an inverted index + embedding matrix once, reuse across many queries."""
    cands = tuple(candidates)
    n = len(cands)
    doc_len = tuple(len(c.doc_tokens) for c in cands)
    avgdl = sum(doc_len) / max(n, 1)
    postings: dict[str, list[tuple[int, int]]] = {}
    doc_token_sets: list[frozenset[str]] = []
    for i, c in enumerate(cands):
        tf_map: dict[str, int] = {}
        for tok in c.doc_tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        for tok, tf in tf_map.items():
            postings.setdefault(tok, []).append((i, tf))
        doc_token_sets.append(frozenset(tf_map))
    df = {tok: len(plist) for tok, plist in postings.items()}

    emb_matrix: Any = None
    dim = next((len(c.embedding) for c in cands if c.embedding is not None), 0)
    if dim:
        try:
            import numpy as np

            mat = np.zeros((n, dim), dtype=np.float32)
            for i, c in enumerate(cands):
                if c.embedding is not None:
                    mat[i] = c.embedding
            emb_matrix = mat
        except ImportError:  # pragma: no cover - numpy is a transitive dep
            emb_matrix = None
    return ToolIndex(cands, doc_len, avgdl, df, postings, emb_matrix, tuple(doc_token_sets))


def bm25_scores_indexed(index: ToolIndex, query_tokens: list[str]) -> list[float]:
    """BM25 over the inverted index — identical formula to ``bm25_over_docs``."""
    n = len(index)
    scores = [0.0] * n
    if not query_tokens or n == 0:
        return scores
    k1, b = 1.2, 0.75
    avgdl = max(index.avgdl, 1e-9)
    qcounts: dict[str, int] = {}
    for qt in query_tokens:
        qcounts[qt] = qcounts.get(qt, 0) + 1
    for tok, count in qcounts.items():
        dfreq = index.df.get(tok, 0)
        if dfreq == 0:
            continue
        idf = math.log((n - dfreq + 0.5) / (dfreq + 0.5) + 1.0)
        for doc_idx, tf in index.postings[tok]:
            dl = index.doc_len[doc_idx]
            scores[doc_idx] += count * idf * tf * (k1 + 1.0) / (
                tf + k1 * (1.0 - b + b * dl / avgdl)
            )
    return scores


def boost_scores_indexed(index: ToolIndex, boost_terms: Sequence[str]) -> list[float]:
    """Count distinct boost terms present per doc, using precomputed token sets."""
    bset = set(boost_terms)
    if not bset:
        return [0.0] * len(index)
    return [float(len(bset & dts)) for dts in index.doc_token_sets]


def semantic_scores_indexed(
    index: ToolIndex, query_embedding: list[float] | None
) -> list[float]:
    """Vectorized cosine (matrix @ normalized query) over the embedding matrix."""
    n = len(index)
    if query_embedding is None or index.emb_matrix is None:
        return [0.0] * n
    import numpy as np

    q = np.asarray(query_embedding, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        return [0.0] * n
    q = q / norm
    # Candidate rows are L2-normalized (or zero for missing) → dot == cosine.
    return [float(x) for x in (index.emb_matrix @ q)]


def fuse_channels(
    candidates: Sequence[ToolCandidate],
    channels: list[tuple[str, float, list[float]]],
    *,
    top_k: int,
) -> list[ToolResult]:
    """Fuse named, weighted score channels via RRF — the single fusion path.

    Each channel is ``(name, weight, per-candidate scores)``. Scores become ranks
    (``rrf_ranks``) and contribute ``weight / (RRF_K + rank)``. Per-channel
    contributions are kept on each ``ToolResult`` for ranking observability.
    Shared by query-only (condition A) and memory-conditioned (condition B) so the
    arithmetic is provably identical across conditions.
    """
    if not candidates:
        return []
    n = len(candidates)
    ranked = [(name, weight, rrf_ranks(scores)) for name, weight, scores in channels]
    # Accumulate fused scores in a flat array — no per-candidate dict/object.
    fused_scores = [0.0] * n
    for _name, weight, ranks in ranked:
        for i in range(n):
            fused_scores[i] += weight / (RRF_K + ranks[i])
    # Select only the top_k (score desc, tie-break tool_id) — build ToolResult,
    # with per-channel contributions, for those alone (not all N candidates).
    order = sorted(range(n), key=lambda i: (-fused_scores[i], candidates[i].tool_id))[:top_k]
    out: list[ToolResult] = []
    for i in order:
        contrib = {name: weight / (RRF_K + ranks[i]) for name, weight, ranks in ranked}
        contrib["final"] = fused_scores[i]
        out.append(ToolResult(candidates[i].tool_id, candidates[i].name, fused_scores[i], contrib))
    return out


def semantic_scores(
    query_embedding: list[float] | None, candidates: Sequence[ToolCandidate]
) -> list[float]:
    """Cosine of the query vector against each candidate's tool embedding."""
    if query_embedding is None:
        return [0.0] * len(candidates)
    return [
        cosine(query_embedding, c.embedding) if c.embedding is not None else 0.0
        for c in candidates
    ]


def rank_query_only(
    candidates: Sequence[ToolCandidate],
    *,
    query: str,
    query_embedding: list[float] | None,
    top_k: int = 10,
    w_semantic: float = 0.5,
    w_bm25: float = 0.5,
) -> list[ToolResult]:
    """Query-only RRF over the registry: semantic + BM25 (the baseline condition)."""
    if not candidates:
        return []
    bm25 = bm25_over_docs(tokenize_for_bm25(query), [c.doc_tokens for c in candidates])
    sem = semantic_scores(query_embedding, candidates)
    w_sem = w_semantic if any(s != 0.0 for s in sem) else 0.0
    return fuse_channels(
        candidates,
        [("semantic", w_sem, sem), ("bm25", w_bm25, bm25)],
        top_k=top_k,
    )


def tooldoc_from_mcp(record: dict[str, Any], *, server: str) -> ToolDoc:
    """Map a native MCP tool definition (``{name, description, inputSchema}``)."""
    return ToolDoc(
        name=str(record["name"]),
        description=str(record.get("description", "")),
        parameters=record.get("inputSchema"),
        parent_mcp_server=server,
        source="mcp",
        source_dataset=server,
        source_tool_id=str(record["name"]),
        metadata={"raw_mcp": record},
    )


def ingest_mcp_tools_list(
    conn: sqlite3.Connection,
    data: Any,
    *,
    server: str = "mcp",
    now: int | None = None,
) -> int:
    """Ingest an MCP ``tools/list`` payload (``{"tools": [...]}`` or a bare list)."""
    raw = data.get("tools") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("MCP tools/list must be a list or contain a 'tools' list")
    tools = cast("list[dict[str, Any]]", raw)
    return upsert_tools(conn, [tooldoc_from_mcp(t, server=server) for t in tools], now=now)
