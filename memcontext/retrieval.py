"""Query-conditional retrieval over the claim store.

Design:
- Sidecar embeddings table, not inline on claims.
- bge-m3 (MIT) as the default embedding model. Falls back to local
  sentence-transformers when MODAL_BGE_M3_URL is unset.
- Normalize embeddings so cosine reduces to dot product.
- Filter ordering: supersession-active set first, then embed + rank.
- Top-k default 20.

Multi-signal retrieval (retrieve_hybrid):
- Combines semantic (cosine), entity (match on entity_key), temporal
  (recency), and BM25 signals via Reciprocal Rank Fusion (k=60).

Temporal-window event-tuple retrieval (retrieve_event_tuples):
- Projects claims onto EventTuples with temporal validity windows.
- Optional valid_at_ts filter for point-in-time queries.
"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from memcontext.claims import list_active_claims
from memcontext.event_tuples import EventTuple, claim_to_event
from memcontext.schema import Claim

log = structlog.get_logger(__name__)


# --- constants ---------------------------------------------------------------

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_MODEL_REVISION = "main"
BGE_M3_VERSION_TAG = f"{BGE_M3_MODEL_ID}@{BGE_M3_MODEL_REVISION}"
BGE_M3_EMBED_DIM = 1024

MODAL_URL_ENV = "MODAL_BGE_M3_URL"
CACHE_DIR_ENV = "SUBSTRATE_EMBED_CACHE_DIR"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "substrate" / "embeddings"

DEFAULT_TOP_K: int = 20


# --- dataclasses -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RankedClaim:
    """A retrieved claim with its similarity score and vector version."""

    claim: Claim
    similarity_score: float
    embedding_model_version: str


# --- embedding client --------------------------------------------------------


class EmbeddingClient:
    """bge-m3 embedding client.

    Tries Modal first (HTTP POST). Falls back to local sentence-transformers
    when MODAL_BGE_M3_URL is unset.
    """

    def __init__(
        self,
        *,
        modal_url: str | None = None,
        model_version: str = BGE_M3_VERSION_TAG,
    ) -> None:
        self._modal_url = (modal_url or os.environ.get(MODAL_URL_ENV, "")).strip() or None
        self._model_version = model_version
        self._local_model: Any = None

    @property
    def model_version(self) -> str:
        return self._model_version

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._modal_url is not None:
            return self._embed_modal(texts)
        return self._embed_local(texts)

    def _embed_modal(self, texts: list[str]) -> list[list[float]]:
        try:
            import requests  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "requests package not importable; add requests to runtime deps "
                "or unset MODAL_BGE_M3_URL to use the local fallback."
            ) from exc
        url = self._modal_url.rstrip("/") + "/embed"  # type: ignore[union-attr]
        resp = requests.post(url, json={"texts": texts}, timeout=60.0)
        resp.raise_for_status()
        payload = resp.json()
        vectors = payload.get("embeddings") or []
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Modal endpoint returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        server_version = payload.get("model_version")
        if isinstance(server_version, str) and server_version:
            self._model_version = server_version
        return [list(map(float, v)) for v in vectors]

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "sentence_transformers is not installed; either set "
                "MODAL_BGE_M3_URL or install sentence-transformers."
            ) from exc
        if self._local_model is None:
            self._local_model = SentenceTransformer(
                BGE_M3_MODEL_ID, revision=BGE_M3_MODEL_REVISION
            )
        raw = self._local_model.encode(texts, normalize_embeddings=True)
        out: list[list[float]] = []
        for v in raw:
            out.append([float(x) for x in v])
        return out


# --- blob encoding -----------------------------------------------------------


def _encode_vector(vec: list[float]) -> bytes:
    n = len(vec)
    return struct.pack(f"<I{n}f", n, *vec)


def _decode_vector(blob: bytes) -> list[float]:
    if len(blob) < 4:
        raise ValueError("embedding blob shorter than length prefix")
    (n,) = struct.unpack_from("<I", blob, 0)
    expected = 4 + 4 * n
    if len(blob) != expected:
        raise ValueError(
            f"embedding blob length {len(blob)} does not match declared dim {n}"
        )
    return list(struct.unpack_from(f"<{n}f", blob, 4))


# --- cache ------------------------------------------------------------------


def _cache_path() -> Path:
    raw = os.environ.get(CACHE_DIR_ENV, "").strip()
    path = Path(raw) if raw else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_key(model_version: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model_version.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _cache_get(model_version: str, text: str) -> list[float] | None:
    path = _cache_path() / f"{_cache_key(model_version, text)}.bin"
    if not path.is_file():
        return None
    try:
        return _decode_vector(path.read_bytes())
    except ValueError:
        path.unlink(missing_ok=True)
        return None


def _cache_put(model_version: str, text: str, vec: list[float]) -> None:
    path = _cache_path() / f"{_cache_key(model_version, text)}.bin"
    path.write_bytes(_encode_vector(vec))


# --- claim → identity text --------------------------------------------------


def claim_retrieval_text(claim: Claim) -> str:
    """Canonical text fed to the embedder for a claim.

    Includes value (unlike identity_text in supersession_semantic which
    excludes value for identity matching).
    """
    return f"{claim.subject} {claim.predicate} {claim.value}"


# --- write-time embedder -----------------------------------------------------


def embed_and_store(
    conn: sqlite3.Connection,
    claim: Claim,
    *,
    client: EmbeddingClient | None = None,
) -> None:
    """Embed a claim and upsert its vector into the sidecar table.

    On failure, logs a warning and leaves the claim un-embedded so a
    backfill pass can pick it up later.
    """
    effective = client or EmbeddingClient()
    text = claim_retrieval_text(claim)
    try:
        cached = _cache_get(effective.model_version, text)
        if cached is not None:
            vec = cached
        else:
            vec = effective.embed([text])[0]
            _cache_put(effective.model_version, text, vec)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "substrate.embed_and_store_failed",
            claim_id=claim.claim_id,
            error=repr(exc)[:200],
        )
        return

    blob = _encode_vector(vec)
    conn.execute(
        "INSERT OR REPLACE INTO claim_embeddings "
        "(claim_id, embedding, embedding_model_version, embedded_at_unix) "
        "VALUES (?, ?, ?, ?)",
        (claim.claim_id, blob, effective.model_version, int(time.time())),
    )


def backfill_embeddings(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    client: EmbeddingClient | None = None,
) -> int:
    """Embed every active claim not already in the sidecar. Returns count."""
    effective = client or EmbeddingClient()
    active = list_active_claims(conn, session_id)
    missing = _filter_missing_embeddings(conn, active, effective.model_version)
    if not missing:
        return 0

    texts = [claim_retrieval_text(c) for c in missing]
    uncached_texts: list[str] = []
    uncached_claims: list[Claim] = []
    resolved: dict[str, list[float]] = {}
    for c, t in zip(missing, texts, strict=True):
        cached = _cache_get(effective.model_version, t)
        if cached is not None:
            resolved[c.claim_id] = cached
        else:
            uncached_texts.append(t)
            uncached_claims.append(c)

    if uncached_texts:
        try:
            vectors = effective.embed(uncached_texts)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "substrate.backfill_embed_failed",
                session_id=session_id,
                error=repr(exc)[:200],
            )
            return len(resolved)
        for c, t, v in zip(uncached_claims, uncached_texts, vectors, strict=True):
            _cache_put(effective.model_version, t, v)
            resolved[c.claim_id] = v

    now = int(time.time())
    for claim_id, vec in resolved.items():
        conn.execute(
            "INSERT OR REPLACE INTO claim_embeddings "
            "(claim_id, embedding, embedding_model_version, embedded_at_unix) "
            "VALUES (?, ?, ?, ?)",
            (claim_id, _encode_vector(vec), effective.model_version, now),
        )
    return len(resolved)


def _filter_missing_embeddings(
    conn: sqlite3.Connection, claims: Iterable[Claim], model_version: str
) -> list[Claim]:
    out: list[Claim] = []
    for c in claims:
        row = conn.execute(
            "SELECT embedding_model_version FROM claim_embeddings WHERE claim_id = ?",
            (c.claim_id,),
        ).fetchone()
        if row is None:
            out.append(c)
        elif row["embedding_model_version"] != model_version:
            out.append(c)
    return out


# --- retrieval API -----------------------------------------------------------


def _cosine_normalised(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cosine_fallback(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def retrieve_relevant_claims(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    question: str,
    branch: str | None = None,
    k: int = DEFAULT_TOP_K,
    client: EmbeddingClient | None = None,
) -> list[RankedClaim]:
    """Return top-k active claims by cosine similarity to question."""
    if not question or not question.strip():
        return []

    effective = client or EmbeddingClient()
    model_version = effective.model_version

    active = list_active_claims(conn, session_id)
    if branch is not None:
        active = _filter_by_branch(active, branch)
    if not active:
        return []

    ids = tuple(c.claim_id for c in active)
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT claim_id, embedding, embedding_model_version "
        f"FROM claim_embeddings WHERE claim_id IN ({placeholders})",
        ids,
    ).fetchall()
    embedding_by_id: dict[str, tuple[list[float], str]] = {}
    for row in rows:
        try:
            vec = _decode_vector(row["embedding"])
        except ValueError:
            log.warning("substrate.retrieval_decode_failed", claim_id=row["claim_id"])
            continue
        embedding_by_id[row["claim_id"]] = (vec, row["embedding_model_version"])

    q_vec = effective.embed([question])[0]

    scored: list[tuple[float, Claim, str]] = []
    for c in active:
        entry = embedding_by_id.get(c.claim_id)
        if entry is None:
            log.debug("substrate.retrieval_skip_unembedded", claim_id=c.claim_id)
            continue
        vec, version = entry
        if version != model_version:
            log.debug(
                "substrate.retrieval_version_mismatch",
                claim_id=c.claim_id,
                claim_version=version,
                query_version=model_version,
            )
            continue
        score = _cosine_normalised(q_vec, vec)
        if not (-1.01 <= score <= 1.01):
            score = _cosine_fallback(q_vec, vec)
        scored.append((score, c, version))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]
    return [
        RankedClaim(claim=c, similarity_score=s, embedding_model_version=v)
        for s, c, v in top
    ]


def _filter_by_branch(claims: list[Claim], branch: str) -> list[Claim]:
    """Filter claims to a branch's sub-slot predicates via the active pack."""
    try:
        from memcontext.predicate_packs import active_pack
        slots = active_pack().sub_slots
    except Exception:  # noqa: BLE001
        log.debug("substrate.retrieval_branch_pack_unavailable", branch=branch)
        return claims
    allowed = slots.get(branch)
    if allowed is None:
        log.warning(
            "substrate.retrieval_unknown_branch",
            branch=branch,
            known_branches=sorted(slots.keys())[:10],
        )
        return claims
    return [c for c in claims if c.predicate in allowed]


# --- multi-signal retrieval (RRF) --------------------------------------------

RRF_K: int = 60


def _claim_recency_ts(c: Claim) -> int:
    return c.valid_from_ts if c.valid_from_ts is not None else c.created_ts


def _entity_in_query(query_norm: str, entity_key: str) -> bool:
    if not entity_key:
        return False
    tokens = query_norm.replace("_", " ").split()
    needle_tokens = entity_key.replace("_", " ").split()
    if not needle_tokens:
        return False
    return all(nt in tokens for nt in needle_tokens)


def _rrf_ranks(scores: list[float]) -> list[int]:
    indexed = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank_minus_1, idx in enumerate(indexed):
        ranks[idx] = rank_minus_1 + 1
    return ranks


def _tokenize_for_bm25(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_scores(query_tokens: list[str], claims: list[Claim], *, k1: float = 1.2, b: float = 0.75) -> list[float]:
    if not query_tokens or not claims:
        return [0.0] * len(claims)
    docs = [_tokenize_for_bm25(claim_retrieval_text(c)) for c in claims]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(n, 1)
    df: dict[str, int] = {}
    for qt in set(query_tokens):
        df[qt] = sum(1 for d in docs if qt in d)
    scores: list[float] = []
    for d in docs:
        score = 0.0
        dl = len(d)
        tf_map: dict[str, int] = {}
        for token in d:
            tf_map[token] = tf_map.get(token, 0) + 1
        for qt in query_tokens:
            if qt not in df or df[qt] == 0:
                continue
            idf = math.log((n - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
            tf = tf_map.get(qt, 0)
            score += idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9)))
        scores.append(score)
    return scores


def retrieve_hybrid(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    query: str,
    entity_hint: str | None = None,
    top_k: int = 20,
    valid_at_ts: int | None = None,  # noqa: ARG001
    weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    embedding_client: EmbeddingClient | None = None,
    include_superseded: bool = False,
) -> list[tuple[Claim, float]]:
    """Multi-signal retrieval: semantic + entity + temporal + BM25 via RRF."""
    if not query or not query.strip():
        return []

    effective = embedding_client or EmbeddingClient()
    model_version = effective.model_version

    if include_superseded:
        from memcontext.claims import list_claims_with_lifecycle
        active = list_claims_with_lifecycle(conn, session_id, "historical_truth")
    else:
        active = list_active_claims(conn, session_id)
    if not active:
        return []

    ids = tuple(c.claim_id for c in active)
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT claim_id, embedding, embedding_model_version "
        f"FROM claim_embeddings WHERE claim_id IN ({placeholders})", ids,
    ).fetchall()
    embedding_by_id: dict[str, tuple[list[float], str]] = {}
    for row in rows:
        try:
            vec = _decode_vector(row["embedding"])
        except ValueError:
            continue
        embedding_by_id[row["claim_id"]] = (vec, row["embedding_model_version"])

    q_vec = effective.embed([query])[0]

    sem_scores: list[float] = []
    for c in active:
        entry = embedding_by_id.get(c.claim_id)
        if entry is None:
            sem_scores.append(0.0)
            continue
        vec, version = entry
        if version != model_version:
            sem_scores.append(0.0)
            continue
        score = _cosine_normalised(q_vec, vec)
        if not (-1.01 <= score <= 1.01):
            score = _cosine_fallback(q_vec, vec)
        sem_scores.append(score)

    meta_rows = conn.execute(
        f"SELECT claim_id, entity_key FROM claim_metadata WHERE claim_id IN ({placeholders})", ids,
    ).fetchall()
    entity_by_id: dict[str, str] = {r["claim_id"]: r["entity_key"] for r in meta_rows}

    from memcontext.claims import _normalise_subject
    hint_norm = _normalise_subject(entity_hint) if entity_hint else ""
    query_norm = query.strip().lower()

    ent_scores: list[float] = []
    for c in active:
        ek = entity_by_id.get(c.claim_id, "")
        match = (hint_norm and ek == hint_norm) or _entity_in_query(query_norm, ek)
        ent_scores.append(1.0 if match else 0.0)

    tmp_scores: list[float] = [float(_claim_recency_ts(c)) for c in active]
    bm25_raw = _bm25_scores(_tokenize_for_bm25(query), active)

    sem_ranks = _rrf_ranks(sem_scores)
    ent_ranks = _rrf_ranks(ent_scores)
    tmp_ranks = _rrf_ranks(tmp_scores)
    bm25_ranks = _rrf_ranks(bm25_raw)

    w_sem = weights[0] if len(weights) > 0 else 1.0
    w_ent = weights[1] if len(weights) > 1 else 1.0
    w_tmp = weights[2] if len(weights) > 2 else 1.0
    w_bm25 = weights[3] if len(weights) > 3 else 0.0

    fused: list[tuple[Claim, float]] = []
    for i, c in enumerate(active):
        fused_score = (
            w_sem / (RRF_K + sem_ranks[i])
            + w_ent / (RRF_K + ent_ranks[i])
            + w_tmp / (RRF_K + tmp_ranks[i])
            + w_bm25 / (RRF_K + bm25_ranks[i])
        )
        fused.append((c, fused_score))

    fused.sort(key=lambda x: (-x[1], x[0].claim_id))
    return fused[:top_k]


# --- temporal-window event-tuple retrieval -----------------------------------


def _claim_valid_at(claim: Claim, valid_at_ts: int) -> bool:
    if claim.valid_from_ts is not None and claim.valid_from_ts > valid_at_ts:
        return False
    if claim.valid_until_ts is not None and valid_at_ts >= claim.valid_until_ts:
        return False
    return True


def retrieve_event_tuples(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    query: str,
    top_k: int = 16,
    valid_at_ts: int | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[tuple[EventTuple, float]]:
    """Retrieve event tuples ranked by query-claim cosine similarity."""
    if not query or not query.strip():
        return []

    effective = embedding_client or EmbeddingClient()
    model_version = effective.model_version

    active = list_active_claims(conn, session_id)
    if valid_at_ts is not None:
        active = [c for c in active if _claim_valid_at(c, valid_at_ts)]
    if not active:
        return []

    ids = tuple(c.claim_id for c in active)
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT claim_id, embedding, embedding_model_version "
        f"FROM claim_embeddings WHERE claim_id IN ({placeholders})",
        ids,
    ).fetchall()
    embedding_by_id: dict[str, tuple[list[float], str]] = {}
    for row in rows:
        try:
            vec = _decode_vector(row["embedding"])
        except ValueError:
            log.warning("substrate.retrieve_event_tuples_decode_failed", claim_id=row["claim_id"])
            continue
        embedding_by_id[row["claim_id"]] = (vec, row["embedding_model_version"])

    q_vec = effective.embed([query])[0]

    scored: list[tuple[float, EventTuple]] = []
    for c in active:
        entry = embedding_by_id.get(c.claim_id)
        if entry is None:
            log.debug("substrate.retrieve_event_tuples_skip_unembedded", claim_id=c.claim_id)
            continue
        vec, version = entry
        if version != model_version:
            log.debug(
                "substrate.retrieve_event_tuples_version_mismatch",
                claim_id=c.claim_id,
                claim_version=version,
                query_version=model_version,
            )
            continue
        score = _cosine_normalised(q_vec, vec)
        if not (-1.01 <= score <= 1.01):
            score = _cosine_fallback(q_vec, vec)
        scored.append((score, claim_to_event(c)))

    scored.sort(key=lambda x: (-x[0], x[1].claim_id))
    top = scored[:top_k]
    return [(et, s) for s, et in top]


# --- event-frame retrieval ---------------------------------------------------


def backfill_event_frame_embeddings(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    client: EmbeddingClient | None = None,
) -> int:
    """Embed every event frame not already embedded. Returns count."""
    from memcontext.event_frames import list_event_frames

    effective = client or EmbeddingClient()
    frames = list_event_frames(conn, session_id)
    if not frames:
        return 0

    existing = set()
    for f in frames:
        row = conn.execute(
            "SELECT embedding_model_version FROM event_frame_embeddings WHERE event_id = ?",
            (f.event_id,),
        ).fetchone()
        if row is not None and row["embedding_model_version"] == effective.model_version:
            existing.add(f.event_id)

    missing = [f for f in frames if f.event_id not in existing]
    if not missing:
        return 0

    texts = [f.frame_text() for f in missing]
    try:
        vectors = effective.embed(texts)
    except Exception as exc:  # noqa: BLE001
        log.warning("substrate.event_frame_embed_failed", error=repr(exc)[:200])
        return 0

    now = int(time.time())
    for frame, vec in zip(missing, vectors, strict=True):
        conn.execute(
            "INSERT OR REPLACE INTO event_frame_embeddings "
            "(event_id, embedding, embedding_model_version, embedded_at_unix) "
            "VALUES (?, ?, ?, ?)",
            (frame.event_id, _encode_vector(vec), effective.model_version, now),
        )
    return len(missing)


def retrieve_event_frames(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    query: str,
    top_k: int = 8,
    embedding_client: EmbeddingClient | None = None,
) -> list[tuple["EventFrame", float]]:
    """Retrieve event frames ranked by query-frame cosine similarity."""
    from memcontext.event_frames import EventFrame, list_event_frames

    if not query or not query.strip():
        return []

    effective = embedding_client or EmbeddingClient()
    frames = list_event_frames(conn, session_id)
    if not frames:
        return []

    rows = conn.execute(
        "SELECT event_id, embedding, embedding_model_version FROM event_frame_embeddings "
        "WHERE event_id IN ({})".format(",".join("?" for _ in frames)),
        [f.event_id for f in frames],
    ).fetchall()

    embedding_by_id: dict[str, list[float]] = {}
    for row in rows:
        if row["embedding_model_version"] != effective.model_version:
            continue
        try:
            embedding_by_id[row["event_id"]] = _decode_vector(row["embedding"])
        except ValueError:
            continue

    q_vec = effective.embed([query])[0]

    scored: list[tuple[float, EventFrame]] = []
    for f in frames:
        vec = embedding_by_id.get(f.event_id)
        if vec is None:
            continue
        score = _cosine_normalised(q_vec, vec)
        if not (-1.01 <= score <= 1.01):
            score = _cosine_fallback(q_vec, vec)
        scored.append((score, f))

    scored.sort(key=lambda x: (-x[0], x[1].event_id))
    return [(f, s) for s, f in scored[:top_k]]


__all__ = [
    "BGE_M3_EMBED_DIM",
    "BGE_M3_MODEL_ID",
    "BGE_M3_MODEL_REVISION",
    "BGE_M3_VERSION_TAG",
    "DEFAULT_TOP_K",
    "RRF_K",
    "EmbeddingClient",
    "RankedClaim",
    "backfill_embeddings",
    "backfill_event_frame_embeddings",
    "claim_retrieval_text",
    "embed_and_store",
    "retrieve_event_frames",
    "retrieve_event_tuples",
    "retrieve_hybrid",
    "retrieve_relevant_claims",
]
