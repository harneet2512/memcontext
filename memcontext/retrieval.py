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
"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import struct
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import structlog

from memcontext.claims import list_active_claims
from memcontext.schema import Claim, Turn

log = structlog.get_logger(__name__)


# --- constants ---------------------------------------------------------------

# Product default: BGE-M3 is the benchmark/product embedder. It is heavier than
# small efficiency models, but gives the retrieval layer the strongest default
# signal for long-memory workloads.
_DEFAULT_EMBED_MODEL = "BAAI/bge-m3"

BGE_M3_MODEL_ID = os.environ.get("MEMCONTEXT_EMBED_MODEL", _DEFAULT_EMBED_MODEL)
BGE_M3_MODEL_REVISION = "main"
BGE_M3_VERSION_TAG = f"{BGE_M3_MODEL_ID}@{BGE_M3_MODEL_REVISION}"


def _embedding_dim_for(model_id: str) -> int:
    """Best-effort dimension for common supported models.

    The runtime stores actual vectors as blobs, so this is mainly metadata for
    callers that want to pre-size indexes. Keep BGE-M3 as the default path.
    """
    mid = model_id.lower()
    if "minilm" in mid or "bge-small" in mid or "arctic-embed-s" in mid:
        return 384
    if "arctic-embed-m" in mid or "e5-base" in mid:
        return 768
    return 1024


BGE_M3_EMBED_DIM = _embedding_dim_for(BGE_M3_MODEL_ID)


def _query_prefix_for(model_id: str) -> str:
    """Return the asymmetric-retrieval query prompt for ``model_id``.

    Asymmetric retrieval models (arctic-embed, bge-*-en, e5) expect the SEARCH
    QUERY to carry a task prompt while DOCUMENTS are embedded bare. Symmetric
    models (MiniLM, gte) get no prefix.
    """
    mid = model_id.lower()
    if "arctic-embed" in mid or ("bge-" in mid and "-en" in mid):
        return "Represent this sentence for searching relevant passages: "
    if "e5-" in mid:
        return "query: "
    return ""  # symmetric models (MiniLM, gte) — no prefix


_QUERY_PREFIX = _query_prefix_for(BGE_M3_MODEL_ID)


def apply_query_prefix(text: str) -> str:
    """Prepend the asymmetric-retrieval query prompt to a SEARCH QUERY before
    embedding. No-op for symmetric models and for documents (never prefixed)."""
    return _QUERY_PREFIX + text if _QUERY_PREFIX else text

MODAL_URL_ENV = "MODAL_BGE_M3_URL"
CACHE_DIR_ENV = "SUBSTRATE_EMBED_CACHE_DIR"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "substrate" / "embeddings"

DEFAULT_TOP_K: int = 20

# Cross-session fusion (``retrieve_memory_across``): each queried session keeps
# at least its top-``DEFAULT_PER_SESSION_K`` hits, so a global ``top_k`` cap can
# no longer collapse a session to one turn when the queried sessions outnumber
# the budget (the session-starvation that loses rank-2+ answer turns). The total
# is bounded by ``MAX_ACROSS_HITS`` to guard a pathological session count.
DEFAULT_PER_SESSION_K: int = 3
MAX_ACROSS_HITS: int = 300

_BUILTIN_WEIGHTS = (0.5, 0.2, 0.1, 0.2)  # semantic, entity, temporal, BM25


def _default_weights() -> tuple[float, ...]:
    """Read retrieval weights from MEMCONTEXT_RETRIEVAL_WEIGHTS or use built-in defaults."""
    raw = os.environ.get("MEMCONTEXT_RETRIEVAL_WEIGHTS", "").strip()
    if raw:
        try:
            parts = [float(x.strip()) for x in raw.split(",")]
            if len(parts) >= 3:
                return tuple(parts)
        except ValueError:
            log.warning("substrate.bad_retrieval_weights", raw=raw)
    return _BUILTIN_WEIGHTS


# --- dataclasses -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RankedClaim:
    """A retrieved claim with its similarity score and vector version."""

    claim: Claim
    similarity_score: float
    embedding_model_version: str


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """One unified retrieval result — a fact or an episode, source-tagged.

    `kind` distinguishes the tier; `text` is the NL surface used for ranking
    (fact `text` or episode text); `id` is the claim_id or turn_id;
    `source_turn_id` links back to the originating episode for both kinds.
    """

    kind: Literal["fact", "episode"]
    id: str
    text: str
    source_turn_id: str


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


# --- default embedder (memoised) ---------------------------------------------


_default_client: EmbeddingClient | None = None


def _default_embedding_client() -> EmbeddingClient:
    """Process-wide default embedder, constructed once and reused.

    Constructing a fresh ``EmbeddingClient`` per call re-initialises the local
    sentence-transformers model (~hundreds of ms each). Entry points that are
    not handed an explicit client share this singleton, so the model loads
    once instead of on every retrieve/ingest call.
    """
    global _default_client
    if _default_client is None:
        _default_client = EmbeddingClient()
    return _default_client


EPISODE_EMBED_ENV = "MEMCONTEXT_EMBED_EPISODES"


def episode_embedder() -> EmbeddingClient | None:
    """The embedder production entry points pass to `on_new_turn` for Tier-1.

    Returns the default local embedding client so episodes embed synchronously
    at ingest. Returns None when ``MEMCONTEXT_EMBED_EPISODES=0`` — the test suite
    sets this (via conftest) so CI never loads or downloads a model. Production
    leaves it unset (defaults on).
    """
    if os.environ.get(EPISODE_EMBED_ENV, "1") == "0":
        return None
    return _default_embedding_client()


def semantic_supersession():
    """Pass-2 semantic supersession for the production ingest path, reusing the
    episode embedder. Returns None when no real embedder is configured —
    NullEmbedder's cosine is always 1.0, so wiring it would supersede everything.
    Active only when episode embeddings are on (a real model is loaded).
    """
    emb = episode_embedder()
    if emb is None:
        return None
    from memcontext.supersession_semantic import SemanticSupersession
    return SemanticSupersession(emb)


def semantic_enabled() -> bool:
    """True when a real embedder is configured — i.e. semantic retrieval and Pass-2
    supersession are active. False means the engine is in lexical-only (BM25) mode."""
    return episode_embedder() is not None


def enforce_semantic_policy() -> bool:
    """Serving guard: MemContext IS semantic memory, so running without an embedder
    is a DEGRADED lexical-only mode, not normal operation. Returns True when
    semantic is on. When off: raises if MEMCONTEXT_REQUIRE_EMBEDDINGS=1 (strict),
    else emits a loud warning. Call at serving/ingest entry points so the degraded
    mode is never silent.
    """
    if semantic_enabled():
        return True
    msg = (
        "MemContext is running WITHOUT embeddings: semantic retrieval and Pass-2 "
        "supersession are DISABLED. This is degraded, lexical-only (BM25) mode — "
        "not the product's normal operation. Set MEMCONTEXT_EMBED_EPISODES=1 (the "
        "default) with an embedding model available to enable semantic memory."
    )
    if os.environ.get("MEMCONTEXT_REQUIRE_EMBEDDINGS", "") == "1":
        raise RuntimeError(msg)
    log.warning("substrate.semantic_disabled", detail=msg)
    return False


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
    """Canonical text fed to the embedder / BM25 for a fact.

    NL-first: prefer the fact's natural-language ``text`` (always present from
    v4 on, including for NL-only facts with no structured triple). Falls back to
    the synthesized ``subject predicate value`` triple for structured facts that
    predate ``text`` (pre-v4 rows). Includes value (unlike identity_text in
    supersession_semantic, which excludes value for identity matching).
    """
    if claim.text:
        return claim.text
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
    effective = client or _default_embedding_client()
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
    effective = client or _default_embedding_client()
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


# --- Tier-1 episode embedding ------------------------------------------------


def embed_and_store_episode(
    conn: sqlite3.Connection,
    turn: Turn,
    *,
    client: EmbeddingClient | None = None,
) -> None:
    """Embed an episode (turn) and upsert its vector into ``turn_embeddings``.

    Mirrors :func:`embed_and_store` for claims, keyed on the episode's raw NL
    ``text``. Called synchronously in the Tier-1 write path (the always-on,
    zero-LLM floor); embedding is a local model inference (~tens of ms with
    all-MiniLM-L6-v2, instant with NullEmbedder in tests), never an LLM call.
    On failure, logs and leaves the episode un-embedded for a backfill pass.
    """
    effective = client or _default_embedding_client()
    text = turn.text
    try:
        cached = _cache_get(effective.model_version, text)
        if cached is not None:
            vec = cached
        else:
            vec = effective.embed([text])[0]
            _cache_put(effective.model_version, text, vec)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "substrate.embed_episode_failed",
            turn_id=turn.turn_id,
            error=repr(exc)[:200],
        )
        return

    conn.execute(
        "INSERT OR REPLACE INTO turn_embeddings "
        "(turn_id, embedding, embedding_model_version, embedded_at_unix) "
        "VALUES (?, ?, ?, ?)",
        (turn.turn_id, _encode_vector(vec), effective.model_version, int(time.time())),
    )


def backfill_episode_embeddings(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    client: EmbeddingClient | None = None,
) -> int:
    """Embed every episode in the session not already in ``turn_embeddings``.

    Returns the number of episodes embedded. Used to backfill legacy turns
    migrated to episodes (v3) that predate write-time episode embedding.
    """
    from memcontext.claims import row_to_turn

    effective = client or _default_embedding_client()
    rows = conn.execute(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY ts ASC",
        (session_id,),
    ).fetchall()
    turns = [row_to_turn(r) for r in rows]
    missing = [
        t
        for t in turns
        if _episode_embedding_version(conn, t.turn_id) != effective.model_version
    ]
    if not missing:
        return 0

    resolved: dict[str, list[float]] = {}
    uncached_turns: list[Turn] = []
    uncached_texts: list[str] = []
    for t in missing:
        cached = _cache_get(effective.model_version, t.text)
        if cached is not None:
            resolved[t.turn_id] = cached
        else:
            uncached_turns.append(t)
            uncached_texts.append(t.text)

    if uncached_texts:
        try:
            vectors = effective.embed(uncached_texts)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "substrate.backfill_episode_embed_failed",
                session_id=session_id,
                error=repr(exc)[:200],
            )
            return len(resolved)
        for t, text, v in zip(uncached_turns, uncached_texts, vectors, strict=True):
            _cache_put(effective.model_version, text, v)
            resolved[t.turn_id] = v

    now = int(time.time())
    for turn_id, vec in resolved.items():
        conn.execute(
            "INSERT OR REPLACE INTO turn_embeddings "
            "(turn_id, embedding, embedding_model_version, embedded_at_unix) "
            "VALUES (?, ?, ?, ?)",
            (turn_id, _encode_vector(vec), effective.model_version, now),
        )
    return len(resolved)


def _episode_embedding_version(conn: sqlite3.Connection, turn_id: str) -> str | None:
    row = conn.execute(
        "SELECT embedding_model_version FROM turn_embeddings WHERE turn_id = ?",
        (turn_id,),
    ).fetchone()
    return row["embedding_model_version"] if row is not None else None


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

    effective = client or _default_embedding_client()
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

    q_vec = effective.embed([apply_query_prefix(question)])[0]

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


# --- temporal parsing ---------------------------------------------------------

import re as _re
from datetime import datetime, timedelta, timezone

_TEMPORAL_PATTERNS: list[tuple[str, str]] = [
    (r"(?:last|past)\s+(\d+)\s+days?", "last_n_days"),
    (r"(?:last|past)\s+(\d+)\s+weeks?", "last_n_weeks"),
    (r"(?:last|past)\s+(\d+)\s+months?", "last_n_months"),
    (r"(?:last|past)\s+week\b", "last_week"),
    (r"(?:last|past)\s+month\b", "last_month"),
    (r"(?:last|past)\s+year\b", "last_year"),
    (r"\byesterday\b", "yesterday"),
    (r"\bthis\s+week\b", "this_week"),
    (r"(\d+)\s+weeks?\s+ago", "n_weeks_ago"),
    (r"(\d+)\s+months?\s+ago", "n_months_ago"),
    (r"(\d+)\s+days?\s+ago", "n_days_ago"),
]


def parse_temporal_scope(
    query: str,
    reference_ts: int | None = None,
) -> tuple[int | None, int | None]:
    """Extract a temporal window from a query string.

    Returns (start_ns, end_ns) in nanoseconds, or (None, None) if no
    temporal expression is found.
    """
    if reference_ts is None:
        reference_ts = int(time.time() * 1e9)

    ref_dt = datetime.fromtimestamp(reference_ts / 1e9, tz=timezone.utc)
    q_lower = query.lower()

    for pattern, kind in _TEMPORAL_PATTERNS:
        m = _re.search(pattern, q_lower)
        if not m:
            continue

        start_dt: datetime
        end_dt: datetime = ref_dt

        if kind == "yesterday":
            start_dt = ref_dt - timedelta(days=1)
            end_dt = ref_dt
        elif kind == "last_week":
            start_dt = ref_dt - timedelta(weeks=1)
        elif kind == "last_month":
            start_dt = ref_dt - timedelta(days=30)
        elif kind == "last_year":
            start_dt = ref_dt - timedelta(days=365)
        elif kind == "this_week":
            start_dt = ref_dt - timedelta(days=ref_dt.weekday())
        elif kind == "last_n_days":
            start_dt = ref_dt - timedelta(days=int(m.group(1)))
        elif kind == "last_n_weeks":
            start_dt = ref_dt - timedelta(weeks=int(m.group(1)))
        elif kind == "last_n_months":
            start_dt = ref_dt - timedelta(days=int(m.group(1)) * 30)
        elif kind == "n_weeks_ago":
            n = int(m.group(1))
            start_dt = ref_dt - timedelta(weeks=n)
            end_dt = ref_dt - timedelta(weeks=max(n - 1, 0))
        elif kind == "n_months_ago":
            n = int(m.group(1))
            start_dt = ref_dt - timedelta(days=n * 30)
            end_dt = ref_dt - timedelta(days=max(n - 1, 0) * 30)
        elif kind == "n_days_ago":
            n = int(m.group(1))
            start_dt = ref_dt - timedelta(days=n)
            end_dt = ref_dt - timedelta(days=max(n - 1, 0))
        else:
            continue

        return (int(start_dt.timestamp() * 1e9), int(end_dt.timestamp() * 1e9))

    return (None, None)


# --- query depth routing -----------------------------------------------------

_AGGREGATION_KEYWORDS = frozenset({
    "how many", "list all", "all the", "every", "count",
    "summarize", "overview", "history", "timeline",
    "throughout", "across all",
})

_TEMPORAL_QUERY_KEYWORDS = frozenset({
    "when", "last time", "first time", "how long ago",
    "most recent", "latest", "earliest",
})


def classify_query_depth(query: str) -> tuple[str, int]:
    """Classify query type and return recommended top_k."""
    q_lower = query.lower().strip()
    for kw in _AGGREGATION_KEYWORDS:
        if kw in q_lower:
            return ("aggregation", 50)
    for kw in _TEMPORAL_QUERY_KEYWORDS:
        if kw in q_lower:
            return ("temporal", 30)
    return ("factual", 15)


# --- predicate-aware query routing -------------------------------------------

_QUERY_PREDICATE_MAP: list[tuple[tuple[str, ...], set[str], str]] = [
    (("recommend", "suggest", "advise", "told me to", "assistant said",
      "assistant told", "you said", "you told", "you recommend", "did you",
      "remind me of", "remind me what"),
     {"assistant_recommendation", "assistant_action"}, "assistant_recall"),
    (("prefer", "like", "want", "favorite", "enjoy", "hobby", "interest"),
     {"user_preference"}, "preference"),
    (("when", "timeline", "order", "sequence", "date", "how long"),
     {"user_event"}, "temporal"),
    (("changed", "updated", "used to", "switched", "current", "latest"),
     {"user_fact", "user_preference"}, "knowledge_update"),
    (("goal", "plan", "trying to", "working on", "aim"),
     {"user_goal"}, "fact_recall"),
]


_HISTORY_INTENT = _re.compile(
    r"\b(before|previously|used to|use to|formerly|former|earlier|prior|"
    r"originally|in the past|history of|no longer)\b"
)


def detect_history_intent(query: str) -> bool:
    """True when a query asks about PAST/superseded state rather than the current
    value, so retrieval should include superseded facts (temporal history mode).
    Deterministic, zero-LLM.
    """
    return bool(_HISTORY_INTENT.search(query.lower()))


def classify_query_predicates(query: str) -> tuple[set[str], str]:
    """Map a query to target predicate families and a query type label.

    Returns (target_predicates, query_type). Both are deterministic —
    no LLM call.
    """
    q_lower = query.lower()
    matched_preds: set[str] = set()
    query_type = "fact_recall"
    for keywords, predicates, qtype in _QUERY_PREDICATE_MAP:
        for kw in keywords:
            if kw in q_lower:
                matched_preds |= predicates
                query_type = qtype
                break
    return matched_preds, query_type


# --- multi-signal retrieval (RRF) --------------------------------------------

RRF_K: int = 60


def _claim_recency_ts(c: Claim) -> int:
    if c.event_ts is not None:
        return c.event_ts
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


def _bm25_over_docs(
    query_tokens: list[str],
    docs: list[list[str]],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """BM25 over pre-tokenised documents. The single BM25 implementation.

    Shared by `_bm25_scores` (claims) and `retrieve_episodes`/`search_raw_turns`
    (episodes) so the scoring formula lives in exactly one place.
    """
    if not query_tokens or not docs:
        return [0.0] * len(docs)
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


def _bm25_scores(query_tokens: list[str], claims: list[Claim], *, k1: float = 1.2, b: float = 0.75) -> list[float]:
    docs = [_tokenize_for_bm25(claim_retrieval_text(c)) for c in claims]
    return _bm25_over_docs(query_tokens, docs, k1=k1, b=b)


def retrieve_hybrid(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    query: str,
    entity_hint: str | None = None,
    top_k: int = 20,
    valid_at_ts: int | None = None,
    weights: tuple[float, ...] | None = None,
    embedding_client: EmbeddingClient | None = None,
    include_superseded: bool = False,
    reranker: Callable[[str, list[str]], list[float]] | None = None,
    explain: dict[str, dict[str, float]] | None = None,
    include_demoted: bool = False,
) -> list[tuple[Claim, float]]:
    """Multi-signal retrieval: semantic + entity + temporal + BM25 + importance via RRF.

    Pass an empty ``explain`` dict to capture the per-signal RRF contribution for
    every ranked claim (ranking observability); it is filled in place.

    If ``reranker`` is provided, the top results from RRF are re-scored by
    the reranker and re-sorted.  The callable signature is
    ``(query, list_of_texts) -> list_of_scores`` (higher = more relevant).
    """
    if not query or not query.strip():
        return []

    weights = weights or _default_weights()

    effective = embedding_client or _default_embedding_client()
    model_version = effective.model_version

    if include_superseded:
        from memcontext.claims import list_claims_with_lifecycle
        active = list_claims_with_lifecycle(conn, session_id, "historical_truth")
    else:
        active = list_active_claims(conn, session_id)
    if valid_at_ts is not None:
        active = [c for c in active if _claim_valid_at(c, valid_at_ts)]
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

    has_embeddings = bool(embedding_by_id)
    if has_embeddings:
        q_vec: list[float] | None = effective.embed([apply_query_prefix(query)])[0]
    else:
        q_vec = None
        log.debug("substrate.retrieve_hybrid_no_embeddings", session_id=session_id)

    sem_scores: list[float] = []
    for c in active:
        if q_vec is None:
            sem_scores.append(0.0)
            continue
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
        f"SELECT claim_id, entity_key, COALESCE(importance_score, 0.5) AS importance_score,"
        f" COALESCE(access_count, 0) AS access_count,"
        f" COALESCE(demoted, 0) AS demoted,"
        f" COALESCE(source_trust, 0.5) AS source_trust"
        f" FROM claim_metadata WHERE claim_id IN ({placeholders})", ids,
    ).fetchall()
    entity_by_id: dict[str, str] = {r["claim_id"]: r["entity_key"] for r in meta_rows}
    importance_by_id: dict[str, float] = {
        r["claim_id"]: float(r["importance_score"]) for r in meta_rows
    }
    usage_by_id: dict[str, float] = {
        r["claim_id"]: float(r["access_count"]) for r in meta_rows
    }
    # Utility-weighted retention: demoted claims leave active retrieval.
    demoted_ids: set[str] = {r["claim_id"] for r in meta_rows if r["demoted"]}
    # Source trust: how much to trust each claim by where it came from (Phase 3).
    trust_by_id: dict[str, float] = {
        r["claim_id"]: float(r["source_trust"]) for r in meta_rows
    }

    from memcontext.claims import _normalise_subject
    hint_norm = _normalise_subject(entity_hint) if entity_hint else ""
    query_norm = query.strip().lower()

    ent_scores: list[float] = []
    for c in active:
        ek = entity_by_id.get(c.claim_id, "")
        match = (hint_norm and ek == hint_norm) or _entity_in_query(query_norm, ek)
        ent_scores.append(1.0 if match else 0.0)

    try:
        from memcontext.entities import extract_entities
        entity_rows = conn.execute(
            f"SELECT claim_id, entity_text FROM claim_entities WHERE claim_id IN ({placeholders})",
            ids,
        ).fetchall()
        entities_by_claim: dict[str, set[str]] = {}
        for r in entity_rows:
            entities_by_claim.setdefault(r["claim_id"], set()).add(r["entity_text"])
        query_entities = {e.text.lower() for e in extract_entities(query)}
        query_words = set(query_norm.split())
        match_set = query_entities | query_words
        # Entity/graph expansion (Cycle A): pull in claims of entities that
        # co-occur with a query entity (1-hop graph neighbors), so a 2-hop query
        # surfaces related facts the flat token match misses. Gated on the query
        # naming a known entity; deterministic, zero-LLM, best-effort.
        neighbor_claim_ids: set[str] = set()
        if query_entities:
            try:
                from memcontext.entity_graph import EntityGraph
                graph = EntityGraph(conn, session_id)
                for qe in query_entities:
                    ek = _normalise_subject(qe)
                    if graph.has_entity(ek):
                        neighbor_claim_ids |= graph.neighbor_claim_ids(ek, max_hops=1)
            except Exception:  # noqa: BLE001
                neighbor_claim_ids = set()
        for i, c in enumerate(active):
            claim_ents = entities_by_claim.get(c.claim_id, set())
            if claim_ents & match_set:
                ent_scores[i] = max(ent_scores[i], 1.0)
            elif c.claim_id in neighbor_claim_ids:
                ent_scores[i] = max(ent_scores[i], 0.6)  # 2-hop graph neighbor
    except Exception:  # noqa: BLE001
        pass

    tmp_scores: list[float] = [float(_claim_recency_ts(c)) for c in active]
    bm25_raw = _bm25_scores(_tokenize_for_bm25(query), active)

    scope_start, scope_end = parse_temporal_scope(query)
    scope_scores: list[float] = []
    if scope_start is not None and scope_end is not None:
        for c in active:
            ts = _claim_recency_ts(c)
            scope_scores.append(1.0 if scope_start <= ts <= scope_end else 0.0)
        w_scope = 0.3
    else:
        scope_scores = [0.0] * len(active)
        w_scope = 0.0

    target_preds, _query_type = classify_query_predicates(query)
    if target_preds:
        pred_scores: list[float] = [1.0 if c.predicate in target_preds else 0.0 for c in active]
        w_pred = 0.2
    else:
        pred_scores = [0.0] * len(active)
        w_pred = 0.0

    conf_scores: list[float] = [c.confidence for c in active]

    freq_counts: dict[tuple[str | None, str | None], int] = {}
    for c in active:
        key = (c.subject, c.predicate)
        freq_counts[key] = freq_counts.get(key, 0) + 1
    freq_scores: list[float] = [float(freq_counts[(c.subject, c.predicate)]) for c in active]

    # Importance: computed at ingest (importance.py) + stored in claim_metadata.
    # Now a first-class ranking signal (previously read only by digests/profiles).
    imp_scores: list[float] = [importance_by_id.get(c.claim_id, 0.5) for c in active]
    # Usage: access_count, incremented when a claim is served (handle_memory_query).
    # Cue-dependent reinforcement — frequently-retrieved claims rank up over time.
    usage_scores: list[float] = [usage_by_id.get(c.claim_id, 0.0) for c in active]
    trust_scores: list[float] = [trust_by_id.get(c.claim_id, 0.5) for c in active]

    sem_ranks = _rrf_ranks(sem_scores)
    ent_ranks = _rrf_ranks(ent_scores)
    tmp_ranks = _rrf_ranks(tmp_scores)
    bm25_ranks = _rrf_ranks(bm25_raw)
    scope_ranks = _rrf_ranks(scope_scores)
    pred_ranks = _rrf_ranks(pred_scores)
    conf_ranks = _rrf_ranks(conf_scores)
    freq_ranks = _rrf_ranks(freq_scores)
    imp_ranks = _rrf_ranks(imp_scores)
    usage_ranks = _rrf_ranks(usage_scores)
    trust_ranks = _rrf_ranks(trust_scores)

    w_sem = weights[0] if len(weights) > 0 else 1.0
    w_ent = weights[1] if len(weights) > 1 else 1.0
    w_tmp = weights[2] if len(weights) > 2 else 1.0
    w_bm25 = weights[3] if len(weights) > 3 else 0.0
    w_conf = 0.1
    w_freq = 0.1
    w_imp = 0.15
    w_usage = 0.1
    w_trust = 0.12

    # Freshness: knowledge-update / temporal queries ask for the CURRENT value, so
    # weight recency (the temporal channel) higher — the latest active fact wins.
    if _query_type in ("knowledge_update", "temporal"):
        w_tmp *= 3.0

    if not has_embeddings:
        w_sem = 0.0

    fused: list[tuple[Claim, float]] = []
    for i, c in enumerate(active):
        if c.claim_id in demoted_ids and not include_demoted:
            continue
        contrib = {
            "semantic": w_sem / (RRF_K + sem_ranks[i]),
            "entity": w_ent / (RRF_K + ent_ranks[i]),
            "temporal": w_tmp / (RRF_K + tmp_ranks[i]),
            "bm25": w_bm25 / (RRF_K + bm25_ranks[i]),
            "scope": w_scope / (RRF_K + scope_ranks[i]),
            "predicate": w_pred / (RRF_K + pred_ranks[i]),
            "confidence": w_conf / (RRF_K + conf_ranks[i]),
            "frequency": w_freq / (RRF_K + freq_ranks[i]),
            "importance": w_imp / (RRF_K + imp_ranks[i]),
            "usage": w_usage / (RRF_K + usage_ranks[i]),
            "source_trust": w_trust / (RRF_K + trust_ranks[i]),
        }
        fused_score = sum(contrib.values())
        if explain is not None:
            contrib["final"] = fused_score
            explain[c.claim_id] = contrib
        fused.append((c, fused_score))

    fused.sort(key=lambda x: (-x[1], x[0].claim_id))

    if reranker is not None:
        candidates = fused[:top_k]
        texts = [claim_retrieval_text(c) for c, _ in candidates]
        try:
            rerank_scores = reranker(query, texts)
            candidates = [
                (c, float(rs)) for (c, _), rs in zip(candidates, rerank_scores)
            ]
            candidates.sort(key=lambda x: (-x[1], x[0].claim_id))
        except Exception:
            log.warning("reranker failed, falling back to RRF order")
        return candidates

    return fused[:top_k]


# --- Tier-1 episode retrieval ------------------------------------------------


def retrieve_episodes(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    valid_at_ts: int | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[tuple[Turn, float]]:
    """Rank episodes (turns) for a query via RRF over NL-text signals.

    Channels: semantic (``turn_embeddings`` cosine), BM25 over ``turn.text``,
    entity overlap (deterministic, on-the-fly — no stored sidecar needed), and
    temporal recency (``turn.ts``). This is the Tier-1 floor: it needs no
    structured fields and works whether or not any facts have been extracted.
    Degrades to BM25 + recency when no episode embeddings exist yet.
    """
    if not query or not query.strip():
        return []

    from memcontext.claims import row_to_turn

    rows = conn.execute(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY ts ASC",
        (session_id,),
    ).fetchall()
    episodes: list[Turn] = [row_to_turn(r) for r in rows]
    if valid_at_ts is not None:
        episodes = [t for t in episodes if t.ts <= valid_at_ts]
    if not episodes:
        return []

    effective = embedding_client or _default_embedding_client()
    model_version = effective.model_version

    ids = tuple(t.turn_id for t in episodes)
    placeholders = ",".join("?" for _ in ids)
    emb_rows = conn.execute(
        f"SELECT turn_id, embedding, embedding_model_version "
        f"FROM turn_embeddings WHERE turn_id IN ({placeholders})",
        ids,
    ).fetchall()
    embedding_by_id: dict[str, tuple[list[float], str]] = {}
    for row in emb_rows:
        try:
            vec = _decode_vector(row["embedding"])
        except ValueError:
            continue
        embedding_by_id[row["turn_id"]] = (vec, row["embedding_model_version"])

    has_embeddings = bool(embedding_by_id)
    q_vec = effective.embed([apply_query_prefix(query)])[0] if has_embeddings else None

    sem_scores: list[float] = []
    for t in episodes:
        entry = embedding_by_id.get(t.turn_id)
        if q_vec is None or entry is None or entry[1] != model_version:
            sem_scores.append(0.0)
            continue
        score = _cosine_normalised(q_vec, entry[0])
        if not (-1.01 <= score <= 1.01):
            score = _cosine_fallback(q_vec, entry[0])
        sem_scores.append(score)

    bm25_raw = _bm25_over_docs(
        _tokenize_for_bm25(query), [_tokenize_for_bm25(t.text) for t in episodes]
    )

    query_norm = query.strip().lower()
    match_set = set(query_norm.split())
    try:
        from memcontext.entities import extract_entities
        match_set |= {e.text.lower() for e in extract_entities(query)}
    except Exception:  # noqa: BLE001
        pass
    ent_scores: list[float] = []
    for t in episodes:
        doc_tokens = set(_tokenize_for_bm25(t.text))
        ent_scores.append(1.0 if doc_tokens & match_set else 0.0)

    tmp_scores: list[float] = [float(t.ts) for t in episodes]

    sem_ranks = _rrf_ranks(sem_scores)
    bm25_ranks = _rrf_ranks(bm25_raw)
    ent_ranks = _rrf_ranks(ent_scores)
    tmp_ranks = _rrf_ranks(tmp_scores)

    w_sem = 0.5 if has_embeddings else 0.0
    w_bm25 = 0.25
    w_ent = 0.15
    w_tmp = 0.1

    fused: list[tuple[Turn, float]] = []
    for i, t in enumerate(episodes):
        fused_score = (
            w_sem / (RRF_K + sem_ranks[i])
            + w_bm25 / (RRF_K + bm25_ranks[i])
            + w_ent / (RRF_K + ent_ranks[i])
            + w_tmp / (RRF_K + tmp_ranks[i])
        )
        fused.append((t, fused_score))

    fused.sort(key=lambda x: (-x[1], x[0].turn_id))
    return fused[:top_k]


# --- deterministic query expansion -------------------------------------------


_STOPWORDS = frozenset(
    "i me my we our you your he she it they them the a an is are was were "
    "be been have has had do did does will would can could should may might "
    "in on at to for of with from by about into through during before after "
    "and or but not no nor so if then than that this these those what which "
    "who whom how when where why all any each every some many much more most "
    "very also just only even still already yet again too quite really".split()
)


def _extract_query_entities(query: str) -> list[str]:
    """Extract likely entity words from a query (deterministic, no LLM)."""
    tokens = _re.findall(r"[a-z0-9]+", query.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def retrieve_expanded(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    query: str,
    top_k: int = 50,
    weights: tuple[float, ...] | None = None,
    embedding_client: EmbeddingClient | None = None,
    max_expansions: int = 3,
) -> list[tuple[Claim, float]]:
    """Expand query with related claim subjects, merge multiple retrievals.

    1. Run primary retrieval with the original query.
    2. Extract entity-like words from the query.
    3. Find claim subjects that contain those words.
    4. Run sub-queries for each matched subject (up to max_expansions).
    5. Merge all results, deduplicate, re-sort by best score.
    """
    primary = retrieve_hybrid(
        conn,
        session_id=session_id,
        query=query,
        top_k=top_k,
        weights=weights,
        embedding_client=embedding_client,
    )

    seen: dict[str, float] = {c.claim_id: s for c, s in primary}
    all_results: list[tuple[Claim, float]] = list(primary)

    entity_words = _extract_query_entities(query)
    if not entity_words:
        return primary

    active = list_active_claims(conn, session_id)
    subjects = {c.subject for c in active}

    expansion_subjects: list[str] = []
    for subj in subjects:
        subj_lower = subj.lower().replace("_", " ")
        if any(w in subj_lower for w in entity_words):
            expansion_subjects.append(subj)

    for subj in expansion_subjects[:max_expansions]:
        sub_results = retrieve_hybrid(
            conn,
            session_id=session_id,
            query=f"{subj} {query}",
            top_k=top_k // 2,
            weights=weights,
            embedding_client=embedding_client,
        )
        for c, s in sub_results:
            if c.claim_id not in seen or s > seen[c.claim_id]:
                seen[c.claim_id] = s
                all_results.append((c, s))

    deduped: dict[str, tuple[Claim, float]] = {}
    for c, s in all_results:
        if c.claim_id not in deduped or s > deduped[c.claim_id][1]:
            deduped[c.claim_id] = (c, s)

    result = sorted(deduped.values(), key=lambda x: (-x[1], x[0].claim_id))
    return result[:top_k]


# --- context expansion -------------------------------------------------------


def expand_claim_context(
    conn: sqlite3.Connection,
    claim: Claim,
    *,
    window: int = 2,
) -> list[Any]:
    """Fetch neighboring turns around a claim's source turn.

    Returns up to *window* turns before and after the source turn
    (by timestamp within the same session), plus the source turn itself.
    """
    from memcontext.claims import get_turn
    from memcontext.schema import Speaker, Turn

    source_turn = get_turn(conn, claim.source_turn_id)
    if source_turn is None:
        return []

    rows = conn.execute(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY ts ASC",
        (source_turn.session_id,),
    ).fetchall()

    turns: list[Turn] = []
    source_idx: int | None = None
    for i, row in enumerate(rows):
        t = Turn(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            speaker=Speaker(row["speaker"]),
            text=row["text"],
            ts=row["ts"],
            asr_confidence=row["asr_confidence"],
        )
        turns.append(t)
        if row["turn_id"] == source_turn.turn_id:
            source_idx = i

    if source_idx is None:
        return [source_turn]

    start = max(0, source_idx - window)
    end = min(len(turns), source_idx + window + 1)
    return turns[start:end]


# --- multi-resolution retrieval ----------------------------------------------


def search_raw_turns(
    conn: sqlite3.Connection,
    session_id: str,
    query: str,
    *,
    top_k: int = 15,
) -> list[tuple[Any, float]]:
    """BM25 search over raw turn (episode) text as retrieval fallback."""
    from memcontext.claims import row_to_turn

    rows = conn.execute(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY ts ASC",
        (session_id,),
    ).fetchall()

    turns: list[Turn] = [row_to_turn(row) for row in rows]
    if not turns:
        return []

    query_tokens = _tokenize_for_bm25(query)
    if not query_tokens:
        return []

    docs = [_tokenize_for_bm25(t.text) for t in turns]
    scores = _bm25_over_docs(query_tokens, docs)
    scored: list[tuple[Turn, float]] = list(zip(turns, scores, strict=True))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def bump_access(
    conn: sqlite3.Connection, claim_ids: list[str], *, now_ts: int | None = None
) -> None:
    """Reinforce served claims: increment access_count + stamp last_accessed_ts.

    Best-effort, non-blocking, zero-LLM — a usage-tracking failure must never
    break a read. The serving door calls this for the fact claims it returns; the
    counts feed the usage ranking channel (cue-dependent reinforcement).
    """
    if not claim_ids:
        return
    if now_ts is None:
        from memcontext.claims import now_ns
        now_ts = now_ns()
    try:
        ph = ",".join("?" for _ in claim_ids)
        conn.execute(
            f"UPDATE claim_metadata SET access_count = COALESCE(access_count, 0) + 1,"
            f" last_accessed_ts = ? WHERE claim_id IN ({ph})",
            (now_ts, *claim_ids),
        )
    except Exception:  # noqa: BLE001
        log.warning("substrate.usage_bump_failed")


def retrieve_memory(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    valid_at_ts: int | None = None,
    embedding_client: EmbeddingClient | None = None,
    explain: dict[str, dict[str, float]] | None = None,
    include_superseded: bool = False,
) -> list[tuple[MemoryHit, float]]:
    """Unified Tier-1 + Tier-2 retrieval: facts AND episodes, source-tagged.

    Ranks facts (`retrieve_hybrid`) and episodes (`retrieve_episodes`)
    independently, then fuses them with a second-level RRF — facts get a slight
    weight edge so a fact outranks the episode it came from on a tie. Needs no
    structured field; when no facts exist (extraction disabled or still pending)
    the episode hits carry retrieval entirely — the Tier-1 floor.
    """
    if not query or not query.strip():
        return []

    # Temporal truth, universally: a past-tense query ("what was X before") pulls in
    # superseded facts for EVERY caller (cli query, MCP door), not just the door.
    if not include_superseded:
        include_superseded = detect_history_intent(query)

    facts = retrieve_hybrid(
        conn, session_id=session_id, query=query, top_k=top_k,
        valid_at_ts=valid_at_ts,
        embedding_client=embedding_client, explain=explain,
        include_superseded=include_superseded,
    )
    episodes = retrieve_episodes(
        conn, session_id=session_id, query=query, top_k=top_k,
        valid_at_ts=valid_at_ts,
        embedding_client=embedding_client,
    )
    return _fuse_memory(facts, episodes, top_k)


def retrieve_memory_across(
    conn: sqlite3.Connection,
    *,
    session_ids: list[str],
    query: str,
    top_k: int = DEFAULT_TOP_K,
    per_session_k: int = DEFAULT_PER_SESSION_K,
    valid_at_ts: int | None = None,
    embedding_client: EmbeddingClient | None = None,
    explain: dict[str, dict[str, float]] | None = None,
    include_superseded: bool = False,
) -> list[tuple[MemoryHit, float]]:
    """Unified retrieval across MANY sessions — fuse per-session rankings by RANK.

    Each session is an independent ranking source, and **raw hybrid scores are
    not comparable across sessions**: BM25/semantic magnitudes scale with a
    session's size and content, so a long, on-topic-but-irrelevant session
    carries larger raw scores than a terse session that happens to hold the
    answer. Reciprocal Rank Fusion exists precisely to fuse such lists — it uses
    only **rank**, never raw score. So we run the within-session fact+episode
    fusion (`retrieve_memory`, whose scores are `w / (RRF_K + rank_within_its_
    session)`) for each session and merge the results.

    Consequence: a session's rank-1 hit scores `w / (RRF_K + 1)` regardless of
    session, so every queried session is represented (up to `top_k`) and the
    answer surfaces even from a low-raw-score session. Pooling all sessions and
    sorting by raw score (the earlier approach) silenced whole sessions whose
    score magnitudes were smaller — a session-size bias, not relevance.

    This is plain RRF (Cormack et al.): rank within each source, fuse by rank.
    It needs no per-source score calibration and no tuning, so it generalizes to
    any number/shape of sessions.
    """
    if not query or not query.strip() or not session_ids:
        return []
    # Per-session depth guarantee. A single global ``fused[:top_k]`` collapses to
    # ONE hit per session once the queried sessions reach ``top_k``: every
    # session's rank-1 scores ``w/(RRF_K+1)``, which sorts above ANY rank-2
    # (``w/(RRF_K+2)``), so the cap admits only rank-1s and starves an answer
    # turn that is rank-2+ within its own session (measured: 33%->72% answer-turn
    # recall on 53-session haystacks once each session may keep its top-3).
    #
    # We do NOT instead score-rank the sessions and give depth only to the
    # "relevant" ones: raw cross-session scores are not comparable (see above),
    # so session selection drops the answer session - measured WORSE than breadth
    # (17% vs 33%). Breadth is load-bearing; depth is layered on top of it.
    #
    # Two passes: each session RESERVES its top-``per_session_k`` (the guarantee);
    # the remainder share whatever budget is left up to ``top_k``. So few-session
    # queries keep their old depth (budget >= guarantee) and many-session queries
    # stop starving (budget grows to the guarantee), bounded by ``MAX_ACROSS_HITS``.
    per_session_k = max(1, per_session_k)
    tie = lambda h: (-h[1], h[0].kind != "fact", h[0].id)  # noqa: E731
    reserved: list[tuple[MemoryHit, float]] = []
    overflow: list[tuple[MemoryHit, float]] = []
    for sid in session_ids:
        hits = retrieve_memory(
            conn, session_id=sid, query=query, top_k=top_k,
            valid_at_ts=valid_at_ts,
            embedding_client=embedding_client, explain=explain,
            include_superseded=include_superseded,
        )
        reserved.extend(hits[:per_session_k])
        overflow.extend(hits[per_session_k:])
    reserved.sort(key=tie)
    overflow.sort(key=tie)
    # Never cut below the per-session guarantee for the queried breadth.
    budget = min(max(top_k, len(reserved)), MAX_ACROSS_HITS)
    return (reserved + overflow)[:budget]


def _fuse_memory(
    facts: list[tuple[Claim, float]],
    episodes: list[tuple[Turn, float]],
    top_k: int,
) -> list[tuple[MemoryHit, float]]:
    """Second-level RRF over already-ranked fact and episode pools.

    Each pool must arrive sorted best-first; fusion weights its *rank position*
    (RRF), so the per-channel raw scores need not be comparable. Facts carry a
    slight weight edge so a fact outranks the episode it was extracted from on a
    tie — but episodes still interleave by rank (the Tier-1 floor).
    """
    w_fact, w_ep = 1.0, 0.9
    fused: list[tuple[MemoryHit, float]] = []
    for rank, (claim, _score) in enumerate(facts, start=1):
        fused.append((
            MemoryHit(
                kind="fact",
                id=claim.claim_id,
                text=claim_retrieval_text(claim),
                source_turn_id=claim.source_turn_id,
            ),
            w_fact / (RRF_K + rank),
        ))
    for rank, (turn, _score) in enumerate(episodes, start=1):
        fused.append((
            MemoryHit(
                kind="episode",
                id=turn.turn_id,
                text=turn.text,
                source_turn_id=turn.turn_id,
            ),
            w_ep / (RRF_K + rank),
        ))
    # -score, then facts before episodes on exact ties, then id for determinism.
    fused.sort(key=lambda h: (-h[1], h[0].kind != "fact", h[0].id))
    return fused[:top_k]


def retrieve_with_fallback(
    conn: sqlite3.Connection,
    session_id: str,
    query: str,
    *,
    top_k: int = 15,
    embedding_client: "EmbeddingClient | None" = None,
) -> list[dict[str, Any]]:
    """Multi-resolution retrieval: claims first, raw turns as fallback."""
    claim_results = retrieve_hybrid(
        conn,
        session_id=session_id,
        query=query,
        top_k=top_k,
        embedding_client=embedding_client,
    )

    # If claims are sufficient, return claims only
    if len(claim_results) >= 3 and claim_results[0][1] >= 0.3:
        return [
            {
                "type": "claim",
                "text": f"[{c.predicate}] {c.subject}: {c.value}",
                "score": score,
                "claim_id": c.claim_id,
                "source_turn_id": c.source_turn_id,
            }
            for c, score in claim_results
        ]

    # Fallback: also search raw turns
    turn_results = search_raw_turns(conn, session_id, query, top_k=top_k)

    # Build merged result list
    merged: list[dict[str, Any]] = []

    # Collect source_turn_ids from claims for deduplication
    claim_source_turn_ids: set[str] = set()
    for c, score in claim_results:
        claim_source_turn_ids.add(c.source_turn_id)
        merged.append({
            "type": "claim",
            "text": f"[{c.predicate}] {c.subject}: {c.value}",
            "score": score,
            "claim_id": c.claim_id,
            "source_turn_id": c.source_turn_id,
        })

    # Add turns, skipping those already represented by a claim
    for t, score in turn_results:
        if t.turn_id in claim_source_turn_ids:
            continue
        merged.append({
            "type": "turn",
            "text": t.text,
            "score": score,
            "turn_id": t.turn_id,
            "speaker": t.speaker.value,
        })

    # Sort merged by score descending
    merged.sort(key=lambda x: -x["score"])
    return merged[:top_k]


# --- temporal-window event-tuple retrieval -----------------------------------


def _claim_valid_at(claim: Claim, valid_at_ts: int) -> bool:
    if claim.valid_from_ts is not None and claim.valid_from_ts > valid_at_ts:
        return False
    if claim.valid_until_ts is not None and valid_at_ts >= claim.valid_until_ts:
        return False
    return True


# --- event-frame retrieval ---------------------------------------------------


def backfill_event_frame_embeddings(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    client: EmbeddingClient | None = None,
) -> int:
    """Embed every event frame not already embedded. Returns count."""
    from memcontext.event_frames import list_event_frames

    effective = client or _default_embedding_client()
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

    effective = embedding_client or _default_embedding_client()
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

    q_vec = effective.embed([apply_query_prefix(query)])[0]

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


# --- Public reuse surface (Session 2 tool/activation track) ------------------
# Generic, domain-agnostic IR primitives intentionally re-exported under public
# names so the additive tool registry reuses the *exact* fusion arithmetic and
# vector codec instead of forking a parallel retrieval stack. These are thin
# aliases — no behavior, signatures, or memory internals change.
rrf_ranks = _rrf_ranks
bm25_over_docs = _bm25_over_docs
tokenize_for_bm25 = _tokenize_for_bm25
encode_vector = _encode_vector
decode_vector = _decode_vector


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
    "bm25_over_docs",
    "claim_retrieval_text",
    "decode_vector",
    "embed_and_store",
    "encode_vector",
    "retrieve_event_frames",
    "retrieve_hybrid",
    "retrieve_relevant_claims",
    "retrieve_with_fallback",
    "rrf_ranks",
    "search_raw_turns",
    "tokenize_for_bm25",
]
