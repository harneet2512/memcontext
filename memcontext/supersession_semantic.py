"""Supersession Pass 2 — semantic identity via embeddings.

Identity embedding is `subject + predicate + context`, **never** `value`.
If value were in the embedding, "onset: 3 days" and "onset: 4 days" would
embed differently and supersession could never match.

Cosine threshold: 0.88 (tuned for paraphrase recall).

Embedder interface: `Embedder.embed(texts: list[str]) -> list[list[float]]`.
Two implementations:
- NullEmbedder — constant vector, cosine always 1.0. For tests.
- E5Embedder — intfloat/e5-small-v2 via sentence_transformers. Lazy import.
"""
from __future__ import annotations

import math
import sqlite3
from typing import Any, Protocol, cast, runtime_checkable

import structlog

from memcontext.claims import set_claim_status
from memcontext.schema import Claim, ClaimStatus, EdgeType, SupersessionEdge
from memcontext.supersession import write_supersession_edge

log = structlog.get_logger(__name__)


DEFAULT_COSINE_THRESHOLD = 0.88


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedder interface. Implementations must be deterministic."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class NullEmbedder:
    """Constant-vector embedder. Cosine is always 1.0. For tests/CI."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._vec = [1.0 / math.sqrt(dim)] * dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


class E5Embedder:
    """intfloat/e5-small-v2 via sentence_transformers (MIT)."""

    MODEL_ID = "intfloat/e5-small-v2"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            log.error("substrate.e5_import_failed", error=str(exc))
            raise

        self._model = SentenceTransformer(self.MODEL_ID)
        self._prefix = "passage: "

    def embed(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self._prefix + t for t in texts]
        raw = cast(Any, self._model).encode(prefixed, normalize_embeddings=True)
        out: list[list[float]] = []
        for v in raw:
            out.append([float(x) for x in v])
        return out


# -------------------------------------------------------------- cosine sim ---


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
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


# ------------------------------------------------- SemanticSupersession API ---


def identity_text(claim: Claim, context: str) -> str:
    """Build the identity embedding input — subject predicate [context]. Value excluded."""
    ctx = context.strip().replace("\n", " ")
    if len(ctx) > 160:
        ctx = ctx[:160]
    return f"{claim.subject} {claim.predicate} {ctx}"


class SemanticSupersession:
    """Pass-2 semantic supersession with a pluggable embedder."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        threshold: float = DEFAULT_COSINE_THRESHOLD,
    ) -> None:
        self._embedder: Embedder = embedder or NullEmbedder()
        self._threshold = threshold
        # Identity-vector cache. Pass-2 compares each NEW claim against its session's
        # active candidates, so the SAME candidate identity texts get re-encoded for
        # every new claim in the session — O(claims x candidates) encodes. With the
        # heavy bge-m3 model (~80ms/encode) that re-encoding dominated ingest at
        # haystack scale (~21k encodes for a 500-turn haystack, mostly duplicates) and
        # blew the shard timeout. Caching by text encodes each identity ONCE; candidates
        # become cache hits. Per-instance, so it lives exactly as long as one ingest.
        self._vec_cache: dict[str, list[float]] = {}

    def _embed_cached(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts`, reusing the per-instance cache so each text is encoded once.

        Encodes only cache-misses (deduped, one batched call), then returns vectors in
        the requested order. Soft-capped to bound memory if the instance is long-lived;
        eviction never touches texts in the current request, so the return is always valid.
        """
        missing = [t for t in dict.fromkeys(texts) if t not in self._vec_cache]
        if missing:
            new_vecs = self._embedder.embed(missing)
            for t, v in zip(missing, new_vecs, strict=True):
                self._vec_cache[t] = v
            if len(self._vec_cache) > 50_000:
                keep = set(texts)
                for k in list(self._vec_cache):
                    if len(self._vec_cache) <= 50_000:
                        break
                    if k not in keep:
                        del self._vec_cache[k]
        return [self._vec_cache[t] for t in texts]

    def detect(
        self,
        conn: sqlite3.Connection,
        new_claim: Claim,
        *,
        new_turn_text: str = "",
    ) -> SupersessionEdge | None:
        """Find the nearest prior active claim and supersede it if close enough.

        Two modes, picked by whether the new fact carries a structured predicate:

        - STRUCTURED: candidates share the same predicate family; identity is
          ``subject predicate [context]`` (value excluded so "3 days" vs "4 days"
          still match). Unchanged from the original Pass-2.
        - NL-ONLY (no predicate): the always-available fallback. Candidates are
          all other active facts in the session; identity is the fact's NL
          ``text``. Needs no structured field — determinism comes from the
          deterministic embedder + threshold, not from predicate typing.

        Returns the typed SEMANTIC_REPLACE edge (and marks the old claim
        superseded) if cosine >= threshold, else None. Same-turn and self
        candidates are excluded in both modes.
        """
        nl_mode = not new_claim.predicate
        if nl_mode:
            rows = conn.execute(
                "SELECT * FROM claims WHERE session_id = ?"
                " AND status IN ('active','confirmed')"
                " AND claim_id != ?"
                " AND source_turn_id != ?",
                (
                    new_claim.session_id,
                    new_claim.claim_id,
                    new_claim.source_turn_id,
                ),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM claims WHERE session_id = ?"
                " AND predicate = ?"
                " AND status IN ('active','confirmed')"
                " AND claim_id != ?"
                " AND source_turn_id != ?",
                (
                    new_claim.session_id,
                    new_claim.predicate,
                    new_claim.claim_id,
                    new_claim.source_turn_id,
                ),
            ).fetchall()
        if not rows:
            return None

        from memcontext.claims import row_to_claim
        from memcontext.supersession import _event_blocks

        candidates = [row_to_claim(r) for r in rows]

        # Temporal guard (parity with Pass-1 detect_pass1): two claims that are
        # distinct DATED events — both carry an explicit, DIFFERING event_ts — are
        # separate occurrences and must never supersede each other, even at
        # byte-identical identity text. Pass-2 runs live alongside Pass-1, so
        # without this filter the semantic path would silently retire valid dated
        # history that the Pass-1 guard protects. Drop such candidates before
        # scoring; conservative (fires only when BOTH sides are dated and differ).
        candidates = [c for c in candidates if not _event_blocks(new_claim, c)]

        # FRACTURE B guard (parity with Pass-1): under a coarse predicate the
        # semantic candidate pool is the whole session, so the embedding could
        # match two facts that name DIFFERENT attribute slots (residence vs
        # employer) when their surrounding context is similar. Drop candidates
        # whose value demonstrably names a different slot than the new value.
        # attributes_conflict abstains when either value has no derivable slot, so
        # NL-only / slot-less facts and same-slot updates are unaffected.
        from memcontext.attribute_key import attributes_conflict

        candidates = [
            c for c in candidates
            if not attributes_conflict(new_claim.value, c.value)
        ]
        if not candidates:
            return None

        if nl_mode:
            new_text = new_claim.text or new_turn_text
            cand_texts = [
                c.text or self._lookup_turn_text(conn, c.source_turn_id)
                for c in candidates
            ]
        else:
            new_text = identity_text(new_claim, new_turn_text)
            cand_texts = [
                identity_text(c, self._lookup_turn_text(conn, c.source_turn_id))
                for c in candidates
            ]

        vecs = self._embed_cached([new_text, *cand_texts])
        if len(vecs) != 1 + len(candidates):
            log.error(
                "substrate.semantic_embed_length_mismatch",
                expected=1 + len(candidates),
                actual=len(vecs),
            )
            return None
        new_vec = vecs[0]
        cand_vecs = vecs[1:]

        best_idx = -1
        best_score = -1.0
        for i, v in enumerate(cand_vecs):
            score = cosine(new_vec, v)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0 or best_score < self._threshold:
            log.debug(
                "substrate.semantic_no_match",
                session_id=new_claim.session_id,
                claim_id=new_claim.claim_id,
                best_score=best_score,
                threshold=self._threshold,
            )
            return None

        old_claim = candidates[best_idx]
        if old_claim.status is ClaimStatus.SUPERSEDED:
            return None

        edge = write_supersession_edge(
            conn,
            old_claim_id=old_claim.claim_id,
            new_claim_id=new_claim.claim_id,
            edge_type=EdgeType.SEMANTIC_REPLACE,
            identity_score=best_score,
        )
        set_claim_status(conn, old_claim.claim_id, ClaimStatus.SUPERSEDED)
        log.info(
            "substrate.supersession_pass2",
            session_id=new_claim.session_id,
            old_claim_id=old_claim.claim_id,
            claim_id=new_claim.claim_id,
            cosine=best_score,
        )
        return edge

    @staticmethod
    def _lookup_turn_text(conn: sqlite3.Connection, turn_id: str) -> str:
        row = conn.execute("SELECT text FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        return row["text"] if row is not None else ""
