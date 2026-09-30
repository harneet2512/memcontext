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

from memcontext.schema import Claim, ClaimStatus, EdgeType, SupersessionEdge
from memcontext.supersession import (
    claim_trust,
    normalize_value,
    retire_claim,
    write_supersession_edge,
)

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
            from sentence_transformers import (  # pyright: ignore[reportMissingImports]
                SentenceTransformer,
            )
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
            # NL-only identity is fuzzy, so it stays session-scoped — and always
            # inside the new claim's namespace: a same-named session in another
            # namespace belongs to a different tenant.
            rows = conn.execute(
                "SELECT c.* FROM claims c JOIN turns t ON t.turn_id = c.source_turn_id"
                " WHERE c.session_id = ?"
                " AND t.namespace = (SELECT namespace FROM turns WHERE turn_id = ?)"
                " AND c.status IN ('active','confirmed')"
                " AND c.claim_id != ?"
                " AND c.source_turn_id != ?"
                " AND c.created_ts < ?",
                (
                    new_claim.session_id,
                    new_claim.source_turn_id,
                    new_claim.claim_id,
                    new_claim.source_turn_id,
                    new_claim.created_ts,
                ),
            ).fetchall()
        else:
            # Structured claims share Pass-1's scope: the namespace (project) of the
            # new claim's episode, across sessions; only strictly older candidates.
            rows = conn.execute(
                "SELECT c.* FROM claims c JOIN turns t ON t.turn_id = c.source_turn_id"
                " WHERE t.namespace = (SELECT namespace FROM turns WHERE turn_id = ?)"
                " AND c.predicate = ?"
                " AND c.status IN ('active','confirmed')"
                " AND c.claim_id != ?"
                " AND c.source_turn_id != ?"
                " AND c.created_ts < ?",
                (
                    new_claim.source_turn_id,
                    new_claim.predicate,
                    new_claim.claim_id,
                    new_claim.source_turn_id,
                    new_claim.created_ts,
                ),
            ).fetchall()
        if not rows:
            return None

        from memcontext.claims import row_to_claim

        candidates = [row_to_claim(r) for r in rows]

        # An identical restatement is recurrence evidence (Pass 1 keeps both copies
        # on purpose, and consolidation counts them), not a replacement.
        def _surface(c: Claim) -> str:
            return normalize_value(c.value if c.predicate else (c.text or ""))

        new_surface = _surface(new_claim)
        if any(_surface(c) == new_surface and (not c.predicate or c.subject == new_claim.subject)
               for c in candidates):
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

        vecs = self._embedder.embed([new_text, *cand_texts])
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

        # Same source-trust guard as Pass 1: a markedly lower-trust source (an
        # assistant restatement, a web page) never retires a higher-trust fact.
        if claim_trust(conn, new_claim.claim_id) + 0.2 < claim_trust(conn, old_claim.claim_id):
            log.info("substrate.supersession_pass2_blocked_low_trust",
                     old_claim_id=old_claim.claim_id, new_claim_id=new_claim.claim_id)
            return None

        edge = write_supersession_edge(
            conn,
            old_claim_id=old_claim.claim_id,
            new_claim_id=new_claim.claim_id,
            edge_type=EdgeType.SEMANTIC_REPLACE,
            identity_score=best_score,
        )
        retire_claim(conn, old_claim.claim_id, edge.created_ts)
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
