"""Pass-2 semantic supersession over NL-only facts (the always-available path).

An NL-only fact (no structured predicate) supersedes a prior active fact when
their natural-language text is close enough under the embedder + threshold —
no structured field required. Structured facts keep the predicate-scoped path.
"""
from __future__ import annotations

import math
import re
import sqlite3
from typing import cast

from memcontext.claims import get_claim, insert_fact, insert_turn, new_turn_id, now_ns
from memcontext.schema import ClaimStatus, EdgeType, Speaker, Turn, open_database
from memcontext.supersession_semantic import Embedder, SemanticSupersession


class _HashEmbedder:
    """Deterministic normalised bag-of-words embedder (content-aware cosine)."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    @staticmethod
    def _bucket(tok: str, dim: int) -> int:
        acc = 0
        for ch in tok:
            acc = (acc * 31 + ord(ch)) % dim
        return acc

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self._dim
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                v[self._bucket(tok, self._dim)] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def _conn() -> sqlite3.Connection:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _turn(conn: sqlite3.Connection, sid: str, text: str) -> Turn:
    turn = Turn(
        turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER,
        text=text, ts=now_ns(),
    )
    insert_turn(conn, turn)
    return turn


def _sem() -> SemanticSupersession:
    # Lower threshold than the MiniLM-tuned 0.88: bag-of-words cosine for a
    # one-word-different paraphrase is ~0.87, while unrelated text is ~0.1.
    return SemanticSupersession(embedder=cast(Embedder, _HashEmbedder()), threshold=0.6)


def _nl_fact(conn: sqlite3.Connection, sid: str, text: str):
    turn = _turn(conn, sid, text)
    return insert_fact(
        conn, session_id=sid, source_turn_id=turn.turn_id, confidence=0.8, text=text
    )


def test_nl_only_fact_supersedes_a_paraphrase():
    conn = _conn()
    sid = "s1"
    sem = _sem()

    a = _nl_fact(conn, sid, "the production database password rotates every thirty days")
    assert sem.detect(conn, a) is None  # nothing prior

    b = _nl_fact(conn, sid, "the production database password rotates every ninety days")
    edge = sem.detect(conn, b)
    assert edge is not None, "near-duplicate NL fact should supersede the prior one"
    assert edge.edge_type is EdgeType.SEMANTIC_REPLACE
    assert edge.old_claim_id == a.claim_id and edge.new_claim_id == b.claim_id
    assert edge.identity_score is not None and edge.identity_score >= 0.6

    old = get_claim(conn, a.claim_id)
    assert old is not None and old.status is ClaimStatus.SUPERSEDED


def test_nl_only_unrelated_fact_does_not_supersede():
    conn = _conn()
    sid = "s2"
    sem = _sem()

    a = _nl_fact(conn, sid, "the production database password rotates every thirty days")
    sem.detect(conn, a)
    c = _nl_fact(conn, sid, "lunch options near the office are mostly thai and ramen")
    assert sem.detect(conn, c) is None, "unrelated NL fact must not supersede"

    old = get_claim(conn, a.claim_id)
    assert old is not None and old.status is ClaimStatus.ACTIVE


def test_structured_path_stays_predicate_scoped():
    """A structured fact only matches candidates in the SAME predicate family —
    it does not fall into the NL all-facts comparison."""
    conn = _conn()
    sid = "s3"
    sem = _sem()

    t1 = _turn(conn, sid, "context one")
    t2 = _turn(conn, sid, "context two")
    f1 = insert_fact(
        conn, session_id=sid, source_turn_id=t1.turn_id, confidence=0.9,
        subject="user", predicate="user_preference", value="dark mode",
    )
    sem.detect(conn, f1)
    # Same subject/value text but a DIFFERENT predicate family -> not a candidate.
    f2 = insert_fact(
        conn, session_id=sid, source_turn_id=t2.turn_id, confidence=0.9,
        subject="user", predicate="user_fact", value="dark mode",
    )
    assert sem.detect(conn, f2) is None, "structured match must stay predicate-scoped"
    old = get_claim(conn, f1.claim_id)
    assert old is not None and old.status is ClaimStatus.ACTIVE
