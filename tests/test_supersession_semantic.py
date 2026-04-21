from __future__ import annotations

import math
import sqlite3

from memcontext.claims import get_claim, insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.schema import ClaimStatus, EdgeType, Speaker, Turn
from memcontext.supersession_semantic import (
    NullEmbedder,
    SemanticSupersession,
    cosine,
    identity_text,
)


def _make_turn(db: sqlite3.Connection, session_id: str, speaker: Speaker, text: str) -> Turn:
    turn = Turn(
        turn_id=new_turn_id(),
        session_id=session_id,
        speaker=speaker,
        text=text,
        ts=now_ns(),
        asr_confidence=None,
    )
    insert_turn(db, turn)
    return turn


def test_null_embedder_constant_vectors():
    emb = NullEmbedder(dim=4)
    result = emb.embed(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 4
    assert result[0] == result[1]


def test_cosine_unit_vectors():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert abs(cosine(a, b) - 0.0) < 1e-6

    c = [1.0, 0.0, 0.0]
    d = [1.0, 0.0, 0.0]
    assert abs(cosine(c, d) - 1.0) < 1e-6


def test_cosine_zero_vectors():
    a = [0.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert cosine(a, b) == 0.0


def test_identity_text_excludes_value():
    from memcontext.schema import Claim, ClaimStatus
    claim = Claim(
        claim_id="cl_test",
        session_id="s1",
        subject="user",
        predicate="user_preference",
        value="prefers dark mode",
        value_normalised=None,
        confidence=0.9,
        source_turn_id="tu_test",
        status=ClaimStatus.ACTIVE,
        created_ts=1,
        char_start=None,
        char_end=None,
        valid_from_ts=1,
        valid_until_ts=None,
    )
    text = identity_text(claim, "I prefer dark mode")
    assert "user" in text.lower()
    assert "user_preference" in text.lower()
    assert "prefers dark mode" not in text


def test_semantic_detect_with_null_embedder(
    db: sqlite3.Connection, session_id: str,
):
    ss = SemanticSupersession(embedder=NullEmbedder(dim=4), threshold=0.5)

    t1 = _make_turn(db, session_id, Speaker.USER, "I like coffee")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="likes coffee", confidence=0.9, source_turn_id=t1.turn_id,
    )

    t2 = _make_turn(db, session_id, Speaker.USER, "I prefer tea")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="prefers tea", confidence=0.9, source_turn_id=t2.turn_id,
    )

    edge = ss.detect(db, c2, new_turn_text=t2.text)
    assert edge is not None
    assert edge.edge_type == EdgeType.SEMANTIC_REPLACE
    assert edge.old_claim_id == c1.claim_id


def test_semantic_detect_high_threshold_no_match(
    db: sqlite3.Connection, session_id: str,
):
    ss = SemanticSupersession(embedder=NullEmbedder(dim=4), threshold=2.0)

    t1 = _make_turn(db, session_id, Speaker.USER, "I like coffee")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="likes coffee", confidence=0.9, source_turn_id=t1.turn_id,
    )

    t2 = _make_turn(db, session_id, Speaker.USER, "I prefer tea")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="prefers tea", confidence=0.9, source_turn_id=t2.turn_id,
    )

    edge = ss.detect(db, c2, new_turn_text=t2.text)
    assert edge is None


def test_semantic_marks_old_superseded(
    db: sqlite3.Connection, session_id: str,
):
    ss = SemanticSupersession(embedder=NullEmbedder(dim=4), threshold=0.5)

    t1 = _make_turn(db, session_id, Speaker.USER, "I like coffee")
    c1 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="likes coffee", confidence=0.9, source_turn_id=t1.turn_id,
    )

    t2 = _make_turn(db, session_id, Speaker.USER, "I prefer tea")
    c2 = insert_claim(
        db, session_id=session_id, subject="user", predicate="user_preference",
        value="prefers tea", confidence=0.9, source_turn_id=t2.turn_id,
    )

    ss.detect(db, c2, new_turn_text=t2.text)

    old = get_claim(db, c1.claim_id)
    assert old is not None
    assert old.status == ClaimStatus.SUPERSEDED
