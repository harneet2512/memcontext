"""Tier-2 NL-first facts: out-of-vocab / NL-only facts are stored, never dropped.

A fact ALWAYS has NL `text` and links to its source episode. The structured
triple is an optional precision layer: present only on an in-vocab, complete
triple; an out-of-vocab predicate DEMOTES to an NL-only fact rather than dropping
it. NL-only facts are retrievable with no structured field.
"""
from __future__ import annotations

import math
import sqlite3
from typing import cast

from memcontext.claims import get_claim, insert_fact, insert_turn, new_turn_id, now_ns
from memcontext.mcp_tools import handle_memory_correct
from memcontext.retrieval import EmbeddingClient, retrieve_hybrid
from memcontext.schema import ClaimStatus, Speaker, Turn, open_database


class _StubEmbedder:
    """model_version only; embed() never called (no claim embeddings here)."""

    model_version = "test-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 / math.sqrt(8)] * 8 for _ in texts]


def _client() -> EmbeddingClient:
    return cast(EmbeddingClient, _StubEmbedder())


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


def test_nl_only_fact_stored_with_null_triple():
    conn = _conn()
    sid = "s1"
    turn = _turn(conn, sid, "the deploy runbook is pinned in the ops channel")
    fact = insert_fact(
        conn,
        session_id=sid,
        source_turn_id=turn.turn_id,
        confidence=0.8,
        text="the deploy runbook is pinned in the ops channel",
    )
    # Dataclass surfaces the empty-string sentinel; storage holds NULL.
    assert fact.subject == "" and fact.predicate == "" and fact.value == ""
    assert fact.text == "the deploy runbook is pinned in the ops channel"
    assert fact.source_turn_id == turn.turn_id  # provenance to the episode
    row = conn.execute(
        "SELECT subject, predicate, value, text FROM claims WHERE claim_id = ?",
        (fact.claim_id,),
    ).fetchone()
    assert row["subject"] is None and row["predicate"] is None and row["value"] is None
    assert row["text"] == fact.text
    # NL-only facts now DO get a claim_metadata row (entity/temporal indexing +
    # an importance slot) — anchored on the top entity, predicate_family='nl'.
    meta = conn.execute(
        "SELECT predicate_family, temporal_bin FROM claim_metadata WHERE claim_id = ?",
        (fact.claim_id,),
    ).fetchone()
    assert meta is not None and meta["predicate_family"] == "nl"


def test_out_of_vocab_predicate_demotes_not_drops():
    conn = _conn()
    sid = "s2"
    turn = _turn(conn, sid, "context")
    fact = insert_fact(
        conn,
        session_id=sid,
        source_turn_id=turn.turn_id,
        confidence=0.7,
        subject="user",
        predicate="totally_made_up_predicate",
        value="some value",
    )
    # Demoted: triple dropped, NL text synthesised from the triple.
    assert fact.predicate == "" and fact.subject == "" and fact.value == ""
    assert fact.text and "totally_made_up_predicate" in fact.text


def test_in_vocab_triple_stays_structured():
    conn = _conn()
    sid = "s3"
    turn = _turn(conn, sid, "I prefer dark mode")
    fact = insert_fact(
        conn,
        session_id=sid,
        source_turn_id=turn.turn_id,
        confidence=0.9,
        subject="user",
        predicate="user_preference",
        value="dark mode",
    )
    # Structured: triple retained (subject normalised), metadata written.
    assert fact.subject == "user" and fact.predicate == "user_preference"
    assert fact.value == "dark mode"
    assert fact.text == "user user_preference dark mode"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM claim_metadata WHERE claim_id = ?", (fact.claim_id,)
    ).fetchone()["n"] == 1


def test_nl_only_fact_populates_entities_from_text():
    conn = _conn()
    sid = "s4"
    turn = _turn(conn, sid, "context")
    fact = insert_fact(
        conn,
        session_id=sid,
        source_turn_id=turn.turn_id,
        confidence=0.8,
        text="Priya owns the Helsinki migration project",
    )
    ents = {
        r["entity_text"]
        for r in conn.execute(
            "SELECT entity_text FROM claim_entities WHERE claim_id = ?",
            (fact.claim_id,),
        )
    }
    # Lightweight entity extraction (no LLM) ran over the NL text.
    assert any(e in ents for e in ("priya", "helsinki")), ents


def test_nl_only_fact_is_retrievable_without_structure():
    conn = _conn()
    sid = "s5"
    t1 = _turn(conn, sid, "a")
    t2 = _turn(conn, sid, "b")
    insert_fact(
        conn, session_id=sid, source_turn_id=t1.turn_id, confidence=0.8,
        text="the staging database password rotates every ninety days",
    )
    insert_fact(
        conn, session_id=sid, source_turn_id=t2.turn_id, confidence=0.8,
        text="lunch options near the office are mostly thai and ramen",
    )
    # retrieve_hybrid ranks on NL text (BM25/entity) with no structured field
    # and no claim embeddings present (semantic channel off).
    results = retrieve_hybrid(
        conn, session_id=sid, query="how often does the staging database password rotate",
        top_k=5, embedding_client=_client(),
    )
    assert results, "NL-only facts must be retrievable"
    top_text = results[0][0].text or ""
    assert "staging database password" in top_text


def test_nl_only_fact_surfaces_in_session_digest():
    """The summary layer is NL-aware: an NL-only fact appears in the digest by
    its text rather than rendering as an empty triple."""
    from memcontext.digests import build_session_digest, format_digest

    conn = _conn()
    sid = "s7"
    turn = _turn(conn, sid, "context")
    insert_fact(
        conn, session_id=sid, source_turn_id=turn.turn_id, confidence=0.95,
        text="the product launch is scheduled for next friday",
    )
    digest = build_session_digest(conn, sid)
    rendered = format_digest(digest)
    assert "the product launch is scheduled for next friday" in rendered
    # And it is NOT rendered as a broken empty triple.
    assert "[] " not in rendered


def test_correcting_an_nl_only_fact_does_not_raise():
    """Regression: correcting an NL-only fact (empty triple) must not feed an
    invalid partial triple back into the writer — it corrects as NL text."""
    conn = _conn()
    sid = "s6"
    turn = _turn(conn, sid, "context")
    fact = insert_fact(
        conn, session_id=sid, source_turn_id=turn.turn_id, confidence=0.8,
        text="the old note",
    )
    result = handle_memory_correct(
        conn, claim_id=fact.claim_id, action="correct", new_value="the corrected note",
    )
    assert "error" not in result, result
    new_fact = get_claim(conn, result["new_claim_id"])
    assert new_fact is not None
    assert new_fact.text == "the corrected note"
    assert new_fact.predicate == ""  # stays NL-only
    # The original is superseded.
    old = get_claim(conn, fact.claim_id)
    assert old is not None and old.status is ClaimStatus.SUPERSEDED
