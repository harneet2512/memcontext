"""Offline proof for the opt-in precision re-rank in retrieve_memory_across.

Diagnosis it guards: the cross-session breadth guarantee serves ~per_session_k ×
n_sessions memories (~140 on a real haystack). Recall is fine (the needle is in
the served set), but the reader drowns and mis-reads/abstains. The opt-in
``rerank_top_k`` re-ranks the breadth pool by query-text cosine and serves only
the top-k — needle survives (recall), flood is cut (precision).

Deterministic, network-free, model-free: a bag-of-words stub embedder.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3

import pytest

from memcontext.claims import insert_turn, new_turn_id, now_ns
from memcontext.retrieval import embed_and_store_episode, retrieve_memory_across
from memcontext.schema import Speaker, Turn, open_database

_DIM = 64


class StubEmbedder:
    """Normalised bag-of-words vectors so cosine == query/text word overlap.

    Deterministic (md5-hashed word buckets), no model download. Distinct enough
    that the needle text out-scores noise text on the query.
    """

    model_version = "stub-bow-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * _DIM
            for raw in t.lower().split():
                w = "".join(ch for ch in raw if ch.isalnum())
                if not w:
                    continue
                idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % _DIM
                v[idx] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


NEEDLE_SID = "sess-needle"
NEEDLE_TEXT = "The romantic Italian restaurant I always recommend is Roscioli near the market."
QUERY = "what is the name of the romantic Italian restaurant"

# 20 noise sessions, each one off-topic turn — the haystack the needle hides in.
NOISE = [
    "I went hiking up the steep mountain trail early on Saturday morning.",
    "My quarterly tax filing spreadsheet needs three more deduction rows.",
    "The puppy chewed through another pair of running shoes last night.",
    "We rebalanced the investment portfolio toward index funds this month.",
    "The community theater is staging a new science fiction play in spring.",
    "I replaced the laptop battery and reinstalled the operating system.",
    "Our camping trip to the national park got rained out on day two.",
    "The yoga instructor introduced a new breathing routine for anxiety.",
    "I assembled a one twenty-fourth scale model car kit over the weekend.",
    "The grocery store moved the coffee creamer to a different aisle.",
    "My commute downtown takes about forty minutes by the express train.",
    "She graduated with a degree in mechanical engineering last June.",
    "The Spotify playlist for focus music now has sixty-two tracks.",
    "We compared three different mortgage lenders for the pre-approval.",
    "The Korean barbecue place near campus added a new lunch special.",
    "I watched all the superhero movies over a long holiday weekend.",
    "The shift rotation sheet assigns the early slot on alternating days.",
    "My aunt mailed a hand-knitted scarf for my birthday this year.",
    "The conference on cloud infrastructure was rescheduled to autumn.",
    "I picked up the dry-cleaning and exchanged a pair of winter boots.",
]


def _add_turn(conn: sqlite3.Connection, sid: str, text: str, emb: StubEmbedder) -> str:
    t = Turn(turn_id=new_turn_id(), session_id=sid, speaker=Speaker.USER, text=text, ts=now_ns())
    insert_turn(conn, t)
    embed_and_store_episode(conn, t, client=emb)
    return t.turn_id


@pytest.fixture()
def conn_and_sessions():
    emb = StubEmbedder()
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    sessions: list[str] = []
    needle_turn_id = _add_turn(conn, NEEDLE_SID, NEEDLE_TEXT, emb)
    sessions.append(NEEDLE_SID)
    for i, txt in enumerate(NOISE):
        sid = f"sess-noise-{i:02d}"
        _add_turn(conn, sid, txt, emb)
        sessions.append(sid)
    try:
        yield conn, sessions, emb, needle_turn_id
    finally:
        conn.close()


def _needle_present(hits, needle_turn_id: str) -> bool:
    return any(h.source_turn_id == needle_turn_id or h.id == needle_turn_id for h, _ in hits)


def test_rerank_cuts_flood_and_keeps_needle(conn_and_sessions):
    conn, sessions, emb, needle_turn_id = conn_and_sessions
    n_sessions = len(sessions)  # 21

    # ---- BASELINE: legacy full breadth (rerank_top_k=None) -> FLOOD ----
    flood = retrieve_memory_across(
        conn, session_ids=sessions, query=QUERY, top_k=10, embedding_client=emb,
    )
    # one episode reserved per session -> the reader is flooded with ~all sessions
    assert len(flood) >= n_sessions - 1, (
        f"baseline should flood (~{n_sessions} memories), got {len(flood)}"
    )
    assert _needle_present(flood, needle_turn_id), "recall: needle must be in the breadth pool"

    # ---- PRECISION: opt-in rerank_top_k=8 -> small, relevant set ----
    precise = retrieve_memory_across(
        conn, session_ids=sessions, query=QUERY, top_k=10, embedding_client=emb,
        rerank_top_k=8,
    )
    # flood is cut...
    assert len(precise) <= 8, f"precision: served set must be <= 8, got {len(precise)}"
    assert len(precise) < len(flood), "precision must serve fewer than the flood"
    # ...without losing the needle (recall preserved) ...
    assert _needle_present(precise, needle_turn_id), (
        "the needle must survive the precision cut — recall is preserved by the reserve"
    )
    # ...and the needle ranks at the TOP by query cosine (it is the relevant one).
    assert precise[0][0].source_turn_id == needle_turn_id or precise[0][0].id == needle_turn_id, (
        "the needle should rank #1 after the cosine re-rank"
    )


def test_default_behaviour_unchanged(conn_and_sessions):
    """Opt-in: with no rerank_top_k the result is byte-identical to legacy."""
    conn, sessions, emb, _ = conn_and_sessions
    a = retrieve_memory_across(conn, session_ids=sessions, query=QUERY, top_k=10, embedding_client=emb)
    b = retrieve_memory_across(conn, session_ids=sessions, query=QUERY, top_k=10, embedding_client=emb,
                               rerank_top_k=None)
    assert [(h.id, h.kind) for h, _ in a] == [(h.id, h.kind) for h, _ in b]
