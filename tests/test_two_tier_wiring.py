"""Characterization tests for the two-tier WIRING (Phase 1 + 2).

These pin behavior that did NOT exist before the wiring — each test is written
so it would be RED on the pre-wiring code:

- `handle_memory_query` used to return ``episodes: []`` whenever a session had
  any active claims (mcp_tools.py:177 old). Tests here assert episodes ARE
  returned alongside facts → red before.
- cross-session used `retrieve_hybrid` only (facts), merged per-session →
  episodes never appeared. Test asserts episodes survive the global fusion → red
  before.
- `session_digests` had no writer in production. Test asserts the digest tool
  persists a row → red before (table stayed empty).

A passing full suite means none of this regressed; these tests mean the wiring
actually does what it claims.
"""
from __future__ import annotations

import sqlite3

from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import handle_memory_digest, handle_memory_query
from memcontext.on_new_turn import on_new_turn
from memcontext.retrieval import retrieve_memory_across
from memcontext.schema import Speaker, open_database


def _ingest(conn, session_id, text, subject, predicate, value):
    on_new_turn(
        conn, session_id=session_id, speaker=Speaker.USER, text=text,
        extractor=PassthroughExtractor(
            [{"subject": subject, "predicate": predicate,
              "value": value, "confidence": 0.9}]
        ),
    )


def _fresh():
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_query_door_returns_facts_AND_episodes():
    """The live door serves the unified two-tier — not facts-only with episodes:[].

    RED before: old handle_memory_query returned episodes:[] when claims existed.
    """
    conn = _fresh()
    _ingest(conn, "s1", "I live in Berlin and hike on weekends",
            "user", "user_location", "Berlin")
    _ingest(conn, "s1", "I work as a data engineer in Berlin",
            "user", "user_occupation", "data engineer")

    res = handle_memory_query(conn, query="where does the user live in Berlin",
                              session_id="s1", top_k=10)

    assert res["claims"], "facts must surface"
    assert res["episodes"], "episodes MUST surface (would be [] on old code)"
    # episodes carry the real turn text + source_type
    assert all("text" in e and "source_type" in e for e in res["episodes"])


def test_cross_session_episodes_survive_global_fusion():
    """Episodes are not drowned by facts across many sessions.

    RED before: cross-session merged per-session retrieve_hybrid (facts only),
    so episodes never appeared. The global single-fusion lets an episode at
    global rank 1 outrank a fact at rank ~8.
    """
    conn = _fresh()
    for i in range(6):
        _ingest(conn, f"s{i}", f"In session {i} I visited the Berlin museum of art",
                "user", "user_activity", f"museum visit {i}")

    res = handle_memory_query(conn, query="Berlin museum visit", session_id=None,
                              top_k=20)
    assert res["claims"], "facts surface cross-session"
    assert res["episodes"], "episodes survive the global fusion (red on old merge)"


def test_retrieve_memory_across_interleaves_kinds():
    """The cross-session primitive returns BOTH kinds, source-tagged."""
    conn = _fresh()
    for i in range(4):
        _ingest(conn, f"s{i}", f"I bought a camera lens number {i} in Berlin",
                "user", "user_purchase", f"lens {i}")

    hits = retrieve_memory_across(
        conn, session_ids=[f"s{i}" for i in range(4)],
        query="camera lens Berlin", top_k=20,
    )
    kinds = {h.kind for h, _ in hits}
    assert "fact" in kinds, "facts present"
    assert "episode" in kinds, "episodes present (not drowned)"
    # fused scores are descending
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_digest_tool_builds_and_persists():
    """memory_digest builds + PERSISTS the summary layer.

    RED before: session_digests had no writer; the table stayed empty in prod.
    """
    conn = _fresh()
    _ingest(conn, "s1", "I moved to Berlin", "user", "user_location", "Berlin")
    _ingest(conn, "s1", "I prefer tea over coffee",
            "user", "user_preference", "tea")

    before = conn.execute("SELECT COUNT(*) FROM session_digests").fetchone()[0]
    out = handle_memory_digest(conn, session_id="s1")
    after = conn.execute("SELECT COUNT(*) FROM session_digests").fetchone()[0]

    assert before == 0
    assert after == 1, "digest must be persisted (table was always empty before)"
    assert out["total_claims"] >= 2
    assert out["key_facts"], "summary surfaces key facts"
    assert out["text"], "rendered digest text present"


def test_life_events_tool_detects_and_persists():
    """life_events tool detects a burst of diverse predicate changes + PERSISTS.

    RED before: detect_life_events had no caller → life_events table empty.
    Uses IN-VOCAB predicates so they stay structured (NL-only facts don't cluster).
    """
    from memcontext.mcp_tools import handle_memory_life_events

    conn = _fresh()
    # 4 distinct in-vocab predicates for 'user' in one window -> a life event
    for pred, val in [("user_event", "started a new job"),
                      ("user_goal", "learn German"),
                      ("user_relationship", "met Anna"),
                      ("user_fact", "turned thirty")]:
        _ingest(conn, "s1", f"Today I {val}", "user", pred, val)

    before = conn.execute("SELECT COUNT(*) FROM life_events").fetchone()[0]
    out = handle_memory_life_events(conn, subject="user", min_predicates=3)
    after = conn.execute("SELECT COUNT(*) FROM life_events").fetchone()[0]

    assert before == 0
    assert out["count"] >= 1, "a 4-predicate burst must surface a life event"
    assert after >= 1, "life_events must be persisted (table was always empty before)"


def test_event_frames_tool_assembles_and_persists():
    """event_frames tool groups co-referent claims into event records + PERSISTS.

    RED before: assemble_event_frames had no caller → event_frames table empty.
    """
    from memcontext.mcp_tools import handle_memory_events

    conn = _fresh()
    # A purchase-shaped cluster in one turn-neighbourhood (item/amount/location).
    _ingest(conn, "s1", "I bought a Sony camera for $500 in Tokyo",
            "user", "user_event", "bought a Sony camera")
    _ingest(conn, "s1", "The camera cost five hundred dollars",
            "user", "user_fact", "camera cost $500")

    before = conn.execute("SELECT COUNT(*) FROM event_frames").fetchone()[0]
    out = handle_memory_events(conn, session_id="s1")
    after = conn.execute("SELECT COUNT(*) FROM event_frames").fetchone()[0]

    assert before == 0
    assert isinstance(out["events"], list)
    # assemble self-persists; if it produced any frame the table reflects it
    assert after == out["count"], "persisted count must match returned count"
    assert out["count"] >= 1, "co-referent claims should assemble >=1 event frame"


def test_volatility_tool_classifies():
    """volatility tool is reachable + classifies a slot from supersession history.

    RED before: classify_predicate had no serving door.
    """
    from memcontext.mcp_tools import handle_memory_volatility

    conn = _fresh()
    _ingest(conn, "s1", "I prefer tea", "user", "user_preference", "tea")
    out = handle_memory_volatility(conn, subject="user", predicate="user_preference")
    assert out["classification"] in {"stable", "evolving", "volatile"}
    assert out["classification"] == "stable"  # single fact, no supersession yet
    assert "change_count" in out and out["change_count"] == 0


def test_tuples_tool_projects_active_facts():
    """event-tuple tool projects active facts into (subject, action, object) rows.

    RED before: claims_to_events had no serving door.
    """
    from memcontext.mcp_tools import handle_memory_tuples

    conn = _fresh()
    _ingest(conn, "s1", "I started a job", "user", "user_event", "started a job")
    _ingest(conn, "s1", "My goal is German", "user", "user_goal", "learn German")
    out = handle_memory_tuples(conn, session_id="s1")
    assert out["count"] >= 2
    assert all({"subject", "action", "obj", "claim_id"} <= set(t) for t in out["tuples"])


def test_entity_graph_tool_returns_shape():
    """entity-graph tool builds the co-occurrence graph + returns neighbors shape.

    RED before: EntityGraph had no serving door.
    """
    from memcontext.mcp_tools import handle_memory_entity_graph

    conn = _fresh()
    _ingest(conn, "s1", "Anna and Bob visited Berlin together",
            "user", "user_relationship", "knows Anna and Bob")
    out = handle_memory_entity_graph(conn, session_id="s1", entity="anna")
    assert out["entity"] == "anna"
    assert isinstance(out["neighbors"], list)


def test_cross_session_fusion_is_rank_based_not_raw_score():
    """retrieve_memory_across must fuse sessions by RANK (RRF), not raw score.

    A verbose session with many query-matching facts has larger raw BM25/semantic
    scores than a terse session that actually holds the answer. Correct RRF gives
    each session's rank-1 the same weight, so the answer session is represented.

    RED before the fix: the function pooled all sessions' facts and sorted by raw
    score, so the high-score 'noise' session filled top_k and the 'answer'
    session was dropped entirely. This is a general IR property, not a
    LongMemEval quirk.
    """
    from memcontext.claims import get_turn

    conn = _fresh()
    # Two symmetric sessions, each holding one query-matching fact. Whatever their
    # raw scores, correct RRF gives EACH session's rank-1 the same reciprocal-rank
    # weight (1/(RRF_K+1)). Global raw-score pooling instead assigns them GLOBAL
    # ranks 1 and 2 -> different scores -> one session is implicitly demoted.
    _ingest(conn, "sA", "I live in Berlin", "user", "user_fact", "lives in Berlin")
    _ingest(conn, "sB", "I work in Berlin", "user", "user_fact", "works in Berlin")

    hits = retrieve_memory_across(conn, session_ids=["sA", "sB"],
                                  query="Berlin", top_k=10)

    top_fact_score: dict = {}
    for h, s in hits:
        if h.kind != "fact":
            continue
        t = get_turn(conn, h.source_turn_id)
        sid = t.session_id if t is not None else None
        top_fact_score.setdefault(sid, s)  # first = highest (list is score-sorted)

    assert set(top_fact_score) >= {"sA", "sB"}, "both sessions must be represented"
    scores = [top_fact_score["sA"], top_fact_score["sB"]]
    assert abs(scores[0] - scores[1]) < 1e-9, (
        "per-session rank-1 facts must score EQUALLY (rank-based RRF); unequal "
        "scores mean the fusion ranked by raw score globally and demoted a session"
    )
