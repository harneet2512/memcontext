"""HAR-95 slice 1: a Memory (evidence) with derived claims, served without duplication.

A Memory is the stored turn (verbatim evidence + source); its claims are the
structured state derived from it. These tests pin the representation
(Memory <- Claim provenance, state after correction, trust) and the serving
invariants: no claim is served twice, the top-level claims list stays complete,
and a served Memory says what became of the state derived from it.
"""
from __future__ import annotations

import copy

import pytest

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import handle_memory_query, handle_memory_store, handle_memory_trace
from memcontext.memories import get_memories, get_memory, memory_for_claim
from memcontext.on_new_turn import on_new_turn, run_extraction
from memcontext.predicate_packs import active_pack
from memcontext.retrieval import classify_query_depth, detect_history_intent, retrieve_memory
from memcontext.schema import SourceType, Speaker, Turn, open_database
from memcontext.serving import MAX_NESTED_CLAIMS, annotate_served_evidence
from tests.test_har95_baseline import (
    Q_QUOTE,
    Q_STATUS,
    Q_WHY,
    QUESTIONS,
    SESSION,
    T1_TEXT,
    TB_TEXT,
    Scenario,
    build_scenario,
    claim_rows,
    write_renewal_pack,
)


@pytest.fixture
def renewal_pack(tmp_path, monkeypatch):
    def _use(*, single_valued: bool = True) -> Scenario:
        packs = write_renewal_pack(tmp_path, single_valued=single_valued)
        monkeypatch.setenv("SUBSTRATE_PACKS_DIR", str(packs))
        monkeypatch.setenv("ACTIVE_PACK", "renewal")
        active_pack.cache_clear()
        return build_scenario()
    return _use


@pytest.fixture
def sc(renewal_pack) -> Scenario:
    return renewal_pack(single_valued=True)


def _claim_of(sc: Scenario, label: str, predicate: str) -> str:
    return next(r["claim_id"] for r in claim_rows(sc, predicate=predicate)
                if r["source_turn_id"] == sc.turns[label])


# ---------------------------------------------------------- representation ---


def test_memory_preserves_the_evidence_and_its_source(sc):
    m = get_memory(sc.conn, sc.turns["T1"])
    assert m is not None
    assert m.content == T1_TEXT
    assert m.source_type == "conversation" and m.speaker == "user"
    assert m.trust == 1.0 and not m.quarantined


def test_memory_derives_several_claims_with_their_current_status(sc):
    m = get_memory(sc.conn, sc.turns["T1"])
    status = {c.fact: c.status for c in m.claims}
    assert status == {
        "acme renewal_status probable": "superseded",
        "acme renewal_blocker security_approval": "active",
    }
    assert m.state == "mixed"


def test_corrected_memory_is_current_and_the_old_one_keeps_its_evidence(sc):
    t3 = get_memory(sc.conn, sc.turns["T3"])
    assert t3.state == "current"
    # the superseded claim's evidence is still fully available
    assert get_memory(sc.conn, sc.turns["T1"]).content == T1_TEXT


def test_claim_traces_to_its_memory(sc):
    cid = _claim_of(sc, "T1", "renewal_blocker")
    m = memory_for_claim(sc.conn, cid)
    assert m is not None and m.memory_id == sc.turns["T1"]
    assert cid in {c.claim_id for c in m.claims}


def test_memory_without_claims_is_allowed():
    conn = open_database(":memory:")
    r = on_new_turn(conn, session_id="s", speaker=Speaker.USER,
                    text="We talked through the Q3 roadmap at a high level.",
                    extractor=lambda turn: [])
    assert r.turn is not None
    m = get_memory(conn, r.turn.turn_id)
    assert m is not None and m.claims == () and m.state == "no_claims"
    assert m.content == "We talked through the Q3 roadmap at a high level."


def test_inline_browser_claim_keeps_its_own_quarantine_flag(renewal_pack):
    """A superseded browser claim is inlined under its evidence, still quarantined."""
    sc = renewal_pack(single_valued=True)
    tb = Turn(turn_id=new_turn_id(), session_id=SESSION, speaker=Speaker.USER, ts=now_ns(),
              text="Scraped page: Initech is churning next month.", source_type=SourceType.BROWSER)
    insert_turn(sc.conn, tb)
    run_extraction(sc.conn, episode_id=tb.turn_id, session_id=SESSION, extractor=PassthroughExtractor(
        [{"subject": "Initech", "predicate": "renewal_status", "value": "churning"}]))
    fix = handle_memory_store(sc.conn, session_id=SESSION, text="Initech renewed today.",
                              claims=[{"subject": "Initech", "predicate": "renewal_status",
                                       "value": "renewed"}])
    assert fix["supersessions"] == 1  # the user's correction retires the web claim
    out = handle_memory_query(sc.conn, query="Initech churning scraped page", session_id=SESSION)
    ep = next(e for e in out["episodes"] if e["turn_id"] == tb.turn_id)
    assert ep["quarantined"] is True and ep["state"] == "historical"
    assert [(c["status"], c["quarantined"]) for c in ep["claims"]] == [("superseded", True)]


def test_browser_memory_is_quarantined(sc):
    m = get_memory(sc.conn, sc.turns["TB"])
    assert m.content == TB_TEXT
    assert m.source_type == "browser" and m.quarantined


def test_get_memories_batches_and_skips_unknown_ids(sc):
    got = get_memories(sc.conn, [sc.turns["T1"], "tu_missing", sc.turns["T3"]])
    assert set(got) == {sc.turns["T1"], sc.turns["T3"]}


def test_unknown_memory_is_none(sc):
    assert get_memory(sc.conn, "tu_missing") is None
    assert memory_for_claim(sc.conn, "cl_missing") is None


# ----------------------------------------------------------------- serving ---


def _all_served_claim_ids(out: dict) -> list[str]:
    ids = [c["claim_id"] for c in out["claims"]]
    ids += [c["claim_id"] for e in out["episodes"] for c in e["claims"]]
    return ids


@pytest.mark.parametrize("question", QUESTIONS)
def test_no_claim_is_served_twice(sc, question):
    ids = _all_served_claim_ids(handle_memory_query(sc.conn, query=question, session_id=SESSION))
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("question", [Q_WHY, Q_QUOTE])
def test_served_evidence_carries_its_derived_state(sc, question):
    out = handle_memory_query(sc.conn, query=question, session_id=SESSION)
    t1 = next(e for e in out["episodes"] if e["turn_id"] == sc.turns["T1"])
    assert t1["state"] == "mixed"
    top = {c["claim_id"]: c for c in out["claims"]}
    # every derived claim is visible exactly once: linked (already top-level) or inline
    shown = {cid: (top[cid]["fact"], top[cid]["status"]) for cid in t1["linked_claim_ids"]}
    shown |= {c["claim_id"]: (c["fact"], c["status"]) for c in t1["claims"]}
    assert set(shown.values()) == {
        ("acme renewal_status probable", "superseded"),
        ("acme renewal_blocker security_approval", "active"),
    }
    assert not set(t1["linked_claim_ids"]) & {c["claim_id"] for c in t1["claims"]}


_PRE_HAR95_CLAIM_KEYS = {
    "claim_id", "subject", "predicate", "value", "fact", "confidence", "status", "score",
    "durability", "consolidated", "trust", "quarantined",
}


def _ranked(sc: Scenario, question: str):
    _, k = classify_query_depth(question)
    return retrieve_memory(sc.conn, session_id=SESSION, query=question, top_k=k,
                           include_superseded=detect_history_intent(question))


@pytest.mark.parametrize("question", QUESTIONS)
def test_top_level_claims_are_the_complete_ranked_state(sc, question):
    """Compat: top-level claims are exactly the ranker's fact hits, in order, with the
    pre-HAR-95 keys plus source_turn_id. Consumers (hooks) read only this list."""
    hits = _ranked(sc, question)
    out = handle_memory_query(sc.conn, query=question, session_id=SESSION)
    assert [c["claim_id"] for c in out["claims"]] == [h.id for h, _ in hits if h.kind == "fact"]
    assert all(set(c) == _PRE_HAR95_CLAIM_KEYS | {"source_turn_id"} for c in out["claims"])


def test_top_level_claims_name_their_memory(sc):
    out = handle_memory_query(sc.conn, query=Q_STATUS, session_id=SESSION)
    for c in out["claims"]:
        assert memory_for_claim(sc.conn, c["claim_id"]).memory_id == c["source_turn_id"]


def test_served_episodes_carry_trust(sc):
    for q in QUESTIONS:
        for e in handle_memory_query(sc.conn, query=q, session_id=SESSION)["episodes"]:
            assert e["quarantined"] is (e["turn_id"] == sc.turns["TB"])
            assert 0.0 <= e["trust"] <= 1.0


def test_state_question_still_serves_current_state(sc):
    out = handle_memory_query(sc.conn, query=Q_STATUS, session_id=SESSION)
    confirmed = [c for c in out["claims"] if c["fact"] == "acme renewal_status confirmed"]
    confirmed += [c for e in out["episodes"] for c in e["claims"]
                  if c["fact"] == "acme renewal_status confirmed"]
    assert confirmed and all(c["status"] == "active" for c in confirmed)


@pytest.mark.parametrize("question", QUESTIONS)
def test_annotation_adds_no_slots_and_only_historical_lines(sc, question):
    """Served items are exactly the ranked hits; the only added content lines are
    historical claims (the measured +7 tokens on evidence questions)."""
    hits = _ranked(sc, question)
    out = handle_memory_query(sc.conn, query=question, session_id=SESSION)
    assert out["token_report"]["served_items"] == len(hits)
    nested = [c for e in out["episodes"] for c in e["claims"]]
    assert out["token_report"]["nested_claims"] == len(nested)
    assert all(c["status"] in {"superseded", "dismissed"} for c in nested)


def test_serve_ledger_covers_nested_claims(sc):
    out = handle_memory_query(sc.conn, query=Q_QUOTE, session_id=SESSION)
    assert any(e["claims"] for e in out["episodes"])  # precondition: something is inlined
    ledger = {r[0] for r in sc.conn.execute(
        "SELECT claim_id FROM serve_events WHERE query = ?", (Q_QUOTE,))}
    assert set(_all_served_claim_ids(out)) <= ledger


def test_trace_from_a_served_claim_reaches_memory_and_source(sc):
    out = handle_memory_query(sc.conn, query=Q_STATUS, session_id=SESSION)
    cid = out["claims"][0]["claim_id"]
    trace = handle_memory_trace(sc.conn, claim_id=cid, session_id=SESSION)
    assert trace["source_turn"]["turn_id"] == out["claims"][0]["source_turn_id"]
    assert trace["source_turn"]["source_type"] == "conversation"


def test_default_pack_nl_only_claims_link_too():
    sc = build_scenario()  # general pack: every claim is text-only
    out = handle_memory_query(sc.conn, query=Q_QUOTE, session_id=SESSION)
    t1 = next(e for e in out["episodes"] if e["turn_id"] == sc.turns["T1"])
    t1_top = {c["claim_id"] for c in out["claims"] if c["source_turn_id"] == sc.turns["T1"]}
    assert t1_top and set(t1["linked_claim_ids"]) == t1_top and t1["claims"] == []
    assert out["token_report"]["served_memories"] < out["token_report"]["served_items"]


def test_cross_session_query_groups_too(sc):
    other = handle_memory_store(sc.conn, text="Acme also asked about SOC2 evidence packs.",
                                session_id="other", claims=[{"subject": "Acme", "predicate": "observation",
                                                             "value": "asked about SOC2 evidence packs"}])
    out = handle_memory_query(sc.conn, query="Acme SOC2 evidence packs", session_id=None)
    ep = next(e for e in out["episodes"] if e["turn_id"] == other["turn_id"])
    assert ep["state"] == "current" and ep["trust"] == 1.0
    ids = _all_served_claim_ids(out)
    assert len(ids) == len(set(ids))


def test_namespace_denial_is_unchanged(sc):
    out = handle_memory_query(sc.conn, query=Q_STATUS, session_id=SESSION, namespace="tenant-b")
    assert out == {"claims": [], "episodes": [], "total": 0, "denied": "namespace"}


def test_unrelated_browser_turn_without_claims_is_quarantined_evidence():
    conn = open_database(":memory:")
    t = Turn(turn_id=new_turn_id(), session_id="s", speaker=Speaker.USER, ts=now_ns(),
             text="Scraped page: Initech pricing doubled.", source_type=SourceType.BROWSER)
    insert_turn(conn, t)
    out = handle_memory_query(conn, query="Initech pricing", session_id="s")
    ep = next(e for e in out["episodes"] if e["turn_id"] == t.turn_id)
    assert ep["quarantined"] is True and ep["state"] == "no_claims"
    assert ep["claims"] == [] and ep["linked_claim_ids"] == []


# ------------------------------------------------------- unit: annotation ---


def _turn_with_claims(conn, n: int, *, session: str = "u") -> str:
    turn = Turn(turn_id=new_turn_id(), session_id=session, speaker=Speaker.USER,
                text="A long planning note.", ts=now_ns())
    insert_turn(conn, turn)
    for i in range(n):
        insert_claim(conn, session_id=session, subject=f"item{i}", predicate="observation",
                     value=f"value {i}", confidence=0.9, source_turn_id=turn.turn_id)
    return turn.turn_id


def test_annotation_caps_historical_claims_newest_first():
    conn = open_database(":memory:")
    tid = _turn_with_claims(conn, 10)
    conn.execute("UPDATE claims SET status = 'superseded' WHERE source_turn_id = ?", (tid,))
    [ep] = annotate_served_evidence(conn, [], [{"turn_id": tid, "text": "x"}])
    assert len(ep["claims"]) == MAX_NESTED_CLAIMS
    assert ep["historical_claims_omitted"] == 10 - MAX_NESTED_CLAIMS
    assert ep["claims"][0]["fact"].startswith("item9")  # newest first
    assert ep["state"] == "historical"


def test_claim_without_metadata_row_defaults_to_neutral_trust():
    conn = open_database(":memory:")
    tid = _turn_with_claims(conn, 1)
    conn.execute("UPDATE claims SET status = 'superseded' WHERE source_turn_id = ?", (tid,))
    conn.execute("DELETE FROM claim_metadata")
    [ep] = annotate_served_evidence(conn, [], [{"turn_id": tid, "text": "x"}])
    assert ep["claims"][0]["trust"] == 0.5 and ep["claims"][0]["quarantined"] is False


def test_annotation_does_not_mutate_inputs_and_handles_empty():
    conn = open_database(":memory:")
    tid = _turn_with_claims(conn, 2)
    claims = [{"claim_id": "cl_x", "fact": "f"}]
    episodes = [{"turn_id": tid, "text": "x"}]
    before = copy.deepcopy((claims, episodes))
    annotate_served_evidence(conn, claims, episodes)
    assert (claims, episodes) == before
    assert annotate_served_evidence(conn, claims, []) == []


def test_vanished_episode_is_skipped_not_served_unannotated():
    conn = open_database(":memory:")
    assert annotate_served_evidence(conn, [], [{"turn_id": "tu_gone", "text": "x"}]) == []


def test_memory_never_lists_a_claim_from_another_session():
    conn = open_database(":memory:")
    tid = _turn_with_claims(conn, 1, session="a")
    stray = insert_claim(conn, session_id="b", subject="x", predicate="observation",
                         value="foreign", confidence=0.9, source_turn_id=tid)
    m = get_memory(conn, tid)
    assert stray.claim_id not in {c.claim_id for c in m.claims} and len(m.claims) == 1
