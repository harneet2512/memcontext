"""HAR-95 baseline: what the claim-centric store preserves, fragments, or loses.

Runs one nuanced business scenario through the real store/query/trace handlers
(no mocks) and characterizes current behavior before any Memory/evidence layer
exists. Passing tests pin behavior that already works and must survive HAR-95.
GAP-1a and GAP-2..4 were closed by HAR-95 slice 1 (memcontext/memories.py + grouped serving)
and are now plain regression tests. Strict-xfail tests are the remaining gaps: each one flips to XPASS (and fails) the
moment the gap is closed, so it must then be promoted to a plain test.

Scenario (one session, 20 unrelated distractor turns so top_k actually binds):
  T1  user    "Sarah said Acme will probably renew, but only if security approves
               SOC2 before Friday."  -> (Acme, renewal_status, probable)
                                       (Acme, renewal_blocker, security_approval)
  T2  user    Globex note             -> (Globex, renewal_status, evaluating)
  T3  user    correction              -> (Acme, renewal_status, confirmed)
  T4  user    restatement             -> (Acme, renewal_status, confirmed)
  TB  browser forum post (low trust)  -> (Acme, renewal_status, churning)

Two vocabularies: the default ``general`` pack (renewal_* is out of vocab) and a
test-local pack that declares renewal_* with renewal_status single-valued.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from memcontext.claims import insert_turn, new_turn_id, now_ns
from memcontext.extractors import PassthroughExtractor
from memcontext.mcp_tools import handle_memory_query, handle_memory_store, handle_memory_trace
from memcontext.on_new_turn import run_extraction
from memcontext.predicate_packs import active_pack
from memcontext.schema import SourceType, Speaker, Turn, open_database

SESSION = "har95"
PACKS_DIR = Path(__file__).resolve().parent.parent / "predicate_packs"

T1_TEXT = "Sarah said Acme will probably renew, but only if security approves SOC2 before Friday."
T2_TEXT = "Globex asked for a quote on the enterprise tier; no renewal decision yet."
T3_TEXT = "Update: security approved SOC2 on Wednesday and Acme confirmed they are renewing."
T4_TEXT = "Acme is renewing, confirmed again on the call today."
TB_TEXT = "Forum post: Acme is churning to a competitor next quarter."

Q_STATUS = "What is Acme's current renewal status?"
Q_BLOCKER = "What is blocking renewal?"
Q_WHY = "Why does Sarah think Acme will renew?"
Q_QUOTE = "What exactly did Sarah say?"
Q_CHANGE = "How did Acme's renewal situation change?"
QUESTIONS = (Q_STATUS, Q_BLOCKER, Q_WHY, Q_QUOTE, Q_CHANGE)

_ACCOUNTS = ("Initech", "Umbrella", "Hooli", "Stark", "Wayne")
_TOPICS = (
    "asked to move the quarterly business review to next Tuesday",
    "reported a billing discrepancy on the March invoice",
    "wants a demo of the analytics add-on for their ops team",
    "is migrating their SSO provider next month",
)


@dataclass(frozen=True)
class Scenario:
    conn: sqlite3.Connection
    turns: dict[str, str]  # label -> turn_id
    store_results: dict[str, dict]

    def label_of(self, turn_id: str) -> str:
        return next((k for k, v in self.turns.items() if v == turn_id), "distractor")


def write_renewal_pack(root: Path, *, single_valued: bool) -> Path:
    """Materialize a packs dir holding `general` plus a `renewal` test pack."""
    shutil.copytree(PACKS_DIR / "general", root / "general", dirs_exist_ok=True)
    general = json.loads((PACKS_DIR / "general" / "predicates.json").read_text(encoding="utf-8"))
    pack = root / "renewal"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "predicates.json").write_text(json.dumps({
        "pack_id": "renewal",
        "predicate_families": general["predicate_families"] + ["renewal_status", "renewal_blocker"],
        "single_valued": ["renewal_status"] if single_valued else [],
    }), encoding="utf-8")
    (pack / "few_shot_examples.json").write_text("[]", encoding="utf-8")
    return root


def _claim(subject: str, predicate: str, value: str) -> dict:
    return {"subject": subject, "predicate": predicate, "value": value}


def build_scenario() -> Scenario:
    """Ingest the scenario through the production handlers. Uses the active pack."""
    conn = open_database(":memory:")
    results: dict[str, dict] = {}

    def store(label: str, text: str, claims: list[dict]) -> None:
        results[label] = handle_memory_store(conn, text=text, session_id=SESSION, claims=claims)

    store("T1", T1_TEXT, [_claim("Acme", "renewal_status", "probable"),
                          _claim("Acme", "renewal_blocker", "security_approval")])
    store("T2", T2_TEXT, [_claim("Globex", "renewal_status", "evaluating")])
    for i, (acct, topic) in enumerate((a, t) for a in _ACCOUNTS for t in _TOPICS):
        store(f"D{i}", f"{acct} {topic}.", [_claim(acct, "observation", topic)])
    store("T3", T3_TEXT, [_claim("Acme", "renewal_status", "confirmed")])
    store("T4", T4_TEXT, [_claim("Acme", "renewal_status", "confirmed")])

    # No product door ingests browser content yet; insert the episode the way a
    # browser source would and run the real extraction tail over it.
    tb = Turn(turn_id=new_turn_id(), session_id=SESSION, speaker=Speaker.USER,
              text=TB_TEXT, ts=now_ns(), source_type=SourceType.BROWSER)
    insert_turn(conn, tb)
    run_extraction(conn, episode_id=tb.turn_id, session_id=SESSION,
                   extractor=PassthroughExtractor([_claim("Acme", "renewal_status", "churning")]))

    turns = {k: v["turn_id"] for k, v in results.items() if not k.startswith("D")}
    turns["TB"] = tb.turn_id
    return Scenario(conn=conn, turns=turns, store_results=results)


def claim_rows(sc: Scenario, *, predicate: str | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM claims WHERE session_id = ?"
    args: list = [SESSION]
    if predicate:
        sql += " AND predicate = ?"
        args.append(predicate)
    return sc.conn.execute(sql + " ORDER BY created_ts", args).fetchall()


def served_sources(sc: Scenario, out: dict) -> list[tuple[str, str]]:
    """(kind, source label) for every served item, in served order."""
    items = []
    for c in out["claims"]:
        src = sc.conn.execute(
            "SELECT source_turn_id FROM claims WHERE claim_id = ?", (c["claim_id"],)
        ).fetchone()[0]
        items.append(("claim", sc.label_of(src)))
    items.extend(("episode", sc.label_of(e["turn_id"])) for e in out["episodes"])
    return items


def baseline_matrix(sc: Scenario) -> dict[str, dict]:
    """Measured slot/token behavior per question (feeds docs/har95/BASELINE.md)."""
    matrix = {}
    for q in QUESTIONS:
        out = handle_memory_query(sc.conn, query=q, session_id=SESSION)
        items = served_sources(sc, out)
        matrix[q] = {
            "served_items": out["token_report"]["served_items"],
            "total_tokens": out["token_report"]["total_tokens"],
            "t1_slots": sum(1 for _, lbl in items if lbl == "T1"),
            "t1_episode_served": ("episode", "T1") in items,
            "superseded_claims_served": sum(1 for c in out["claims"] if c["status"] == "superseded"),
            "items": items,
        }
    return matrix


# ------------------------------------------------------------------ fixtures ---


@pytest.fixture
def renewal_pack(tmp_path, monkeypatch):
    def _use(*, single_valued: bool = True) -> Scenario:
        monkeypatch.setenv("SUBSTRATE_PACKS_DIR", str(write_renewal_pack(tmp_path, single_valued=single_valued)))
        monkeypatch.setenv("ACTIVE_PACK", "renewal")
        active_pack.cache_clear()
        return build_scenario()
    return _use


@pytest.fixture
def sc(renewal_pack) -> Scenario:
    return renewal_pack(single_valued=True)


# ------------------------------------------- preserved (must survive HAR-95) ---


def test_original_evidence_is_stored_verbatim(sc):
    row = sc.conn.execute("SELECT text FROM turns WHERE turn_id = ?", (sc.turns["T1"],)).fetchone()
    assert row["text"] == T1_TEXT


@pytest.mark.parametrize("question", [Q_WHY, Q_QUOTE])
def test_evidence_questions_serve_the_original_turn(sc, question):
    out = handle_memory_query(sc.conn, query=question, session_id=SESSION)
    assert any(e["text"] == T1_TEXT for e in out["episodes"])


def test_every_derived_claim_traces_to_its_source_turn(sc):
    t1_claims = [r for r in claim_rows(sc) if r["source_turn_id"] == sc.turns["T1"]]
    assert {r["predicate"] for r in t1_claims} == {"renewal_status", "renewal_blocker"}
    for r in t1_claims:
        trace = handle_memory_trace(sc.conn, claim_id=r["claim_id"], session_id=SESSION)
        assert trace["source_turn"]["text"] == T1_TEXT


def test_correction_supersedes_single_valued_status(sc):
    status = {r["value"]: r["status"] for r in claim_rows(sc, predicate="renewal_status")
              if r["subject"] == "acme" and r["source_turn_id"] != sc.turns["TB"]}
    assert status == {"probable": "superseded", "confirmed": "active"}
    # the correction's lineage reaches back to the superseded claim and its evidence
    new = next(r for r in claim_rows(sc, predicate="renewal_status")
               if r["source_turn_id"] == sc.turns["T3"])
    trace = handle_memory_trace(sc.conn, claim_id=new["claim_id"], session_id=SESSION)
    assert [s["value"] for s in trace["lineage"]] == ["confirmed", "probable"]
    assert trace["lineage"][-1]["text"] == T1_TEXT


def test_low_trust_browser_claim_cannot_retire_user_state(sc):
    rows = {r["value"]: r for r in claim_rows(sc, predicate="renewal_status") if r["subject"] == "acme"}
    assert rows["churning"]["status"] == "active"
    assert rows["confirmed"]["status"] == "active"
    edge = sc.conn.execute(
        "SELECT edge_type FROM supersession_edges WHERE new_claim_id = ?",
        (rows["churning"]["claim_id"],),
    ).fetchone()
    assert edge["edge_type"] == "contradicts"
    # whichever question surfaces the browser claim, top-level or nested under its
    # evidence, it is flagged, never authoritative
    outs = [handle_memory_query(sc.conn, query=q, session_id=SESSION) for q in QUESTIONS]
    served = [c for out in outs
              for c in out["claims"] + [n for e in out["episodes"] for n in e.get("claims", ())]]
    browser = [c for c in served if c["fact"] == "acme renewal_status churning"]
    assert browser and all(c["quarantined"] for c in browser)
    assert all(not c["quarantined"] for c in served if c["fact"] == "acme renewal_status confirmed")
    assert all(out["contradictions"]["count"] >= 1 for out in outs)


def test_default_pack_demotes_renewal_predicates_to_text_only():
    """Out-of-vocab predicates keep the fact text but lose the structured slot,
    so the correction cannot supersede: current state is fragmented."""
    sc = build_scenario()  # conftest pins ACTIVE_PACK=general
    assert any("renewal_status" in w for w in sc.store_results["T1"]["warnings"])
    acme = [r for r in claim_rows(sc) if "acme" in (r["text"] or "").lower()]
    assert acme and all(r["predicate"] is None for r in acme)
    assert all(r["status"] == "active" for r in acme)
    texts = " | ".join(r["text"] for r in acme)
    assert "probable" in texts and "confirmed" in texts  # both "current"


def test_undeclared_cardinality_treats_categorical_update_as_additive(renewal_pack):
    """By design (supersession.py multi-valued branch): without single_valued,
    'probable' -> 'confirmed' shares no content token, so both stay active.
    Status-like predicates are only correct if the pack declares cardinality."""
    sc = renewal_pack(single_valued=False)
    statuses = {r["value"]: r["status"] for r in claim_rows(sc, predicate="renewal_status")
                if r["subject"] == "acme"}
    assert statuses["probable"] == "active" and statuses["confirmed"] == "active"


# ---------- confirmed gaps: GAP-1a, 2..4 closed by slice 1; GAP-1b, 5..8 still xfail ---


# GAP-1a (closed by slice 1): a served evidence object repeated its claims' content
@pytest.mark.parametrize("question", [Q_WHY, Q_QUOTE])  # the questions that serve T1's evidence
def test_evidence_links_its_served_claims_instead_of_repeating_them(sc, question):
    out = handle_memory_query(sc.conn, query=question, session_id=SESSION)
    items = served_sources(sc, out)
    assert ("episode", "T1") in items  # precondition: not vacuous
    t1 = next(e for e in out["episodes"] if e["turn_id"] == sc.turns["T1"])
    t1_top = {c["claim_id"] for c in out["claims"] if c["source_turn_id"] == sc.turns["T1"]}
    assert set(t1["linked_claim_ids"]) == t1_top
    assert not t1_top & {c["claim_id"] for c in t1["claims"]}


@pytest.mark.xfail(strict=True, reason="GAP-1b: an episode and its ranked claims still count as "
                   "separate top-k items; removing that needs a claims-consumer API migration")
@pytest.mark.parametrize("question", [Q_WHY, Q_QUOTE])
def test_one_evidence_object_takes_at_most_one_slot(sc, question):
    out = handle_memory_query(sc.conn, query=question, session_id=SESSION)
    items = served_sources(sc, out)
    assert ("episode", "T1") in items  # precondition: not vacuous
    assert sum(1 for _, lbl in items if lbl == "T1") <= 1


# GAP-2 (closed by slice 1): served episodes carried no source-trust / quarantine flag
def test_served_episodes_carry_source_trust(sc):
    out = handle_memory_query(sc.conn, query=Q_STATUS, session_id=SESSION)
    browser = next(e for e in out["episodes"] if e["turn_id"] == sc.turns["TB"])
    assert browser["quarantined"] is True


# GAP-3 (closed by slice 1): served claims did not name their source evidence
def test_served_claims_name_their_source_evidence(sc):
    out = handle_memory_query(sc.conn, query=Q_STATUS, session_id=SESSION)
    assert all("source_turn_id" in c for c in out["claims"])


# GAP-4 (closed by slice 1): evidence whose state was superseded was served as if current
def test_evidence_with_superseded_state_is_marked_historical(sc):
    out = handle_memory_query(sc.conn, query=Q_QUOTE, session_id=SESSION)
    t1 = next(e for e in out["episodes"] if e["turn_id"] == sc.turns["T1"])
    assert "superseded" in json.dumps(t1)


@pytest.mark.xfail(strict=True, reason="GAP-5: slot trace picks the newest active claim, even a quarantined one")
def test_slot_trace_head_is_the_trusted_current_value(sc):
    trace = handle_memory_trace(sc.conn, subject="acme", predicate="renewal_status", session_id=SESSION)
    assert trace["claim"]["value"] == "confirmed"
    assert "probable" in [s["value"] for s in trace["lineage"]]


@pytest.mark.xfail(strict=True, reason="GAP-6: 'how did X change' is not detected as a history query")
def test_evolution_question_serves_superseded_state(sc):
    out = handle_memory_query(sc.conn, query=Q_CHANGE, session_id=SESSION)
    assert any(c["status"] == "superseded" for c in out["claims"])


@pytest.mark.xfail(strict=True, reason="GAP-7: no retraction primitive; a resolved blocker stays active")
def test_resolved_blocker_is_no_longer_active(sc):
    blocker = claim_rows(sc, predicate="renewal_blocker")
    assert all(r["status"] != "active" for r in blocker)


@pytest.mark.xfail(strict=True, reason="GAP-8: a restated value is served once per copy")
def test_restated_value_is_served_once():
    sc = build_scenario()  # general pack: T3 and T4 both yield the identical fact text
    out = handle_memory_query(sc.conn, query=Q_STATUS, session_id=SESSION)
    labels = [lbl for kind, lbl in served_sources(sc, out) if kind == "claim"]
    assert {"T3", "T4"} <= set(labels)  # precondition: both copies are in play
    facts = [c["fact"] for c in out["claims"] if c["status"] == "active"]
    assert len(facts) == len(set(facts))


if __name__ == "__main__":  # python -m tests.test_har95_baseline  -> matrix for BASELINE.md
    import tempfile

    os.environ.setdefault("MEMCONTEXT_EMBED_EPISODES", "0")
    for label, single in (("general", None), ("renewal/single_valued", True)):
        if single is None:
            os.environ["SUBSTRATE_PACKS_DIR"] = str(PACKS_DIR)
            os.environ["ACTIVE_PACK"] = "general"
        else:
            os.environ["SUBSTRATE_PACKS_DIR"] = str(write_renewal_pack(Path(tempfile.mkdtemp()), single_valued=single))
            os.environ["ACTIVE_PACK"] = "renewal"
        active_pack.cache_clear()
        print(f"== pack: {label}")
        for q, row in baseline_matrix(build_scenario()).items():
            print(json.dumps({"q": q, **row}))
