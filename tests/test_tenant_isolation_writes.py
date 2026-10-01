"""Write-side tenant isolation: background passes must stay inside one namespace.

Regressions (pre-existing, found by the HAR-95 security audit):
- consolidate_facts grouped identical (subject, predicate, value) claims across the
  whole DB, so tenant A's fact became the canonical copy and tenant B's copies were
  demoted out of B's retrieval, by accident or on purpose.
- the extractor's prior-turn context was read by session id alone, so with a
  colliding session id (hooks default to "hooks") tenant B's turns were sent to the
  LLM as context for tenant A's turn.
"""
from __future__ import annotations

from memcontext.claims import insert_turn, new_turn_id, now_ns
from memcontext.consolidate import consolidate_facts
from memcontext.mcp_tools import handle_memory_store
from memcontext.on_new_turn import on_new_turn
from memcontext.schema import Speaker, Turn, open_database

FACT = {"subject": "proj", "predicate": "user_fact", "value": "uses postgres"}


def _store(conn, session: str, namespace: str) -> str:
    return handle_memory_store(conn, text="We use postgres.", session_id=session,
                               namespace=namespace, claims=[FACT])["claim_ids"][0]


def _meta(conn, claim_id: str):
    return conn.execute("SELECT consolidated, demoted FROM claim_metadata WHERE claim_id = ?",
                        (claim_id,)).fetchone()


def test_consolidation_never_demotes_another_tenants_fact():
    conn = open_database(":memory:")
    a = _store(conn, "s1", "tenantA")
    b1, b2 = _store(conn, "s2", "tenantB"), _store(conn, "s3", "tenantB")
    consolidate_facts(conn)  # 3 sessions only when tenants are pooled
    for cid in (a, b1, b2):
        assert not _meta(conn, cid)["demoted"]
        assert not _meta(conn, cid)["consolidated"]


def test_consolidation_still_works_within_one_tenant():
    conn = open_database(":memory:")
    a = _store(conn, "s0", "tenantA")
    b = [_store(conn, s, "tenantB") for s in ("s1", "s2", "s3")]
    assert consolidate_facts(conn) == 1
    assert _meta(conn, b[0])["consolidated"] and all(_meta(conn, c)["demoted"] for c in b[1:])
    assert not _meta(conn, a)["demoted"] and not _meta(conn, a)["consolidated"]


def test_a_contested_slot_in_another_tenant_does_not_block_consolidation():
    conn = open_database(":memory:")
    b = [_store(conn, s, "tenantB") for s in ("s1", "s2", "s3")]
    handle_memory_store(conn, text="We use mysql.", session_id="x", namespace="tenantA",
                        claims=[{**FACT, "value": "uses mysql"}])
    assert consolidate_facts(conn) == 1
    assert _meta(conn, b[0])["consolidated"]


class _ContextRecorder:
    """An extractor that records the prior turns it is given as context."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def set_context(self, prior_turns) -> None:
        self.seen = [t.text for t in prior_turns]

    def __call__(self, turn):
        return []


def test_extractor_context_never_includes_another_tenants_turns():
    conn = open_database(":memory:")
    insert_turn(conn, Turn(turn_id=new_turn_id(), session_id="hooks", speaker=Speaker.USER,
                           text="Tenant B secret: the acquisition closes Friday.", ts=now_ns()),
                namespace="tenantB")
    insert_turn(conn, Turn(turn_id=new_turn_id(), session_id="hooks", speaker=Speaker.USER,
                           text="Tenant A earlier note about the billing service.", ts=now_ns()),
                namespace="tenantA")
    rec = _ContextRecorder()
    on_new_turn(conn, session_id="hooks", speaker=Speaker.USER, namespace="tenantA",
                text="Tenant A: what did we decide about billing?", extractor=rec)
    assert rec.seen == ["Tenant A earlier note about the billing service."]


# ── derived caches (HAR-95 option 2: no migration) ────────────────────────────


def _ten_turns(conn, *, namespace: str, session: str, subject: str, start: int = 0) -> None:
    """Ten writes in one session: the 10th triggers the derived-cache rebuild."""
    for i in range(start, start + 10):
        handle_memory_store(conn, text=f"The user mentioned that the {subject} plan item {i} "
                            "matters for next quarter.", session_id=session,
                            namespace=namespace,
                            claims=[{"subject": "user", "predicate": "user_fact",
                                     "value": f"{subject} fact {i}"}])


def _cache_rows(conn) -> tuple[int, int, int]:
    return tuple(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                 for t in ("profiles", "session_digests", "life_events"))


def test_tenant_writes_never_build_the_shared_caches():
    conn = open_database(":memory:")
    _ten_turns(conn, namespace="tenantB", session="b1", subject="tenant-b")
    assert _cache_rows(conn)[:2] == (0, 0)
    assert _cache_rows(conn)[2] == 0


def test_shared_caches_never_include_tenant_data():
    conn = open_database(":memory:")
    _ten_turns(conn, namespace="tenantB", session="b1", subject="tenant-b")
    _ten_turns(conn, namespace="default", session="d1", subject="local")
    text = conn.execute("SELECT profile_text FROM profiles WHERE subject = 'user'").fetchone()[0]
    assert "local fact" in text and "tenant-b" not in text


def test_single_tenant_still_builds_the_caches():
    conn = open_database(":memory:")
    _ten_turns(conn, namespace="default", session="d1", subject="local")
    profiles, digests, _ = _cache_rows(conn)
    assert profiles == 1 and digests == 1
