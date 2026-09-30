"""Supersession defects surfaced by evals/product/supersession_matrix.py (dev split).

- Pass-2 NL-only candidates were filtered by session id only, so a same-named
  session in ANOTHER namespace could retire a tenant's fact (xn-06/xn-07).
- Pass-1 deliberately keeps an identical restatement (recurrence evidence), but
  Pass-2 then ran and retired it at cosine ~0.9-0.97; Pass-2 also had no source-
  trust guard, so an assistant echo could retire a user fact (dup-*, tru-10).
- A value differing only by trailing punctuation counted as a new value (dup-10).
- The historical-record guard ran only in the attribute-slot branch, and only
  recognised closed date ranges: "managed the payments team from 2019 to 2021" was
  clobbered through the token-overlap branch, and "used to live in Denver" by
  "lives in Austin" (his-*).
"""
from __future__ import annotations

import sqlite3

import pytest

from memcontext.claims import get_claim, insert_claim, insert_fact, insert_turn, new_turn_id, now_ns
from memcontext.predicate_packs import active_pack
from memcontext.schema import ClaimStatus, Speaker, Turn
from memcontext.supersession import detect_pass1
from memcontext.supersession_semantic import SemanticSupersession


@pytest.fixture(autouse=True)
def packs(monkeypatch):
    monkeypatch.setenv("ACTIVE_PACK", "general,developer")
    active_pack.cache_clear()
    yield
    active_pack.cache_clear()


class _SameVector:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _store(db, value, *, session="s1", namespace="default", speaker=Speaker.USER,
           subject="user", predicate="user_fact", text=None, pass2=False):
    t = Turn(turn_id=new_turn_id(), session_id=session, speaker=speaker,
             text=text or f"{subject} {value}", ts=now_ns())
    insert_turn(db, t, namespace=namespace)
    if predicate is None:
        c = insert_fact(db, session_id=session, text=value, confidence=0.9,
                        source_turn_id=t.turn_id)
    else:
        c = insert_claim(db, session_id=session, subject=subject, predicate=predicate,
                         value=value, confidence=0.9, source_turn_id=t.turn_id)
    edge = detect_pass1(db, c)
    if edge is None and pass2:
        edge = SemanticSupersession(_SameVector()).detect(db, c, new_turn_text=t.text)
    return c, edge


def _active(db, claim) -> bool:
    return get_claim(db, claim.claim_id).status is ClaimStatus.ACTIVE


# ── Pass 2: namespace, restatement, trust ──────────────────────────────────

def test_pass2_nl_facts_never_cross_namespaces(db: sqlite3.Connection):
    a, _ = _store(db, "the quarterly report is due on Friday", session="shared",
                  namespace="tenantA", predicate=None)
    _, edge = _store(db, "the quarterly report is due on Monday", session="shared",
                     namespace="tenantB", predicate=None, pass2=True)
    assert edge is None and _active(db, a)


def test_pass2_keeps_an_identical_restatement(db: sqlite3.Connection):
    first, _ = _store(db, "prefers dark mode in every editor")
    second, edge = _store(db, "prefers dark mode in every editor", session="s2", pass2=True)
    assert edge is None and _active(db, first) and _active(db, second)


def test_pass2_lower_trust_source_never_retires_a_user_fact(db: sqlite3.Connection):
    user, _ = _store(db, "the launch is planned for March", predicate=None)
    _, edge = _store(db, "the launch is planned for April", predicate=None,
                     speaker=Speaker.ASSISTANT, pass2=True)
    assert edge is None and _active(db, user)


# ── Pass 1: normalisation and history ─────────────────────────────────────

def test_trailing_punctuation_is_not_a_new_value(db: sqlite3.Connection):
    first, _ = _store(db, "use SQLite for local development",
                      subject="dev database", predicate="decision_made")
    second, edge = _store(db, "use SQLite for local development.", session="s2",
                          subject="dev database", predicate="decision_made")
    assert edge is None and _active(db, first) and _active(db, second)


def test_closed_window_history_survives_the_overlap_branch(db: sqlite3.Connection):
    past, _ = _store(db, "managed the payments team from 2019 to 2021")
    _, edge = _store(db, "manages the payments team")
    assert edge is None and _active(db, past)


def test_used_to_marks_a_historical_record(db: sqlite3.Connection):
    past, _ = _store(db, "used to live in Denver")
    current, edge = _store(db, "lives in Austin")
    assert edge is None and _active(db, past) and _active(db, current)


def test_no_longer_is_still_an_update(db: sqlite3.Connection):
    # "no longer" states the CURRENT value; it is not a historical marker.
    old, _ = _store(db, "lives in Denver")
    _, edge = _store(db, "moved to Austin, no longer lives in Denver")
    assert edge is not None and not _active(db, old)
