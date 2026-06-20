"""Byte-identity proof: the PARALLEL faithful ingest produces the SAME claims +
supersession as the SERIAL faithful ingest (queue+drain).

The serial faithful path (what timed out in CI) is:
    q = InlineQueue(conn, extractor, semantic)
    for turn: on_new_turn(..., queue=q)   # store + embed + enqueue
    q.drain()                              # run_extraction per episode (set_context-aware)

The parallel faithful path keeps fidelity but moves the SLOW LLM extract off the
critical path:
    1. store all turns (on_new_turn queue path — no extraction yet)
    2. PRE-FETCH each turn's prior-turn context SERIALLY (main thread; avoids
       SQLite :memory: thread-safety — workers never touch the DB)
    3. PARALLEL extract — each worker uses its OWN extractor instance + set_context
       (the slow part, fanned out)
    4. SERIAL insert in ts order via run_extraction(PassthroughExtractor(claims)) —
       deterministic Pass-1/Pass-2 supersession

If the two paths produce identical claims + edges, "parallel" is faithful, not
degraded. Deterministic stub extractor; no network, no model, :memory: only.
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from memcontext.extraction_queue import InlineQueue
from memcontext.on_new_turn import ExtractedClaim, on_new_turn, run_extraction
from memcontext.schema import Speaker, Turn, open_database


class _Precomputed:
    """Inject already-extracted ExtractedClaim objects into run_extraction's
    insert path, preserving EVERY field (value_normalised, char offsets) — unlike
    a dict round-trip. Not deferrable + no set_context, so run_extraction inserts
    these claims verbatim and does not re-extract."""

    is_deferrable = False

    def __init__(self, claims: list[ExtractedClaim]) -> None:
        self._claims = claims

    def __call__(self, turn: Turn) -> list[ExtractedClaim]:
        return self._claims

SESSION = "s"
# Mix of superseding ('city=') and additive turns, enough prior history that a
# later turn's set_context is non-empty (proves context delivery parity).
TURNS = [
    "city=NYC i just moved into a new apartment here",
    "i really enjoy eating pizza on the weekends",
    "city=Boston i relocated for a new job recently",
    "some notes about my ongoing work project deadlines",
    "city=Seattle now living near the waterfront downtown",
    "my favourite hobby is hiking in the mountains",
]


def _prior_turns_for(conn: sqlite3.Connection, session_id: str, before_ts: int, limit: int = 8):
    """The product's set_context fetch (run_extraction's SELECT), reused verbatim."""
    rows = conn.execute(
        "SELECT * FROM turns WHERE session_id = ? AND ts < ? ORDER BY ts DESC LIMIT ?",
        (session_id, before_ts, limit),
    ).fetchall()
    return [
        Turn(
            turn_id=r["turn_id"], session_id=r["session_id"], speaker=Speaker(r["speaker"]),
            text=r["text"], ts=r["ts"], asr_confidence=r["asr_confidence"],
        )
        for r in reversed(rows)
    ]


class StubExtractor:
    """Deterministic, deferrable extractor. Output depends ONLY on (prior, turn),
    so parallel per-worker instances and the serial shared instance must agree.
    Encodes the prior-turn count into the claim value so set_context parity is
    observable in the snapshot."""

    is_deferrable = True

    def __init__(self) -> None:
        self._prior: list[Turn] = []

    def set_context(self, prior_turns: list[Turn]) -> None:
        self._prior = list(prior_turns)

    def __call__(self, turn: Turn) -> list[ExtractedClaim]:
        n = len(self._prior)
        if "city=" in turn.text:
            city = turn.text.split("city=", 1)[1].split()[0]
            # single slot -> exercises the supersession path across the 3 city turns
            return [ExtractedClaim(subject="user", predicate="lives_in", value=city, confidence=0.9)]
        return [ExtractedClaim(subject="user", predicate="note", value=f"ctx{n}:{turn.text[:18]}", confidence=0.8)]


def _ingest_serial(conn: sqlite3.Connection) -> None:
    """Faithful SERIAL path (queue + drain) — the current adapter behaviour."""
    stub = StubExtractor()
    q = InlineQueue(conn, extractor=stub, semantic=None)
    for txt in TURNS:
        on_new_turn(conn, session_id=SESSION, speaker=Speaker.USER, text=txt,
                    extractor=stub, queue=q, embedder=None)
    q.drain()


def _ingest_parallel(conn: sqlite3.Connection, workers: int = 4) -> dict[str, int]:
    """Faithful PARALLEL path — store, pre-fetch context, parallel extract, serial insert."""
    # Phase 1 — store all turns (no extraction yet)
    stub0 = StubExtractor()
    q = InlineQueue(conn, extractor=stub0, semantic=None)
    stored: list[Turn] = []
    for txt in TURNS:
        r = on_new_turn(conn, session_id=SESSION, speaker=Speaker.USER, text=txt,
                        extractor=stub0, queue=q, embedder=None)
        assert r.turn is not None
        stored.append(r.turn)

    # Phase 2 — pre-fetch each turn's prior context SERIALLY (main thread)
    ctx = {t.turn_id: _prior_turns_for(conn, t.session_id, t.ts) for t in stored}

    # Phase 3 — PARALLEL extract, each worker with its OWN extractor (no set_context race)
    def _worker(t: Turn):
        ex = StubExtractor()               # per-worker instance — thread-isolated state
        ex.set_context(ctx[t.turn_id])
        return (t.turn_id, ex(t))          # ExtractedClaim objects, preserved verbatim

    with ThreadPoolExecutor(max_workers=workers) as pool:
        claims_by_tid = dict(pool.map(_worker, stored))

    # Phase 4 — SERIAL insert in ts order (deterministic supersession)
    for t in sorted(stored, key=lambda x: x.ts):
        run_extraction(conn, episode_id=t.turn_id, session_id=t.session_id,
                       extractor=_Precomputed(claims_by_tid[t.turn_id]), semantic=None)
    return {t.turn_id: len(ctx[t.turn_id]) for t in stored}


def _claim_snapshot(conn: sqlite3.Connection):
    """Semantic content (claim_ids are non-deterministic, so compare values)."""
    return sorted(
        tuple(r) for r in conn.execute(
            "SELECT subject, predicate, value, status FROM claims"
        ).fetchall()
    )


def _edge_snapshot(conn: sqlite3.Connection):
    rows = conn.execute(
        "SELECT e.edge_type, o.value AS old_v, n.value AS new_v "
        "FROM supersession_edges e "
        "JOIN claims o ON o.claim_id = e.old_claim_id "
        "JOIN claims n ON n.claim_id = e.new_claim_id"
    ).fetchall()
    return sorted((r["edge_type"], r["old_v"], r["new_v"]) for r in rows)


@pytest.fixture()
def conns():
    a = open_database(":memory:"); a.row_factory = sqlite3.Row
    b = open_database(":memory:"); b.row_factory = sqlite3.Row
    try:
        yield a, b
    finally:
        a.close(); b.close()


def test_parallel_ingest_is_byte_identical_to_serial(conns):
    serial_conn, parallel_conn = conns
    _ingest_serial(serial_conn)
    prior_counts = _ingest_parallel(parallel_conn)

    # set_context actually delivered context in the parallel path (a later turn
    # received >0 prior turns) — not a cold extraction.
    assert max(prior_counts.values()) > 0, "parallel path must deliver prior-turn context"

    # THE PROOF: identical claims AND identical supersession edges.
    s_claims, p_claims = _claim_snapshot(serial_conn), _claim_snapshot(parallel_conn)
    assert p_claims == s_claims, (
        "parallel ingest produced different CLAIMS than serial — not faithful\n"
        f"serial={s_claims}\nparallel={p_claims}"
    )
    s_edges, p_edges = _edge_snapshot(serial_conn), _edge_snapshot(parallel_conn)
    assert p_edges == s_edges, (
        "parallel ingest produced different SUPERSESSION than serial — not faithful\n"
        f"serial={s_edges}\nparallel={p_edges}"
    )
