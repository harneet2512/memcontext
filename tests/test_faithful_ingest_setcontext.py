"""No-API proof that the faithful (deferred-queue) ingest path runs the
product's set_context-aware extraction.

This mirrors EXACTLY what the patched AMB adapter does:

    q = InlineQueue(conn, extractor=stub, semantic=None)
    for turn in session:
        on_new_turn(conn, session_id=..., speaker=..., text=...,
                    extractor=stub, queue=q, embedder=None)
    q.drain()

and proves three product invariants with a pure-stdlib stub extractor:

  ASSERT 1 — set_context fires through the deferred path: a later turn's
             extraction received >0 prior turns. The OLD cold/parallel pool
             extracted each turn in isolation and would record 0.
  ASSERT 2 — episode floor: a turn whose extractor returns [] still persists
             a retrievable `turns` row (episode survives with zero claims).
  ASSERT 3 — deferred, not inline: immediately after on_new_turn (pre-drain)
             the episode is PENDING and NO claims exist; after drain claims
             exist. Falls back to a claim-count check if the status enum is
             unavailable.

Network-free, model-free, :memory: only.
"""
from __future__ import annotations

import sqlite3

import pytest

from memcontext.extraction_queue import InlineQueue
from memcontext.on_new_turn import ExtractedClaim, on_new_turn
from memcontext.schema import Speaker, open_database

try:  # exact status enum if present; test still works without it
    from memcontext.schema import ExtractionStatus

    _PENDING_VALUE = ExtractionStatus.PENDING.value
except Exception:  # noqa: BLE001
    ExtractionStatus = None  # type: ignore[assignment]
    _PENDING_VALUE = "pending"


class StubDeferrableExtractor:
    """A deferrable extractor that records the prior-turn context it was handed.

    `is_deferrable = True` forces on_new_turn onto the queue path (it stores +
    enqueues, never extracts inline). `set_context` is the product hook that
    `run_extraction` calls with up to 8 prior same-session turns BEFORE
    `__call__`; we record, per turn text, how many prior turns we received.

    A claim is emitted ONLY when prior context was actually supplied, so a
    context-aware (deferred/faithful) run is observably distinct from a cold,
    context-free (old parallel-pool) run, which would supply none.
    """

    is_deferrable = True

    def __init__(self) -> None:
        # turn text -> number of prior turns set_context handed us for that turn
        self.prior_counts: dict[str, int] = {}
        self._pending_context: list = []

    def set_context(self, prior_turns: list) -> None:
        # Store the received prior turns for the NEXT __call__.
        self._pending_context = list(prior_turns)

    def __call__(self, turn) -> list[ExtractedClaim]:
        n_prior = len(self._pending_context)
        self.prior_counts[turn.text] = n_prior
        # consume the context so it does not leak into the next episode
        self._pending_context = []

        # The marker turn deliberately produces NOTHING -> exercises the
        # episode floor (Tier-1 persistence with zero claims).
        if turn.text == EMPTY_TURN_TEXT:
            return []

        # Emit a claim ONLY when we actually got prior context. This makes
        # "context-aware" provable: cold extraction (0 prior) yields no claim.
        if n_prior > 0:
            return [
                ExtractedClaim(
                    subject="user",
                    predicate="prefers",
                    value=f"ctx:{n_prior}:{turn.text}",
                    confidence=0.9,
                )
            ]
        return []


SESSION_ID = "sess-faithful"
EMPTY_TURN_TEXT = "this marker turn yields no claims"

# Several non-trivial turns so admission keeps them and there is real prior
# history for a later turn. Index 1 is the empty/floor marker.
TURN_TEXTS = [
    "I really love hiking in the mountains every summer weekend.",
    EMPTY_TURN_TEXT,
    "My favorite programming language for backend work is Python.",
    "I prefer my coffee black with no sugar in the morning.",
]


def _claim_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]


def _turn_status(conn: sqlite3.Connection, turn_id: str) -> str | None:
    row = conn.execute(
        "SELECT extraction_status FROM turns WHERE turn_id = ?", (turn_id,)
    ).fetchone()
    return None if row is None else row[0]


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = open_database(":memory:")
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def test_faithful_path_runs_setcontext_aware_extraction(conn: sqlite3.Connection) -> None:
    stub = StubDeferrableExtractor()
    # Faithful path: deferrable extractor + a queue -> on_new_turn defers.
    q = InlineQueue(conn, extractor=stub, semantic=None)

    turn_ids: list[str] = []
    for text in TURN_TEXTS:
        result = on_new_turn(
            conn,
            session_id=SESSION_ID,
            speaker=Speaker.USER,
            text=text,
            extractor=stub,
            queue=q,
            embedder=None,
        )
        assert result.admitted, f"turn was rejected by admission: {text!r}"
        assert result.turn is not None
        turn_ids.append(result.turn.turn_id)

        # ASSERT 3a (deferred, not inline): the episode is enqueued, NOT
        # extracted inline. No claims exist yet, and the episode is PENDING.
        assert result.created_claims == (), "deferred path must not create claims inline"
        assert stub.prior_counts == {}, "extractor must not run before drain()"

    # Pre-drain invariants across the whole session.
    assert _claim_count(conn) == 0, "no claims must exist before drain()"
    for tid in turn_ids:
        status = _turn_status(conn, tid)
        assert status == _PENDING_VALUE, (
            f"episode {tid} should be PENDING before drain, got {status!r}"
        )

    # Run the deferred Tier-2 tail (this is where set_context fires).
    q.drain()

    # ----- ASSERT 1: set_context fired through the deferred path -----
    # Every non-empty turn was extracted...
    for text in TURN_TEXTS:
        assert text in stub.prior_counts, f"extractor never ran for {text!r}"
    # ...and a LATER turn received >0 prior turns. A cold/parallel path
    # (per-turn isolation, no set_context) would record 0 here.
    last_text = TURN_TEXTS[-1]
    assert stub.prior_counts[last_text] > 0, (
        "the faithful path must hand prior-turn context to a later episode; "
        f"got {stub.prior_counts[last_text]} prior turns -> looks like cold extraction"
    )
    # First turn has no predecessors -> exactly 0 prior turns (sanity bound).
    assert stub.prior_counts[TURN_TEXTS[0]] == 0

    # ----- ASSERT 3b: claims exist only AFTER drain, one per context-bearing turn -----
    # The stub emits a claim ONLY for a turn that received prior context, so the
    # claim count equals the number of context-bearing turns. Turn 0 (no
    # predecessors) and turn 1 (the empty marker, short-circuited) emit nothing;
    # turns 2 and 3 each received prior context and emit one claim -> exactly 2.
    # That linkage is itself the proof set_context delivered context to the right
    # turns. We assert on COUNT, not the value string, because the product demotes
    # the unknown 'prefers' predicate to NL and rewrites the stored value.
    n_context_turns = sum(
        1 for t in TURN_TEXTS if t != EMPTY_TURN_TEXT and stub.prior_counts.get(t, 0) > 0
    )
    assert n_context_turns == 2, f"expected 2 context-bearing turns, got {n_context_turns}"
    assert _claim_count(conn) == n_context_turns, (
        f"claims ({_claim_count(conn)}) must equal context-bearing turns "
        f"({n_context_turns}); a claim was minted iff set_context delivered context"
    )

    # ----- ASSERT 2: episode floor -----
    # The empty-marker turn produced zero claims yet its episode row persists
    # and is retrievable.
    empty_idx = TURN_TEXTS.index(EMPTY_TURN_TEXT)
    empty_turn_id = turn_ids[empty_idx]
    floor_row = conn.execute(
        "SELECT turn_id, text FROM turns WHERE turn_id = ?", (empty_turn_id,)
    ).fetchone()
    assert floor_row is not None, "episode floor: zero-claim turn must still persist"
    assert floor_row["text"] == EMPTY_TURN_TEXT
    # And it contributed no claims of its own.
    floor_claims = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE source_turn_id = ?", (empty_turn_id,)
    ).fetchone()[0]
    assert floor_claims == 0, "the empty-marker episode must have zero claims"
