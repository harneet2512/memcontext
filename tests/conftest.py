from __future__ import annotations

import os
import sqlite3

import pytest

from memcontext.claims import insert_claim, insert_turn, new_turn_id, now_ns
from memcontext.event_bus import EventBus
from memcontext.on_new_turn import ExtractedClaim
from memcontext.schema import Speaker, Turn, open_database
from memcontext.supersession_semantic import NullEmbedder


@pytest.fixture()
def db() -> sqlite3.Connection:
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture()
def session_id() -> str:
    return "test-session-001"


@pytest.fixture()
def sample_turn(db: sqlite3.Connection, session_id: str) -> Turn:
    turn = Turn(
        turn_id=new_turn_id(),
        session_id=session_id,
        speaker=Speaker.USER,
        text="I prefer dark mode and my favorite language is Python",
        ts=now_ns(),
        asr_confidence=None,
    )
    insert_turn(db, turn)
    return turn


@pytest.fixture()
def sample_claim(db: sqlite3.Connection, session_id: str, sample_turn: Turn):
    return insert_claim(
        db,
        session_id=session_id,
        subject="user",
        predicate="user_preference",
        value="prefers dark mode",
        confidence=0.9,
        source_turn_id=sample_turn.turn_id,
    )


@pytest.fixture()
def null_embedder() -> NullEmbedder:
    return NullEmbedder(dim=8)


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def extractor_fn():
    def _extract(turn: Turn) -> list[ExtractedClaim]:
        return [
            ExtractedClaim(
                subject="user",
                predicate="user_preference",
                value="prefers dark mode",
                confidence=0.9,
            ),
        ]
    return _extract


@pytest.fixture(autouse=True)
def _set_packs_dir(monkeypatch: pytest.MonkeyPatch):
    packs_dir = os.path.join(os.path.dirname(__file__), "..", "predicate_packs")
    monkeypatch.setenv("SUBSTRATE_PACKS_DIR", os.path.abspath(packs_dir))
    monkeypatch.setenv("ACTIVE_PACK", "general")
    # Never load/download an embedding model via the production episode-embed path
    # in CI; tests that exercise embedding inject an explicit stub client instead.
    monkeypatch.setenv("MEMCONTEXT_EMBED_EPISODES", "0")
    # Tests should not auto-select a cloud extractor just because the developer
    # shell has router credentials configured.
    for name in (
        "MEMCONTEXT_EXTRACTOR_API_KEY",
        "MEMCONTEXT_EXTRACTOR_BACKEND",
        "MEMCONTEXT_EXTRACTOR_ENDPOINT",
        "MEMCONTEXT_EXTRACTOR_MODEL",
        "MEMCONTEXT_EXTRACTOR_REASONING_EFFORT",
        "MEMCONTEXT_EXTRACTOR_REASONING_EXCLUDE",
    ):
        monkeypatch.delenv(name, raising=False)
    from memcontext.predicate_packs import active_pack
    active_pack.cache_clear()
    yield
    active_pack.cache_clear()
