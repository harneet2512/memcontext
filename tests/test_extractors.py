from __future__ import annotations

from memcontext.extractors import PassthroughExtractor, SimpleExtractor
from memcontext.schema import Speaker, Turn


def _make_turn(text: str) -> Turn:
    return Turn(
        turn_id="tu_test",
        session_id="s1",
        speaker=Speaker.USER,
        text=text,
        ts=1,
        asr_confidence=None,
    )


def test_passthrough_returns_claims():
    claims = [
        {"subject": "user", "predicate": "user_fact", "value": "lives in Toronto", "confidence": 0.9},
        {"subject": "user", "predicate": "user_preference", "value": "prefers dark mode", "confidence": 0.8},
    ]
    ext = PassthroughExtractor(claims)
    result = ext(_make_turn("I live in Toronto and prefer dark mode"))
    assert len(result) == 2
    assert result[0].subject == "user"
    assert result[0].value == "lives in Toronto"
    assert result[1].predicate == "user_preference"


def test_passthrough_empty():
    ext = PassthroughExtractor([])
    result = ext(_make_turn("anything"))
    assert result == []


def test_passthrough_minimal_claim():
    ext = PassthroughExtractor([{"value": "some fact"}])
    result = ext(_make_turn("text"))
    assert len(result) == 1
    assert result[0].value == "some fact"
    assert result[0].subject == "user"
    assert result[0].predicate == "user_fact"


def test_simple_extractor_preference():
    ext = SimpleExtractor()
    result = ext(_make_turn("I prefer dark mode"))
    assert len(result) >= 1
    preds = [c.predicate for c in result]
    assert "user_preference" in preds


def test_simple_extractor_fact():
    ext = SimpleExtractor()
    result = ext(_make_turn("I am a software engineer"))
    assert len(result) >= 1
    preds = [c.predicate for c in result]
    assert "user_fact" in preds


def test_simple_extractor_fallback():
    ext = SimpleExtractor()
    result = ext(_make_turn("The weather is nice today in downtown"))
    assert len(result) == 1
    assert result[0].predicate == "user_fact"
    assert "weather" in result[0].value.lower()


def test_simple_extractor_low_confidence():
    ext = SimpleExtractor()
    result = ext(_make_turn("I prefer dark mode and I am a developer"))
    for claim in result:
        assert claim.confidence <= 0.5


def test_simple_extractor_empty_text():
    ext = SimpleExtractor()
    result = ext(_make_turn(""))
    assert result == []


def test_simple_extractor_no_network():
    """SimpleExtractor works with no network — it's regex only."""
    ext = SimpleExtractor()
    result = ext(_make_turn("I live in Toronto"))
    assert len(result) >= 1
    assert all(c.confidence <= 0.5 for c in result)
