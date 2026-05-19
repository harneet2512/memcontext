from __future__ import annotations

from memcontext.extractors import (
    LLMExtractor,
    PassthroughExtractor,
    SimpleExtractor,
    _find_char_span,
    _parse_claims,
    _to_extracted_claims,
    auto_extractor,
)
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


# --- JSON parsing (ported from RobbyMD) ---


def test_parse_claims_bare_list():
    result = _parse_claims('[{"subject": "user", "predicate": "user_fact", "value": "x", "confidence": 0.9}]')
    assert len(result) == 1
    assert result[0]["value"] == "x"


def test_parse_claims_wrapper_dict():
    result = _parse_claims('{"claims": [{"subject": "u", "predicate": "p", "value": "v", "confidence": 0.5}]}')
    assert len(result) == 1


def test_parse_claims_single_claim():
    result = _parse_claims('{"subject": "u", "predicate": "p", "value": "v", "confidence": 0.8}')
    assert len(result) == 1


def test_parse_claims_empty_dict():
    result = _parse_claims("{}")
    assert result == []


def test_parse_claims_malformed():
    result = _parse_claims("not json at all")
    assert result == []


def test_parse_claims_empty_list():
    result = _parse_claims("[]")
    assert result == []


# --- Claim validation ---


def test_to_extracted_claims_valid():
    raw = [{"subject": "user", "predicate": "user_fact", "value": "lives in Toronto", "confidence": 0.9}]
    turn = _make_turn("I live in Toronto")
    result = _to_extracted_claims(raw, turn, frozenset({"user_fact", "user_preference"}))
    assert len(result) == 1
    assert result[0].value == "lives in Toronto"


def test_to_extracted_claims_invalid_predicate():
    raw = [{"subject": "user", "predicate": "invalid_xyz", "value": "x", "confidence": 0.9}]
    turn = _make_turn("text")
    result = _to_extracted_claims(raw, turn, frozenset({"user_fact"}))
    assert result == []


def test_to_extracted_claims_bad_confidence():
    raw = [{"subject": "user", "predicate": "user_fact", "value": "x", "confidence": 1.5}]
    turn = _make_turn("text")
    result = _to_extracted_claims(raw, turn, frozenset({"user_fact"}))
    assert result == []


def test_to_extracted_claims_missing_fields():
    raw = [{"subject": "user"}]  # missing predicate, value, confidence
    turn = _make_turn("text")
    result = _to_extracted_claims(raw, turn, frozenset({"user_fact"}))
    assert result == []


# --- Char span resolution ---


def test_find_char_span_exact():
    start, end = _find_char_span("I live in Toronto", "Toronto")
    assert start == 10
    assert end == 17


def test_find_char_span_case_insensitive():
    start, end = _find_char_span("I live in toronto", "Toronto")
    assert start == 10
    assert end == 17


def test_find_char_span_not_found():
    start, end = _find_char_span("I live in Toronto", "Vancouver")
    assert start is None
    assert end is None


# --- Auto-extractor ---


def test_auto_extractor_returns_callable():
    ext = auto_extractor()
    assert callable(ext)
    # Without Ollama, should fall back to SimpleExtractor
    assert isinstance(ext, (LLMExtractor, SimpleExtractor))


# --- LLMExtractor class exists and has right interface ---


def test_llm_extractor_has_is_available():
    assert hasattr(LLMExtractor, "is_available")
    avail = LLMExtractor.is_available()
    assert isinstance(avail, bool)


def test_llm_extractor_callable():
    ext = LLMExtractor(model="test", base_url="http://localhost:99999")
    assert callable(ext)
