"""Behavioral prompt unit tests -- synthetic plumbing tests.

These verify that prompt routing, formatting, and category selection work.
They are NOT benchmark evidence and do NOT substitute for real LongMemEval runs.
"""
from __future__ import annotations

from evals.longmemeval_prompts import PROMPTS, format_claims_for_prompt, get_prompt


def test_prompt_exists_for_all_categories():
    required = [
        "single_session_user_fact",
        "single_session_preference",
        "cross_session_preference",
        "cross_session_user_fact",
        "temporal_ordering",
        "knowledge_update",
        "abstention",
    ]
    for cat in required:
        assert cat in PROMPTS, f"Missing prompt for {cat}"


def test_get_prompt_formats_correctly():
    prompt = get_prompt("single_session_preference", "1. prefers dark mode", "What does the user prefer?")
    assert "prefers dark mode" in prompt
    assert "What does the user prefer?" in prompt


def test_preference_prompt_synthesizes_implicit():
    prompt = PROMPTS["single_session_preference"]
    assert "IMPLICIT" in prompt or "implicit" in prompt


def test_abstention_prompt_mentions_not_available():
    prompt = PROMPTS["abstention"]
    assert "not available" in prompt


def test_knowledge_update_prompt_mentions_recent():
    prompt = PROMPTS["knowledge_update"]
    assert "most recent" in prompt.lower() or "current" in prompt.lower() or "updated" in prompt.lower()


def test_temporal_prompt_mentions_order():
    prompt = PROMPTS["temporal_ordering"]
    assert "order" in prompt.lower() or "timing" in prompt.lower() or "time" in prompt.lower()


def test_format_claims_for_prompt():
    claims = [
        {"subject": "user", "predicate": "user_preference", "value": "dark mode", "confidence": 0.9},
        {"subject": "user", "predicate": "user_fact", "value": "lives in Toronto"},
    ]
    text = format_claims_for_prompt(claims)
    assert "dark mode" in text
    assert "Toronto" in text
    assert "1." in text
    assert "2." in text


def test_format_claims_empty():
    text = format_claims_for_prompt([])
    assert "no claims" in text.lower()


def test_get_prompt_unknown_category_uses_fallback():
    prompt = get_prompt("unknown_category_xyz", "claims", "question?")
    assert "question?" in prompt  # should use fallback, not crash


def test_new_prompt_single_session_assistant():
    assert "single_session_assistant" in PROMPTS
    prompt = PROMPTS["single_session_assistant"]
    assert "assistant" in prompt.lower()


def test_new_prompt_default():
    assert "default" in PROMPTS
    prompt = PROMPTS["default"]
    assert "{claims}" in prompt
    assert "{question}" in prompt


def test_category_map_exists_and_covers_dataset_names():
    from evals.longmemeval_prompts import CATEGORY_MAP

    dataset_categories = [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "temporal-reasoning",
        "knowledge-update",
        "multi-session",
        "abstention",
    ]
    for cat in dataset_categories:
        assert cat in CATEGORY_MAP, f"CATEGORY_MAP missing dataset category: {cat}"
        prompt_key = CATEGORY_MAP[cat]
        assert prompt_key in PROMPTS, f"CATEGORY_MAP maps {cat} to {prompt_key} which is not in PROMPTS"


def test_get_prompt_resolves_dataset_category_names():
    """Verify that get_prompt works with hyphenated dataset category names."""
    prompt = get_prompt("single-session-user", "1. some claim", "What is user's name?")
    assert "What is user's name?" in prompt
    assert "some claim" in prompt
