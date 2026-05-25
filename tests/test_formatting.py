"""Tests for memcontext.formatting — JSON context formatting for reader LLMs."""
from __future__ import annotations

import json

from memcontext.formatting import format_context_for_reader, format_context_json, format_reader_prompt


def test_format_empty():
    """No inputs produces empty JSON array."""
    result = format_context_for_reader()
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed == []


def test_format_with_claims():
    """Passing claims produces valid JSON with claim data."""
    claims = [
        {
            "claim_id": "cl_abc123",
            "subject": "user",
            "predicate": "user_fact",
            "value": "works at Google",
            "confidence": 0.92,
        },
    ]
    result = format_context_for_reader(claims=claims)
    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["content"] == "works at Google"
    assert parsed[0]["type"] == "claim"


def test_format_with_turns():
    """Passing turns produces valid JSON with turn data."""
    turns = [
        {"speaker": "user", "text": "Hello, my name is Alice"},
        {"speaker": "assistant", "text": "Nice to meet you!"},
    ]
    result = format_context_for_reader(turns=turns)
    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0]["type"] == "turn"
    assert "Alice" in parsed[0]["content"]


def test_reader_prompt_has_chain_of_note():
    """The reader prompt uses Chain-of-Note structure."""
    prompt = format_reader_prompt(
        context_json="[]",
        question="What is my name?",
        question_date="2024/03/15",
    )
    assert "Step 1" in prompt
    assert "Step 2" in prompt
    assert "Step 3" in prompt
    assert "Notes:" in prompt
    assert "Question: What is my name?" in prompt
    assert "2024/03/15" in prompt
