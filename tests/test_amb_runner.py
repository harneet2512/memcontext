"""Smoke tests for the AMB runner.

Uses inline datasets and :memory: SQLite. No external API keys or models required.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from evals.amb_runner import (
    AMBConversation,
    AMBQuestion,
    AMBResult,
    load_amb_dataset,
)


# ---------------------------------------------------------------------------
# Inline dataset fixtures
# ---------------------------------------------------------------------------

GROUPED_DATASET = [
    {
        "conversation_id": "conv_001",
        "messages": [
            {"role": "user", "content": "My favorite color is blue."},
            {"role": "assistant", "content": "Got it, blue is your favorite color!"},
            {"role": "user", "content": "I work as a software engineer."},
            {"role": "assistant", "content": "Nice, software engineering is a great field."},
        ],
        "questions": [
            {
                "question_id": "q_001",
                "question": "What is the user's favorite color?",
                "gold_answer": "blue",
                "category": "preference",
            },
            {
                "question_id": "q_002",
                "question": "What does the user do for work?",
                "gold_answer": "software engineer",
                "category": "fact",
            },
        ],
    },
    {
        "conversation_id": "conv_002",
        "messages": [
            {"role": "user", "content": "I have two cats named Luna and Milo."},
            {"role": "assistant", "content": "Luna and Milo, lovely names!"},
        ],
        "questions": [
            {
                "question_id": "q_003",
                "question": "What are the names of the user's cats?",
                "gold_answer": "Luna and Milo",
                "category": "fact",
            },
        ],
    },
]

FLAT_DATASET = [
    {
        "question_id": "q_flat_001",
        "conversation_id": "conv_flat_001",
        "question": "What is the user's favorite programming language?",
        "gold_answer": "Python",
        "category": "preference",
        "messages": [
            {"role": "user", "content": "I love programming in Python."},
            {"role": "assistant", "content": "Python is a great language!"},
        ],
    },
    {
        "question_id": "q_flat_002",
        "conversation_id": "conv_flat_001",
        "question": "What language does the user like?",
        "expected_answer": "Python",
        "type": "preference",
        "conversation": [
            {"role": "user", "content": "I love programming in Python."},
            {"role": "assistant", "content": "Python is a great language!"},
        ],
    },
]


@pytest.fixture()
def grouped_dataset_path(tmp_path: Path) -> str:
    path = tmp_path / "amb_grouped.json"
    path.write_text(json.dumps(GROUPED_DATASET), encoding="utf-8")
    return str(path)


@pytest.fixture()
def flat_dataset_path(tmp_path: Path) -> str:
    path = tmp_path / "amb_flat.json"
    path.write_text(json.dumps(FLAT_DATASET), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Tests: data structures
# ---------------------------------------------------------------------------


def test_amb_conversation_dataclass():
    conv = AMBConversation(
        conversation_id="c1",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert conv.conversation_id == "c1"
    assert len(conv.messages) == 1


def test_amb_question_dataclass():
    q = AMBQuestion(
        question_id="q1",
        conversation_id="c1",
        question="What?",
        gold_answer="answer",
        category="fact",
    )
    assert q.question_id == "q1"
    assert q.gold_answer == "answer"


def test_amb_result_dataclass():
    r = AMBResult(
        question_id="q1",
        category="fact",
        predicted_answer="answer",
        gold_answer="answer",
        score=1.0,
    )
    assert r.score == 1.0


# ---------------------------------------------------------------------------
# Tests: dataset loading — grouped format
# ---------------------------------------------------------------------------


def test_load_grouped_conversations(grouped_dataset_path: str):
    conversations, questions = load_amb_dataset(grouped_dataset_path)
    assert len(conversations) == 2
    assert conversations[0].conversation_id == "conv_001"
    assert len(conversations[0].messages) == 4
    assert conversations[1].conversation_id == "conv_002"
    assert len(conversations[1].messages) == 2


def test_load_grouped_questions(grouped_dataset_path: str):
    conversations, questions = load_amb_dataset(grouped_dataset_path)
    assert len(questions) == 3
    assert questions[0].question_id == "q_001"
    assert questions[0].conversation_id == "conv_001"
    assert questions[0].gold_answer == "blue"
    assert questions[0].category == "preference"
    assert questions[2].question_id == "q_003"
    assert questions[2].conversation_id == "conv_002"


# ---------------------------------------------------------------------------
# Tests: dataset loading — flat format
# ---------------------------------------------------------------------------


def test_load_flat_dataset(flat_dataset_path: str):
    conversations, questions = load_amb_dataset(flat_dataset_path)
    # Both questions point to the same conversation, so 1 unique conversation
    assert len(conversations) == 1
    assert conversations[0].conversation_id == "conv_flat_001"


def test_load_flat_questions(flat_dataset_path: str):
    conversations, questions = load_amb_dataset(flat_dataset_path)
    assert len(questions) == 2
    assert questions[0].question_id == "q_flat_001"
    assert questions[0].gold_answer == "Python"
    assert questions[0].category == "preference"
    # Second question uses "expected_answer" and "type" field names
    assert questions[1].question_id == "q_flat_002"
    assert questions[1].gold_answer == "Python"
    assert questions[1].category == "preference"


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------


def test_load_missing_file():
    with pytest.raises(FileNotFoundError, match="AMB dataset not found"):
        load_amb_dataset("/nonexistent/path/to/data.json")


def test_load_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON list"):
        load_amb_dataset(str(path))


def test_load_empty_list(tmp_path: Path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    conversations, questions = load_amb_dataset(str(path))
    assert conversations == []
    assert questions == []
