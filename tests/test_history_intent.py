"""Evolution questions ("how did X change") switch on history mode.

Regression (HAR-95 GAP-6): the history cue list had only past-state words
(before, previously, used to ...), so "How did Acme's renewal situation change?"
was served current facts only and the superseded value it asks about was missing.
Imperatives and how-to questions ("change the CI provider", "how do I change the
log level") must NOT switch it on: in a coding assistant "change" is mostly a
request, and history mode lets superseded claims take ranked slots.
"""
from __future__ import annotations

import pytest

from memcontext.retrieval import detect_history_intent


@pytest.mark.parametrize("query", [
    "How did Acme's renewal situation change?",
    "How has our CI setup changed over the last quarter?",
    "How have the pricing tiers evolved?",
    "How did Harneet's thinking about the company shift?",
    "What changed about the billing database?",
    "What has changed in the deployment plan?",
    "How did the auth design develop over time?",
    "Show me the timeline of the database decision",
    "What was the database before the migration?",  # existing cue still works
])
def test_evolution_and_past_questions_are_history(query):
    assert detect_history_intent(query) is True


@pytest.mark.parametrize("query", [
    "Change the CI provider to GitLab CI",
    "Please change the database to PostgreSQL",
    "How do I change the log level?",
    "How can we change the retry policy safely?",
    "What is Acme's current renewal status?",
    "Write a migration that changes the column type",
    "Did the tests pass?",
])
def test_requests_and_current_state_questions_are_not_history(query):
    assert detect_history_intent(query) is False
