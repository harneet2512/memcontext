from __future__ import annotations

import os

import pytest

from memcontext.predicate_packs import active_pack, load_pack


def test_load_general_pack():
    pack = load_pack("general")
    assert pack.pack_id == "general"
    assert len(pack.predicate_families) == 10
    assert "user_fact" in pack.predicate_families
    assert "user_preference" in pack.predicate_families
    assert "user_event" in pack.predicate_families
    assert "user_relationship" in pack.predicate_families
    assert "user_goal" in pack.predicate_families
    assert "user_constraint" in pack.predicate_families
    assert "context" in pack.predicate_families
    assert "action" in pack.predicate_families
    assert "observation" in pack.predicate_families
    assert "metadata" in pack.predicate_families


def test_active_pack_default():
    pack = active_pack()
    assert pack.pack_id == "general"
    assert "user_fact" in pack.predicate_families


def test_load_nonexistent_pack_raises():
    with pytest.raises(FileNotFoundError):
        load_pack("nonexistent_pack_xyz")


def test_few_shot_examples_parsed():
    pack = load_pack("general")
    assert len(pack.few_shot_examples) >= 4
    for ex in pack.few_shot_examples:
        assert ex.name
        assert ex.scenario
        assert ex.current_turn
        assert ex.expected_output


def test_pack_has_description():
    pack = load_pack("general")
    assert "general-purpose" in pack.description.lower() or "general" in pack.description.lower()
