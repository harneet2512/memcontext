from __future__ import annotations

import os

import pytest

from memcontext.predicate_packs import active_pack, load_pack, load_packs


# ---------------------------------------------------------------------------
# General pack tests
# ---------------------------------------------------------------------------


def test_load_general_pack():
    pack = load_pack("general")
    assert pack.pack_id == "general"
    assert len(pack.predicate_families) >= 10
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
    active_pack.cache_clear()
    pack = active_pack()
    assert pack.pack_id == "general"
    assert "user_fact" in pack.predicate_families
    active_pack.cache_clear()


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
    assert "general" in pack.description.lower()


# ---------------------------------------------------------------------------
# Developer pack tests
# ---------------------------------------------------------------------------

DEVELOPER_FAMILIES = frozenset(
    [
        "decision_made",
        "bug_fixed",
        "convention_established",
        "file_purpose",
        "dependency_reason",
        "api_contract",
        "todo",
        "blocker",
        "user_preference",
        "project_status",
    ]
)


def test_load_developer_pack():
    pack = load_pack("developer")
    assert pack.pack_id == "developer"
    assert len(pack.predicate_families) >= 10


def test_developer_pack_families():
    pack = load_pack("developer")
    assert pack.predicate_families == DEVELOPER_FAMILIES


def test_developer_few_shot_examples():
    pack = load_pack("developer")
    assert len(pack.few_shot_examples) >= 5
    for ex in pack.few_shot_examples:
        assert ex.name
        assert ex.scenario
        assert ex.current_turn
        assert ex.expected_output


# ---------------------------------------------------------------------------
# Pack composition tests
# ---------------------------------------------------------------------------


def test_load_packs_composition():
    merged = load_packs(["general", "developer"])
    assert merged is not None
    assert merged.pack_id.startswith("merged:")


def test_composed_pack_family_union():
    general = load_pack("general")
    developer = load_pack("developer")
    merged = load_packs(["general", "developer"])

    expected_union = general.predicate_families | developer.predicate_families
    assert merged.predicate_families == expected_union
    assert len(merged.predicate_families) == len(expected_union)


def test_composed_few_shot_concat():
    general = load_pack("general")
    developer = load_pack("developer")
    merged = load_packs(["general", "developer"])

    assert len(merged.few_shot_examples) == len(general.few_shot_examples) + len(
        developer.few_shot_examples
    )


def test_comma_separated_active_pack(monkeypatch: pytest.MonkeyPatch):
    active_pack.cache_clear()
    monkeypatch.setenv("ACTIVE_PACK", "general,developer")
    try:
        pack = active_pack()
        assert pack.pack_id.startswith("merged:")
        assert "user_fact" in pack.predicate_families
        assert "decision_made" in pack.predicate_families
    finally:
        active_pack.cache_clear()


def test_single_pack_backward_compat(monkeypatch: pytest.MonkeyPatch):
    active_pack.cache_clear()
    monkeypatch.setenv("ACTIVE_PACK", "general")
    try:
        pack = active_pack()
        assert pack.pack_id == "general"
        assert len(pack.predicate_families) >= 10
    finally:
        active_pack.cache_clear()


def test_composed_pack_id():
    merged = load_packs(["general", "developer"])
    assert merged.pack_id == "merged:general+developer"


def test_load_packs_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        load_packs([])


def test_load_packs_single_returns_original():
    pack = load_packs(["general"])
    assert pack.pack_id == "general"
