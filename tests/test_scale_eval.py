"""The scale eval's distractors are deterministic, nested and can never be scored."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.product import scale

HIST = json.loads((Path(scale.HERE) / "datasets" / "stale_exposure.json").read_text(encoding="utf-8"))["histories"]
MAX = len(scale.distractor_subjects())


def test_distractors_are_deterministic_and_nested():
    big = scale.make_distractors(2000)
    assert big == scale.make_distractors(2000)
    assert scale.make_distractors(500) == big[:500]  # size N's set contains size M's


def test_distractor_subjects_are_distinct_decisions_in_their_own_sessions():
    ds = scale.make_distractors(MAX)
    assert len({d["claim"]["subject"] for d in ds}) == MAX
    assert len({d["session_id"] for d in ds}) == MAX
    assert all(d["claim"]["predicate"] == "decision_made" for d in ds)


def test_no_distractor_can_match_a_scored_value_or_subject():
    assert scale.collision_issues(HIST, scale.make_distractors(MAX)) == []


def test_collision_check_catches_a_planted_scored_value():
    planted = [{"text": "Decision: for the geo api database, we chose SQLite.",
                "claim": {"subject": "geo api database"}}]
    assert scale.collision_issues(HIST, planted)


def test_aspect_lookup_handles_two_word_kinds():
    assert scale._aspect_of("ledger sync job retry policy") == "retry policy"
    assert scale._aspect_of("geo api database") == "database"


def test_interleave_spreads_distractors_through_the_timeline():
    scored = [{"id": f"s{i}"} for i in range(4)]
    ds = [{"id": f"d{i}"} for i in range(10)]
    out = scale.interleave(scored, ds)
    assert len(out) == 14
    assert [x for x in out if x["id"][0] == "s"] == scored  # scored order kept
    assert [x for x in out if x["id"][0] == "d"] == ds
    first, last = out.index(scored[0]), out.index(scored[-1])
    assert first > 0 and last < len(out) - 1  # distractors before, between and after


@pytest.mark.parametrize(("p", "want"), [(50, 3), (95, 5), (100, 5), (0, 1)])
def test_nearest_rank_percentile(p, want):
    assert scale.pctl([5, 1, 4, 2, 3], p) == want
