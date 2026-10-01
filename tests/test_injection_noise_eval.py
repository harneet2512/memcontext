"""Scoring of the injection-noise eval (evals/product/injection_noise.py) on tiny fake hook output."""
from __future__ import annotations

from evals.product.injection_noise import (
    context_entries,
    entry_is_about,
    score_related,
    score_silence,
    trigger_counts,
)

HEADER = "[MemContext] Current project memory relevant to this prompt:"
CONFLICT = ("CONFLICT: 2 current values recorded for the same decision (newest first). Prefer...:\n"
            "    1. cache_layer / decision_made: Redis 7  (newest)\n"
            "    2. cache_layer / decision_made: Memcached")


def test_context_entries_splits_items_and_keeps_conflicts_whole():
    ctx = f"{HEADER}\n- ci_provider / decision_made: GitHub Actions\n- {CONFLICT}"
    entries = context_entries(ctx)
    assert entries[0] == "ci_provider / decision_made: GitHub Actions"
    assert entries[1].startswith("CONFLICT:") and "Memcached" in entries[1]
    assert context_entries(None) == [] and context_entries("") == []


def test_entry_is_about_matches_subject_prefix_and_conflict_values():
    assert entry_is_about("ci_provider / decision_made: GitHub Actions", "ci_provider")
    assert not entry_is_about("ci_provider_cache / decision_made: x", "ci_provider")
    assert entry_is_about(CONFLICT, "cache_layer")
    assert not entry_is_about(CONFLICT, "ci_provider")


def test_score_silence_counts_injected_cases_and_their_size():
    ctx = f"{HEADER}\n- a / decision_made: x\n- b / decision_made: y"
    s = score_silence([{"context": None}, {"context": ctx}, {"context": ""}])
    assert (s["n"], s["injected"], s["injected_pct"]) == (3, 1, 33.3)
    assert s["lines_total"] == 2 and s["chars_total"] == len(ctx)


def test_score_related_current_stale_and_on_subject_precision():
    on = f"{HEADER}\n- cache_layer / decision_made: Redis 7\n- ci_provider / decision_made: GitHub Actions"
    stale = f"{HEADER}\n- cache_layer / decision_made: Memcached"
    base = {"subject_key": "cache_layer", "current": "Redis 7", "stale": ["Memcached"]}
    s = score_related([{**base, "context": on}, {**base, "context": stale}, {**base, "context": None}])
    assert (s["n"], s["injected"]) == (3, 2)
    assert (s["current_present"], s["stale_present"]) == (1, 1)
    assert (s["on_subject_lines"], s["lines_total"], s["on_subject_pct"]) == (2, 3, 66.7)


def test_current_value_match_is_whole_value_not_substring():
    # "Redis 7" must not count as present inside "Redis 70"
    ctx = f"{HEADER}\n- cache_layer / decision_made: Redis 70"
    s = score_related([{"subject_key": "cache_layer", "current": "Redis 7", "stale": [], "context": ctx}])
    assert s["current_present"] == 0


def test_trigger_counts_ranks_shared_words():
    pairs = [({"error", "message"}, {"error", "tracking"}), ({"error", "loop"}, {"error"}), ({"tool"}, {"tool"})]
    assert trigger_counts(pairs) == [("error", 2), ("tool", 1)]
