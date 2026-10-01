"""Scoring of the provenance-integrity eval on tiny fake memory_trace payloads (no models)."""
from __future__ import annotations

from evals.product.provenance_trace import (
    expected_edge,
    score_changed,
    score_drift,
    score_unchanged,
)


def _step(value: str, status: str, edge: str, text: str | None = None) -> dict:
    return {"value": value, "status": status, "edge_type": edge,
            "text": text if text is not None else f"We decided on {value}."}


def _trace(*oldest_first: dict, conflicts: list[dict] | None = None) -> dict:
    # memory_trace returns its lineage newest-first
    return {"lineage": list(reversed(oldest_first)), "conflicts": conflicts or []}


GOOD = _trace(_step("PostgreSQL 15", "superseded", "user_correction"),
              _step("MySQL 8", "superseded", "user_correction"),
              _step("SQLite", "active", "active"))
VALUES = ["PostgreSQL 15", "MySQL 8", "SQLite"]


def test_full_history_in_order_scores_fully_correct():
    s = score_changed(VALUES, GOOD)
    assert s["fully_correct"] and s["edge_types"] == ["user_correction", "user_correction"]


def test_missing_version_fails_complete_and_ordered():
    t = _trace(_step("PostgreSQL 15", "superseded", "user_correction"), _step("SQLite", "active", "active"))
    s = score_changed(VALUES, t)
    assert not s["complete"] and not s["ordered"] and not s["fully_correct"]


def test_two_live_versions_fail_ordered():
    t = _trace(_step("PostgreSQL 15", "active", "user_correction"),
               _step("MySQL 8", "superseded", "user_correction"), _step("SQLite", "active", "active"))
    assert not score_changed(VALUES, t)["ordered"]


def test_ungrounded_source_text_fails_grounded():
    t = _trace(_step("PostgreSQL 15", "superseded", "user_correction", text="something else"),
               _step("MySQL 8", "superseded", "user_correction"), _step("SQLite", "active", "active"))
    s = score_changed(VALUES, t)
    assert s["complete"] and s["ordered"] and not s["grounded"]


def test_wrong_edge_type_fails_edges():
    t = _trace(_step("PostgreSQL 15", "superseded", "contradicts"),
               _step("MySQL 8", "superseded", "user_correction"), _step("SQLite", "active", "active"))
    assert not score_changed(VALUES, t)["edges"]


def test_expected_edge_follows_documented_refines_rule():
    assert expected_edge("Go 1.22 and Rust", "Rust") == "refines"
    assert expected_edge("PostgreSQL 15", "SQLite") == "user_correction"


def test_error_payload_scores_nothing():
    s = score_changed(VALUES, {"error": "No active claim", "lineage": []})
    assert not any(s[c] for c in ("complete", "ordered", "grounded", "edges"))


def test_unchanged_fact_needs_one_live_unsuperseded_version():
    assert score_unchanged("Sentry", _trace(_step("Sentry", "active", "active")))
    assert not score_unchanged("Sentry", _trace(_step("Rollbar", "superseded", "user_correction"),
                                                _step("Sentry", "active", "active")))


def test_drift_reports_stale_head_and_missing_link():
    old = _trace(_step("PostgreSQL 15", "superseded", "user_correction"), _step("MySQL 8", "active", "active"))
    alt = _trace(_step("SQLite", "active", "active"))
    d = score_drift(VALUES, old, alt)
    assert d == {"old_subject_current": False, "alt_subject_full_history": False, "drift_flagged": False}
    flagged = _trace(_step("MySQL 8", "active", "active"), conflicts=[{"value": "SQLite"}])
    assert score_drift(VALUES, flagged, alt)["drift_flagged"]
