"""Tests for the browser observation sub-package.

All tests use mock/fake page snapshots -- no Playwright dependency required.
"""
from __future__ import annotations

import sqlite3

import pytest

from memcontext.observe.browser import ObservationResult, PageSnapshot, observe_page
from memcontext.observe.extractors import (
    AccessibilityTreeExtractor,
    DOMExtractor,
    _url_to_subject,
)
from memcontext.observe.revisit import ChangeReport, apply_changes, diff_snapshots
from memcontext.schema import open_database


@pytest.fixture()
def db():
    conn = open_database(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _make_snapshot(url="https://example.com", title="Example", tree=None):
    return PageSnapshot(
        url=url,
        title=title,
        timestamp="2026-05-19T15:00:00Z",
        accessibility_tree=tree or {},
    )


# --- AccessibilityTreeExtractor tests ---


def test_a11y_extractor_page_title():
    ext = AccessibilityTreeExtractor()
    snap = _make_snapshot(title="My Dashboard")
    claims = ext.extract(snap)
    assert any("page title: My Dashboard" in c["value"] for c in claims)


def test_a11y_extractor_heading():
    ext = AccessibilityTreeExtractor()
    tree = {"role": "heading", "name": "Project Status", "children": []}
    snap = _make_snapshot(tree=tree)
    claims = ext.extract(snap)
    assert any("heading: Project Status" in c["value"] for c in claims)


def test_a11y_extractor_form_field():
    ext = AccessibilityTreeExtractor()
    tree = {
        "role": "textbox",
        "name": "Email",
        "value": "user@test.com",
        "children": [],
    }
    snap = _make_snapshot(tree=tree)
    claims = ext.extract(snap)
    assert any(
        "Email" in c["value"] and "user@test.com" in c["value"] for c in claims
    )


def test_a11y_extractor_nested_tree():
    ext = AccessibilityTreeExtractor()
    tree = {
        "role": "main",
        "name": "",
        "children": [
            {"role": "heading", "name": "Tasks", "children": []},
            {
                "role": "text",
                "name": "Complete the migration by Friday",
                "children": [],
            },
        ],
    }
    snap = _make_snapshot(tree=tree)
    claims = ext.extract(snap)
    assert len(claims) >= 2  # title + heading + text


def test_a11y_extractor_empty_tree():
    ext = AccessibilityTreeExtractor()
    snap = _make_snapshot(title="", tree={})
    claims = ext.extract(snap)
    assert claims == []


def test_a11y_extractor_confidence_ranges():
    ext = AccessibilityTreeExtractor()
    tree = {"role": "heading", "name": "Test", "children": []}
    snap = _make_snapshot(tree=tree)
    claims = ext.extract(snap)
    for c in claims:
        assert 0 < c["confidence"] <= 1.0


# --- DOMExtractor tests ---


def test_dom_extractor_basic():
    ext = DOMExtractor()
    claims = ext.extract_from_text(
        "https://example.com", "Example", ["Hello world from the page content"]
    )
    assert len(claims) >= 1
    assert any("page title" in c["value"] for c in claims)


def test_dom_extractor_limits_blocks():
    ext = DOMExtractor()
    blocks = [f"Block {i} with enough content to pass" for i in range(50)]
    claims = ext.extract_from_text("https://test.com", "Test", blocks)
    # 1 title + max 20 blocks
    assert len(claims) <= 21


# --- URL to subject ---


def test_url_to_subject():
    assert _url_to_subject("https://example.com/page") == "example.com/page"
    assert _url_to_subject("https://example.com/page?q=1") == "example.com/page"
    assert (
        _url_to_subject("http://localhost:3000/dashboard")
        == "localhost:3000/dashboard"
    )


# --- PageSnapshot ---


def test_snapshot_id_deterministic():
    s1 = _make_snapshot()
    s2 = _make_snapshot()
    assert s1.snapshot_id == s2.snapshot_id


def test_snapshot_id_changes_with_url():
    s1 = _make_snapshot(url="https://a.com")
    s2 = _make_snapshot(url="https://b.com")
    assert s1.snapshot_id != s2.snapshot_id


# --- observe_page integration ---


def test_observe_page_stores_claims(db):
    tree = {"role": "heading", "name": "Sprint Board", "children": []}
    snap = _make_snapshot(
        url="https://jira.example.com/board", title="Sprint Board", tree=tree
    )
    result = observe_page(db, snapshot=snap, session_id="observe-test")
    assert len(result.claims) >= 1
    assert result.turn_id is not None


def test_observe_page_empty_tree(db):
    snap = _make_snapshot(title="", tree={})
    result = observe_page(db, snapshot=snap, session_id="observe-test")
    assert result.claims == []
    assert result.turn_id is None


# --- diff_snapshots ---


def test_diff_detects_additions():
    old = [{"subject": "s", "predicate": "observation", "value": "A"}]
    new = [
        {"subject": "s", "predicate": "observation", "value": "A"},
        {"subject": "s", "predicate": "context", "value": "B"},
    ]
    report = diff_snapshots(old, new, "https://example.com")
    assert len(report.added_claims) == 1
    assert report.unchanged_count == 1


def test_diff_detects_removals():
    old = [
        {"subject": "s", "predicate": "observation", "value": "A"},
        {"subject": "s", "predicate": "context", "value": "B"},
    ]
    new = [{"subject": "s", "predicate": "observation", "value": "A"}]
    report = diff_snapshots(old, new, "https://example.com")
    assert len(report.removed_claims) == 1


def test_diff_detects_changes():
    old = [{"subject": "s", "predicate": "observation", "value": "old value"}]
    new = [{"subject": "s", "predicate": "observation", "value": "new value"}]
    report = diff_snapshots(old, new, "https://example.com")
    assert len(report.changed_claims) == 1
    assert report.changed_claims[0][0]["value"] == "old value"
    assert report.changed_claims[0][1]["value"] == "new value"


def test_diff_no_changes():
    claims = [{"subject": "s", "predicate": "observation", "value": "same"}]
    report = diff_snapshots(claims, claims, "https://example.com")
    assert report.unchanged_count == 1
    assert not report.added_claims
    assert not report.removed_claims
    assert not report.changed_claims


# --- apply_changes ---


def test_apply_changes_additions(db):
    report = ChangeReport(
        url="https://example.com",
        added_claims=[
            {
                "subject": "example.com",
                "predicate": "observation",
                "value": "new heading",
            }
        ],
    )
    stats = apply_changes(db, change_report=report, session_id="revisit-test")
    assert stats["added"] >= 1


def test_apply_changes_empty(db):
    report = ChangeReport(url="https://example.com")
    stats = apply_changes(db, change_report=report, session_id="revisit-test")
    assert stats["added"] == 0
    assert stats["changed"] == 0
