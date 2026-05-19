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
    old = [{"subject": "s", "predicate": "observation", "value": "A", "obs_key": "heading:a"}]
    new = [
        {"subject": "s", "predicate": "observation", "value": "A", "obs_key": "heading:a"},
        {"subject": "s", "predicate": "observation", "value": "B", "obs_key": "link:b"},
    ]
    report = diff_snapshots(old, new, "https://example.com")
    assert len(report.added_claims) == 1
    assert report.unchanged_count == 1


def test_diff_detects_removals():
    old = [
        {"subject": "s", "predicate": "observation", "value": "A", "obs_key": "heading:a"},
        {"subject": "s", "predicate": "observation", "value": "B", "obs_key": "link:b"},
    ]
    new = [{"subject": "s", "predicate": "observation", "value": "A", "obs_key": "heading:a"}]
    report = diff_snapshots(old, new, "https://example.com")
    assert len(report.removed_claims) == 1


def test_diff_detects_changes():
    old = [{"subject": "s", "predicate": "observation", "value": "old value", "obs_key": "field:email"}]
    new = [{"subject": "s", "predicate": "observation", "value": "new value", "obs_key": "field:email"}]
    report = diff_snapshots(old, new, "https://example.com")
    assert len(report.changed_claims) == 1
    assert report.changed_claims[0][0]["value"] == "old value"
    assert report.changed_claims[0][1]["value"] == "new value"


def test_diff_no_changes():
    claims = [{"subject": "s", "predicate": "observation", "value": "same", "obs_key": "heading:x"}]
    report = diff_snapshots(claims, claims, "https://example.com")
    assert report.unchanged_count == 1
    assert not report.added_claims
    assert not report.removed_claims
    assert not report.changed_claims


def test_diff_realistic_multi_claim_page():
    """Black-box test: multiple claims sharing (subject, observation) from a real page."""
    ext = AccessibilityTreeExtractor()
    tree_v1 = {
        "role": "WebArea", "name": "Board", "children": [
            {"role": "heading", "name": "Sprint 42", "children": []},
            {"role": "text", "name": "Migration is 75% complete today", "children": []},
            {"role": "link", "name": "TICKET-123: Auth refactor", "children": []},
            {"role": "textbox", "name": "Search", "value": "auth", "children": []},
        ]
    }
    tree_v2 = {
        "role": "WebArea", "name": "Board", "children": [
            {"role": "heading", "name": "Sprint 42", "children": []},
            {"role": "text", "name": "Migration is 100% complete shipped", "children": []},
            {"role": "link", "name": "TICKET-123: Auth refactor", "children": []},
            {"role": "link", "name": "TICKET-456: Dashboard feature added", "children": []},
            {"role": "textbox", "name": "Search", "value": "dashboard", "children": []},
        ]
    }
    snap_v1 = PageSnapshot(
        url="http://board.test/sprint", title="Board",
        timestamp="2026-01-01T00:00:00Z", accessibility_tree=tree_v1,
    )
    snap_v2 = PageSnapshot(
        url="http://board.test/sprint", title="Board",
        timestamp="2026-01-01T01:00:00Z", accessibility_tree=tree_v2,
    )
    claims_v1 = ext.extract(snap_v1)
    claims_v2 = ext.extract(snap_v2)
    assert len(claims_v1) == 5  # title + heading + text + link + field
    assert len(claims_v2) == 6  # title + heading + text + 2 links + field

    report = diff_snapshots(claims_v1, claims_v2, "http://board.test/sprint")
    # text claims use content-hash keys, so changed text = old removed + new added
    # new link = 1 added. Changed text content = 1 removed + 1 added. Field value change = 1 changed.
    assert len(report.added_claims) == 2, f"Expected 2 added (new link + new text hash), got {len(report.added_claims)}"
    assert any("TICKET-456" in c["value"] for c in report.added_claims)
    assert len(report.removed_claims) == 1, f"Expected 1 removed (old text hash), got {len(report.removed_claims)}"
    assert len(report.changed_claims) == 1, f"Expected 1 changed (field value), got {len(report.changed_claims)}"
    assert report.unchanged_count == 3, f"Title + heading + old link unchanged, got {report.unchanged_count}"


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
