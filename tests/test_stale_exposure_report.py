"""The stale-exposure HTML report shows exactly what the eval measured, safely embedded."""
from __future__ import annotations

import json
import re

from evals.product.report_html import BASELINE, MEMCONTEXT, render, report_data

SUMMARY = {"repo_sha": "0123456789abcdef"}
QUESTIONS = [
    {"q": "Which DB?", "subject": "db", "current": "SQLite", "stale": ["PostgreSQL 15"], "changed": True},
    {"q": "Which CI?", "subject": "ci", "current": "GitHub Actions", "stale": [], "changed": False},
]


def _row(q: str, changed: bool, served: list[str], stale: list[str], cur: bool) -> dict:
    return {"q": q, "subject": q, "changed": changed, "served": served, "stale_hit": stale, "cur_hit": cur}


PERQ = {
    BASELINE: [_row("Which DB?", True, ["we use PostgreSQL 15", "now SQLite"], ["PostgreSQL 15"], True),
               _row("Which CI?", False, ["GitHub Actions"], [], True)],
    MEMCONTEXT: [_row("Which DB?", True, ["db / decision_made: SQLite </script>"], [], True),
                 _row("Which CI?", False, [], [], False)],
}


def test_report_lists_only_changed_questions_with_full_value_history():
    data = report_data(SUMMARY, PERQ, QUESTIONS, n_histories=2)
    assert [r["q"] for r in data["rows"]] == ["Which DB?"]
    row = data["rows"][0]
    assert row["history"] == ["PostgreSQL 15", "SQLite"]  # oldest first, current last
    assert row["base"] == {"served": ["we use PostgreSQL 15", "now SQLite"], "stale": ["PostgreSQL 15"], "cur": True}
    assert row["mc"]["stale"] == [] and row["mc"]["cur"] is True
    assert data["sha"] == "0123456789"


def test_served_text_cannot_close_the_script_tag():
    page = render(SUMMARY, PERQ, QUESTIONS, n_histories=2)
    embedded = page.split("const D = ", 1)[1].split(";\nconst rows", 1)[0]
    assert "</script>" not in embedded
    assert json.loads(embedded)["rows"][0]["mc"]["served"] == ["db / decision_made: SQLite </script>"]


def test_standalone_page_and_artifact_fragment():
    page = render(SUMMARY, PERQ, QUESTIONS, n_histories=2)
    fragment = render(SUMMARY, PERQ, QUESTIONS, n_histories=2, standalone=False)
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    assert not re.search(r"<!doctype|<html|<body", fragment, re.I)
    assert "<title>MemContext Stale Exposure</title>" in fragment
