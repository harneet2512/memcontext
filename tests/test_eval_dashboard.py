"""The eval dashboard reads every committed eval result into one schema and embeds it safely."""
from __future__ import annotations

import json

from evals.product import dashboard

REQUIRED = {"eval", "title", "question", "headline"}


def test_every_collected_eval_has_the_common_schema():
    evals = dashboard.collect()
    assert {e["eval"] for e in evals} >= {"stale_exposure", "supersession_matrix", "claude_code_recall"}
    for e in evals:
        assert e.keys() >= REQUIRED, e["eval"]
        assert {"label", "value"} <= e["headline"].keys(), e["eval"]


def test_adapted_numbers_match_the_result_files():
    det = json.loads((dashboard.RESULTS / "supersession_matrix_deterministic_dev.json").read_text(encoding="utf-8"))
    sm = dashboard.adapt_supersession()
    assert sm is not None
    assert sm["headline"]["value"] == f"{det['overall']['correct']}/{det['overall']['n']}"
    se = dashboard.adapt_stale_exposure()
    assert se is not None
    hook, base = se["headline"]["value"], se["headline"]["baseline"]["value"]
    assert hook.split("/")[1] == base.split("/")[1]  # same question set on both sides


def test_render_escapes_script_close_and_fragment_has_no_document_tags():
    evil = [{"eval": "x", "title": "</script><b>", "question": "q", "headline": {"label": "l", "value": "1/1"}}]
    page = dashboard.render(evil)
    embedded = page.split("const D = ", 1)[1].split(";\nconst PROMISE", 1)[0]
    assert "</script>" not in embedded
    fragment = dashboard.render(evil, standalone=False)
    assert "<!doctype" not in fragment.lower() and "<title>MemContext Product Evals</title>" in fragment
