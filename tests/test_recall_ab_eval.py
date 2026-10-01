"""recall_ab arms: baselines get no memory wiring, are not judged as MemContext, and are scored honestly."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.product import recall_ab


def _project(tmp_path: Path, arm: str) -> Path:
    project = tmp_path / arm
    project.mkdir()
    recall_ab.write_arm_files(project, arm)
    return project


@pytest.mark.parametrize("arm", ["self_notes", "notes"])
def test_baseline_arms_have_no_memory_wiring(tmp_path, arm):
    project = _project(tmp_path, arm)
    settings = json.loads((project / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    assert not (project / ".mcp.json").exists()
    assert "hooks" not in settings and "enableAllProjectMcpServers" not in settings
    assert not any("memcontext" in a for a in settings["permissions"]["allow"])
    assert settings["autoMemoryEnabled"] is False
    claude_md = (project / "CLAUDE.md").read_text(encoding="utf-8").lower()
    assert "memcontext" not in claude_md and "memory_query" not in claude_md


def test_self_notes_arm_has_no_decision_log(tmp_path):
    project = _project(tmp_path, "self_notes")
    assert not (project / recall_ab.NOTES_FILE).exists()
    assert recall_ab.NOTES_FILE not in (project / "CLAUDE.md").read_text(encoding="utf-8")


def test_notes_log_is_append_only_dated_and_ordered(tmp_path):
    project = _project(tmp_path, "notes")
    assert recall_ab.NOTES_FILE in (project / "CLAUDE.md").read_text(encoding="utf-8")
    recall_ab.append_decision(project, "We will use PostgreSQL.", 0)
    recall_ab.append_decision(project, "Change of plan: SQLite instead of PostgreSQL.", 2)
    log = (project / recall_ab.NOTES_FILE).read_text(encoding="utf-8")
    assert log.index("2026-09-01") < log.index("We will use PostgreSQL.") < log.index("2026-09-03")
    assert log.index("2026-09-03") < log.index("SQLite instead")
    assert "We will use PostgreSQL." in log  # the old entry is never deleted


def test_memcontext_arm_is_not_built_by_write_arm_files(tmp_path):
    with pytest.raises(ValueError):
        recall_ab.write_arm_files(tmp_path, "memcontext")


def test_result_usage_sums_tokens_from_the_result_event(tmp_path):
    t = tmp_path / "s.jsonl"
    t.write_text("\n".join([
        json.dumps({"type": "assistant", "message": {"content": []}}),
        json.dumps({"type": "result", "usage": {"input_tokens": 10, "cache_creation_input_tokens": 100,
                                                "cache_read_input_tokens": 1000, "output_tokens": 5}}),
    ]), encoding="utf-8")
    u = recall_ab.result_usage(t)
    assert u["total_tokens"] == 1115 and u["output_tokens"] == 5
    assert recall_ab.result_usage(tmp_path / "missing.jsonl")["total_tokens"] == 0


def _step(index: int, kind: str, ok: bool, answer: str, action: str | None = None) -> dict:
    score = {"answer": answer, "ok": ok, **({"action": action} if action else {})}
    return {"index": index, "kind": kind, "score": score, "final": f"... CURRENT: {answer}",
            "usage": {"total_tokens": 1000}}


def test_examples_pair_a_baseline_miss_with_memcontext_on_the_same_step():
    by_arm = {
        "notes": [{"id": "db", "steps": [_step(3, "act", False, "stale", "stale")]}],
        "memcontext": [{"id": "db", "steps": [_step(3, "act", True, "current", "current")]}],
    }
    ex = recall_ab.pick_examples(by_arm)
    assert [(e["arm"], e["scenario"], e["step"]) for e in ex] == [("notes", "db", "#3 act"),
                                                                    ("memcontext", "db", "#3 act")]
    assert ex[0]["verdict"] == "stale / action stale"


@pytest.mark.parametrize("parts", [("self_notes", "db_engine_change"), ("notes", "two_topics_no_interference"),
                                   ("20261001T195757Z",)])
def test_run_dirs_cannot_start_a_backslash_escape(parts):
    name = recall_ab.safe_dirname(*parts)
    assert name.startswith("p-") and "\\" not in name and "/" not in name


def test_arm_names_and_labels():
    assert recall_ab.ARMS == ("self_notes", "notes", "memcontext")
    assert recall_ab.ARM_LABELS["self_notes"] == "Claude's own notes (no memory tool)"
    assert recall_ab.LEGACY_ARM_NAMES == {"none": "self_notes"}


def test_baseline_steps_carry_no_memcontext_attribution():
    steps = [{"index": 0, "kind": "capture", "attribution": {"layer": "plumbing",
                                                             "reason": "serve-http never answered"}}]
    assert "attribution" not in recall_ab.strip_memcontext_attribution(steps)[0]
    assert "attribution" in steps[0]  # input is not mutated


def test_claude_md_is_credited_only_when_nothing_else_explains_the_answer():
    pats = recall_ab.ccr.Patterns({"kafka": "kafka", "nats": "nats"})
    md = "# Project\n\nEvent broker: NATS (was Kafka).\n"
    assert recall_ab.attribute_source("none", md, pats, "nats") == "claude_md"
    assert recall_ab.attribute_source("file", md, pats, "nats") == "file"
    assert recall_ab.attribute_source("none", "# Project\n", pats, "nats") == "none"
    assert recall_ab.attribute_source("none", md, pats, None) == "none"


def _scen(sid: str, set_name: str, wrong: int, n: int, arm: str) -> dict:
    steps = [{"index": i, "kind": "recall", "expect": {"stale": []}, "cost_usd": 0.01, "wall_s": 5.0,
              "usage": {"total_tokens": 100}, "db_after": dict(recall_ab.NO_DB),
              "evidence": {"source": "file", "injected_current": False, "tool_called": False},
              "score": {"answer": "stale" if i < wrong else "current", "ok": i >= wrong}}
             for i in range(n)]
    return {"id": sid, "arm": arm, "set": set_name, "steps": steps}


def test_findings_say_plainly_when_memcontext_shows_no_advantage():
    per_set = {"long_horizon": {
        a: recall_ab.arm_summary([_scen("x", "long_horizon", w, 9, a)])
        for a, w in (("self_notes", 1), ("notes", 3), ("memcontext", 1))}}
    [f] = recall_ab.findings(per_set)
    assert "MemContext shows no advantage over Claude's own notes (no memory tool)" in f
    assert "Decision log" not in f.split("(wrong steps")[0]


def test_findings_report_a_win_only_when_memcontext_beats_every_baseline():
    per_set = {"short": {
        a: recall_ab.arm_summary([_scen("x", "short", w, 6, a)])
        for a, w in (("self_notes", 2), ("notes", 1), ("memcontext", 0))}}
    [f] = recall_ab.findings(per_set)
    assert "fewer wrong steps than every baseline" in f and "no advantage" not in f


def test_report_has_by_arm_and_by_set_tables_and_a_long_horizon_headline():
    by_arm = {a: [_scen("s", "short", 0, 4, a), _scen("l", "long_horizon", w, 9, a)]
              for a, w in (("self_notes", 2), ("notes", 1), ("memcontext", 0))}
    meta = {"commit": "c", "code_under_test": {"dirty": False}, "timestamp_utc": "t", "split": "dev",
            "model": "haiku", "claude_version": "v", "spent_usd": 1.0}
    r = recall_ab.build_report(by_arm, meta)
    assert [t["title"] for t in r["tables"]] == ["By arm", "By scenario set (short vs long-horizon)"]
    assert r["headline"]["value"] == "2/9 / 1/9 / 0/9" and r["headline"]["label"].startswith("long-horizon")
    sets = [row[0] for row in r["tables"][1]["rows"]]
    assert sets[0].startswith("long-horizon") and any(s.startswith("short") for s in sets)
    assert {"eval", "title", "question", "headline", "metrics", "limits", "findings"} <= r.keys()


def test_imported_raw_renames_the_legacy_arm_and_drops_its_plumbing_blame(tmp_path):
    raw = {"none": [{"id": "db", "steps": [{"index": 0, "kind": "capture",
                                            "attribution": {"layer": "plumbing"}}]}],
           "memcontext": [{"id": "db", "steps": [{"index": 0, "kind": "capture",
                                                  "attribution": {"layer": "capture"}}]}]}
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    out = recall_ab.load_raw(p, "short")
    assert set(out) == {"self_notes", "memcontext"}
    assert "attribution" not in out["self_notes"][0]["steps"][0]
    assert out["memcontext"][0]["steps"][0]["attribution"] == {"layer": "capture"}
    assert out["self_notes"][0]["set"] == "short"


def test_long_horizon_dataset_labels_are_consistent():
    data = json.loads((Path(recall_ab.HERE) / "datasets" / "recall_long_horizon.json").read_text(encoding="utf-8"))
    assert {s["split"] for s in data["scenarios"]} == {"dev", "heldout"}
    for sc in data["scenarios"]:
        pats = recall_ab.ccr.Patterns(sc["values"])
        keys = set(sc["values"])
        decided = [s for s in sc["steps"] if s["kind"] in ("capture", "change")]
        scored = [s for s in sc["steps"] if s["kind"] in ("recall", "act")]
        assert len({s["current"] for s in decided if s["kind"] == "capture"}) >= 10, sc["id"]
        assert sum(s["kind"] == "change" for s in decided) >= 8 and len(scored) >= 7
        for s in sc["steps"]:
            assert s.get("current") in keys and set(s.get("stale", [])) <= keys, (sc["id"], s["prompt"])
            if s["kind"] == "act":
                assert s["file"].startswith("config/") and s["var"].isupper()
        # a value's regex never matches another value's canonical spelling
        for k in keys:
            assert not any(pats.hit(k, other) for other in keys - {k} if len(other) > 3), (sc["id"], k)
        # every scored question comes after all the sessions that decided it
        first_scored = sc["steps"].index(scored[0])
        assert all(sc["steps"].index(s) < first_scored for s in decided)
