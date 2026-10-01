"""`memcontext watch`: a live, read-only view of the memory pipeline.

Shows each stage so a person can see what memory is doing while an agent works:
what was captured, which decisions are current, what changed (supersession, with
the typed reason), and what was served back to the agent.
"""
from __future__ import annotations

from click.testing import CliRunner

from memcontext.cli import main
from memcontext.mcp_tools import handle_memory_query, handle_memory_store
from memcontext.schema import open_database
from memcontext.watch import render_memory_view

OLD = "Tied ranking scores share a rank, newest claim first."
NEW = "Tied ranking scores share a rank, most trusted source first, then newest."


def _seed(conn):
    for text, value in (("We decided: tied ranking scores share a rank, newest claim first.", OLD),
                        ("Change of plan: most trusted source first, then newest.", NEW)):
        handle_memory_store(conn, text=text, session_id="har",
                            claims=[{"subject": "gap9 tie ranking", "predicate": "user_fact",
                                     "value": value}])
    handle_memory_query(conn, query="how are tied ranking scores ordered", session_id="har")


def test_view_shows_every_pipeline_stage():
    conn = open_database(":memory:")
    _seed(conn)
    view = render_memory_view(conn)

    assert "CAPTURED" in view and "Change of plan" in view
    assert "CURRENT DECISIONS" in view and "most trusted source first" in view
    assert "CHANGES" in view and "user_correction" in view
    assert "SERVED TO AGENT" in view and "how are tied ranking scores ordered" in view


def test_superseded_value_is_shown_only_as_history():
    conn = open_database(":memory:")
    _seed(conn)
    view = render_memory_view(conn)
    current = view.split("CURRENT DECISIONS", 1)[1].split("CHANGES", 1)[0]
    assert "newest claim first" not in current  # stale value never listed as current
    assert "newest claim first" in view.split("CHANGES", 1)[1]  # but kept as history


def test_view_fits_a_recording_terminal():
    conn = open_database(":memory:")
    _seed(conn)
    assert all(len(line) <= 100 for line in render_memory_view(conn, width=100).splitlines())


def test_empty_memory_renders():
    assert "no memory yet" in render_memory_view(open_database(":memory:"))


def test_cli_watch_once(tmp_path):
    db = str(tmp_path / "m.db")
    conn = open_database(db)
    _seed(conn)
    conn.close()
    r = CliRunner().invoke(main, ["watch", "--db", db, "--once"])
    assert r.exit_code == 0, r.output
    assert "CURRENT DECISIONS" in r.output


def test_one_request_is_one_served_line():
    conn = open_database(":memory:")
    for topic in ("ranking ties", "ranking cache", "ranking weights"):
        handle_memory_store(conn, text=f"We decided the {topic} policy for retrieval ranking.",
                            session_id="har",
                            claims=[{"subject": f"retrieval {topic}", "predicate": "user_fact",
                                     "value": f"retrieval ranking {topic} policy decided"}])
    handle_memory_query(conn, query="what did we decide about retrieval ranking", session_id="har")
    served = render_memory_view(conn).split("SERVED TO AGENT", 1)[1]
    assert served.count("what did we decide about retrieval ranking") == 1


def test_long_values_keep_the_part_that_changed():
    # Live demo: old and new rules were both cut to "...tied claims ar..." in the pane,
    # so viewers could not see which version the agent got.
    conn = open_database(":memory:")
    old = "Tied ranking scores share a rank; tied claims are then ordered newest first."
    new = ("Tied ranking scores share a rank; tied claims are then ordered by source trust "
           "first, then newest first, so a user-stated fact beats a newer web snippet.")
    for text, value in ((old, old), (new, new)):
        handle_memory_store(conn, text=text, session_id="har",
                            claims=[{"subject": "gap9 tie ranking", "predicate": "user_fact",
                                     "value": value}])
    for compact in (False, True):
        view = " ".join(render_memory_view(conn, width=80, compact=compact).split())
        assert "newer web snippet" in view.split("CHANGES", 1)[0]  # current, its tail visible
        changes = view.split("CHANGES", 1)[1]
        assert "ordered newest first" in changes and "newer web snippet" in changes
