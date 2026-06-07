"""The reindex-embeddings CLI wires the previously-dormant embedding backfills
(backfill_embeddings / backfill_episode_embeddings / backfill_event_frame_embeddings).
It is gated on a real embedder so it never loads a model in CI.
"""
from __future__ import annotations

from click.testing import CliRunner

from memcontext.cli import main
from memcontext.schema import open_database


def test_reindex_embeddings_is_gated_off_without_a_model(tmp_path):
    db = str(tmp_path / "m.db")
    open_database(db).close()  # embeddings off in CI (MEMCONTEXT_EMBED_EPISODES=0)

    r = CliRunner().invoke(main, ["reindex-embeddings", "--db", db])
    assert r.exit_code == 0, r.output
    assert "Embeddings are disabled" in r.output  # short-circuits, no model load
