"""v8 drops the dormant `decisions` table (created since v0, never read/written)."""
from __future__ import annotations

from memcontext.schema import SCHEMA_VERSION, open_database


def test_v8_drops_dormant_decisions_table(tmp_path):
    db = str(tmp_path / "m.db")
    conn = open_database(db)  # fresh DB at the current version
    # simulate a pre-v8 database that still carries the dormant decisions table
    conn.execute("CREATE TABLE IF NOT EXISTS decisions (decision_id TEXT PRIMARY KEY, ts INTEGER)")
    conn.execute("PRAGMA user_version = 7")
    conn.commit()
    conn.close()

    conn2 = open_database(db)  # re-open -> v8 migration drops decisions
    tables = {
        r[0] for r in conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "decisions" not in tables, "v8 drops the dormant decisions table"
    assert conn2.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn2.close()
