"""Durability: locked/corrupt DB surfaces a clean error, concurrent open works."""
from __future__ import annotations

import pytest

from memcontext.schema import DatabaseUnavailableError, open_database


def test_corrupt_db_raises_clean(tmp_path):
    p = tmp_path / "corrupt.db"
    p.write_bytes(b"NOT A SQLITE FILE " * 50)
    with pytest.raises(DatabaseUnavailableError):
        open_database(str(p))


def test_concurrent_second_client_no_crash(tmp_path):
    p = tmp_path / "shared.db"
    c1 = open_database(str(p))
    c2 = open_database(str(p))  # a second client opening the same file
    assert c2.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
    c1.close()
    c2.close()


def test_open_creates_file(tmp_path):
    p = tmp_path / "fresh.db"
    open_database(str(p)).close()
    assert p.exists()
