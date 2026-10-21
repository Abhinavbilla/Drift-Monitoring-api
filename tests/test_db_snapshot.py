"""
db/snapshot.py: the database is saved to and restored from DRIFT_DB_BACKUP_PATH.

Run:
    python -m pytest tests/test_db_snapshot.py -v
"""

import sqlite3

import pytest

from db import crud, snapshot


@pytest.fixture
def paths(tmp_path, monkeypatch):
    live, backup = tmp_path / "local" / "drift.db", tmp_path / "share" / "drift.db"
    live.parent.mkdir()
    monkeypatch.setattr(crud, "DB_PATH", str(live))
    monkeypatch.setenv("DRIFT_DB_BACKUP_PATH", str(backup))
    return live, backup


def _rows(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT v FROM t ORDER BY v").fetchall()
    finally:
        conn.close()


def test_save_then_restore_into_a_fresh_container(paths):
    live, backup = paths
    conn = sqlite3.connect(live)
    conn.execute("PRAGMA journal_mode=WAL")  # like the real database
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
    conn.commit()
    assert snapshot.save() and not backup.with_name("drift.db.tmp").exists()
    conn.execute("INSERT INTO t VALUES (3)")  # written after the snapshot
    conn.commit()
    conn.close()

    for f in live.parent.iterdir():  # container replaced: local disk is empty
        f.unlink()
    assert snapshot.restore()
    assert _rows(live) == [(1,), (2,)]


def test_restore_never_overwrites_a_local_database(paths):
    live, backup = paths
    backup.parent.mkdir()
    for path, value in ((backup, 1), (live, 2)):
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (v INTEGER)")
        conn.execute("INSERT INTO t VALUES (?)", (value,))
        conn.commit()
        conn.close()
    assert not snapshot.restore()
    assert _rows(live) == [(2,)]


def test_everything_is_a_no_op_without_a_backup_path(tmp_path, monkeypatch):
    monkeypatch.setattr(crud, "DB_PATH", str(tmp_path / "drift.db"))
    monkeypatch.delenv("DRIFT_DB_BACKUP_PATH", raising=False)
    assert not snapshot.restore() and not snapshot.save()
    assert not list(tmp_path.iterdir())
