"""
Keeps the SQLite database across container restarts on hosts whose only
persistent storage is a network share (Azure App Service's /home).

SQLite must not run on a network filesystem -- its locks aren't reliable
there, and WAL mode needs shared memory -- so the live database stays on
the container's local disk and a consistent copy is saved to the share:
every DRIFT_DB_BACKUP_SECONDS (default 300) and on shutdown. On startup,
an absent local database is restored from that copy. A crash (rather than
a normal stop or redeploy) loses at most the last interval of writes.

Enabled by setting DRIFT_DB_BACKUP_PATH (e.g. /home/drift/drift.db);
without it every function here is a no-op.
"""

import logging
import os
import shutil
import sqlite3
import threading

from db import crud

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_started = False


def _backup_path():
    return os.getenv("DRIFT_DB_BACKUP_PATH")


def restore() -> bool:
    """Copies the saved database into place if there's no local one yet."""
    backup = _backup_path()
    if not backup or os.path.exists(crud.DB_PATH) or not os.path.exists(backup):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(crud.DB_PATH)), exist_ok=True)
    shutil.copyfile(backup, crud.DB_PATH)
    logger.info("Restored database from %s", backup)
    return True


def save() -> bool:
    """Writes a consistent copy of the live database to the backup path."""
    backup = _backup_path()
    if not backup or not os.path.exists(crud.DB_PATH):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(backup)), exist_ok=True)
    tmp = backup + ".tmp"
    with _lock:
        src, dst = crud.get_connection(), sqlite3.connect(tmp)
        try:
            src.backup(dst)  # consistent even while other threads write
        finally:
            dst.close()
            src.close()
        os.replace(tmp, backup)  # never leave a half-written copy in place
    return True


def start_periodic_save():
    global _started
    if _started or not _backup_path():
        return
    _started = True
    interval = int(os.getenv("DRIFT_DB_BACKUP_SECONDS", "300"))

    def loop():
        stop = threading.Event()
        while not stop.wait(interval):
            try:
                save()
            except Exception:
                logger.exception("Database backup failed")

    threading.Thread(target=loop, name="db-snapshot", daemon=True).start()
