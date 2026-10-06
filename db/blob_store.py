"""
Storage for large table-path artifacts: staged uploads, reference
embeddings, row-aligned reference rows. SQLite rows hold only the keys.

Filesystem backend under DATA_DIR (default ./data/blobs -- docker-compose
already mounts ./data). The small put/get/delete interface is the seam for a
future object-storage backend; nothing outside this module touches paths
except `path()`, used where a real file is needed (zipfile, streaming).
"""

import hashlib
import io
import os
import re
import shutil
import uuid
from typing import BinaryIO, Optional, Tuple

import numpy as np

DATA_DIR = os.getenv("DRIFT_DATA_DIR", os.path.join("data", "blobs"))
_KEY_RE = re.compile(r"^[0-9a-f]{16}/[a-z_]+/[0-9a-f]{32}\.[a-z0-9]+$")
_CHUNK = 1024 * 1024


class BlobTooLarge(Exception):
    pass


def _project_dir(project_id: str) -> str:
    # Hashed so owner emails (part of internal project ids) never appear in paths.
    return hashlib.sha256(project_id.encode()).hexdigest()[:16]


def new_key(project_id: str, kind: str, ext: str) -> str:
    return f"{_project_dir(project_id)}/{kind}/{uuid.uuid4().hex}.{ext}"


def path(key: str) -> str:
    if not _KEY_RE.match(key):
        raise ValueError("Invalid blob key.")
    return os.path.join(DATA_DIR, *key.split("/"))


def put_stream(key: str, src: BinaryIO, max_bytes: int) -> Tuple[int, str]:
    """Copies src to the blob in chunks, never holding it all in memory.
    Returns (bytes written, sha256 hex). Deletes the partial file and raises
    BlobTooLarge past max_bytes."""
    dest = path(key)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    written = 0
    digest = hashlib.sha256()
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = src.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise BlobTooLarge()
                digest.update(chunk)
                out.write(chunk)
    except BaseException:
        if os.path.exists(dest):
            os.remove(dest)
        raise
    return written, digest.hexdigest()


def put_bytes(key: str, data: bytes) -> int:
    return put_stream(key, io.BytesIO(data), max_bytes=len(data))[0]


def get_bytes(key: str) -> bytes:
    with open(path(key), "rb") as f:
        return f.read()


def put_array(key: str, arr: np.ndarray) -> int:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return put_bytes(key, buf.getvalue())


def get_array(key: str) -> np.ndarray:
    return np.load(io.BytesIO(get_bytes(key)), allow_pickle=False)


def delete(key: Optional[str]) -> None:
    if key and os.path.exists(path(key)):
        os.remove(path(key))


def delete_project(project_id: str) -> None:
    shutil.rmtree(os.path.join(DATA_DIR, _project_dir(project_id)), ignore_errors=True)
