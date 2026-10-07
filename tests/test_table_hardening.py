"""
Robustness of the unified table path (milestone M4, phase 12): fuzzed tables
and ZIPs never crash the worker or leak values; damaged ZIP entries become
per-row 'corrupt' statuses; archive limits hold. Seeded.

Run:
    python -m pytest tests/test_table_hardening.py -v
"""

import io
import logging
import random
import sys
import zipfile

import pytest

sys.path.insert(0, "tests")
import ingest.images as images  # noqa: E402
from ingest.images import ArchiveError, ImageArchive, load_image  # noqa: E402
from test_table_unified import H, PREFIX, SENTINEL, _png, _wait, client  # noqa: E402

EXTENSIONS = [".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".json", ".jsonl", ".arff", ".gz", ".zip", ".dat"]


def test_fuzzed_tables_fail_cleanly_without_echoing_data(caplog):
    rng = random.Random(7)
    for i in range(len(EXTENSIONS) * 2):
        ext = EXTENSIONS[i % len(EXTENSIONS)]
        junk = bytes(rng.getrandbits(8) for _ in range(rng.randint(20, 400))) + SENTINEL.encode()
        resp = client.post(f"/tables/{PREFIX}fuzz/stage", headers=H, files={"file": (f"t{ext}", junk)})
        assert resp.status_code == 202, (ext, resp.text)
        with caplog.at_level(logging.ERROR, logger="drift.jobs"):
            job = _wait(resp.json()["job_id"])
        assert job["status"] in ("failed", "succeeded"), (ext, job)
        assert SENTINEL not in (job["error"] or "")
        if job["status"] == "succeeded":
            # Junk that happens to parse: its profile shows the owner sample values until the
            # stage is discarded -- discarding must purge them.
            client.delete(f"/tables/stages/{resp.json()['stage_id']}", headers=H)
            assert client.get(f"/jobs/{job['job_id']}", headers=H).json()["result"] is None
    assert SENTINEL not in caplog.text


def test_fuzzed_zip_entries_become_row_statuses(tmp_path):
    rng = random.Random(8)
    path = tmp_path / "fuzz.zip"
    with zipfile.ZipFile(path, "w") as z:
        for i in range(20):
            z.writestr(f"f{i}.png", bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 300))))
        z.writestr("ok.png", _png(1))
    archive = ImageArchive(str(path))
    statuses = [load_image(f"f{i}.png", archive)[0] for i in range(20)]
    assert set(statuses) <= {"corrupt", "unsupported"} and load_image("ok.png", archive)[0] == "ok"
    archive.close()


def test_damaged_compressed_entry_is_corrupt_not_a_crash(tmp_path):
    path = tmp_path / "damaged.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.png", _png(1))
    data = bytearray(path.read_bytes())
    header_end = 30 + len("a.png")  # local file header + name; compressed data follows
    for k in range(header_end + 10, header_end + 40):
        data[k] ^= 0xFF
    path.write_bytes(bytes(data))
    archive = ImageArchive(str(path))
    assert load_image("a.png", archive) == ("corrupt", None)
    archive.close()


def test_entry_count_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "MAX_ZIP_ENTRIES", 5)
    path = tmp_path / "many.zip"
    with zipfile.ZipFile(path, "w") as z:
        for i in range(6):
            z.writestr(f"{i}.png", b"x")
    with pytest.raises(ArchiveError, match="entries"):
        ImageArchive(str(path))


def test_oversized_entry_rejected_by_header_and_by_read(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "MAX_ENTRY_BYTES", 1000)
    path = tmp_path / "big.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("big.png", _png(1) + bytes(2000))
    archive = ImageArchive(str(path))
    assert archive.rejected == {"too_large": 1}
    archive.close()


def test_server_survives_bad_requests():
    assert client.post(f"/tables/{PREFIX}fuzz/fit", headers=H, json={"stage_id": "nope", "columns": []}).status_code == 404
    assert client.post(f"/tables/{PREFIX}fuzz/fit", headers=H, json={"columns": "x"}).status_code == 422
    assert client.get("/jobs/does-not-exist", headers=H).status_code == 404
    assert client.post(f"/tables/{PREFIX}fuzz/analyze", headers=H, files={"file": ("b.csv", b"a\n1")}).status_code == 404
    assert client.get("/jobs/x").status_code in (401, 403)  # unauthenticated
