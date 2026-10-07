"""
Unified table path, milestone M1 (docs/unified_table_plan.md): profiling,
safe image-ZIP ingestion, human-confirmed schema, fit/analyze jobs,
versioned storage, equivalence with the tabular endpoints, isolation,
idempotency, and no raw values persisted.

Run:
    HF_HUB_OFFLINE=1 python -m pytest tests/test_table_unified.py -v
"""

import io
import os
import random
import sqlite3
import sys
import tempfile
import time
import zipfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import jobs  # noqa: E402
from db import blob_store, crud  # noqa: E402
from ingest.images import ImageArchive, load_image, validate_image_bytes  # noqa: E402
from main import app, _internal_project_key  # noqa: E402
from utils.profiler import profile_columns, profile_table  # noqa: E402

crud.init_db()
blob_store.DATA_DIR = tempfile.mkdtemp(prefix="drift_blobs_test_")
client = TestClient(app)

EMAIL, EMAIL_B = "table-test-a@example.com", "table-test-b@example.com"
H = {"Authorization": f"Bearer {mint_session_token(EMAIL)}"}
H_B = {"Authorization": f"Bearer {mint_session_token(EMAIL_B)}"}
PREFIX = "test_table_unified_"
SENTINEL = "zqxsentinelvalue"
IDEM_KEY = f"table-idem-{time.time_ns()}"
BREEDS = ["labrador", "beagle", "poodle", "terrier"]
WORDS = ("friendly playful calm gentle loyal energetic curious shy smart happy dog home family "
         "children walks garden loves needs good with very looking for a").split()


def _png(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _table(n: int, seed: int, sentinel_row: bool = False) -> pd.DataFrame:
    rng = random.Random(seed)
    desc = [" ".join(rng.choice(WORDS) for _ in range(8)) + f" {i}" for i in range(n)]
    if sentinel_row:
        desc[0] = f"friendly dog {SENTINEL} looking for home"
    return pd.DataFrame({
        "PetID": [f"P{seed}x{i:05d}" for i in range(n)],
        "Age": [rng.randint(1, 120) for _ in range(n)],
        "Fee": [round(rng.uniform(0, 500), 2) for _ in range(n)],
        "Breed": [rng.choice(BREEDS) for _ in range(n)],
        "Description": desc,
        "Photo": [f"img_{seed}_{i}.png" for i in range(n)],
    })


def _zip_for(df: pd.DataFrame, corrupt: tuple = (), skip: tuple = ()) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, name in enumerate(df["Photo"]):
            if name in skip:
                continue
            z.writestr(name, b"not an image at all" if name in corrupt else _png(i))
    return buf.getvalue()


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _wait(job_id: str, headers=H, timeout: float = 300) -> dict:
    deadline = time.time() + timeout
    while True:
        jobs.run_pending()
        body = client.get(f"/jobs/{job_id}", headers=headers).json()
        if body["status"] in ("succeeded", "failed", "interrupted") or time.time() > deadline:
            return body
        time.sleep(0.2)


def _stage(project: str, df: pd.DataFrame, zip_bytes=None, headers=H) -> dict:
    files = {"file": ("train.csv", _csv(df), "text/csv")}
    if zip_bytes is not None:
        files["images"] = ("images.zip", zip_bytes, "application/zip")
    resp = client.post(f"/tables/{project}/stage", files=files, headers=headers)
    assert resp.status_code == 202, resp.text
    return resp.json()


def _confirmed(profile: dict, overrides: dict = None) -> list:
    overrides = overrides or {}
    return [{"name": c["name"], "type": overrides.get(c["name"], c["proposed_type"]),
             "monitor": overrides.get(c["name"], c["proposed_type"]) != "ignore"} for c in profile["columns"]]


def _fit(project: str, df: pd.DataFrame, zip_bytes=None, overrides=None, accept_proposed_relationships=False) -> dict:
    staged = _stage(project, df, zip_bytes)
    profile = _wait(staged["job_id"])
    assert profile["status"] == "succeeded", profile
    relationships = [{"col_a": r["col_a"], "col_b": r["col_b"], "monitor": True}
                     for r in profile["result"]["relationships"] if r["proposed"]] if accept_proposed_relationships else []
    resp = client.post(f"/tables/{project}/fit", headers=H, json={
        "stage_id": staged["stage_id"], "columns": _confirmed(profile["result"], overrides),
        "relationships": relationships})
    assert resp.status_code == 202, resp.text
    fit = _wait(resp.json()["job_id"])
    assert fit["status"] == "succeeded", fit
    return {"staged": staged, "profile": profile["result"], "fit": fit["result"]}


def _analyze(project: str, df: pd.DataFrame, zip_bytes=None, key=None):
    files = {"file": ("batch.csv", _csv(df), "text/csv")}
    if zip_bytes is not None:
        files["images"] = ("images.zip", zip_bytes, "application/zip")
    headers = {**H, "Idempotency-Key": key} if key else H
    return client.post(f"/tables/{project}/analyze", files=files, headers=headers)


@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    for email, headers in ((EMAIL, H), (EMAIL_B, H_B)):
        conn = sqlite3.connect(crud.DB_PATH)
        ids = [r[0] for r in conn.execute("SELECT id FROM projects WHERE id LIKE ?", (f"%{PREFIX}%",))]
        conn.close()
        for internal in ids:
            if internal.startswith(email):
                client.delete(f"/projects/{internal.split('::', 1)[1]}", headers=headers)


@pytest.fixture(scope="module")
def fitted():
    project = PREFIX + "main"
    train = _table(48, seed=1, sentinel_row=True)
    out = _fit(project, train, _zip_for(train))
    out["project"] = project
    return out


# ---------------------------------------------------------
# Profiler
# ---------------------------------------------------------
def test_profile_table_proposes_each_type(tmp_path):
    df = _table(60, seed=3)
    df["Homepage"] = [f"https://example.com/pet/{i}" for i in range(60)]
    df["Listed"] = pd.date_range("2024-01-01", periods=60).strftime("%Y-%m-%d")
    zpath = tmp_path / "imgs.zip"
    zpath.write_bytes(_zip_for(df))
    archive = ImageArchive(str(zpath))
    by_name = {p["name"]: p for p in profile_table(df, archive)}
    archive.close()

    assert by_name["Age"]["proposed_type"] == "numeric"
    assert by_name["Fee"]["proposed_type"] == "numeric"
    assert by_name["Breed"]["proposed_type"] == "categorical"
    assert by_name["Description"]["proposed_type"] == "text"
    assert by_name["Photo"]["proposed_type"] == "image" and by_name["Photo"]["proposed_monitor"]
    assert by_name["Photo"]["evidence"]["image_match_ratio"] == 1.0
    for ignored in ("PetID", "Homepage", "Listed"):
        assert by_name[ignored]["proposed_type"] == "ignore", ignored
        assert not by_name[ignored]["proposed_monitor"]
    for p in by_name.values():
        assert 0 <= p["confidence"] <= 1 and p["reason"] and "n_rows" in p["evidence"]


def test_profile_table_without_zip_does_not_monitor_filenames():
    photo = next(p for p in profile_table(_table(40, seed=4)) if p["name"] == "Photo")
    assert photo["proposed_type"] == "image" and not photo["proposed_monitor"]
    assert "no image ZIP" in photo["reason"]


def test_profile_table_matches_profile_columns_for_numeric_categorical():
    df = _table(60, seed=5)[["Age", "Fee", "Breed"]]
    old = {p["name"]: p["monitor"] for p in profile_columns(df)}
    new = {p["name"]: p["proposed_type"] for p in profile_table(df)}
    expected = {True: "numeric", "Categorical": "categorical"}
    assert {k: expected[v] for k, v in old.items()} == new


# ---------------------------------------------------------
# Image ZIP security and association
# ---------------------------------------------------------
def test_zip_rejects_unsafe_and_suspicious_entries(tmp_path):
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("good.png", _png(1))
        z.writestr("../escape.png", _png(2))
        z.writestr("/abs.png", _png(3))
        z.writestr("C:/drive.png", _png(4))
        z.writestr("__MACOSX/._good.png", b"x")
        z.writestr("notes.txt", b"hello")
        z.writestr("bomb.png", b"\0" * (5 * 1024 * 1024))  # ~1000:1 compression
        link = zipfile.ZipInfo("link.png")
        link.external_attr = (0o120777 << 16)
        z.writestr(link, "good.png")
    archive = ImageArchive(str(zpath))
    assert list(archive.by_path) == ["good.png"]
    assert archive.rejected == {"unsafe_path": 3, "os_metadata": 1, "not_an_image_file": 1,
                                "suspicious_compression_ratio": 1, "symlink": 1}
    archive.close()


def test_zip_association_ambiguous_missing_and_paths(tmp_path):
    zpath = tmp_path / "dirs.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("a/x.png", _png(1))
        z.writestr("b/x.png", _png(2))
        z.writestr("c/unique.png", _png(3))
    archive = ImageArchive(str(zpath))
    assert archive.resolve("x.png") == ("ambiguous", None)
    assert archive.resolve("a/x.png") == ("ok", "a/x.png")
    assert archive.resolve("unique.png") == ("ok", "c/unique.png")
    assert archive.resolve("nope.png") == ("missing", None)
    assert archive.resolve("../a/x.png") == ("missing", None)
    archive.close()


def test_image_validation_corrupt_inline_and_missing():
    import base64
    assert validate_image_bytes(b"garbage") == "corrupt"
    assert validate_image_bytes(_png(1)[:40]) == "corrupt"  # truncated
    data_uri = "data:image/png;base64," + base64.b64encode(_png(1)).decode()
    assert load_image(data_uri, None)[0] == "ok"
    assert load_image("x.png", None) == ("missing", None)
    assert load_image(None, None) == ("missing", None)


# ---------------------------------------------------------
# End-to-end fit / analyze
# ---------------------------------------------------------
def test_fit_records_confirmed_schema_and_cleans_up_stage(fitted):
    fit = fitted["fit"]
    assert fit["monitored"] == {"numeric": ["Age", "Fee"], "categorical": ["Breed"],
                                "text": ["Description"], "image": ["Photo"]}
    assert fit["not_monitored"] == ["PetID"]

    body = client.get(f"/tables/{fitted['project']}/baseline", headers=H).json()
    assert body["version"] == fit["version"]
    schema = {r["column_name"]: r for r in body["schema"]}
    assert schema["PetID"]["proposed_type"] == "ignore" and schema["PetID"]["final_monitor"] is False
    assert schema["Photo"]["final_type"] == "image" and schema["Photo"]["decided_by"] == EMAIL
    assert "sample_values" not in schema["Description"]["evidence"]
    assert body["reference_rows"] == 48

    stage = crud.get_stage(fitted["staged"]["stage_id"])
    assert stage["status"] == "consumed" and stage["table_blob"] is None and stage["profile"] is None
    assert crud.get_job(fitted["staged"]["job_id"])["result"] is None  # profile sample values cleared


def test_analyze_reports_columns_and_invalid_images(fitted):
    batch = _table(40, seed=2)
    bad, missing = batch["Photo"][0], batch["Photo"][1]
    resp = _analyze(fitted["project"], batch, _zip_for(batch, corrupt=(bad,), skip=(missing,)))
    assert resp.status_code == 202, resp.text
    job = _wait(resp.json()["job_id"])
    assert job["status"] == "succeeded", job
    report = job["result"]
    assert set(report["column_drift"]) == {"Age", "Fee", "Breed", "Description", "Photo"}
    assert report["column_drift"]["Photo"]["test"] == "DCT"
    assert report["column_drift"]["Photo"]["in_family"] is True  # calibrated real-embedding null (M3 decision)
    assert report["column_drift"]["Age"]["in_family"] is True  # calibrated tabular default
    assert report["data_quality"]["Photo"]["invalid"] == {"corrupt": 1, "missing": 1}
    assert any(i["issue"] == "invalid_images" for i in report["schema_report"]["Photo"])
    assert report["relationship_drift"] == {}
    assert report["family"]["method"] == "holm"

    runs = client.get(f"/history/{fitted['project']}", headers=H).json()
    conn = sqlite3.connect(crud.DB_PATH)
    kind = conn.execute("SELECT report_kind FROM analysis_runs WHERE job_id = ?", (job["job_id"],)).fetchone()
    conn.close()
    assert kind == ("table",) and runs


def test_analyze_tiny_batch_marks_embedding_columns_not_tested(fitted):
    batch = _table(2, seed=9)
    job = _wait(_analyze(fitted["project"], batch, _zip_for(batch)).json()["job_id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["column_drift"]["Description"]["status"] == "NOT_TESTED"


def test_idempotent_replay_and_conflict(fitted):
    batch = _table(30, seed=11)
    zip_bytes = _zip_for(batch)
    first = _analyze(fitted["project"], batch, zip_bytes, key=IDEM_KEY).json()
    _wait(first["job_id"])
    replay = _analyze(fitted["project"], batch, zip_bytes, key=IDEM_KEY).json()
    assert replay == {"job_id": first["job_id"], "stage_id": None, "expires_at": None, "replay": True}
    other = _table(30, seed=12)
    assert _analyze(fitted["project"], other, _zip_for(other), key=IDEM_KEY).status_code == 409


def test_no_raw_values_persisted(fitted):
    batch = _table(30, seed=13, sentinel_row=True)
    _wait(_analyze(fitted["project"], batch, _zip_for(batch)).json()["job_id"])
    conn = sqlite3.connect(crud.DB_PATH)
    for table in ("jobs", "analysis_runs", "table_schemas", "table_column_baselines", "staged_uploads",
                  "alert_events", "webhook_deliveries"):
        for row in conn.execute(f"SELECT * FROM {table}"):
            assert SENTINEL not in " ".join(str(v) for v in row), table
    conn.close()


# ---------------------------------------------------------
# Equivalence with the existing tabular endpoints
# ---------------------------------------------------------
def test_numeric_categorical_table_matches_tabular_endpoints():
    rng = random.Random(21)
    train = pd.DataFrame({"x": [rng.gauss(50, 10) for _ in range(200)],
                          "y": [rng.gauss(0, 1) for _ in range(200)],
                          "c": [rng.choice("abc") for _ in range(200)]})
    batch = pd.DataFrame({"x": [rng.gauss(55, 10) for _ in range(150)],
                          "y": [rng.gauss(0, 1) for _ in range(150)],
                          "c": [rng.choice("abcc") for _ in range(150)]})
    tab, tbl = PREFIX + "equiv_tabular", PREFIX + "equiv_table"
    assert client.post(f"/fit/{tab}/upload", files={"file": ("t.csv", _csv(train))}, headers=H).status_code == 200
    expected = client.post(f"/analyze/{tab}/upload", files={"file": ("b.csv", _csv(batch))}, headers=H).json()

    _fit(tbl, train)
    job = _wait(_analyze(tbl, batch).json()["job_id"])
    report = job["result"]
    for col, metric in expected["feature_metrics"].items():
        got = report["column_drift"][col]
        for field in ("statistic", "p_value", "p_value_adjusted", "drift_detected", "effect_size",
                      "effect_floor", "significant", "material", "decision_mode", "threshold_used"):
            assert got[field] == metric[field], (col, field)
    assert report["overall"]["alert"] == expected["system_alert_triggered"]
    assert report["schema_report"] == expected["schema_report"]


# ---------------------------------------------------------
# Isolation, validation, malformed input, lifecycle
# ---------------------------------------------------------
def test_two_user_isolation(fitted):
    staged = _stage(PREFIX + "iso", _table(20, seed=30))
    assert client.get(f"/jobs/{staged['job_id']}", headers=H_B).status_code == 404
    assert client.delete(f"/tables/stages/{staged['stage_id']}", headers=H_B).status_code == 404
    resp = client.post(f"/tables/{PREFIX}iso/fit", headers=H_B,
                       json={"stage_id": staged["stage_id"], "columns": [{"name": "Age", "type": "numeric", "monitor": True}]})
    assert resp.status_code == 404
    assert client.get(f"/tables/{fitted['project']}/baseline", headers=H_B).status_code == 404
    assert client.delete(f"/tables/stages/{staged['stage_id']}", headers=H).status_code == 200


def test_fit_validation_errors():
    project = PREFIX + "validation"
    staged = _stage(project, _table(20, seed=31))
    fit = lambda cols: client.post(f"/tables/{project}/fit", headers=H, json={"stage_id": staged["stage_id"], "columns": cols})
    assert fit([{"name": "Age", "type": "numeric", "monitor": True}]).status_code == 409  # not profiled yet
    _wait(staged["job_id"])
    assert fit([{"name": "Nope", "type": "numeric", "monitor": True}]).status_code == 422
    assert fit([{"name": "Age", "type": "ignore", "monitor": True}]).status_code == 422
    assert fit([{"name": "Photo", "type": "image", "monitor": True}]).status_code == 422  # no ZIP
    assert fit([{"name": "Age", "type": "bogus", "monitor": True}]).status_code == 422


def test_malformed_uploads():
    project = PREFIX + "malformed"
    resp = client.post(f"/tables/{project}/stage", headers=H, files={
        "file": ("t.csv", _csv(_table(10, seed=40))), "images": ("images.tar", b"x")})
    assert resp.status_code == 422
    assert client.post(f"/tables/{project}/stage", headers=H, files={"file": ("t.csv", b"")}).status_code == 422
    staged = client.post(f"/tables/{project}/stage", headers=H, files={
        "file": ("t.parquet", f"{SENTINEL} not parquet".encode())}).json()
    job = _wait(staged["job_id"])
    assert job["status"] == "failed" and SENTINEL not in job["error"]
    staged = client.post(f"/tables/{project}/stage", headers=H, files={
        "file": ("t.csv", _csv(_table(10, seed=41))), "images": ("i.zip", b"not a zip")}).json()
    job = _wait(staged["job_id"])
    assert job["status"] == "failed" and "could not be opened" in job["error"]


def test_stage_expiry_and_restart_recovery():
    staged = _stage(PREFIX + "expiry", _table(10, seed=50))
    jobs.run_pending()
    stage = crud.get_stage(staged["stage_id"])
    conn = sqlite3.connect(crud.DB_PATH)
    conn.execute("UPDATE staged_uploads SET expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (stage["id"],))
    conn.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (staged["job_id"],))
    conn.commit()
    conn.close()
    assert jobs.sweep_expired_stages() >= 1
    stage_after = crud.get_stage(stage["id"])
    assert stage_after["status"] == "expired" and not os.path.exists(blob_store.path(stage["table_blob"]))
    crud.mark_running_jobs_interrupted()
    assert crud.get_job(staged["job_id"])["status"] == "interrupted"


def test_delete_project_removes_table_artifacts_and_blobs():
    project = PREFIX + "delete"
    df = _table(20, seed=60)[["Age", "Fee", "Breed"]]
    _fit(project, df)
    internal = _internal_project_key(project, EMAIL)
    project_dir = os.path.join(blob_store.DATA_DIR, blob_store._project_dir(internal))
    assert os.path.isdir(project_dir)
    assert client.delete(f"/projects/{project}", headers=H).status_code == 200
    assert not os.path.exists(project_dir)
    assert crud.get_table_version(internal, 1) is None


# ---------------------------------------------------------
# Relationship drift (milestone M2)
# ---------------------------------------------------------
def _rel_table(n: int, seed: int, shuffle_y: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(50, 10, n)
    y = 0.9 * (x - 50) / 10 + 0.45 * rng.normal(size=n)
    breed = rng.choice(["lab", "beagle", "poodle"], n)
    fee = np.array([{"lab": 100, "beagle": 150, "poodle": 200}[b] for b in breed]) + rng.normal(0, 10, n)
    if shuffle_y:
        y = rng.permutation(y)  # y's own distribution is untouched; only its link to x breaks
    return pd.DataFrame({"x": x.round(3), "y": y.round(4), "breed": breed, "fee": fee.round(2)})


def test_relationship_drift_end_to_end():
    project = PREFIX + "relationships"
    out = _fit(project, _rel_table(1500, 70), accept_proposed_relationships=True)
    proposed = {(r["col_a"], r["col_b"]) for r in out["profile"]["relationships"] if r["proposed"]}
    assert {("x", "y"), ("fee", "breed")} <= proposed
    assert "x<->y" in out["fit"]["relationships"]

    stored = client.get(f"/tables/{project}/baseline", headers=H).json()["relationships"]
    assert {(r["col_a"], r["col_b"]) for r in stored if r["final_monitor"]} == proposed

    same = _wait(_analyze(project, _rel_table(300, 71)).json()["job_id"])["result"]
    assert all(r["status"] == "STABLE" for r in same["relationship_drift"].values()), same["relationship_drift"]
    assert "x<->y" in same["family"]["members"]

    job = _wait(_analyze(project, _rel_table(300, 72, shuffle_y=True)).json()["job_id"])
    shuffled = job["result"]
    assert shuffled["relationship_drift"]["x<->y"]["status"] == "DRIFT"
    assert shuffled["column_drift"]["y"]["status"] == "STABLE"   # marginal unchanged: no column alarm
    assert shuffled["relationship_drift"]["fee<->breed"]["status"] == "STABLE"
    assert "x<->y" in shuffled["overall"]["triggered_by"]

    conn = sqlite3.connect(crud.DB_PATH)
    stored_run = conn.execute("SELECT relationship_metrics FROM analysis_runs WHERE job_id = ?", (job["job_id"],)).fetchone()
    conn.close()
    assert "x<->y" in stored_run[0]


def test_relationship_validation_errors():
    project = PREFIX + "rel_validation"
    staged = _stage(project, _table(60, seed=80))
    profile = _wait(staged["job_id"])["result"]
    cols = _confirmed(profile)
    fit = lambda rels: client.post(f"/tables/{project}/fit", headers=H,
                                   json={"stage_id": staged["stage_id"], "columns": cols, "relationships": rels})
    assert fit([{"col_a": "Age", "col_b": "Description"}]).status_code == 422          # text column
    assert fit([{"col_a": "Age", "col_b": "PetID"}]).status_code == 422                # not monitored
    assert fit([{"col_a": "Age", "col_b": "Fee"}, {"col_a": "Fee", "col_b": "Age"}]).status_code == 422  # duplicate


def test_fit_keeps_distinct_records_that_share_values():
    """Regression: dedup used to run on the monitored columns only, collapsing
    distinct records (different IDs) that share low-cardinality values."""
    rng = np.random.default_rng(90)
    df = pd.DataFrame({"PetID": [f"id{i}" for i in range(400)],
                       "size": rng.choice(["s", "m", "l"], 400), "kind": rng.choice(["dog", "cat"], 400)})
    out = _fit(PREFIX + "dedup", df)
    assert out["fit"]["duplicate_rows_dropped"] == 0
    body = client.get(f"/tables/{PREFIX}dedup/baseline", headers=H).json()
    assert body["reference_rows"] == 400


# ---------------------------------------------------------
# Text/image relationships (milestone M3, report-only)
# ---------------------------------------------------------
def _breed_text_table(n: int, seed: int, shuffle_text: bool = False) -> pd.DataFrame:
    rng = random.Random(seed)
    breeds = [rng.choice(BREEDS) for _ in range(n)]
    desc = [f"{b} {' '.join(rng.choice(WORDS) for _ in range(6))} {i}" for i, b in enumerate(breeds)]
    if shuffle_text:
        rng.shuffle(desc)  # each text keeps its words; it just no longer describes its own row's breed
    weight = [{"labrador": 30, "beagle": 10, "poodle": 20, "terrier": 8}[b] + rng.gauss(0, 1) for b in breeds]
    return pd.DataFrame({"PetID": [f"T{seed}x{i}" for i in range(n)], "Breed": breeds, "Description": desc,
                         "Weight": [round(w, 2) for w in weight]})


def test_text_probes_categorical_alerts_numeric_is_report_only():
    project = PREFIX + "probe"
    out = _fit(project, _breed_text_table(240, 100), accept_proposed_relationships=True)
    proposed = {(r["col_a"], r["col_b"], r["kind"]) for r in out["profile"]["relationships"] if r["proposed"]}
    assert ("Description", "Breed", "probe") in proposed
    assert "Description<->Breed" in out["fit"]["relationships"]

    report = _wait(_analyze(project, _breed_text_table(120, 101, shuffle_text=True)).json()["job_id"])["result"]
    probe = report["relationship_drift"]["Description<->Breed"]
    assert probe["status"] == "DRIFT" and probe["in_family"] and not probe.get("report_only")
    assert probe["current_value"] < probe["reference_value"]
    assert "Description<->Breed" in report["overall"]["triggered_by"]
    numeric = report["relationship_drift"]["Description<->Weight"]
    assert numeric["report_only"] and numeric["in_family"] is False
    assert "Description<->Weight" not in report["overall"]["triggered_by"]
    assert isinstance(report["column_drift"]["Description"]["p_value"], float)  # real-embedding null p
