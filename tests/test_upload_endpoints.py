"""
Step 3b: multipart upload endpoints for tabular /fit and /analyze.

Run:
    python -m pytest tests/test_upload_endpoints.py -v
"""

import io
import json
import random
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import db.crud as crud
import main as main_module
from main import app, _internal_project_key

client = TestClient(app)
EMAIL = "upload-endpoint-test@example.com"
TOKEN = mint_session_token(EMAIL)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def _internal(public_id):
    return _internal_project_key(public_id, EMAIL)

# Non-monotonic values -- a strictly increasing/linear sequence gets
# excluded by the profiler as "Monotonic sequence (Likely Time or Row
# Index)" (see tests/test_new_project_default_mode.py for the same
# pattern), which would silently drop the column from monitoring.
_rng = random.Random(7)
_X_VALUES = [_rng.uniform(0, 100) for _ in range(60)]
REFERENCE_CSV = (
    "x,cat\n" + "\n".join(f"{_X_VALUES[i]},{'a' if i % 2 == 0 else 'b'}" for i in range(60)) + "\n"
).encode()


@pytest.fixture(autouse=True)
def cleanup():
    yield
    conn = crud.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM baselines WHERE project_id LIKE '%test_upload_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE '%test_upload_%'")
    conn.commit()
    conn.close()


class TestFitUpload:
    def test_csv_upload_fits_baseline(self):
        resp = client.post(
            "/fit/test_upload_csv_proj/upload",
            files={"file": ("reference.csv", REFERENCE_CSV, "text/csv")},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["inferred_feature_types"]["x"] == "continuous"
        assert body["inferred_feature_types"]["cat"] == "categorical"
        state = crud.get_baseline(_internal("test_upload_csv_proj"))
        assert state is not None
        assert len(state["reference_data"]["x"]) == 60

    def test_parquet_upload_fits_baseline(self):
        pytest.importorskip("pyarrow")
        df = pd.DataFrame({"x": _X_VALUES,
                            "cat": ["a" if i % 2 == 0 else "b" for i in range(60)]})
        buf = io.BytesIO()
        df.to_parquet(buf)
        resp = client.post(
            "/fit/test_upload_parquet_proj/upload",
            files={"file": ("reference.parquet", buf.getvalue(), "application/octet-stream")},
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_calibration_config_form_field_applied(self):
        resp = client.post(
            "/fit/test_upload_calib_proj/upload",
            files={"file": ("reference.csv", REFERENCE_CSV, "text/csv")},
            data={"calibration_config": json.dumps({"decision_mode": "legacy"})},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        state = crud.get_baseline(_internal("test_upload_calib_proj"))
        assert state["calibration_config"]["decision_mode"] == "legacy"

    def test_malformed_calibration_config_json_rejected(self):
        resp = client.post(
            "/fit/test_upload_badcalib_proj/upload",
            files={"file": ("reference.csv", REFERENCE_CSV, "text/csv")},
            data={"calibration_config": "{not valid json"},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_empty_file_rejected(self):
        resp = client.post(
            "/fit/test_upload_empty_proj/upload",
            files={"file": ("reference.csv", b"", "text/csv")},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_oversized_file_rejected(self, monkeypatch):
        monkeypatch.setattr(main_module, "MAX_UPLOAD_SIZE_BYTES", 10)
        resp = client.post(
            "/fit/test_upload_oversized_proj/upload",
            files={"file": ("reference.csv", REFERENCE_CSV, "text/csv")},
            headers=HEADERS,
        )
        assert resp.status_code == 413

    def test_unparseable_file_rejected_cleanly(self):
        resp = client.post(
            "/fit/test_upload_badfile_proj/upload",
            files={"file": ("reference.parquet", b"this is not a real parquet file", "application/octet-stream")},
            headers=HEADERS,
        )
        assert resp.status_code == 422


class TestAnalyzeUpload:
    def test_csv_upload_analyzes_batch(self):
        client.post(
            "/fit/test_upload_analyze_proj/upload",
            files={"file": ("reference.csv", REFERENCE_CSV, "text/csv")},
            headers=HEADERS,
        )
        batch_csv = ("x,cat\n" + "\n".join(f"{v},a" for v in _X_VALUES) + "\n").encode()
        resp = client.post(
            "/analyze/test_upload_analyze_proj/upload",
            files={"file": ("batch.csv", batch_csv, "text/csv")},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert "system_alert_triggered" in resp.json()

    def test_analyze_upload_without_baseline_404(self):
        batch_csv = ("x,cat\n" + "\n".join(f"{v},a" for v in _X_VALUES) + "\n").encode()
        resp = client.post(
            "/analyze/test_upload_nobaseline_proj/upload",
            files={"file": ("batch.csv", batch_csv, "text/csv")},
            headers=HEADERS,
        )
        assert resp.status_code == 404
