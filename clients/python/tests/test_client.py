"""
Tests for drift_monitor_client.DriftClient against a LOCALLY RUNNING
backend (python -m uvicorn main:app --port 8000 from the repo root) --
per instruction, not mocked. Skipped automatically if the server isn't
reachable.

Run (from the repo root, with the backend already running):
    python -m pytest clients/python/tests/test_client.py -v
"""

import os
import random
import sys

import pandas as pd
import pytest
import requests

_HERE = os.path.dirname(__file__)
_CLIENT_PKG_ROOT = os.path.join(_HERE, "..")
_REPO_ROOT = os.path.join(_HERE, "..", "..", "..")
sys.path.insert(0, _CLIENT_PKG_ROOT)
sys.path.insert(0, _REPO_ROOT)

from drift_monitor_client import DriftClient, DriftClientError  # noqa: E402

import db.crud as crud  # noqa: E402
from auth.tokens import generate_token  # noqa: E402

BASE_URL = os.environ.get("DRIFT_API_BASE_URL", "http://127.0.0.1:8000")


def _server_reachable() -> bool:
    try:
        requests.get(f"{BASE_URL}/docs", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _server_reachable(), reason=f"No server reachable at {BASE_URL}")


@pytest.fixture()
def token():
    crud.init_db()
    full_token, prefix, token_hash = generate_token()
    token_id = f"clienttest-{prefix}"
    crud.create_api_token(
        token_id=token_id, user_email="drift-client-test@example.com", name="client-tests",
        prefix=prefix, token_hash=token_hash, project_scope=["*"],
        created_at="2026-01-01T00:00:00+00:00", expires_at=None,
    )
    yield full_token
    conn = crud.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM api_tokens WHERE id = ?", (token_id,))
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def cleanup_projects():
    yield
    conn = crud.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM baselines WHERE project_id LIKE 'test_client_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE 'test_client_%'")
    conn.commit()
    conn.close()


def _reference_df(n=80):
    rng = random.Random(3)
    return pd.DataFrame({
        "x": [rng.uniform(0, 100) for _ in range(n)],
        "cat": [rng.choice(["a", "b", "c"]) for _ in range(n)],
    })


class TestFitAndAnalyze:
    def test_fit_json_path_small_frame(self, token):
        client = DriftClient(BASE_URL, token)
        result = client.fit("test_client_small", _reference_df(80))
        assert result["status"] == "success"
        assert result["inferred_feature_types"]["x"] == "continuous"

    def test_fit_upload_path_large_frame(self, token):
        pytest.importorskip("pyarrow")
        client = DriftClient(BASE_URL, token, large_frame_row_threshold=50)
        result = client.fit("test_client_large", _reference_df(80))
        assert result["status"] == "success"

    def test_analyze_after_fit(self, token):
        client = DriftClient(BASE_URL, token)
        client.fit("test_client_analyze", _reference_df(100))
        result = client.analyze("test_client_analyze", _reference_df(50))
        assert "system_alert_triggered" in result
        assert "x" in result["feature_metrics"]

    def test_calibration_config_passed_through(self, token):
        client = DriftClient(BASE_URL, token)
        client.fit("test_client_calib", _reference_df(80), calibration_config={"decision_mode": "legacy"})
        state = crud.get_baseline("test_client_calib")
        assert state["calibration_config"]["decision_mode"] == "legacy"

    def test_categorical_column_survives_json_fit_path(self, token):
        """Regression test: a column the server profiles as categorical
        (like 'cat' here, 3 low-cardinality values) must not be silently
        dropped by the JSON fit path -- see client.py's _fit_via_json
        docstring for the server-side quirk this works around."""
        client = DriftClient(BASE_URL, token)
        result = client.fit("test_client_catcol", _reference_df(80))
        assert result["inferred_feature_types"].get("cat") == "categorical"
        # crud.get_baseline merges continuous + categorical values into one
        # "reference_data" key at the storage layer (see db/crud.py's
        # insert_baseline docstring) -- so a surviving categorical column
        # shows up there, not under a separate key.
        state = crud.get_baseline("test_client_catcol")
        assert "cat" in state["reference_data"]
        assert len(state["reference_data"]["cat"]) == 80


class TestErrors:
    def test_analyze_without_fit_raises_client_error(self, token):
        client = DriftClient(BASE_URL, token)
        with pytest.raises(DriftClientError) as exc_info:
            client.analyze("test_client_never_fit", _reference_df(10))
        assert exc_info.value.status_code == 404

    def test_bad_token_raises_client_error(self):
        client = DriftClient(BASE_URL, "dm_deadbeef_notreal", max_retries=0)
        with pytest.raises(DriftClientError) as exc_info:
            client.list_projects()
        assert exc_info.value.status_code == 401


class TestProjectManagement:
    def test_list_and_delete_project(self, token):
        client = DriftClient(BASE_URL, token)
        client.fit("test_client_delete_me", _reference_df(60))
        assert "test_client_delete_me" in client.list_projects()
        client.delete_project("test_client_delete_me")
        assert "test_client_delete_me" not in client.list_projects()
