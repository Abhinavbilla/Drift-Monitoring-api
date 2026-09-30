"""
Tests for the 2026-09-30 decision: NEW projects default to
decision_mode="calibrated"; EXISTING projects (calibration_config already
NULL in the DB, i.e. fit before this change) stay legacy permanently, with
no migration. This lives in main.py's /fit handler (NEW_PROJECT_DEFAULT_DECISION_MODE
in drift/calibration.py), not in CalibrationConfig's own default (which
must stay "legacy" forever, since that's what every pre-existing NULL row
resolves to).

Run:
    python -m pytest tests/test_new_project_default_mode.py -v
"""

import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import db.crud as crud
from main import app

client = TestClient(app)
TOKEN = mint_session_token("new-project-default-test@example.com")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def cleanup():
    yield
    conn = crud.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM baselines WHERE project_id LIKE 'test_new_default_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE 'test_new_default_%'")
    conn.commit()
    conn.close()


class TestNewProjectDefaultsToCalibrated:
    def test_brand_new_project_no_config_gets_calibrated(self):
        resp = client.post(
            "/fit/test_new_default_brand_new",
            json={"reference_data": {"x": [1, 2, 3, 4, 5] * 20}},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        state = crud.get_baseline("test_new_default_brand_new")
        assert state["calibration_config"]["decision_mode"] == "calibrated"

    def test_brand_new_project_explicit_legacy_respected(self):
        """A caller can still explicitly opt a new project into legacy."""
        resp = client.post(
            "/fit/test_new_default_explicit_legacy",
            json={"reference_data": {"x": [1, 2, 3, 4, 5] * 20},
                  "calibration_config": {"decision_mode": "legacy"}},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        state = crud.get_baseline("test_new_default_explicit_legacy")
        assert state["calibration_config"]["decision_mode"] == "legacy"

    def test_refit_of_new_default_project_stays_calibrated(self):
        """Re-fitting a project that got the calibrated default, without
        mentioning calibration_config again, must not revert it."""
        import random
        rng = random.Random(1)
        first_data = [rng.uniform(0, 1) for _ in range(100)]  # non-monotonic, high-cardinality -> continuous
        client.post("/fit/test_new_default_refit",
                    json={"reference_data": {"x": first_data}}, headers=HEADERS)
        assert crud.get_baseline("test_new_default_refit")["calibration_config"]["decision_mode"] == "calibrated"

        second_data = [rng.uniform(1000, 1001) for _ in range(100)]
        client.post("/fit/test_new_default_refit",
                    json={"reference_data": {"x": second_data}}, headers=HEADERS)
        state = crud.get_baseline("test_new_default_refit")
        assert state["calibration_config"]["decision_mode"] == "calibrated"
        assert state["reference_data"]["x"][0] >= 1000  # the re-fit's new data did apply


class TestExistingProjectStaysLegacy:
    def test_project_with_null_config_refit_without_mention_stays_legacy(self):
        """Simulates a project that existed BEFORE this default changed:
        calibration_config is NULL in the DB (as every project fit before
        2026-09-30 has). Re-fitting it via /fit with no calibration_config
        in the request must NOT pick up the new-project default -- it's
        not a new project, it already exists."""
        # Directly create the row the way pre-2026-09-30 code would have
        # (no calibration_config argument at all) -- bypasses the endpoint
        # to simulate a genuinely pre-existing row.
        crud.insert_baseline("test_new_default_preexisting", {"x": "continuous"},
                              {"x": [1, 2, 3, 4, 5] * 20})
        assert crud.get_baseline("test_new_default_preexisting")["calibration_config"] is None

        # Now re-fit it via the actual API, exactly as an ML team's existing
        # pipeline would, without ever mentioning calibration_config.
        resp = client.post(
            "/fit/test_new_default_preexisting",
            json={"reference_data": {"x": [10, 20, 30, 40, 50] * 20}},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        state = crud.get_baseline("test_new_default_preexisting")
        assert state["calibration_config"] is None  # still NULL -> resolves to legacy
