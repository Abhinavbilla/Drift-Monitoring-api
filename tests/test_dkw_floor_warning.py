"""
Step 5 Part 1 item 7: /fit warns when the KS materiality floor is below
the DKW bound sqrt(ln(2/0.05)/(2m)), and reports the minimum reference
size that would make the configured floor DKW-safe.

Run:
    python -m pytest tests/test_dkw_floor_warning.py -v
"""

import math
import random
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import db.crud as crud
from drift.calibration import dkw_bound, min_reference_size_for_dkw_floor
from main import app

client = TestClient(app)
EMAIL = "dkw-test@example.com"
TOKEN = mint_session_token(EMAIL)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def cleanup():
    yield
    conn = crud.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM baselines WHERE project_id LIKE '%test_dkw_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE '%test_dkw_%'")
    cur.execute("DELETE FROM baseline_versions WHERE project_id LIKE '%test_dkw_%'")
    cur.execute("DELETE FROM baseline_active_version WHERE project_id LIKE '%test_dkw_%'")
    conn.commit()
    conn.close()


def _x(seed, n):
    rng = random.Random(seed)
    return [rng.uniform(0, 100) for _ in range(n)]


class TestDkwBoundFormula:
    def test_matches_literal_formula(self):
        for m in (10, 100, 5000, 50000):
            expected = math.sqrt(math.log(2 / 0.05) / (2 * m))
            assert dkw_bound(m) == pytest.approx(expected)

    def test_min_reference_size_is_the_crossover_point(self):
        floor = 0.05
        m_min = min_reference_size_for_dkw_floor(floor)
        assert dkw_bound(m_min) <= floor
        assert dkw_bound(m_min - 1) > floor

    def test_bound_decreases_with_more_data(self):
        assert dkw_bound(100) > dkw_bound(10000)

    def test_non_positive_inputs_rejected(self):
        with pytest.raises(ValueError):
            dkw_bound(0)
        with pytest.raises(ValueError):
            min_reference_size_for_dkw_floor(0)


class TestFitWarnsOnSmallReference:
    def test_small_reference_triggers_dkw_warning(self):
        """Default ks_d floor is 0.05; min_reference_size_for_dkw_floor(0.05)
        is in the low thousands -- a 60-row reference is far below it."""
        pid = "test_dkw_small"
        resp = client.post(f"/fit/{pid}", json={"reference_data": {"x": _x(1, 60)}}, headers=HEADERS)
        body = resp.json()
        info = body["calibration_info"]["x"]
        assert info["floor_below_dkw_bound"] is True
        assert info["minimum_reference_size_for_dkw_safe_floor"] == min_reference_size_for_dkw_floor(0.05)
        assert "DKW bound" in body["message"]

    def test_large_reference_does_not_trigger_dkw_warning(self):
        needed = min_reference_size_for_dkw_floor(0.05)
        pid = "test_dkw_large"
        resp = client.post(f"/fit/{pid}", json={"reference_data": {"x": _x(2, needed + 500)}}, headers=HEADERS)
        body = resp.json()
        info = body["calibration_info"]["x"]
        assert info["floor_below_dkw_bound"] is False
        assert info["minimum_reference_size_for_dkw_safe_floor"] is None
        assert "DKW bound" not in body["message"]

    def test_custom_wider_floor_can_avoid_the_warning_at_small_n(self):
        """A caller who explicitly widens the floor for this feature can
        make even a small reference DKW-safe."""
        pid = "test_dkw_widefloor"
        n = 60
        wide_floor = dkw_bound(n) * 1.1  # deliberately just inside the safe zone
        resp = client.post(
            f"/fit/{pid}",
            json={
                "reference_data": {"x": _x(3, n)},
                "calibration_config": {"per_feature_effect_floors": {"x": wide_floor}},
            },
            headers=HEADERS,
        )
        info = resp.json()["calibration_info"]["x"]
        assert info["floor_below_dkw_bound"] is False

    def test_categorical_features_unaffected(self):
        """DKW/KS-floor concepts are continuous-only; a categorical
        feature's calibration_info must not carry these fields."""
        pid = "test_dkw_categorical"
        resp = client.post(
            f"/fit/{pid}",
            json={"reference_data": {}, "categorical_data": {"c": ["a", "b"] * 30}},
            headers=HEADERS,
        )
        info = resp.json()["calibration_info"]["c"]
        assert info["floor_below_dkw_bound"] is None
        assert info["minimum_reference_size_for_dkw_safe_floor"] is None
