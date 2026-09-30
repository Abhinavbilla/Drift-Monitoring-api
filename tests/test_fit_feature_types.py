"""
Tests for /fit's explicit feature_types override and the "never silently
drop a column" fix (2026-09-30 hardening pass, item 3).

Run:
    python -m pytest tests/test_fit_feature_types.py -v
"""

import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import db.crud as crud
from main import app

client = TestClient(app)
TOKEN = mint_session_token("feature-types-test@example.com")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def cleanup():
    yield
    conn = crud.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM baselines WHERE project_id LIKE 'test_ftypes_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE 'test_ftypes_%'")
    conn.commit()
    conn.close()


import random
_rng = random.Random(9)
_X = [_rng.uniform(0, 100) for _ in range(60)]
_CAT = ["p" if i % 2 == 0 else "q" for i in range(60)]


class TestNoSilentDrop:
    def test_categorical_column_submitted_under_reference_data_survives(self):
        """The root-cause fix: 'cat' is submitted ONLY under
        reference_data (the 'wrong' dict for a categorical column, by the
        old convention), with no feature_types override at all -- it must
        still survive, not be silently dropped."""
        resp = client.post(
            "/fit/test_ftypes_wrongdict",
            json={"reference_data": {"x": _X, "cat": _CAT}},
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["inferred_feature_types"]["cat"] == "categorical"
        state = crud.get_baseline("test_ftypes_wrongdict")
        assert "cat" in state["reference_data"]
        assert len(state["reference_data"]["cat"]) == 60

    def test_continuous_column_submitted_under_categorical_data_survives(self):
        resp = client.post(
            "/fit/test_ftypes_wrongdict2",
            json={"reference_data": {}, "categorical_data": {"x": _X, "cat": _CAT}},
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["inferred_feature_types"]["x"] == "continuous"
        state = crud.get_baseline("test_ftypes_wrongdict2")
        assert "x" in state["reference_data"]
        assert len(state["reference_data"]["x"]) == 60


class TestFeatureTypesOverride:
    def test_override_forces_categorical(self):
        resp = client.post(
            "/fit/test_ftypes_override",
            json={"reference_data": {"x": _X}, "categorical_data": {"cat": _CAT}},
            headers=HEADERS,
        )
        assert resp.json()["inferred_feature_types"]["x"] == "continuous"

        resp2 = client.post(
            "/fit/test_ftypes_override",
            json={"reference_data": {"x": _X}, "categorical_data": {"cat": _CAT},
                  "feature_types": {"x": "categorical"}},
            headers=HEADERS,
        )
        assert resp2.status_code == 200, resp2.text
        assert resp2.json()["inferred_feature_types"]["x"] == "categorical"

    def test_override_unknown_type_value_422(self):
        resp = client.post(
            "/fit/test_ftypes_badvalue",
            json={"reference_data": {"x": _X}, "feature_types": {"x": "ordinal"}},
            headers=HEADERS,
        )
        assert resp.status_code == 422
        assert "x" in resp.json()["detail"]

    def test_override_unknown_column_422(self):
        resp = client.post(
            "/fit/test_ftypes_badcol",
            json={"reference_data": {"x": _X}, "feature_types": {"does_not_exist": "continuous"}},
            headers=HEADERS,
        )
        assert resp.status_code == 422
        assert "does_not_exist" in resp.json()["detail"]

    def test_override_named_columns_unaffected(self):
        """A column not named in feature_types keeps its normal profiled
        classification."""
        resp = client.post(
            "/fit/test_ftypes_partial",
            json={"reference_data": {"x": _X}, "categorical_data": {"cat": _CAT},
                  "feature_types": {"x": "categorical"}},
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text
        types = resp.json()["inferred_feature_types"]
        assert types["x"] == "categorical"
        assert types["cat"] == "categorical"  # unaffected, still profiler-classified
