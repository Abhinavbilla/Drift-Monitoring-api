"""
Reducing tabular /fit limitations (2026-10-05): categorical whitespace
normalization, exact-duplicate-row removal, and graceful (not whole-fit-
aborting) exclusion of an over-cardinality categorical column -- plus a
regression test for a profiling-order bug found while building this
(deduplicating before profiling could manufacture a false "monotonic
sequence" verdict on a column that just cycles through a few repeating
values).

Run:
    python -m pytest tests/test_tabular_cleaning_improvements.py -v
"""

import random
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import db.crud as crud
from main import app, _internal_project_key

client = TestClient(app)
EMAIL = "tabular-cleaning-test@example.com"
TOKEN = mint_session_token(EMAIL)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def _internal(public_id):
    return _internal_project_key(public_id, EMAIL)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    conn = crud.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM baselines WHERE project_id LIKE '%test_tabclean_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE '%test_tabclean_%'")
    cur.execute("DELETE FROM baseline_versions WHERE project_id LIKE '%test_tabclean_%'")
    cur.execute("DELETE FROM baseline_active_version WHERE project_id LIKE '%test_tabclean_%'")
    conn.commit()
    conn.close()


def _x(seed=1, n=60, lo=0, hi=100):
    rng = random.Random(seed)
    return [rng.uniform(lo, hi) for _ in range(n)]


class TestCategoricalWhitespaceNormalization:
    def test_whitespace_variants_merge_into_one_category(self):
        pid = "test_tabclean_whitespace"
        cats = (["USA", " USA", "USA ", " USA "] * 15)  # 60 values, 1 real category
        resp = client.post(
            f"/fit/{pid}", json={"reference_data": {"x": _x()}, "categorical_data": {"country": cats}},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        state = crud.get_baseline(_internal(pid))
        fence = next(f for f in state["iqr_fences"] if f["feature_name"] == "country")
        assert fence["allowed_values"] == ["USA"]
        # The STORED values are normalized too, not just the fence --
        # otherwise /analyze's PSI comparison would disagree with what
        # the fence says is "allowed".
        assert set(state["reference_data"]["country"]) == {"USA"}

    def test_case_is_not_folded(self):
        """Whitespace is stripped, but case is left alone -- it can be a
        real, intended distinction, unlike incidental whitespace."""
        pid = "test_tabclean_case"
        cats = (["USA", "usa"] * 30)
        resp = client.post(
            f"/fit/{pid}", json={"reference_data": {"x": _x()}, "categorical_data": {"country": cats}},
            headers=HEADERS,
        )
        state = crud.get_baseline(_internal(pid))
        fence = next(f for f in state["iqr_fences"] if f["feature_name"] == "country")
        assert set(fence["allowed_values"]) == {"USA", "usa"}
        assert resp.status_code == 200


class TestDuplicateRowRemoval:
    def test_exact_duplicate_rows_removed_and_reported(self):
        pid = "test_tabclean_dedup"
        x = [1.0, 2.0, 3.0, 4.0, 5.0] * 20  # 100 rows, only 5 unique rows
        resp = client.post(f"/fit/{pid}", json={"reference_data": {"x": x}}, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["duplicate_rows_dropped"] == 95
        assert "duplicate row" in resp.json()["message"]

    def test_no_duplicates_reports_zero(self):
        pid = "test_tabclean_nodedup"
        resp = client.post(f"/fit/{pid}", json={"reference_data": {"x": _x(seed=2)}}, headers=HEADERS)
        assert resp.json()["duplicate_rows_dropped"] == 0

    def test_dedup_does_not_manufacture_a_false_monotonic_verdict(self):
        """Regression: deduplicating BEFORE profiling collapsed a
        cycling-but-not-monotonic column down to just its unique values
        in first-occurrence order, which could look monotonic by
        accident and get the column wrongly excluded. Profiling now
        happens on the real, pre-dedup data."""
        pid = "test_tabclean_dedup_order"
        x = [100.0, 200.0, 300.0, 400.0, 500.0] * 20  # cycles, not monotonic; dedups to a monotonic-looking 5
        resp = client.post(f"/fit/{pid}", json={"reference_data": {"x": x}}, headers=HEADERS)
        assert resp.status_code == 200
        assert "x" in resp.json()["inferred_feature_types"]
        assert resp.json()["excluded_columns"] == {}
        state = crud.get_baseline(_internal(pid))
        assert len(state["reference_data"]["x"]) == 5  # still deduplicated for storage


class TestHighCardinalityColumnExcludedNotRejected:
    def test_over_cap_column_excluded_rest_of_fit_still_succeeds(self):
        """The profiler's OWN "likely an identifier" heuristic already
        catches an obvious case like this before it ever reaches
        categorical_features -- so to actually exercise the 50-value cap
        itself (the thing that used to abort the whole fit), force the
        column into 'categorical' via an explicit override, overruling
        the profiler's own judgment, the way a user insisting "no,
        monitor this one anyway" would."""
        pid = "test_tabclean_cardinality"
        free_text_ids = [f"id_{i}" for i in range(60)]  # 60 unique -- over the 50 cap
        resp = client.post(
            f"/fit/{pid}",
            json={
                "reference_data": {"x": _x(seed=3)},
                "categorical_data": {"user_id": free_text_ids},
                "feature_types": {"user_id": "categorical"},
            },
            headers=HEADERS,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "x" in body["inferred_feature_types"]
        assert "user_id" not in body["inferred_feature_types"]
        assert "user_id" in body["excluded_columns"]
        assert "50" in body["excluded_columns"]["user_id"]

    def test_profiler_own_identifier_heuristic_already_excludes_without_an_override(self):
        """Without an override, the profiler's own cardinality_ratio
        check catches this first and excludes it with its own reason --
        the fit still succeeds, just never reaches the 50-value cap
        check at all for this unforced case."""
        pid = "test_tabclean_cardinality_profiler_own"
        free_text_ids = [f"id_{i}" for i in range(60)]
        resp = client.post(
            f"/fit/{pid}",
            json={"reference_data": {"x": _x(seed=7)}, "categorical_data": {"user_id": free_text_ids}},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert "user_id" in resp.json()["excluded_columns"]
        assert "identifier" in resp.json()["excluded_columns"]["user_id"].lower()

    def test_excluded_column_in_a_later_analyze_batch_is_reported_not_a_crash(self):
        pid = "test_tabclean_cardinality_analyze"
        free_text_ids = [f"id_{i}" for i in range(60)]
        client.post(
            f"/fit/{pid}",
            json={"reference_data": {"x": _x(seed=4)}, "categorical_data": {"user_id": free_text_ids}},
            headers=HEADERS,
        )
        resp = client.post(
            f"/analyze/{pid}",
            json={"production_data": {"x": _x(seed=5), "user_id": [f"id_{i}" for i in range(60, 120)]}},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert "user_id" in resp.json()["schema_report"]
        assert resp.json()["schema_report"]["user_id"][0]["issue"] == "unexpected_column"

    def test_numeric_column_forced_categorical_is_not_subject_to_the_cap(self):
        """A continuous-valued column force-labeled 'categorical' via
        feature_types is stored and analyzed as continuous regardless
        (db/crud.py's _calculate_boundaries decides by attempting numeric
        coercion first, not by the caller's label) -- the cardinality cap
        only ever applies to genuinely non-numeric categorical data, so
        this must not be excluded just for having many distinct floats."""
        pid = "test_tabclean_numeric_override"
        x = _x(seed=6, n=60)  # 60 distinct floats, never repeats
        resp = client.post(
            f"/fit/{pid}",
            json={"reference_data": {"x": x}, "feature_types": {"x": "categorical"}},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["inferred_feature_types"]["x"] == "categorical"
        assert resp.json()["excluded_columns"] == {}
