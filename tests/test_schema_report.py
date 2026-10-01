"""
Step 5 Part 1 item 4: schema_report on every tabular /analyze response.

Run:
    python -m pytest tests/test_schema_report.py -v
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
EMAIL = "schema-report-test@example.com"
EMAIL_B = "schema-report-test-b@example.com"
TOKEN = mint_session_token(EMAIL)
TOKEN_B = mint_session_token(EMAIL_B)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
HEADERS_B = {"Authorization": f"Bearer {TOKEN_B}"}


def _internal(public_id, email=EMAIL):
    return _internal_project_key(public_id, email)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    conn = crud.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM baselines WHERE project_id LIKE '%test_schema_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE '%test_schema_%'")
    cur.execute("DELETE FROM analysis_runs WHERE project LIKE '%test_schema_%'")
    conn.commit()
    conn.close()


def _x_values(n=60, seed=5):
    rng = random.Random(seed)
    return [rng.uniform(0, 100) for _ in range(n)]


def _fit(project_id, cat=False, headers=HEADERS, schema_policy=None):
    payload = {"reference_data": {"x": _x_values()}}
    if cat:
        payload["categorical_data"] = {"c": ["a" if i % 2 == 0 else "b" for i in range(60)]}
    if schema_policy is not None:
        payload["schema_policy"] = schema_policy
    return client.post(f"/fit/{project_id}", json=payload, headers=headers)


def _analyze(project_id, production_data, headers=HEADERS):
    return client.post(f"/analyze/{project_id}", json={"production_data": production_data}, headers=headers)


class TestMissingAndUnexpectedColumns:
    def test_missing_column_flagged_alert_by_default(self):
        pid = "test_schema_missing"
        _fit(pid, cat=True)
        resp = _analyze(pid, {"x": _x_values(60, seed=9)})  # 'c' omitted
        assert resp.status_code == 200
        report = resp.json()["schema_report"]
        issues = report["c"]
        assert any(i["issue"] == "missing_column" and i["severity"] == "alert" for i in issues)

    def test_unexpected_column_flagged_warn_by_default(self):
        pid = "test_schema_unexpected"
        _fit(pid)
        resp = _analyze(pid, {"x": _x_values(60, seed=9), "surprise_col": [1] * 60})
        body = resp.json()
        assert resp.status_code == 200
        issues = body["schema_report"]["surprise_col"]
        assert any(i["issue"] == "unexpected_column" and i["severity"] == "warn" for i in issues)
        assert "surprise_col" not in body["feature_metrics"]  # never fed to the detector

    def test_extra_column_never_crashes_valid_columns_still_analyzed(self):
        pid = "test_schema_extra_no_crash"
        _fit(pid)
        resp = _analyze(pid, {"x": _x_values(60, seed=9), "junk": ["a", "b"] * 30})
        assert resp.status_code == 200
        assert "x" in resp.json()["feature_metrics"]


class TestDtypeChangeAndNulls:
    def test_dtype_change_flagged_for_continuous_column(self):
        pid = "test_schema_dtype"
        _fit(pid)
        batch = _x_values(59, seed=9) + ["not_a_number"]
        resp = _analyze(pid, {"x": batch})
        assert resp.status_code == 200
        body = resp.json()
        issues = body["schema_report"]["x"]
        assert any(i["issue"] == "dtype_change" and i["dropped"] == 1 for i in issues)
        assert "x" in body["feature_metrics"]  # still analyzed on the valid values

    def test_null_values_cleaned_before_statistics_computed(self):
        """2026-10-01 NULL policy decision: /analyze drops nulls before
        computing statistics, instead of silently producing a meaningless
        nan statistic."""
        pid = "test_schema_nullclean"
        _fit(pid)
        batch = _x_values(59, seed=9) + [None]
        resp = _analyze(pid, {"x": batch})
        assert resp.status_code == 200
        metrics = resp.json()["feature_metrics"]["x"]
        assert metrics["statistic"] is not None  # not nan/None -- a real computed value

    def test_null_rate_increase_flagged_against_baseline(self):
        pid = "test_schema_nullrate"
        _fit(pid)  # baseline has 0% nulls
        batch = _x_values(30, seed=9) + [None] * 30  # 50% nulls
        resp = _analyze(pid, {"x": batch})
        issues = resp.json()["schema_report"]["x"]
        assert any(i["issue"] == "null_rate" for i in issues)


class TestUnseenCategoriesAndConstantColumn:
    def test_unseen_category_flagged(self):
        pid = "test_schema_unseen"
        _fit(pid, cat=True)
        resp = _analyze(pid, {"x": _x_values(60, seed=9), "c": ["z"] * 60})
        issues = resp.json()["schema_report"]["c"]
        assert any(i["issue"] == "unseen_categories" and "z" in i["values"] for i in issues)

    def test_constant_column_flagged(self):
        pid = "test_schema_constant"
        _fit(pid)
        resp = _analyze(pid, {"x": [42.0] * 60})
        issues = resp.json()["schema_report"]["x"]
        assert any(i["issue"] == "constant_column" for i in issues)

    def test_clean_batch_has_no_schema_issues(self):
        pid = "test_schema_clean"
        _fit(pid, cat=True)
        resp = _analyze(pid, {"x": _x_values(60, seed=9), "c": ["a", "b"] * 30})
        assert resp.json()["schema_report"] == {}


class TestSchemaPolicy:
    def test_ignore_policy_suppresses_the_issue(self):
        pid = "test_schema_ignorepolicy"
        _fit(pid, cat=True, schema_policy={"missing_columns": "ignore"})
        resp = _analyze(pid, {"x": _x_values(60, seed=9)})  # 'c' missing
        assert resp.json()["schema_report"] == {}

    def test_custom_default_severity_applied(self):
        pid = "test_schema_customdefault"
        _fit(pid, schema_policy={"default": "alert"})
        resp = _analyze(pid, {"x": [1.0] * 60})  # constant column -> "default" bucket
        issues = resp.json()["schema_report"]["x"]
        assert any(i["issue"] == "constant_column" and i["severity"] == "alert" for i in issues)

    def test_invalid_schema_policy_key_rejected(self):
        pid = "test_schema_badkey"
        resp = _fit(pid, schema_policy={"nonexistent_key": "alert"})
        assert resp.status_code == 422

    def test_invalid_schema_policy_severity_rejected(self):
        pid = "test_schema_badseverity"
        resp = _fit(pid, schema_policy={"default": "explode"})
        assert resp.status_code == 422

    def test_policy_persists_across_refit_without_mention(self):
        pid = "test_schema_persist"
        _fit(pid, cat=True, schema_policy={"missing_columns": "ignore"})
        _fit(pid, cat=True)  # re-fit without mentioning schema_policy
        resp = _analyze(pid, {"x": _x_values(60, seed=9)})  # 'c' missing
        assert resp.json()["schema_report"] == {}  # policy still applied


class TestSchemaReportHistoryAndIdempotency:
    def test_schema_report_persisted_in_history(self):
        pid = "test_schema_history"
        _fit(pid)
        _analyze(pid, {"x": [42.0] * 60})
        hist = client.get(f"/history/{pid}", headers=HEADERS).json()
        assert "constant_column" in str(hist["runs"][0]["schema_report"])

    def test_idempotent_replay_includes_schema_report(self):
        pid = "test_schema_idem"
        _fit(pid)
        batch = {"x": [42.0] * 60}
        r1 = client.post(f"/analyze/{pid}", json={"production_data": batch},
                          headers={**HEADERS, "Idempotency-Key": "schema-idem-key"})
        r2 = client.post(f"/analyze/{pid}", json={"production_data": batch},
                          headers={**HEADERS, "Idempotency-Key": "schema-idem-key"})
        assert r1.json()["schema_report"] == r2.json()["schema_report"]


class TestSchemaReportIsolation:
    def test_user_b_schema_policy_independent_of_user_a(self):
        pid = "test_schema_iso"
        _fit(pid, cat=True, schema_policy={"missing_columns": "ignore"}, headers=HEADERS)
        _fit(pid, cat=True, headers=HEADERS_B)  # B's own project, default policy

        resp_a = _analyze(pid, {"x": _x_values(60, seed=9)}, headers=HEADERS)
        resp_b = _analyze(pid, {"x": _x_values(60, seed=9)}, headers=HEADERS_B)

        assert resp_a.json()["schema_report"] == {}  # A's ignore policy applied
        assert any(i["issue"] == "missing_column" for i in resp_b.json()["schema_report"]["c"])
