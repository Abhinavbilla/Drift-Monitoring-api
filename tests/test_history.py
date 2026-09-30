"""
Step 5 Part 1 item 2: analysis_runs history table + GET /history/{project_id}.

Run:
    python -m pytest tests/test_history.py -v
"""

import random
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import db.crud as crud
from main import app, _internal_project_key

client = TestClient(app)
EMAIL = "history-test@example.com"
EMAIL_B = "history-test-b@example.com"
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
    cur.execute("DELETE FROM baselines WHERE project_id LIKE '%test_hist_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE '%test_hist_%'")
    cur.execute("DELETE FROM analysis_runs WHERE project LIKE '%test_hist_%'")
    conn.commit()
    conn.close()


def _fit(project_id, headers=HEADERS):
    rng = random.Random(5)
    x = [rng.uniform(0, 100) for _ in range(60)]
    return client.post(f"/fit/{project_id}", json={"reference_data": {"x": x}}, headers=headers)


def _analyze(project_id, values, headers=HEADERS):
    return client.post(f"/analyze/{project_id}", json={"production_data": {"x": values}}, headers=headers)


class TestHistoryRecording:
    def test_one_row_per_analyze_call(self):
        pid = "test_hist_rowcount"
        _fit(pid)
        _analyze(pid, [1.0, 2.0, 3.0] * 20)
        _analyze(pid, [4.0, 5.0, 6.0] * 20)
        resp = client.get(f"/history/{pid}", headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["runs"]) == 2

    def test_upload_analyze_also_recorded(self):
        pid = "test_hist_upload"
        _fit(pid)
        batch_csv = ("x\n" + "\n".join(str(v) for v in [1.0, 2.0, 3.0] * 20) + "\n").encode()
        resp = client.post(f"/analyze/{pid}/upload", files={"file": ("b.csv", batch_csv, "text/csv")},
                            headers=HEADERS)
        assert resp.status_code == 200
        hist = client.get(f"/history/{pid}", headers=HEADERS)
        assert hist.json()["total"] == 1

    def test_run_never_stores_raw_production_values(self):
        """Statistics only -- the exact production values must not appear
        anywhere in the stored feature_results blob."""
        pid = "test_hist_norawvals"
        _fit(pid)
        needle = 918273.456
        _analyze(pid, [needle] + [1.0, 2.0, 3.0] * 19)
        conn = crud.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT feature_results FROM analysis_runs WHERE project = ?",
                    (_internal(pid),))
        row = cur.fetchone()
        conn.close()
        assert str(needle) not in row[0]

    def test_response_fields_present(self):
        pid = "test_hist_fields"
        _fit(pid)
        _analyze(pid, [1.0, 2.0, 3.0] * 20)
        run = client.get(f"/history/{pid}", headers=HEADERS).json()["runs"][0]
        for field in ("id", "ts", "baseline_version", "batch_size", "decision_mode",
                      "system_alert", "sustained_alert", "feature_metrics", "schema_report"):
            assert field in run
        assert run["batch_size"] == 60


class TestHistoryFilters:
    def test_since_until_bound_results(self):
        pid = "test_hist_sinceuntil"
        _fit(pid)
        _analyze(pid, [1.0, 2.0, 3.0] * 20)
        mid_ts = client.get(f"/history/{pid}", headers=HEADERS).json()["runs"][0]["ts"]
        time.sleep(0.01)
        _analyze(pid, [4.0, 5.0, 6.0] * 20)

        only_after = client.get(f"/history/{pid}", params={"since": mid_ts}, headers=HEADERS).json()
        assert only_after["total"] == 2  # since is inclusive of the boundary run

        only_before = client.get(f"/history/{pid}", params={"until": mid_ts}, headers=HEADERS).json()
        assert only_before["total"] == 1

    def test_alert_only_filters_to_triggered_runs(self):
        pid = "test_hist_alertonly"
        _fit(pid)
        _analyze(pid, [1.0, 2.0, 3.0] * 20)  # near-identical to reference, unlikely to alert
        resp = client.get(f"/history/{pid}", params={"alert_only": True}, headers=HEADERS)
        assert resp.status_code == 200
        for run in resp.json()["runs"]:
            assert run["system_alert"] is True

    def test_feature_filter_narrows_metrics_and_time_series(self):
        pid = "test_hist_featurefilter"
        rng = random.Random(9)
        x = [rng.uniform(0, 100) for _ in range(60)]
        cat = ["a" if i % 2 == 0 else "b" for i in range(60)]
        client.post(f"/fit/{pid}", json={"reference_data": {"x": x}, "categorical_data": {"cat": cat}},
                    headers=HEADERS)
        client.post(f"/analyze/{pid}",
                    json={"production_data": {"x": [1.0, 2.0, 3.0] * 20, "cat": ["a"] * 60}},
                    headers=HEADERS)

        resp = client.get(f"/history/{pid}", params={"feature": "x"}, headers=HEADERS)
        body = resp.json()
        assert body["runs"][0]["feature_metrics"].keys() == {"x"}
        assert set(body["feature_time_series"].keys()) == {"x"}
        assert "statistic" in body["feature_time_series"]["x"][0]

    def test_pagination_limit_and_offset(self):
        pid = "test_hist_pagination"
        _fit(pid)
        for _ in range(3):
            _analyze(pid, [1.0, 2.0, 3.0] * 20)
        page1 = client.get(f"/history/{pid}", params={"limit": 2, "offset": 0}, headers=HEADERS).json()
        page2 = client.get(f"/history/{pid}", params={"limit": 2, "offset": 2}, headers=HEADERS).json()
        assert page1["total"] == 3
        assert len(page1["runs"]) == 2
        assert len(page2["runs"]) == 1
        assert {r["id"] for r in page1["runs"]}.isdisjoint({r["id"] for r in page2["runs"]})

    def test_invalid_limit_rejected(self):
        pid = "test_hist_badlimit"
        _fit(pid)
        resp = client.get(f"/history/{pid}", params={"limit": 0}, headers=HEADERS)
        assert resp.status_code == 422


class TestHistoryIsolation:
    """Every new endpoint gets a two-user isolation test, per instruction."""

    def test_user_b_cannot_see_user_as_history(self):
        pid = "test_hist_iso"
        _fit(pid, headers=HEADERS)
        _analyze(pid, [1.0, 2.0, 3.0] * 20, headers=HEADERS)
        resp = client.get(f"/history/{pid}", headers=HEADERS_B)
        assert resp.status_code == 404

    def test_user_b_own_same_named_project_has_independent_history(self):
        pid = "test_hist_iso_shared"
        _fit(pid, headers=HEADERS)
        _analyze(pid, [1.0, 2.0, 3.0] * 20, headers=HEADERS)
        _fit(pid, headers=HEADERS_B)
        _analyze(pid, [4.0, 5.0, 6.0] * 20, headers=HEADERS_B)
        _analyze(pid, [7.0, 8.0, 9.0] * 20, headers=HEADERS_B)

        a_hist = client.get(f"/history/{pid}", headers=HEADERS).json()
        b_hist = client.get(f"/history/{pid}", headers=HEADERS_B).json()
        assert a_hist["total"] == 1
        assert b_hist["total"] == 2
