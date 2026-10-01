"""
Step 5 Part 1 item 5: baseline versioning -- /fit creates version n+1 (old
versions kept), GET /baselines/{project_id}, POST /baselines/{project_id}
/activate, /analyze's optional baseline_version, history records the
version used.

Run:
    python -m pytest tests/test_baseline_versioning.py -v
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
EMAIL = "baseline-version-test@example.com"
EMAIL_B = "baseline-version-test-b@example.com"
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
    cur.execute("DELETE FROM baselines WHERE project_id LIKE '%test_bver_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE '%test_bver_%'")
    cur.execute("DELETE FROM analysis_runs WHERE project LIKE '%test_bver_%'")
    cur.execute("DELETE FROM baseline_versions WHERE project_id LIKE '%test_bver_%'")
    cur.execute("DELETE FROM baseline_active_version WHERE project_id LIKE '%test_bver_%'")
    conn.commit()
    conn.close()


def _x(seed, lo=0, hi=100, n=60):
    rng = random.Random(seed)
    return [rng.uniform(lo, hi) for _ in range(n)]


def _fit(project_id, x_values, headers=HEADERS, model_version_label=None):
    payload = {"reference_data": {"x": x_values}}
    if model_version_label is not None:
        payload["model_version_label"] = model_version_label
    return client.post(f"/fit/{project_id}", json=payload, headers=headers)


class TestFitCreatesVersions:
    def test_first_fit_is_version_one(self):
        pid = "test_bver_first"
        resp = _fit(pid, _x(1))
        assert resp.json()["version"] == 1

    def test_refit_creates_version_two_old_kept(self):
        pid = "test_bver_refit"
        _fit(pid, _x(1))
        resp = _fit(pid, _x(2))
        assert resp.json()["version"] == 2

        versions = client.get(f"/baselines/{pid}", headers=HEADERS).json()["versions"]
        assert {v["version"] for v in versions} == {1, 2}

    def test_new_fit_is_active_by_default(self):
        pid = "test_bver_newactive"
        _fit(pid, _x(1))
        _fit(pid, _x(2))
        versions = client.get(f"/baselines/{pid}", headers=HEADERS).json()["versions"]
        active = [v for v in versions if v["active"]]
        assert len(active) == 1
        assert active[0]["version"] == 2

    def test_model_version_label_stored(self):
        pid = "test_bver_label"
        _fit(pid, _x(1), model_version_label="v1-initial")
        versions = client.get(f"/baselines/{pid}", headers=HEADERS).json()["versions"]
        assert versions[0]["model_version_label"] == "v1-initial"

    def test_label_is_optional(self):
        pid = "test_bver_nolabel"
        _fit(pid, _x(1))
        versions = client.get(f"/baselines/{pid}", headers=HEADERS).json()["versions"]
        assert versions[0]["model_version_label"] is None


class TestListBaselinesEndpoint:
    def test_backfills_version_one_for_pre_existing_project(self):
        """A baseline inserted directly (simulating a pre-item-5 project)
        has no baseline_versions rows until first touched here."""
        pid = "test_bver_legacy"
        internal = _internal(pid)
        crud.create_project(internal, "legacy", EMAIL)
        crud.insert_baseline(internal, {"x": "continuous"}, {"x": _x(3)})
        versions = client.get(f"/baselines/{pid}", headers=HEADERS).json()["versions"]
        assert len(versions) == 1
        assert versions[0]["version"] == 1
        assert versions[0]["active"] is True

    def test_nonexistent_project_404s(self):
        resp = client.get("/baselines/test_bver_never_existed", headers=HEADERS)
        assert resp.status_code == 404


class TestActivateEndpoint:
    def test_activate_old_version_makes_it_active(self):
        pid = "test_bver_activate"
        _fit(pid, _x(1, lo=0, hi=10))     # version 1: small values
        _fit(pid, _x(2, lo=1000, hi=1010))  # version 2: large values

        resp = client.post(f"/baselines/{pid}/activate", json={"version": 1}, headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["active_version"] == 1

        state = crud.get_baseline(_internal(pid))
        assert max(state["reference_data"]["x"]) < 100  # back to version 1's small values

        versions = client.get(f"/baselines/{pid}", headers=HEADERS).json()["versions"]
        active = [v for v in versions if v["active"]]
        assert active[0]["version"] == 1

    def test_activate_nonexistent_version_404s(self):
        pid = "test_bver_activate_bad"
        _fit(pid, _x(1))
        resp = client.post(f"/baselines/{pid}/activate", json={"version": 99}, headers=HEADERS)
        assert resp.status_code == 404

    def test_activated_version_used_by_analyze_without_explicit_version(self):
        pid = "test_bver_activate_analyze"
        _fit(pid, _x(1, lo=0, hi=10))      # version 1
        _fit(pid, _x(2, lo=1000, hi=1010))  # version 2 (active)
        client.post(f"/baselines/{pid}/activate", json={"version": 1}, headers=HEADERS)

        # production data close to version 1's range should look like NOT drifting
        resp = client.post(f"/analyze/{pid}", json={"production_data": {"x": _x(9, lo=0, hi=10)}},
                            headers=HEADERS)
        assert resp.status_code == 200


class TestAnalyzeExplicitBaselineVersion:
    def test_explicit_baseline_version_overrides_active(self):
        pid = "test_bver_explicit"
        _fit(pid, _x(1, lo=0, hi=10))       # version 1
        _fit(pid, _x(2, lo=1000, hi=1010))  # version 2 (active)

        # Explicitly analyze against version 1, even though 2 is active.
        resp = client.post(f"/analyze/{pid}", params={"baseline_version": 1},
                            json={"production_data": {"x": _x(9, lo=0, hi=10)}}, headers=HEADERS)
        assert resp.status_code == 200

        hist = client.get(f"/history/{pid}", headers=HEADERS).json()
        assert hist["runs"][0]["baseline_version"] == 1

    def test_analyze_without_version_records_active_version_in_history(self):
        pid = "test_bver_historyversion"
        _fit(pid, _x(1))
        _fit(pid, _x(2))  # active is now version 2
        client.post(f"/analyze/{pid}", json={"production_data": {"x": _x(9)}}, headers=HEADERS)
        hist = client.get(f"/history/{pid}", headers=HEADERS).json()
        assert hist["runs"][0]["baseline_version"] == 2

    def test_nonexistent_baseline_version_404s(self):
        pid = "test_bver_badversion"
        _fit(pid, _x(1))
        resp = client.post(f"/analyze/{pid}", params={"baseline_version": 99},
                            json={"production_data": {"x": _x(9)}}, headers=HEADERS)
        assert resp.status_code == 404


class TestBaselineVersioningIsolation:
    def test_user_b_cannot_list_or_activate_user_as_versions(self):
        pid = "test_bver_iso"
        _fit(pid, _x(1), headers=HEADERS)
        resp_list = client.get(f"/baselines/{pid}", headers=HEADERS_B)
        assert resp_list.status_code == 404
        resp_activate = client.post(f"/baselines/{pid}/activate", json={"version": 1}, headers=HEADERS_B)
        assert resp_activate.status_code == 404

    def test_user_b_own_same_named_project_has_independent_versions(self):
        pid = "test_bver_iso_shared"
        _fit(pid, _x(1), headers=HEADERS)
        _fit(pid, _x(2), headers=HEADERS)   # A: 2 versions
        _fit(pid, _x(3), headers=HEADERS_B)  # B: 1 version

        a_versions = client.get(f"/baselines/{pid}", headers=HEADERS).json()["versions"]
        b_versions = client.get(f"/baselines/{pid}", headers=HEADERS_B).json()["versions"]
        assert len(a_versions) == 2
        assert len(b_versions) == 1
