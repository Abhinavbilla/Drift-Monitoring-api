"""
Step 3c: DELETE /projects/{project_id}, and DELETE /models/{model_id} kept
as a deprecated alias with identical behavior.

Run:
    python -m pytest tests/test_delete_project.py -v
"""

import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import db.crud as crud
from main import app

client = TestClient(app)
TOKEN = mint_session_token("delete-project-test@example.com")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def cleanup():
    yield
    conn = crud.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM baselines WHERE project_id LIKE '%test_del_proj_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE '%test_del_proj_%'")
    conn.commit()
    conn.close()


def _fit(project_id):
    return client.post(f"/fit/{project_id}", json={"reference_data": {"x": [1.1, 2.2, 3.3, 4.4, 5.5] * 20}},
                        headers=HEADERS)


class TestDeleteProjects:
    def test_delete_removes_baseline_and_project(self):
        _fit("test_del_proj_a")
        assert client.get("/baseline/test_del_proj_a", headers=HEADERS).status_code == 200

        resp = client.delete("/projects/test_del_proj_a", headers=HEADERS)
        assert resp.status_code == 200
        assert client.get("/baseline/test_del_proj_a", headers=HEADERS).status_code == 404

    def test_deprecated_models_alias_still_works_identically(self):
        _fit("test_del_proj_b")
        resp = client.delete("/models/test_del_proj_b", headers=HEADERS)
        assert resp.status_code == 200
        assert client.get("/baseline/test_del_proj_b", headers=HEADERS).status_code == 404

    def test_recreated_project_starts_empty(self):
        # A model re-created under a deleted model's name used to inherit its
        # history, baseline versions, alert state and still-firing webhooks.
        _fit("test_del_proj_c")
        client.post("/analyze/test_del_proj_c", json={"production_data": {"x": [9.9, 8.8, 7.7] * 20}}, headers=HEADERS)
        assert client.get("/history/test_del_proj_c", headers=HEADERS).json()["total"] == 1
        client.delete("/projects/test_del_proj_c", headers=HEADERS)

        _fit("test_del_proj_c")
        assert client.get("/history/test_del_proj_c", headers=HEADERS).json()["total"] == 0
        versions = client.get("/baselines/test_del_proj_c", headers=HEADERS).json()
        assert [v["version"] for v in versions["versions"]] == [1]
        client.delete("/projects/test_del_proj_c", headers=HEADERS)
