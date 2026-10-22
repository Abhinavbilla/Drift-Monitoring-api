"""
Personal access tokens managed from the dashboard (/tokens).

Run:
    python -m pytest tests/test_dashboard_tokens.py -v
"""

import itertools
import sqlite3
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

from db import crud  # noqa: E402
from main import app  # noqa: E402

crud.init_db()
client = TestClient(app)
EMAIL = "tokens-ui@dashboard-tokens-test.example"
SESSION = {"Authorization": f"Bearer {mint_session_token(EMAIL)}"}
PROJECT = "dashboard_tokens_project"


@pytest.fixture(autouse=True, scope="module")
def project():
    x = [float(v) for v in range(1, 61)]
    assert client.post(f"/fit/{PROJECT}", json={"reference_data": {"x": x}}, headers=SESSION).status_code == 200
    yield
    client.delete(f"/projects/{PROJECT}", headers=SESSION)
    conn = sqlite3.connect(crud.DB_PATH)
    conn.execute("DELETE FROM api_tokens WHERE user_email = ?", (EMAIL,))
    conn.commit()
    conn.close()


_names = itertools.count()


def _create(**body):
    return client.post("/tokens", json={"name": f"ci-{next(_names)}", **body}, headers=SESSION)


def test_created_token_works_once_shown_and_can_be_revoked():
    resp = _create(expires_days=30)
    assert resp.status_code == 200
    token, token_id = resp.json()["token"], resp.json()["id"]
    assert token.startswith("dm_")
    bearer = {"Authorization": f"Bearer {token}"}
    assert PROJECT in client.get("/projects", headers=bearer).json()["projects"]

    listed = client.get("/tokens", headers=SESSION).json()["tokens"]
    row = next(t for t in listed if t["id"] == token_id)
    assert "token" not in row and "token_hash" not in row and row["last_used_at"]

    assert client.delete(f"/tokens/{token_id}", headers=SESSION).status_code == 200
    assert client.get("/projects", headers=bearer).status_code == 401
    assert client.delete(f"/tokens/{token_id}", headers=SESSION).status_code == 404


def test_project_scoped_token_only_reaches_its_project():
    token = _create(project_ids=[PROJECT]).json()["token"]
    bearer = {"Authorization": f"Bearer {token}"}
    assert client.get(f"/baseline/{PROJECT}", headers=bearer).status_code == 200
    assert client.get("/baseline/some_other_project", headers=bearer).status_code == 403


def test_a_token_cannot_manage_tokens():
    token = _create().json()["token"]
    bearer = {"Authorization": f"Bearer {token}"}
    assert client.get("/tokens", headers=bearer).status_code == 403
    assert client.post("/tokens", json={"name": "escalate"}, headers=bearer).status_code == 403


def test_input_is_validated():
    assert _create(expires_days=7).status_code == 422
    assert client.post("/tokens", json={"name": "  "}, headers=SESSION).status_code == 422
    assert _create(project_ids=["not_my_project"]).status_code == 404
    other = {"Authorization": f"Bearer {mint_session_token('someone-else@dashboard-tokens-test.example')}"}
    mine = _create().json()["id"]
    assert client.delete(f"/tokens/{mine}", headers=other).status_code == 404


def test_active_token_names_are_unique_but_reusable_after_revoke():
    first = client.post("/tokens", json={"name": "Nightly"}, headers=SESSION)
    assert first.status_code == 200
    clash = client.post("/tokens", json={"name": "nightly "}, headers=SESSION)  # case/space-insensitive
    assert clash.status_code == 409 and "already have an active token" in clash.json()["detail"]
    assert client.delete(f"/tokens/{first.json()['id']}", headers=SESSION).status_code == 200
    assert client.post("/tokens", json={"name": "Nightly"}, headers=SESSION).status_code == 200
