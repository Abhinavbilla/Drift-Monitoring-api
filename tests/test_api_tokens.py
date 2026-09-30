"""
Tests for Step 3a: personal access tokens.

Covers: a valid scoped token works, an expired token is rejected, a
revoked token is rejected, a token used outside its project scope is
rejected (403, not 401 -- it authenticated fine, it just isn't allowed
here), and that no plaintext token is ever persisted anywhere in the DB.

Run:
    python -m pytest tests/test_api_tokens.py -v
"""

import sys
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import db.crud as crud
from auth.tokens import generate_token, hash_token
from main import app

client = TestClient(app)
SESSION_TOKEN = mint_session_token("api-token-test-owner@example.com")
SESSION_HEADERS = {"Authorization": f"Bearer {SESSION_TOKEN}"}


def _make_token(user_email, name="test-token", project_scope=None, expires_at=None, revoked=False):
    full_token, prefix, token_hash = generate_token()
    token_id = f"test-{prefix}"
    crud.create_api_token(
        token_id=token_id, user_email=user_email, name=name, prefix=prefix,
        token_hash=token_hash, project_scope=project_scope,
        created_at=datetime.now(timezone.utc).isoformat(), expires_at=expires_at,
    )
    if revoked:
        crud.revoke_api_token(token_id, user_email)
    return full_token, token_id


@pytest.fixture(autouse=True)
def cleanup():
    yield
    conn = crud.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM api_tokens WHERE user_email LIKE 'test-pat-%@example.com'")
    cur.execute("DELETE FROM baselines WHERE project_id LIKE 'test_pat_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE 'test_pat_%'")
    conn.commit()
    conn.close()


class TestValidToken:
    def test_valid_scoped_token_can_fit_its_project(self):
        token, _ = _make_token("test-pat-valid@example.com", project_scope=["test_pat_valid_proj"])
        resp = client.post(
            "/fit/test_pat_valid_proj",
            json={"reference_data": {"x": [1.1, 2.2, 3.3, 4.4, 5.5] * 20}},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_wildcard_scope_can_fit_any_project(self):
        token, _ = _make_token("test-pat-wild@example.com", project_scope=["*"])
        resp = client.post(
            "/fit/test_pat_wild_anything",
            json={"reference_data": {"x": [1.1, 2.2, 3.3, 4.4, 5.5] * 20}},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_last_used_is_updated_on_use(self):
        token, token_id = _make_token("test-pat-lastused@example.com", project_scope=["*"])
        before = crud.get_api_token_by_prefix(token_id.replace("test-", ""))
        assert before["last_used_at"] is None
        client.get("/projects", headers={"Authorization": f"Bearer {token}"})
        after = crud.get_api_token_by_prefix(token_id.replace("test-", ""))
        assert after["last_used_at"] is not None


class TestExpiredToken:
    def test_expired_token_rejected(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        token, _ = _make_token("test-pat-expired@example.com", project_scope=["*"], expires_at=past)
        resp = client.get("/projects", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401
        assert "expired" in resp.json()["detail"].lower()

    def test_not_yet_expired_token_accepted(self):
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        token, _ = _make_token("test-pat-notexpired@example.com", project_scope=["*"], expires_at=future)
        resp = client.get("/projects", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200


class TestRevokedToken:
    def test_revoked_token_rejected(self):
        token, _ = _make_token("test-pat-revoked@example.com", project_scope=["*"], revoked=True)
        resp = client.get("/projects", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401
        assert "revoked" in resp.json()["detail"].lower()

    def test_revoke_is_scoped_to_owning_user(self):
        _, token_id = _make_token("test-pat-realowner@example.com", project_scope=["*"])
        ok = crud.revoke_api_token(token_id, "test-pat-attacker@example.com")
        assert ok is False
        row = crud.get_api_token_by_prefix(token_id.replace("test-", ""))
        assert row["revoked"] is False


class TestWrongScope:
    def test_out_of_scope_project_rejected_with_403(self):
        token, _ = _make_token("test-pat-scoped@example.com", project_scope=["test_pat_allowed_proj"])
        resp = client.post(
            "/fit/test_pat_forbidden_proj",
            json={"reference_data": {"x": [1.1, 2.2, 3.3, 4.4, 5.5] * 20}},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403
        assert "not scoped" in resp.json()["detail"].lower()

    def test_empty_scope_list_rejects_everything(self):
        token, _ = _make_token("test-pat-noscope@example.com", project_scope=[])
        resp = client.post(
            "/fit/test_pat_anything",
            json={"reference_data": {"x": [1.1, 2.2, 3.3, 4.4, 5.5] * 20}},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403


class TestInvalidToken:
    def test_bad_token_401(self):
        resp = client.get("/projects", headers={"Authorization": "Bearer dm_deadbeef_notarealtoken12345"})
        assert resp.status_code == 401

    def test_wrong_secret_for_real_prefix_401(self):
        """A presented token with a REAL prefix but wrong secret must not
        authenticate -- confirms the hash comparison, not just the prefix
        lookup, gates access."""
        _, token_id = _make_token("test-pat-hashcheck@example.com", project_scope=["*"])
        real_prefix = token_id.replace("test-", "")
        forged = f"dm_{real_prefix}_wrongsecretwrongsecretwrongsecretwrong"
        resp = client.get("/projects", headers={"Authorization": f"Bearer {forged}"})
        assert resp.status_code == 401


class TestNoPlaintextPersisted:
    def test_only_hash_is_stored_not_plaintext(self):
        token, token_id = _make_token("test-pat-hashonly@example.com", project_scope=["*"])
        row = crud.get_api_token_by_prefix(token_id.replace("test-", ""))
        assert row["token_hash"] == hash_token(token)
        assert row["token_hash"] != token
        assert token not in str(row)  # the row itself never contains the plaintext anywhere

    def test_db_file_never_contains_the_plaintext_token(self):
        """Greps the raw sqlite file bytes for the token substring -- the
        strongest possible check that no code path wrote it anywhere."""
        token, _ = _make_token("test-pat-rawscan@example.com", project_scope=["*"])
        secret_part = token.split("_", 2)[2]  # the high-entropy part, safe to grep for
        with open(crud.DB_PATH, "rb") as f:
            raw = f.read()
        assert secret_part.encode() not in raw
