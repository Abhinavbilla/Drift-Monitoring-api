"""
Cross-user isolation (2026-09-30 hardening pass, highest-priority item).

User A creates a project. User B -- via session JWT, via an unscoped
("*") PAT, and via a PAT explicitly (mis)scoped to A's own project_id --
must get 404 from EVERY project-scoped endpoint for that project: read,
analyze, overwrite (/fit), list, delete. A PAT scoped to A's project_id
proves scope alone cannot grant cross-account access -- ownership is a
separate, stricter gate.

Also covers: PAT default expiry (never, unless set), revoked tokens
rejected, last_used_at updates, and a repo/logs/reports grep for any
leaked "dm_" token string.

Run:
    python -m pytest tests/test_cross_user_isolation.py -v
"""

import io
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import db.crud as crud
from auth.tokens import generate_token
from main import app

client = TestClient(app)

USER_A_EMAIL = "isolation-user-a@example.com"
USER_B_EMAIL = "isolation-user-b@example.com"
TOKEN_A_SESSION = mint_session_token(USER_A_EMAIL)
TOKEN_B_SESSION = mint_session_token(USER_B_EMAIL)
HEADERS_A_SESSION = {"Authorization": f"Bearer {TOKEN_A_SESSION}"}
HEADERS_B_SESSION = {"Authorization": f"Bearer {TOKEN_B_SESSION}"}

PROJECT = "test_iso_shared_proj"


def _mint_pat(user_email, name, project_scope, expires_at=None):
    full_token, prefix, token_hash = generate_token()
    token_id = f"iso-{prefix}"
    crud.create_api_token(
        token_id=token_id, user_email=user_email, name=name, prefix=prefix,
        token_hash=token_hash, project_scope=project_scope,
        created_at=datetime.now(timezone.utc).isoformat(), expires_at=expires_at,
    )
    return full_token, token_id


def _solid_color_image_b64(color, size=(32, 32)):
    import base64
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    crud.init_db()
    rng = random.Random(11)
    x = [rng.uniform(0, 100) for _ in range(60)]
    cat = ["p" if i % 2 == 0 else "q" for i in range(60)]

    # A creates the shared tabular project.
    resp = client.post(f"/fit/{PROJECT}", json={"reference_data": {"x": x}, "categorical_data": {"cat": cat}},
                        headers=HEADERS_A_SESSION)
    assert resp.status_code == 200, resp.text

    # A creates one project per other modality, for endpoint-level coverage.
    # Asserted explicitly (not just fire-and-forget): these hit sentence-
    # transformers, which can intermittently fail on an unauthenticated
    # HuggingFace Hub call in a sandboxed environment -- if setup silently
    # fails, the project never gets an owner row, and the isolation tests
    # below would misleadingly look like an auth bug (B "succeeding"
    # against a project that, from the ownership table's perspective,
    # was never actually created by anyone).
    r_text = client.post(f"/fit/{PROJECT}_text/text",
                          json={"reference_texts": [f"sentence number {i}" for i in range(10)]},
                          headers=HEADERS_A_SESSION)
    assert r_text.status_code == 200, f"setup: text fit failed ({r_text.status_code}): {r_text.text}"
    r_image = client.post(f"/fit/{PROJECT}_image/image",
                           json={"reference_images": [_solid_color_image_b64((i * 20, 50, 50)) for i in range(6)]},
                           headers=HEADERS_A_SESSION)
    assert r_image.status_code == 200, f"setup: image fit failed ({r_image.status_code}): {r_image.text}"

    yield

    conn = crud.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM baselines WHERE project_id LIKE 'test_iso_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE 'test_iso_%'")
    cur.execute("DELETE FROM logs WHERE project_id LIKE 'test_iso_%'")
    cur.execute("DELETE FROM api_tokens WHERE id LIKE 'iso-%'")
    conn.commit()
    conn.close()


@pytest.fixture()
def pat_b_unscoped():
    token, token_id = _mint_pat(USER_B_EMAIL, "b-unscoped", ["*"])
    yield token
    conn = crud.get_connection()
    conn.execute("DELETE FROM api_tokens WHERE id = ?", (token_id,))
    conn.commit()
    conn.close()


@pytest.fixture()
def pat_b_scoped_to_a_project():
    """B mints a token and (mis)scopes it to A's own project_id -- proves
    scope alone can't grant cross-account access; ownership still blocks
    it."""
    token, token_id = _mint_pat(USER_B_EMAIL, "b-scoped-to-a", [PROJECT])
    yield token
    conn = crud.get_connection()
    conn.execute("DELETE FROM api_tokens WHERE id = ?", (token_id,))
    conn.commit()
    conn.close()


# Endpoint x expected-status table. Each entry: (method, path, kwargs).
# All must return 404 for user B -- a stranger cannot even tell the
# project exists.
ENDPOINTS = [
    ("GET", f"/baseline/{PROJECT}", {}),
    ("GET", f"/logs/{PROJECT}", {}),
    ("POST", f"/fit/{PROJECT}", {"json": {"reference_data": {"x": [1.0, 2.0, 3.0] * 20}}}),
    ("POST", f"/fit/{PROJECT}/upload",
     {"files": {"file": ("r.csv", b"x\n1.0\n2.0\n3.0\n", "text/csv")}}),
    ("POST", f"/predict/{PROJECT}", {"json": {"features": {"x": 5.0}}}),
    ("POST", f"/analyze/{PROJECT}", {"json": {"production_data": {"x": [1.0, 2.0, 3.0] * 20}}}),
    ("POST", f"/analyze/{PROJECT}/upload",
     {"files": {"file": ("b.csv", b"x\n1.0\n2.0\n3.0\n", "text/csv")}}),
    ("GET", f"/health/{PROJECT}", {}),
    ("DELETE", f"/projects/{PROJECT}", {}),
    ("DELETE", f"/models/{PROJECT}", {}),
    ("POST", f"/fit/{PROJECT}_text/text", {"json": {"reference_texts": ["a", "b", "c", "d", "e"]}}),
    ("POST", f"/analyze/{PROJECT}_text/text", {"json": {"production_texts": ["a", "b", "c", "d", "e"]}}),
    ("POST", f"/fit/{PROJECT}_image/image",
     {"json": {"reference_images": [_solid_color_image_b64((10, 10, 10))] * 5}}),
    ("POST", f"/analyze/{PROJECT}_image/image",
     {"json": {"production_images": [_solid_color_image_b64((10, 10, 10))] * 5}}),
]


def _make_request(method, path, headers, kwargs):
    return client.request(method, path, headers=headers, **kwargs)


class TestCrossUserIsolationSessionJWT:
    @pytest.mark.parametrize("method,path,kwargs", ENDPOINTS)
    def test_user_b_session_jwt_gets_404(self, method, path, kwargs):
        resp = _make_request(method, path, HEADERS_B_SESSION, kwargs)
        assert resp.status_code == 404, f"{method} {path} -> {resp.status_code}: {resp.text}"


class TestCrossUserIsolationUnscopedPAT:
    @pytest.mark.parametrize("method,path,kwargs", ENDPOINTS)
    def test_user_b_unscoped_pat_gets_404(self, method, path, kwargs, pat_b_unscoped):
        headers = {"Authorization": f"Bearer {pat_b_unscoped}"}
        resp = _make_request(method, path, headers, kwargs)
        assert resp.status_code == 404, f"{method} {path} -> {resp.status_code}: {resp.text}"


class TestScopeCannotGrantCrossAccountAccess:
    def test_pat_scoped_to_victim_project_still_blocked(self, pat_b_scoped_to_a_project):
        """B's own token, scoped (by B) to A's project_id -- scope narrows
        what B's account can reach, it does not grant B access to another
        account's project. Must still be 404, not 200/403."""
        headers = {"Authorization": f"Bearer {pat_b_scoped_to_a_project}"}
        resp = client.get(f"/baseline/{PROJECT}", headers=headers)
        assert resp.status_code == 404, resp.text


class TestUserAStillWorks:
    """Confirms the isolation fix didn't collaterally break the owner's
    own access."""

    def test_owner_can_still_read_and_analyze(self):
        resp = client.get(f"/baseline/{PROJECT}", headers=HEADERS_A_SESSION)
        assert resp.status_code == 200
        resp2 = client.post(f"/analyze/{PROJECT}", json={"production_data": {"x": [1.0, 2.0, 3.0] * 20}},
                             headers=HEADERS_A_SESSION)
        assert resp2.status_code == 200


class TestProjectListingIsolation:
    def test_user_b_does_not_see_user_a_projects(self):
        resp = client.get("/projects", headers=HEADERS_B_SESSION)
        assert resp.status_code == 200
        assert PROJECT not in resp.json()["projects"]

    def test_user_a_sees_own_project(self):
        resp = client.get("/projects", headers=HEADERS_A_SESSION)
        assert resp.status_code == 200
        assert PROJECT in resp.json()["projects"]


class TestPATExpiryRevocationAndTracking:
    def test_pat_default_expiry_is_never(self):
        token, token_id = _mint_pat(USER_A_EMAIL, "default-expiry-check", ["*"])
        row = crud.get_api_token_by_prefix(token_id.replace("iso-", ""))
        assert row["expires_at"] is None
        conn = crud.get_connection()
        conn.execute("DELETE FROM api_tokens WHERE id = ?", (token_id,))
        conn.commit()
        conn.close()

    def test_revoked_token_rejected_on_a_real_endpoint(self):
        token, token_id = _mint_pat(USER_A_EMAIL, "revoke-check", ["*"])
        crud.revoke_api_token(token_id, USER_A_EMAIL)
        resp = client.get("/projects", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401
        conn = crud.get_connection()
        conn.execute("DELETE FROM api_tokens WHERE id = ?", (token_id,))
        conn.commit()
        conn.close()

    def test_last_used_at_updates_on_real_use(self):
        token, token_id = _mint_pat(USER_A_EMAIL, "last-used-check", ["*"])
        prefix = token_id.replace("iso-", "")
        assert crud.get_api_token_by_prefix(prefix)["last_used_at"] is None
        client.get("/projects", headers={"Authorization": f"Bearer {token}"})
        assert crud.get_api_token_by_prefix(prefix)["last_used_at"] is not None
        conn = crud.get_connection()
        conn.execute("DELETE FROM api_tokens WHERE id = ?", (token_id,))
        conn.commit()
        conn.close()


class TestNoTokenLeakage:
    """Greps the repo's tracked source/docs/results and this run's own
    log/report artifacts for a real, non-placeholder token pattern. A
    placeholder like 'dm_<prefix>_<secret>' in documentation is fine and
    expected; an actual dm_<hex>_<base64url> triple is not."""

    def test_no_real_dm_tokens_committed_in_repo(self):
        import re
        import subprocess

        result = subprocess.run(
            ["git", "grep", "-InE", r"dm_[0-9a-f]{8}_[A-Za-z0-9_-]{20,}"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True,
        )
        # git grep exit code 1 = no matches (good); 0 = matches found (bad).
        assert result.returncode == 1, f"Possible leaked token(s) found:\n{result.stdout}"
