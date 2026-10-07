"""
Email + password sign-in (auth/email_auth.py): verification before any
session, codes, lockout, reset, enumeration-safe responses, and sharing a
project identity with Google sign-in for the same address.

Run:
    python -m pytest tests/test_email_auth.py -v
"""

import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

from auth import email_auth  # noqa: E402
from db import crud  # noqa: E402
from main import app  # noqa: E402

crud.init_db()
client = TestClient(app)
DOMAIN = "@email-auth-test.example"
PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def outbox(monkeypatch):
    sent = []
    monkeypatch.setattr(email_auth, "send_email", lambda to, subject, body: sent.append((to, subject, body)))
    yield sent


@pytest.fixture(autouse=True, scope="module")
def cleanup():
    yield
    conn = sqlite3.connect(crud.DB_PATH)
    for table in ("users", "email_codes"):
        conn.execute(f"DELETE FROM {table} WHERE email LIKE ?", (f"%{DOMAIN}",))
    conn.commit()
    conn.close()


def _code(outbox, to):
    body = next(b for t, _, b in reversed(outbox) if t == to)
    return re.search(r"\b(\d{6})\b", body).group(1)


def _register(email, outbox, password=PASSWORD):
    assert client.post("/auth/email/register", json={"email": email, "password": password, "name": "Ada"}).status_code == 200
    return _code(outbox, email.strip().lower())  # mail goes to the normalized address


def _signed_up(email, outbox):
    code = _register(email, outbox)
    resp = client.post("/auth/email/verify", json={"email": email, "code": code})
    assert resp.status_code == 200
    return resp.json()["session_token"]


def test_sign_up_requires_the_emailed_code_and_then_works(outbox):
    email = f"Ada.Sign{DOMAIN}"
    code = _register(email.upper(), outbox)  # case-insensitive
    assert client.post("/auth/email/login", json={"email": email, "password": PASSWORD}).status_code == 403
    assert client.post("/auth/email/verify", json={"email": email, "code": "000000" if code != "000000" else "111111"}).status_code == 400
    resp = client.post("/auth/email/verify", json={"email": email, "code": code})
    assert resp.status_code == 200 and resp.json()["email"] == email.lower() and resp.json()["name"] == "Ada"
    token = resp.json()["session_token"]
    assert client.get("/projects", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert client.post("/auth/email/verify", json={"email": email, "code": code}).status_code == 400  # single use
    assert client.post("/auth/email/login", json={"email": email, "password": PASSWORD}).status_code == 200


def test_wrong_password_and_unknown_email_get_the_same_answer(outbox):
    email = f"same{DOMAIN}"
    _signed_up(email, outbox)
    wrong = client.post("/auth/email/login", json={"email": email, "password": "not the password"})
    unknown = client.post("/auth/email/login", json={"email": f"nobody{DOMAIN}", "password": "whatever123"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"]


def test_lockout_after_repeated_wrong_passwords(outbox):
    email = f"lock{DOMAIN}"
    _signed_up(email, outbox)
    for _ in range(email_auth.MAX_FAILED_LOGINS):
        client.post("/auth/email/login", json={"email": email, "password": "wrong-password"})
    assert client.post("/auth/email/login", json={"email": email, "password": PASSWORD}).status_code == 429


def test_code_attempt_limit_and_expiry(outbox):
    email = f"codes{DOMAIN}"
    _register(email, outbox)
    for _ in range(email_auth.MAX_CODE_ATTEMPTS):
        client.post("/auth/email/verify", json={"email": email, "code": "999999"})
    assert client.post("/auth/email/verify", json={"email": email, "code": "999999"}).status_code == 429

    email2 = f"expired{DOMAIN}"
    code = _register(email2, outbox)
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    conn = sqlite3.connect(crud.DB_PATH)
    conn.execute("UPDATE email_codes SET expires_at = ? WHERE email = ?", (past, email2))
    conn.commit()
    conn.close()
    assert client.post("/auth/email/verify", json={"email": email2, "code": code}).status_code == 400


def test_registering_a_taken_address_changes_nothing(outbox):
    email = f"taken{DOMAIN}"
    _signed_up(email, outbox)
    resp = client.post("/auth/email/register", json={"email": email, "password": "attacker-password"})
    assert resp.status_code == 200  # same answer as a fresh sign-up
    assert "already have an account" in outbox[-1][2]
    assert client.post("/auth/email/login", json={"email": email, "password": "attacker-password"}).status_code == 401
    assert client.post("/auth/email/login", json={"email": email, "password": PASSWORD}).status_code == 200


def test_forgot_and_reset_password(outbox):
    email = f"reset{DOMAIN}"
    _signed_up(email, outbox)
    before = len(outbox)
    assert client.post("/auth/email/forgot", json={"email": f"ghost{DOMAIN}"}).status_code == 200
    assert len(outbox) == before  # nothing sent, same answer
    assert client.post("/auth/email/forgot", json={"email": email}).status_code == 200
    code = _code(outbox, email)
    assert client.post("/auth/email/reset", json={"email": email, "code": code, "new_password": "short"}).status_code == 422
    resp = client.post("/auth/email/reset", json={"email": email, "code": code, "new_password": "a brand new password"})
    assert resp.status_code == 200
    assert client.post("/auth/email/login", json={"email": email, "password": PASSWORD}).status_code == 401
    assert client.post("/auth/email/login", json={"email": email, "password": "a brand new password"}).status_code == 200


def test_resend_is_quiet_during_cooldown(outbox):
    email = f"resend{DOMAIN}"
    _register(email, outbox)
    assert client.post("/auth/email/resend", json={"email": email}).status_code == 200  # within 60 s: no error, no leak


def test_passwords_are_hashed_and_inputs_validated(outbox):
    email = f"hash{DOMAIN}"
    _register(email, outbox)
    stored = crud.get_user(email)["password_hash"]
    assert stored.startswith("scrypt$") and PASSWORD not in stored
    assert email_auth.verify_password(PASSWORD, stored) and not email_auth.verify_password("nope", stored)
    assert client.post("/auth/email/register", json={"email": "not-an-email", "password": PASSWORD}).status_code == 422
    assert client.post("/auth/email/register", json={"email": f"weak{DOMAIN}", "password": "123"}).status_code == 422


def test_without_smtp_email_sign_in_is_refused(monkeypatch):
    monkeypatch.undo()  # restore the real send_email
    monkeypatch.delenv("ALERT_EMAIL", raising=False)
    monkeypatch.delenv("ALERT_PASSWORD", raising=False)
    monkeypatch.delenv("DRIFT_DEV_PRINT_EMAIL_CODES", raising=False)
    assert client.get("/auth/methods").json()["email"] is False
    resp = client.post("/auth/email/register", json={"email": f"nosmtp{DOMAIN}", "password": PASSWORD})
    assert resp.status_code == 503 and "Google" in resp.json()["detail"]


def test_same_address_sees_the_same_projects_as_google(outbox):
    email = f"shared{DOMAIN}"
    google = {"Authorization": f"Bearer {mint_session_token(email)}"}
    project = "email_auth_shared_project"
    x = [float(v) for v in range(1, 61)][::-1][:30] + [float(v) for v in range(1, 31)]
    assert client.post(f"/fit/{project}", json={"reference_data": {"x": x}}, headers=google).status_code == 200
    token = _signed_up(email, outbox)
    projects = client.get("/projects", headers={"Authorization": f"Bearer {token}"}).json()["projects"]
    assert project in projects
    client.delete(f"/projects/{project}", headers=google)
