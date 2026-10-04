"""
React frontend support, 2026-10-04: POST /auth/google verifies a Google
ID token and mints the same session-token shape verify_access expects.

Run:
    python -m pytest tests/test_google_auth.py -v
"""

import jwt
import pytest
from fastapi.testclient import TestClient

import main
from main import app

client = TestClient(app)


class TestGoogleLogin:
    def test_valid_token_mints_session_token(self, monkeypatch):
        def fake_verify(id_token, request, client_id):
            assert id_token == "fake-google-id-token"
            return {"email": "newuser@example.com", "email_verified": True, "name": "New User"}

        import google.oauth2.id_token as real_id_token_module
        monkeypatch.setattr(real_id_token_module, "verify_oauth2_token", fake_verify)

        resp = client.post("/auth/google", json={"id_token": "fake-google-id-token"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "newuser@example.com"
        assert body["name"] == "New User"

        decoded = jwt.decode(body["session_token"], main.COOKIE_KEY, algorithms=["HS256"])
        assert decoded["email"] == "newuser@example.com"

        # the minted token actually works against a real protected endpoint
        resp2 = client.get("/projects", headers={"Authorization": f"Bearer {body['session_token']}"})
        assert resp2.status_code == 200

    def test_invalid_token_rejected(self, monkeypatch):
        import google.oauth2.id_token as real_id_token_module

        def fake_verify(id_token, request, client_id):
            raise ValueError("Token used too early")

        monkeypatch.setattr(real_id_token_module, "verify_oauth2_token", fake_verify)
        resp = client.post("/auth/google", json={"id_token": "garbage"})
        assert resp.status_code == 401

    def test_unverified_email_rejected(self, monkeypatch):
        import google.oauth2.id_token as real_id_token_module

        def fake_verify(id_token, request, client_id):
            return {"email": "someone@example.com", "email_verified": False, "name": "Someone"}

        monkeypatch.setattr(real_id_token_module, "verify_oauth2_token", fake_verify)
        resp = client.post("/auth/google", json={"id_token": "fake"})
        assert resp.status_code == 401
