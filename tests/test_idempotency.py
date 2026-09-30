"""
Step 5 Part 1 item 3: optional Idempotency-Key header on both analyze
endpoints (JSON and upload).

Run:
    python -m pytest tests/test_idempotency.py -v
"""

import random
import sys
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import db.crud as crud
from main import app, _internal_project_key

client = TestClient(app)
EMAIL = "idempotency-test@example.com"
EMAIL_B = "idempotency-test-b@example.com"
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
    cur.execute("DELETE FROM baselines WHERE project_id LIKE '%test_idem_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE '%test_idem_%'")
    cur.execute("DELETE FROM analysis_runs WHERE project LIKE '%test_idem_%'")
    conn.commit()
    conn.close()


def _fit(project_id, headers=HEADERS):
    rng = random.Random(5)
    x = [rng.uniform(0, 100) for _ in range(60)]
    return client.post(f"/fit/{project_id}", json={"reference_data": {"x": x}}, headers=headers)


def _analyze(project_id, values, key=None, headers=HEADERS):
    h = dict(headers)
    if key is not None:
        h["Idempotency-Key"] = key
    return client.post(f"/analyze/{project_id}", json={"production_data": {"x": values}}, headers=h)


class TestIdempotencyReplay:
    def test_same_key_same_payload_returns_stored_result_no_new_row(self):
        pid = "test_idem_replay"
        _fit(pid)
        r1 = _analyze(pid, [1.0, 2.0, 3.0] * 20, key="k1")
        r2 = _analyze(pid, [1.0, 2.0, 3.0] * 20, key="k1")
        assert r1.status_code == r2.status_code == 200
        assert r1.json() == r2.json()

        hist = client.get(f"/history/{pid}", headers=HEADERS).json()
        assert hist["total"] == 1  # only one row, despite two calls

    def test_same_key_different_payload_is_409(self):
        pid = "test_idem_conflict"
        _fit(pid)
        r1 = _analyze(pid, [1.0, 2.0, 3.0] * 20, key="k2")
        assert r1.status_code == 200
        r2 = _analyze(pid, [4.0, 5.0, 6.0] * 20, key="k2")
        assert r2.status_code == 409

        hist = client.get(f"/history/{pid}", headers=HEADERS).json()
        assert hist["total"] == 1  # the conflicting call created no row

    def test_no_key_always_creates_a_new_row(self):
        pid = "test_idem_nokey"
        _fit(pid)
        _analyze(pid, [1.0, 2.0, 3.0] * 20)
        _analyze(pid, [1.0, 2.0, 3.0] * 20)
        hist = client.get(f"/history/{pid}", headers=HEADERS).json()
        assert hist["total"] == 2

    def test_different_keys_both_create_rows(self):
        pid = "test_idem_differentkeys"
        _fit(pid)
        _analyze(pid, [1.0, 2.0, 3.0] * 20, key="a")
        _analyze(pid, [1.0, 2.0, 3.0] * 20, key="b")
        hist = client.get(f"/history/{pid}", headers=HEADERS).json()
        assert hist["total"] == 2

    def test_upload_endpoint_also_honors_idempotency_key(self):
        pid = "test_idem_upload"
        _fit(pid)
        batch_csv = ("x\n" + "\n".join(str(v) for v in [1.0, 2.0, 3.0] * 20) + "\n").encode()
        r1 = client.post(f"/analyze/{pid}/upload", files={"file": ("b.csv", batch_csv, "text/csv")},
                          headers={**HEADERS, "Idempotency-Key": "u1"})
        r2 = client.post(f"/analyze/{pid}/upload", files={"file": ("b.csv", batch_csv, "text/csv")},
                          headers={**HEADERS, "Idempotency-Key": "u1"})
        assert r1.status_code == r2.status_code == 200
        hist = client.get(f"/history/{pid}", headers=HEADERS).json()
        assert hist["total"] == 1


class TestIdempotencyKeyScoping:
    def test_key_scoped_per_project_not_global(self):
        """The same key used on two different projects is independent."""
        pid_a = "test_idem_scope_a"
        pid_b = "test_idem_scope_b"
        _fit(pid_a)
        _fit(pid_b)
        ra = _analyze(pid_a, [1.0, 2.0, 3.0] * 20, key="shared-key")
        rb = _analyze(pid_b, [4.0, 5.0, 6.0] * 20, key="shared-key")
        assert ra.status_code == 200
        assert rb.status_code == 200  # not a 409 -- different project, same key is fine

    def test_key_expires_after_seven_days(self):
        """A key used more than 7 days ago no longer blocks a differing
        payload -- simulated by backdating the stored row's ts directly,
        since we won't wait 7 real days in a test."""
        pid = "test_idem_expiry"
        _fit(pid)
        _analyze(pid, [1.0, 2.0, 3.0] * 20, key="old-key")

        old_ts = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        conn = crud.get_connection()
        cur = conn.cursor()
        cur.execute("UPDATE analysis_runs SET ts = ? WHERE project = ? AND idempotency_key = ?",
                    (old_ts, _internal(pid), "old-key"))
        conn.commit()
        conn.close()

        resp = _analyze(pid, [7.0, 8.0, 9.0] * 20, key="old-key")
        assert resp.status_code == 200  # not 409 -- the old key has expired

        hist = client.get(f"/history/{pid}", headers=HEADERS).json()
        assert hist["total"] == 2  # the backdated original + the new one


class TestIdempotencyIsolation:
    """Every new endpoint behavior gets a two-user isolation check: B's
    key on B's own same-named project must be fully independent of A's."""

    def test_same_key_across_two_users_same_project_name_does_not_conflict(self):
        pid = "test_idem_iso"
        _fit(pid, headers=HEADERS)
        _fit(pid, headers=HEADERS_B)
        ra = _analyze(pid, [1.0, 2.0, 3.0] * 20, key="cross-user-key", headers=HEADERS)
        rb = _analyze(pid, [4.0, 5.0, 6.0] * 20, key="cross-user-key", headers=HEADERS_B)
        assert ra.status_code == 200
        assert rb.status_code == 200  # B's own project -- not a 409 against A's key usage
