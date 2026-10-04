"""
Step 5 Part 2: webhooks for drift alerts.

Delivery-mechanics tests (signature, retry/backoff, give-up, sweep)
monkeypatch drift.webhooks.requests.post rather than hitting a real
server -- the SSRF guard deliberately rejects localhost/private
addresses, so a real local test server can't be the delivery target
anyway; those cases are tested directly via db.crud.create_webhook,
bypassing the HTTP registration endpoint's SSRF check on purpose, to
isolate "does delivery logic work" from "is registration safe."

Run:
    python -m pytest tests/test_webhooks.py -v
"""

import random
import sys
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

import db.crud as crud
import drift.webhooks as webhooks
from main import app, _internal_project_key

client = TestClient(app)
EMAIL = "webhook-test@example.com"
EMAIL_B = "webhook-test-b@example.com"
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
    for table, col in [
        ("baselines", "project_id"), ("projects", "id"), ("analysis_runs", "project"),
        ("baseline_versions", "project_id"), ("baseline_active_version", "project_id"),
        ("alert_events", "project"),
    ]:
        cur.execute(f"DELETE FROM {table} WHERE {col} LIKE '%test_wh_%'")
    cur.execute("DELETE FROM webhooks WHERE project LIKE '%test_wh_%'")
    cur.execute("DELETE FROM webhook_deliveries WHERE webhook_id LIKE 'wh_test_%'")
    conn.commit()
    conn.close()


def _uniform(seed, lo, hi, n=60):
    rng = random.Random(seed)
    return [rng.uniform(lo, hi) for _ in range(n)]


def _fit(project_id, headers=HEADERS):
    return client.post(f"/fit/{project_id}", json={"reference_data": {"x": _uniform(1, 0, 10)}}, headers=headers)


def _analyze(project_id, batch, headers=HEADERS):
    return client.post(f"/analyze/{project_id}", json={"production_data": {"x": batch}}, headers=headers)


NO_ALERT_BATCH = _uniform(2, 0, 10)
ALERT_BATCH = _uniform(3, 5000, 5010)


class TestWebhookRegistration:
    def test_register_rejects_loopback_url(self):
        pid = "test_wh_ssrf_loopback"
        _fit(pid)
        resp = client.post(f"/webhooks/{pid}", json={"url": "http://127.0.0.1:9999/hook"}, headers=HEADERS)
        assert resp.status_code == 422

    def test_register_rejects_localhost_hostname(self):
        pid = "test_wh_ssrf_localhost"
        _fit(pid)
        resp = client.post(f"/webhooks/{pid}", json={"url": "http://localhost/hook"}, headers=HEADERS)
        assert resp.status_code == 422

    def test_register_rejects_link_local_metadata_style_address(self):
        pid = "test_wh_ssrf_metadata"
        _fit(pid)
        resp = client.post(f"/webhooks/{pid}", json={"url": "http://169.254.169.254/latest/meta-data/"}, headers=HEADERS)
        assert resp.status_code == 422

    def test_register_rejects_unknown_scheme(self):
        pid = "test_wh_badscheme"
        _fit(pid)
        resp = client.post(f"/webhooks/{pid}", json={"url": "ftp://example.com/hook"}, headers=HEADERS)
        assert resp.status_code == 422

    def test_register_accepts_public_looking_url_and_returns_secret_once(self):
        pid = "test_wh_register"
        _fit(pid)
        resp = client.post(f"/webhooks/{pid}", json={"url": "https://example.com/drift-hook"}, headers=HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["secret"] is not None
        assert body["event_filter"] == ["opened", "resolved"]

        listed = client.get(f"/webhooks/{pid}", headers=HEADERS).json()["webhooks"]
        assert listed[0]["id"] == body["id"]
        assert listed[0].get("secret") is None  # the real secret, not just the key, is absent

    def test_custom_event_filter_accepted(self):
        pid = "test_wh_customfilter"
        _fit(pid)
        resp = client.post(
            f"/webhooks/{pid}",
            json={"url": "https://example.com/hook", "event_filter": ["still_open"]},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["event_filter"] == ["still_open"]

    def test_invalid_event_filter_value_rejected(self):
        pid = "test_wh_badfilter"
        _fit(pid)
        resp = client.post(
            f"/webhooks/{pid}",
            json={"url": "https://example.com/hook", "event_filter": ["exploded"]},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_empty_event_filter_rejected(self):
        pid = "test_wh_emptyfilter"
        _fit(pid)
        resp = client.post(f"/webhooks/{pid}", json={"url": "https://example.com/hook", "event_filter": []}, headers=HEADERS)
        assert resp.status_code == 422

    def test_nonexistent_project_404s(self):
        resp = client.post("/webhooks/test_wh_never_existed", json={"url": "https://example.com/hook"}, headers=HEADERS)
        assert resp.status_code == 404


class TestWebhookDeletion:
    def test_delete_removes_webhook(self):
        pid = "test_wh_delete"
        _fit(pid)
        created = client.post(f"/webhooks/{pid}", json={"url": "https://example.com/hook"}, headers=HEADERS).json()
        resp = client.delete(f"/webhooks/{pid}/{created['id']}", headers=HEADERS)
        assert resp.status_code == 200
        assert client.get(f"/webhooks/{pid}", headers=HEADERS).json()["webhooks"] == []

    def test_delete_nonexistent_webhook_404s(self):
        pid = "test_wh_deletebad"
        _fit(pid)
        resp = client.delete(f"/webhooks/{pid}/wh_does_not_exist", headers=HEADERS)
        assert resp.status_code == 404


class TestWebhookIsolation:
    def test_user_b_cannot_list_or_register_or_delete_on_user_as_project(self):
        pid = "test_wh_iso"
        _fit(pid, headers=HEADERS)
        created = client.post(f"/webhooks/{pid}", json={"url": "https://example.com/hook"}, headers=HEADERS).json()

        assert client.get(f"/webhooks/{pid}", headers=HEADERS_B).status_code == 404
        # B never fit this project name -- registering on it 404s too,
        # same as every other project-scoped endpoint (item 1's
        # _require_existing_project guard).
        assert client.post(f"/webhooks/{pid}", json={"url": "https://example.com/hook"}, headers=HEADERS_B).status_code == 404
        assert client.delete(f"/webhooks/{pid}/{created['id']}", headers=HEADERS_B).status_code == 404

    def test_user_b_own_same_named_project_has_independent_webhooks(self):
        pid = "test_wh_iso_shared"
        _fit(pid, headers=HEADERS)
        _fit(pid, headers=HEADERS_B)
        client.post(f"/webhooks/{pid}", json={"url": "https://example.com/a-hook"}, headers=HEADERS)
        client.post(f"/webhooks/{pid}", json={"url": "https://example.com/b-hook"}, headers=HEADERS_B)

        a_hooks = client.get(f"/webhooks/{pid}", headers=HEADERS).json()["webhooks"]
        b_hooks = client.get(f"/webhooks/{pid}", headers=HEADERS_B).json()["webhooks"]
        assert len(a_hooks) == 1 and a_hooks[0]["url"] == "https://example.com/a-hook"
        assert len(b_hooks) == 1 and b_hooks[0]["url"] == "https://example.com/b-hook"


class TestTriggerOnTransition:
    def test_opened_and_resolved_enqueue_deliveries_by_default(self):
        pid = "test_wh_trigger"
        _fit(pid, headers=HEADERS)
        internal = _internal(pid)
        webhook_id = "wh_test_trigger"
        crud.create_webhook(webhook_id, internal, "https://example.com/hook", "s3cr3t",
                             ["opened", "resolved"], datetime.now(timezone.utc).isoformat())

        _analyze(pid, ALERT_BATCH)  # opened
        _analyze(pid, NO_ALERT_BATCH)  # resolved

        deliveries = crud.list_webhook_deliveries(webhook_id)
        assert len(deliveries) == 2
        events = {d["event_type"] for d in deliveries}
        assert events == {"opened", "resolved"}

    def test_still_open_not_enqueued_unless_opted_in(self):
        pid = "test_wh_stillopen"
        _fit(pid, headers=HEADERS)
        internal = _internal(pid)
        webhook_id = "wh_test_stillopen"
        crud.create_webhook(webhook_id, internal, "https://example.com/hook", "s3cr3t",
                             ["opened", "resolved"], datetime.now(timezone.utc).isoformat())

        _analyze(pid, ALERT_BATCH)   # opened -> enqueued
        _analyze(pid, ALERT_BATCH)   # still_open -> NOT enqueued (not in filter)

        deliveries = crud.list_webhook_deliveries(webhook_id)
        assert len(deliveries) == 1
        assert deliveries[0]["event_type"] == "opened"

    def test_still_open_enqueued_when_opted_in(self):
        pid = "test_wh_stillopen_optin"
        _fit(pid, headers=HEADERS)
        internal = _internal(pid)
        webhook_id = "wh_test_stillopen_optin"
        crud.create_webhook(webhook_id, internal, "https://example.com/hook", "s3cr3t",
                             ["opened", "still_open", "resolved"], datetime.now(timezone.utc).isoformat())

        _analyze(pid, ALERT_BATCH)
        _analyze(pid, ALERT_BATCH)

        deliveries = crud.list_webhook_deliveries(webhook_id)
        assert [d["event_type"] for d in deliveries] == ["still_open", "opened"]  # newest first

    def test_disabled_webhook_never_fires(self):
        pid = "test_wh_disabled"
        _fit(pid, headers=HEADERS)
        internal = _internal(pid)
        webhook_id = "wh_test_disabled"
        crud.create_webhook(webhook_id, internal, "https://example.com/hook", "s3cr3t",
                             ["opened", "resolved"], datetime.now(timezone.utc).isoformat())
        conn = crud.get_connection()
        conn.execute("UPDATE webhooks SET enabled = 0 WHERE id = ?", (webhook_id,))
        conn.commit()
        conn.close()

        _analyze(pid, ALERT_BATCH)
        assert crud.list_webhook_deliveries(webhook_id) == []

    def test_no_raw_production_rows_in_payload(self):
        """Statistics only -- the exact production values must never
        appear in the enqueued payload."""
        pid = "test_wh_norawdata"
        _fit(pid, headers=HEADERS)
        internal = _internal(pid)
        webhook_id = "wh_test_norawdata"
        crud.create_webhook(webhook_id, internal, "https://example.com/hook", "s3cr3t",
                             ["opened"], datetime.now(timezone.utc).isoformat())

        needle = 918273.456
        batch = [needle] + ALERT_BATCH[1:]
        _analyze(pid, batch)

        deliveries = crud.list_webhook_deliveries(webhook_id)
        assert str(needle) not in str(deliveries[0]["payload"])


class TestDeliverySweep:
    def _make_webhook_and_event(self, suffix):
        pid = f"test_wh_sweep_{suffix}"
        internal = _internal(pid)
        webhook_id = f"wh_test_sweep_{suffix}"
        crud.create_webhook(webhook_id, internal, "https://example.com/hook", "s3cr3t",
                             ["opened"], datetime.now(timezone.utc).isoformat())
        now = datetime.now(timezone.utc).isoformat()
        delivery_id = crud.enqueue_webhook_delivery(webhook_id, None, "opened", {"event": "alert.opened"}, now, now)
        return webhook_id, delivery_id

    def test_successful_delivery_marks_success_no_retry(self, monkeypatch):
        webhook_id, delivery_id = self._make_webhook_and_event("success")

        class FakeResponse:
            status_code = 200
            text = "ok"

        monkeypatch.setattr(webhooks.requests, "post", lambda *a, **kw: FakeResponse())
        webhooks.sweep_due_webhook_deliveries()

        deliveries = crud.list_webhook_deliveries(webhook_id)
        assert deliveries[0]["status"] == "success"
        assert deliveries[0]["success"] is True
        assert deliveries[0]["next_retry_at"] is None

    def test_signature_header_is_correct_hmac(self, monkeypatch):
        webhook_id, delivery_id = self._make_webhook_and_event("sig")
        captured = {}

        class FakeResponse:
            status_code = 200
            text = "ok"

        def fake_post(url, data, headers, timeout):
            captured["headers"] = headers
            captured["data"] = data
            return FakeResponse()

        monkeypatch.setattr(webhooks.requests, "post", fake_post)
        webhooks.sweep_due_webhook_deliveries()

        expected = webhooks.sign_payload("s3cr3t", captured["data"])
        assert captured["headers"]["X-Drift-Signature-256"] == f"sha256={expected}"

    def test_failed_delivery_schedules_retry_with_backoff(self, monkeypatch):
        webhook_id, delivery_id = self._make_webhook_and_event("retry")

        class FakeResponse:
            status_code = 500
            text = "server error"

        monkeypatch.setattr(webhooks.requests, "post", lambda *a, **kw: FakeResponse())
        webhooks.sweep_due_webhook_deliveries()

        deliveries = crud.list_webhook_deliveries(webhook_id)
        d = deliveries[0]
        assert d["status"] == "pending"
        assert d["attempt_number"] == 2
        assert d["status_code"] == 500
        assert d["next_retry_at"] is not None
        next_retry = datetime.fromisoformat(d["next_retry_at"])
        assert next_retry > datetime.now(timezone.utc)  # scheduled in the future

    def test_connection_error_treated_as_failed_attempt(self, monkeypatch):
        webhook_id, delivery_id = self._make_webhook_and_event("connerr")

        import requests as requests_module

        def raise_conn_error(*a, **kw):
            raise requests_module.exceptions.ConnectionError("boom")

        monkeypatch.setattr(webhooks.requests, "post", raise_conn_error)
        webhooks.sweep_due_webhook_deliveries()

        d = crud.list_webhook_deliveries(webhook_id)[0]
        assert d["status"] == "pending"
        assert d["status_code"] is None
        assert "boom" in d["response_snippet"]

    def test_gives_up_after_max_attempts(self, monkeypatch):
        webhook_id, delivery_id = self._make_webhook_and_event("giveup")

        class FakeResponse:
            status_code = 500
            text = "still failing"

        monkeypatch.setattr(webhooks.requests, "post", lambda *a, **kw: FakeResponse())

        conn = crud.get_connection()
        for _ in range(webhooks.MAX_WEBHOOK_ATTEMPTS):
            conn.execute(
                "UPDATE webhook_deliveries SET next_retry_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), delivery_id),
            )
            conn.commit()
            webhooks.sweep_due_webhook_deliveries()

        d = crud.list_webhook_deliveries(webhook_id)[0]
        assert d["status"] == "failed"
        assert d["attempt_number"] == webhooks.MAX_WEBHOOK_ATTEMPTS
        assert d["next_retry_at"] is None
        conn.close()

    def test_not_yet_due_delivery_is_skipped(self, monkeypatch):
        webhook_id = "wh_test_sweep_notdue"
        internal = _internal("test_wh_sweep_notdue")
        crud.create_webhook(webhook_id, internal, "https://example.com/hook", "s3cr3t",
                             ["opened"], datetime.now(timezone.utc).isoformat())
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        crud.enqueue_webhook_delivery(webhook_id, None, "opened", {"event": "alert.opened"},
                                       future, datetime.now(timezone.utc).isoformat())

        called = {"count": 0}
        monkeypatch.setattr(webhooks.requests, "post", lambda *a, **kw: called.__setitem__("count", called["count"] + 1))
        webhooks.sweep_due_webhook_deliveries()
        assert called["count"] == 0

    def test_deleted_webhook_fails_pending_delivery_gracefully(self):
        pid = "test_wh_sweep_deleted"
        internal = _internal(pid)
        webhook_id = "wh_test_sweep_deleted"
        crud.create_webhook(webhook_id, internal, "https://example.com/hook", "s3cr3t",
                             ["opened"], datetime.now(timezone.utc).isoformat())
        now = datetime.now(timezone.utc).isoformat()
        crud.enqueue_webhook_delivery(webhook_id, None, "opened", {"event": "alert.opened"}, now, now)
        crud.delete_webhook(webhook_id, internal)

        webhooks.sweep_due_webhook_deliveries()  # must not raise
        d = crud.list_webhook_deliveries(webhook_id)[0]
        assert d["status"] == "failed"


class TestSweepThreadSingleton:
    def test_starting_twice_does_not_spawn_two_threads(self):
        import threading
        before = sum(1 for t in threading.enumerate() if t.name == "webhook-sweep")
        webhooks.start_webhook_sweep_thread()
        webhooks.start_webhook_sweep_thread()
        after = sum(1 for t in threading.enumerate() if t.name == "webhook-sweep")
        assert after - before <= 1
