"""
Step 5 Part 1 item 6: alert policy {k, m}, sustained_alert, and the
ok/open alert state machine (opened/resolved/still_open) in alert_events.

Run:
    python -m pytest tests/test_alert_state_machine.py -v
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
EMAIL = "alert-fsm-test@example.com"
EMAIL_B = "alert-fsm-test-b@example.com"
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
    cur.execute("DELETE FROM baselines WHERE project_id LIKE '%test_fsm_%'")
    cur.execute("DELETE FROM projects WHERE id LIKE '%test_fsm_%'")
    cur.execute("DELETE FROM analysis_runs WHERE project LIKE '%test_fsm_%'")
    cur.execute("DELETE FROM baseline_versions WHERE project_id LIKE '%test_fsm_%'")
    cur.execute("DELETE FROM baseline_active_version WHERE project_id LIKE '%test_fsm_%'")
    cur.execute("DELETE FROM alert_events WHERE project LIKE '%test_fsm_%'")
    conn.commit()
    conn.close()


def _uniform(seed, lo, hi, n=60):
    rng = random.Random(seed)
    return [rng.uniform(lo, hi) for _ in range(n)]


# A constant column is excluded from monitoring entirely by the profiler
# (no variance to track), so the reference and every batch need real
# spread -- distinct seeds so no two draws are byte-identical (which
# would itself look like a suspicious "duplicate data" artifact).
REF_X = _uniform(1, 0, 10)
NO_ALERT_BATCH = _uniform(2, 0, 10)       # same distribution -- shouldn't alert
ALERT_BATCH = _uniform(3, 5000, 5010)     # wildly different range -- always alerts


def _fit(project_id, headers=HEADERS, alert_policy=None):
    payload = {"reference_data": {"x": REF_X}}
    if alert_policy is not None:
        payload["alert_policy"] = alert_policy
    return client.post(f"/fit/{project_id}", json=payload, headers=headers)


def _analyze(project_id, batch, headers=HEADERS):
    return client.post(f"/analyze/{project_id}", json={"production_data": {"x": batch}}, headers=headers)


class TestDefaultPolicyMatchesTodaysBehavior:
    def test_single_alert_is_sustained_with_default_k1_m1(self):
        pid = "test_fsm_default"
        _fit(pid)
        resp = _analyze(pid, ALERT_BATCH)
        body = resp.json()
        assert body["alert"] is True
        assert body["sustained_alert"] is True
        assert body["windows_considered"] == 1
        assert body["alert_state"] == "open"
        assert body["transition"] == "opened"

    def test_single_non_alert_stays_ok(self):
        pid = "test_fsm_defaultok"
        _fit(pid)
        resp = _analyze(pid, NO_ALERT_BATCH)
        body = resp.json()
        assert body["sustained_alert"] is False
        assert body["alert_state"] == "ok"
        assert body["transition"] is None  # steady ok -> no event logged


class TestStateTransitions:
    def test_opened_then_still_open_then_resolved(self):
        pid = "test_fsm_transitions"
        _fit(pid, alert_policy={"k": 1, "m": 1})

        r1 = _analyze(pid, ALERT_BATCH)
        assert r1.json()["transition"] == "opened"

        r2 = _analyze(pid, ALERT_BATCH)
        assert r2.json()["transition"] == "still_open"
        assert r2.json()["alert_state"] == "open"

        r3 = _analyze(pid, NO_ALERT_BATCH)
        assert r3.json()["transition"] == "resolved"
        assert r3.json()["alert_state"] == "ok"

    def test_re_resolving_an_already_ok_project_logs_no_event(self):
        pid = "test_fsm_noop"
        _fit(pid, alert_policy={"k": 1, "m": 1})
        _analyze(pid, ALERT_BATCH)    # opened
        _analyze(pid, NO_ALERT_BATCH)  # resolved
        r3 = _analyze(pid, NO_ALERT_BATCH)  # still ok -- no new event
        assert r3.json()["transition"] is None
        assert r3.json()["alert_state"] == "ok"


class TestSustainedAlertWithKGreaterThanOne:
    def test_k2_m3_requires_two_of_last_three(self):
        pid = "test_fsm_k2m3"
        _fit(pid, alert_policy={"k": 2, "m": 3})

        r1 = _analyze(pid, ALERT_BATCH)      # [alert] -> 1/1, not sustained
        assert r1.json()["sustained_alert"] is False
        assert r1.json()["windows_considered"] == 1

        r2 = _analyze(pid, NO_ALERT_BATCH)   # [alert, ok] -> 1/2, not sustained
        assert r2.json()["sustained_alert"] is False
        assert r2.json()["windows_considered"] == 2

        r3 = _analyze(pid, ALERT_BATCH)      # [alert, ok, alert] -> 2/3, sustained
        assert r3.json()["sustained_alert"] is True
        assert r3.json()["windows_considered"] == 3
        assert r3.json()["transition"] == "opened"

    def test_windows_considered_caps_at_m_sliding_window(self):
        pid = "test_fsm_sliding"
        _fit(pid, alert_policy={"k": 3, "m": 3})
        _analyze(pid, ALERT_BATCH)
        _analyze(pid, ALERT_BATCH)
        _analyze(pid, ALERT_BATCH)  # 3/3 -> sustained
        r4 = _analyze(pid, NO_ALERT_BATCH)  # window slides: [alert,alert,ok] -> 2/3, not sustained
        assert r4.json()["windows_considered"] == 3
        assert r4.json()["sustained_alert"] is False
        assert r4.json()["transition"] == "resolved"


class TestVersionBoundary:
    def test_sustained_alert_window_resets_on_new_baseline_version(self):
        """A re-fit creates a new baseline version -- the 'last m
        analyses on the SAME baseline version' window must not carry
        alert history over from the previous version."""
        pid = "test_fsm_versionreset"
        _fit(pid, alert_policy={"k": 1, "m": 1})
        r1 = _analyze(pid, ALERT_BATCH)
        assert r1.json()["alert_state"] == "open"

        _fit(pid)  # new baseline version -- fresh window, fresh state

        r2 = _analyze(pid, NO_ALERT_BATCH)
        assert r2.json()["windows_considered"] == 1  # not 2 -- doesn't see version 1's history
        assert r2.json()["alert_state"] == "ok"
        assert r2.json()["transition"] is None  # fresh version starts at ok, not "open"

    def test_explicit_old_baseline_version_has_its_own_independent_state(self):
        pid = "test_fsm_versionindependent"
        _fit(pid, alert_policy={"k": 1, "m": 5})
        _analyze(pid, ALERT_BATCH)  # version 1 -> open
        _fit(pid)  # version 2, active

        # Explicitly analyze against version 1 again -- its state is still open.
        resp = client.post(f"/analyze/{pid}", params={"baseline_version": 1},
                            json={"production_data": {"x": ALERT_BATCH}}, headers=HEADERS)
        assert resp.json()["windows_considered"] == 2  # version 1 now has 2 analyses
        assert resp.json()["alert_state"] == "open"
        assert resp.json()["transition"] == "still_open"


class TestAlertPolicyValidation:
    def test_k_greater_than_m_rejected(self):
        resp = _fit("test_fsm_badkm", alert_policy={"k": 5, "m": 2})
        assert resp.status_code == 422

    def test_zero_k_rejected(self):
        resp = _fit("test_fsm_zerok", alert_policy={"k": 0, "m": 1})
        assert resp.status_code == 422

    def test_missing_key_rejected(self):
        resp = _fit("test_fsm_missingkey", alert_policy={"k": 1})
        assert resp.status_code == 422

    def test_unrecognized_key_rejected(self):
        resp = _fit("test_fsm_badkey", alert_policy={"k": 1, "m": 1, "extra": 1})
        assert resp.status_code == 422

    def test_policy_persists_across_refit_without_mention(self):
        pid = "test_fsm_persist"
        _fit(pid, alert_policy={"k": 2, "m": 2})
        _fit(pid)  # re-fit without mentioning alert_policy
        _analyze(pid, ALERT_BATCH)  # 1/1 so far under k=2,m=2 -> not sustained yet
        r2 = _analyze(pid, ALERT_BATCH)
        assert r2.json()["sustained_alert"] is True  # k=2 still in effect


class TestAlertStateMachineIsolation:
    def test_user_b_alert_state_independent_of_user_a(self):
        pid = "test_fsm_iso"
        _fit(pid, headers=HEADERS)
        _fit(pid, headers=HEADERS_B)

        _analyze(pid, ALERT_BATCH, headers=HEADERS)       # A: open
        resp_b = _analyze(pid, NO_ALERT_BATCH, headers=HEADERS_B)  # B: ok, independent
        assert resp_b.json()["alert_state"] == "ok"

    def test_user_b_cannot_affect_user_as_alert_policy(self):
        pid = "test_fsm_iso_policy"
        _fit(pid, alert_policy={"k": 1, "m": 1}, headers=HEADERS)
        _fit(pid, alert_policy={"k": 5, "m": 5}, headers=HEADERS_B)

        resp_a = _analyze(pid, ALERT_BATCH, headers=HEADERS)
        assert resp_a.json()["sustained_alert"] is True  # A's k=1,m=1 still applies
