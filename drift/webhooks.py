"""
Step 5 Part 2: webhooks for drift alerts.

Fires on alert_events transitions (opened/resolved by default, still_open
opt-in per webhook via event_filter) -- never on every alerting /analyze
call, unlike the existing email alert (drift/alerts.py). Delivery is a
durable queue: every attempt (including the first) goes through the same
sweep path, so there's one code path to reason about and no race between
an inline "try now" attempt and the retry sweep. A background daemon
thread, started once from main.py's lifespan, wakes every
WEBHOOK_SWEEP_INTERVAL_SECONDS and processes whatever's due.
"""

import hashlib
import hmac
import ipaddress
import json
import secrets
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import urlparse

import requests

from db import crud
from utils.validation import ValidationError

WEBHOOK_SECRET_BYTES = 32
MAX_WEBHOOK_ATTEMPTS = 5
# Seconds to wait AFTER attempt N before attempt N+1 -- index 0 is the
# wait after attempt 1 (before attempt 2), etc. 4 entries for 5 total
# attempts.
WEBHOOK_BACKOFF_SECONDS = [1, 4, 16, 64]
WEBHOOK_SWEEP_INTERVAL_SECONDS = 30
WEBHOOK_DELIVERY_TIMEOUT_SECONDS = 10
VALID_WEBHOOK_EVENTS = ("opened", "resolved", "still_open")
DEFAULT_WEBHOOK_EVENT_FILTER = ["opened", "resolved"]


def generate_webhook_secret() -> str:
    return secrets.token_urlsafe(WEBHOOK_SECRET_BYTES)


def sign_payload(secret: str, raw_body: bytes) -> str:
    """Stripe-style: hex HMAC-SHA256 of the exact raw body bytes, under
    the header value `sha256=<this>`."""
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


def validate_event_filter(event_filter: Optional[List[str]]) -> List[str]:
    if event_filter is None:
        return list(DEFAULT_WEBHOOK_EVENT_FILTER)
    if not event_filter:
        raise ValidationError("event_filter cannot be empty -- omit it entirely to use the default.")
    unknown = sorted(set(event_filter) - set(VALID_WEBHOOK_EVENTS))
    if unknown:
        raise ValidationError(f"event_filter has unrecognized value(s) {unknown}; valid values are {VALID_WEBHOOK_EVENTS}.")
    return list(dict.fromkeys(event_filter))  # de-duplicate, preserve order


def validate_webhook_url(url: str) -> None:
    """SSRF guard, mandatory per the Step 5 Part 2 scope -- a feature
    that POSTs server-signed data to an arbitrary caller-supplied URL is
    an SSRF vector by default. Resolves the hostname and rejects it if
    ANY resolved address is loopback/private/link-local/reserved/
    multicast/unspecified -- covers the AWS/GCP/Azure metadata endpoint
    (169.254.169.254, link-local) along with the more obvious
    localhost/127.0.0.1/10.x/192.168.x cases. Does not protect against a
    URL that resolves safely at registration time but is re-pointed via
    DNS to an internal address before a later delivery (DNS rebinding) --
    a real residual risk worth flagging, not silently claiming solved."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValidationError(f"Webhook URL must be http or https, got scheme '{parsed.scheme}'.")
    if not parsed.hostname:
        raise ValidationError("Webhook URL has no hostname.")

    try:
        addr_infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise ValidationError(f"Webhook URL's hostname could not be resolved: {e}")

    for family, _, _, _, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        ip = ipaddress.ip_address(ip_str)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved \
                or ip.is_multicast or ip.is_unspecified:
            raise ValidationError(
                f"Webhook URL's hostname resolves to {ip_str}, a non-public address -- "
                f"webhooks cannot target loopback, private, link-local, reserved, or "
                f"multicast addresses (this also blocks cloud metadata endpoints)."
            )


def _build_payload(event_type: str, project_id: str, baseline_version: int, ts: str,
                    alert_state: str, sustained_alert: bool, windows_considered: int,
                    feature_metrics: dict, relationship_metrics: Optional[dict] = None) -> dict:
    payload = {
        "event": f"alert.{event_type}",
        "project_id": project_id,
        "baseline_version": baseline_version,
        "ts": ts,
        "alert_state": alert_state,
        "sustained_alert": sustained_alert,
        "windows_considered": windows_considered,
        "feature_metrics": feature_metrics,
    }
    if relationship_metrics:  # table projects only; tabular payloads stay unchanged
        payload["relationship_metrics"] = relationship_metrics
    return payload


def enqueue_deliveries_for_transition(
    internal_project_id: str, public_project_id: str, event_id: Optional[int],
    transition: str, baseline_version: int, ts: str, alert_state: str,
    sustained_alert: bool, windows_considered: int, feature_metrics: dict,
    relationship_metrics: Optional[dict] = None,
) -> None:
    """Called right after an alert_events row is written for an actual
    transition (opened/still_open/resolved). Enqueues one delivery task
    per enabled webhook whose event_filter includes this transition --
    due immediately (next_retry_at = now), so the sweep loop picks it up
    on its very next tick."""
    payload = _build_payload(
        transition, public_project_id, baseline_version, ts,
        alert_state, sustained_alert, windows_considered, feature_metrics, relationship_metrics,
    )
    now = datetime.now(timezone.utc).isoformat()
    for webhook in crud.list_webhooks(internal_project_id):
        if not webhook["enabled"]:
            continue
        if transition not in webhook["event_filter"]:
            continue
        crud.enqueue_webhook_delivery(webhook["id"], event_id, transition, payload, now, now)


def _attempt_delivery(webhook: dict, payload: dict) -> "tuple[bool, Optional[int], str]":
    """Returns (success, status_code, response_snippet). Never raises --
    any exception (timeout, connection error, DNS failure) is caught and
    reported as a failed attempt, same as a non-2xx status code."""
    raw_body = json.dumps(payload, sort_keys=True).encode("utf-8")
    signature = sign_payload(webhook["secret"], raw_body)
    headers = {
        "Content-Type": "application/json",
        "X-Drift-Signature-256": f"sha256={signature}",
        "X-Drift-Event": payload["event"],
    }
    try:
        resp = requests.post(
            webhook["url"], data=raw_body, headers=headers, timeout=WEBHOOK_DELIVERY_TIMEOUT_SECONDS,
        )
        success = 200 <= resp.status_code < 300
        return success, resp.status_code, resp.text[:500]
    except requests.exceptions.RequestException as e:
        return False, None, str(e)[:500]


def sweep_due_webhook_deliveries() -> int:
    """Processes every delivery task whose next_retry_at has arrived.
    Returns the number processed. Safe to call directly (tests do this
    for deterministic assertions) or from the background sweep thread."""
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    due = crud.get_due_webhook_deliveries(now_iso)
    for delivery in due:
        webhook = crud.get_webhook(delivery["webhook_id"])
        if not webhook or not webhook["enabled"]:
            # Webhook was deleted/disabled after this delivery was queued.
            crud.record_delivery_attempt(
                delivery["id"], delivery["attempt_number"], success=False,
                status_code=None, response_snippet="webhook no longer exists or is disabled",
                now=now_iso, next_retry_at=None,
            )
            continue

        success, status_code, snippet = _attempt_delivery(webhook, delivery["payload"])
        attempt_number = delivery["attempt_number"]
        if success or attempt_number >= MAX_WEBHOOK_ATTEMPTS:
            crud.record_delivery_attempt(
                delivery["id"], attempt_number, success=success,
                status_code=status_code, response_snippet=snippet, now=now_iso, next_retry_at=None,
            )
        else:
            backoff = WEBHOOK_BACKOFF_SECONDS[min(attempt_number - 1, len(WEBHOOK_BACKOFF_SECONDS) - 1)]
            next_retry_at = (now_dt + timedelta(seconds=backoff)).isoformat()
            crud.record_delivery_attempt(
                delivery["id"], attempt_number + 1, success=False,
                status_code=status_code, response_snippet=snippet, now=now_iso, next_retry_at=next_retry_at,
            )
    return len(due)


_sweep_thread_started = False
_sweep_thread_lock = threading.Lock()


def start_webhook_sweep_thread() -> None:
    """Starts the background sweep loop exactly once per process, no
    matter how many times this is called (main.py's lifespan can fire
    more than once -- e.g. once per TestClient(app) across many test
    files). A plain daemon thread, not asyncio: this codebase's HTTP
    calls (requests), DB access (sqlite3), and email alerts (smtplib)
    are all synchronous already, and mixing a blocking requests.post
    into an asyncio loop would need extra care (run_in_executor) this
    avoids entirely."""
    global _sweep_thread_started
    with _sweep_thread_lock:
        if _sweep_thread_started:
            return
        _sweep_thread_started = True

    def _loop():
        while True:
            try:
                sweep_due_webhook_deliveries()
            except Exception as e:
                print(f"webhook sweep tick failed: {e}")
            time.sleep(WEBHOOK_SWEEP_INTERVAL_SECONDS)

    thread = threading.Thread(target=_loop, daemon=True, name="webhook-sweep")
    thread.start()
