"""
Background jobs for the unified table path (profile / fit / analyze).

One in-process daemon worker runs jobs one at a time, FIFO -- bounded
torch memory, no extra infrastructure (same idempotent-thread pattern as
drift/webhooks.py's sweep). The `jobs` table is the queue, so a future
separate worker process can claim from it unchanged.

Handlers are registered by main.py (register()) to avoid import cycles.
A handler receives the job dict and returns a JSON-serializable result;
any exception marks the job failed with a SANITIZED message -- pandas and
PIL errors often echo cell values, and raw production data must never
reach the DB, so only the exception type plus a safe message is stored.
"""

import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

from db import blob_store, crud

STAGE_TTL_HOURS = 24
_POLL_SECONDS = 2.0
_SWEEP_EVERY_SECONDS = 60.0

_handlers: Dict[str, Callable[[dict], dict]] = {}
_wake = threading.Event()
_started = False
_start_lock = threading.Lock()
log = logging.getLogger("drift.jobs")


class JobError(Exception):
    """An expected failure whose message is safe to show the user (it
    never contains cell values) -- e.g. 'no image ZIP was uploaded'."""


def register(kind: str, handler: Callable[[dict], dict]) -> None:
    _handlers[kind] = handler


def enqueue(owner_email: str, owner_name: str, project_id: str, public_project_id: str, kind: str,
            stage_id: Optional[str], payload: Optional[dict] = None,
            idempotency_key: Optional[str] = None, payload_hash: Optional[str] = None) -> str:
    job_id = uuid.uuid4().hex
    crud.create_job(job_id, owner_email, owner_name, project_id, public_project_id, kind, stage_id,
                    payload, idempotency_key, payload_hash)
    _wake.set()
    return job_id


def progress(job_id: str, pct: int, message: str) -> None:
    crud.update_job(job_id, progress=pct, progress_message=message)


def stage_expiry(hours: int = STAGE_TTL_HOURS) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def discard_stage(stage: dict, status: str) -> None:
    """Deletes a stage's raw uploaded files and its profile (which holds
    sample values), and records why."""
    blob_store.delete(stage.get("table_blob"))
    blob_store.delete(stage.get("zip_blob"))
    crud.update_stage(stage["id"], status=status, table_blob=None, zip_blob=None, profile=None)
    crud.clear_profile_job_results(stage["id"])


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, JobError):
        return str(exc)
    return f"{type(exc).__name__}: the job failed while processing the data. See the server log for details."


def run_job(job: dict) -> None:
    handler = _handlers.get(job["kind"])
    try:
        if handler is None:
            raise JobError(f"No handler registered for job kind '{job['kind']}'.")
        result = handler(job)
        crud.update_job(job["id"], status="succeeded", progress=100, result=result,
                        finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:  # every failure must end the job, never kill the worker
        log.exception("Job %s (%s) failed", job["id"], job["kind"])
        crud.update_job(job["id"], status="failed", error=_safe_error(exc),
                        finished_at=datetime.now(timezone.utc).isoformat())


def run_pending() -> int:
    """Runs queued jobs until none remain; returns how many ran."""
    ran = 0
    while True:
        job = crud.claim_next_job()
        if job is None:
            return ran
        run_job(job)
        ran += 1


def sweep_expired_stages() -> int:
    expired = crud.list_expired_stages(datetime.now(timezone.utc).isoformat())
    for stage in expired:
        discard_stage(stage, "expired")
    return len(expired)


def _loop() -> None:
    last_sweep = 0.0
    while True:
        try:
            run_pending()
            if time.monotonic() - last_sweep > _SWEEP_EVERY_SECONDS:
                sweep_expired_stages()
                last_sweep = time.monotonic()
        except Exception:
            log.exception("Job worker loop error")
        _wake.wait(_POLL_SECONDS)
        _wake.clear()


def start_worker() -> None:
    """Idempotent: safe to call on every lifespan start (TestClient calls it many times)."""
    global _started
    with _start_lock:
        if _started:
            return
        crud.mark_running_jobs_interrupted()
        threading.Thread(target=_loop, name="table-job-worker", daemon=True).start()
        _started = True
