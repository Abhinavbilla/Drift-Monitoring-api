"""
DriftClient: a small Python client for the Drift Monitoring API.

    from drift_monitor_client import DriftClient

    client = DriftClient(base_url="http://localhost:8000", token="dm_...")
    client.fit("my_project", reference_df)
    result = client.analyze("my_project", production_df)

Auth: a personal access token (see scripts/create_token.py in the main
repo) or a session JWT, sent as a Bearer token -- identical to how the
dashboard authenticates, just without a browser.

Large frames (more rows than `large_frame_row_threshold`) are
automatically uploaded as Parquet via the multipart upload endpoints
instead of being inlined as a JSON body, which gets slow and memory-heavy
well before a typical HTTP JSON payload limit is hit.

Retries: a urllib3 Retry policy with exponential backoff is applied to
connection failures and 500/502/503/504 responses (a confirmed non-5xx
response, including a 4xx, is never retried).
"""

import io
import json
from typing import Any, Dict, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_LARGE_FRAME_ROW_THRESHOLD = 10_000
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5


class DriftClientError(Exception):
    """Raised for a non-2xx response, with the parsed error detail (main.py's
    handlers return a clean `detail` string) attached where available."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class DriftClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        large_frame_row_threshold: int = DEFAULT_LARGE_FRAME_ROW_THRESHOLD,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.large_frame_row_threshold = large_frame_row_threshold

        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST", "DELETE"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _raise_for_status(self, resp: requests.Response) -> None:
        if resp.ok:
            return
        detail = resp.text
        try:
            body = resp.json()
            if isinstance(body, dict) and "detail" in body:
                detail = body["detail"]
        except ValueError:
            pass
        raise DriftClientError(resp.status_code, detail)

    def _to_column_dict(self, df: pd.DataFrame) -> Dict[str, list]:
        return {col: df[col].tolist() for col in df.columns}

    def _to_parquet_bytes(self, df: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        df.to_parquet(buf)
        return buf.getvalue()

    # ---------------------------------------------------------
    # fit
    # ---------------------------------------------------------
    def fit(
        self, project_id: str, df: pd.DataFrame,
        calibration_config: Optional[Dict[str, Any]] = None,
        feature_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Fits (or re-fits) a project's baseline from a DataFrame. Every
        column is sent under reference_data and classified server-side
        (continuous vs. categorical) by the profiler. Pass feature_types
        ({column: "continuous"|"categorical"}) to override the profiler's
        classification for specific columns -- e.g. a low-cardinality
        integer column you specifically want monitored as continuous;
        columns not named there are unaffected. Previously this client
        replicated the server's own classification heuristic client-side
        to route each column into the dict the server would actually
        retain it from -- replaced (2026-09-30) by the server itself no
        longer needing that dance (main.py's /fit now looks a column's
        values up regardless of which dict it arrived in), plus this
        explicit override for the cases where you want a specific type
        rather than whatever the profiler infers."""
        if len(df) > self.large_frame_row_threshold:
            return self._fit_via_upload(project_id, df, calibration_config, feature_types)
        return self._fit_via_json(project_id, df, calibration_config, feature_types)

    def _fit_via_json(self, project_id, df, calibration_config, feature_types):
        payload: Dict[str, Any] = {"reference_data": self._to_column_dict(df)}
        if calibration_config is not None:
            payload["calibration_config"] = calibration_config
        if feature_types is not None:
            payload["feature_types"] = feature_types
        resp = self.session.post(f"{self.base_url}/fit/{project_id}", json=payload, timeout=self.timeout)
        self._raise_for_status(resp)
        return resp.json()

    def _fit_via_upload(self, project_id, df, calibration_config, feature_types):
        files = {"file": (f"{project_id}.parquet", self._to_parquet_bytes(df), "application/octet-stream")}
        data = {}
        if calibration_config is not None:
            data["calibration_config"] = json.dumps(calibration_config)
        if feature_types is not None:
            data["feature_types"] = json.dumps(feature_types)
        resp = self.session.post(f"{self.base_url}/fit/{project_id}/upload", files=files, data=data,
                                  timeout=self.timeout)
        self._raise_for_status(resp)
        return resp.json()

    # ---------------------------------------------------------
    # analyze
    # ---------------------------------------------------------
    def analyze(self, project_id: str, df: pd.DataFrame, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """Analyzes a production batch against project_id's baseline.

        idempotency_key (optional): resending the same key with the same
        data returns the previously stored result instead of re-analyzing
        (and doesn't create a new history row); resending it with
        different data raises DriftClientError(409). Keys expire after 7
        days server-side."""
        if len(df) > self.large_frame_row_threshold:
            return self._analyze_via_upload(project_id, df, idempotency_key)
        return self._analyze_via_json(project_id, df, idempotency_key)

    def _idempotency_headers(self, idempotency_key):
        return {"Idempotency-Key": idempotency_key} if idempotency_key else {}

    def _analyze_via_json(self, project_id, df, idempotency_key=None):
        payload = {"production_data": self._to_column_dict(df)}
        resp = self.session.post(f"{self.base_url}/analyze/{project_id}", json=payload, timeout=self.timeout,
                                  headers=self._idempotency_headers(idempotency_key))
        self._raise_for_status(resp)
        return resp.json()

    def _analyze_via_upload(self, project_id, df, idempotency_key=None):
        files = {"file": (f"{project_id}_batch.parquet", self._to_parquet_bytes(df), "application/octet-stream")}
        resp = self.session.post(f"{self.base_url}/analyze/{project_id}/upload", files=files, timeout=self.timeout,
                                  headers=self._idempotency_headers(idempotency_key))
        self._raise_for_status(resp)
        return resp.json()

    # ---------------------------------------------------------
    # management
    # ---------------------------------------------------------
    def delete_project(self, project_id: str) -> Dict[str, Any]:
        resp = self.session.delete(f"{self.base_url}/projects/{project_id}", timeout=self.timeout)
        self._raise_for_status(resp)
        return resp.json()

    def list_projects(self) -> list:
        resp = self.session.get(f"{self.base_url}/projects", timeout=self.timeout)
        self._raise_for_status(resp)
        return resp.json()["projects"]
