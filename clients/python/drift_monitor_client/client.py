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

    def _guess_categorical_columns(self, df: pd.DataFrame) -> set:
        """Best-effort replica of utils/profiler.py's own continuous-vs-
        categorical threshold (text with <=50 uniques, or integer with
        <=20 uniques -> categorical), so the JSON fit path can pre-split
        columns into the dicts the server actually retains them from (see
        _fit_via_json). Sending everything under both dicts is NOT safe --
        the server pd.concat()s reference_data and categorical_data into
        one frame, and a column present in both produces a duplicate
        column name, which crashes profiling with a 500 (found while
        building examples/model_serving/). This heuristic won't always
        agree with the server's own profiler on edge cases (monotonic-
        sequence exclusion, structured-text detection); for large frames,
        fit() uses the upload path instead (see _fit_via_upload), which
        has no such edge case since the server classifies directly from
        one combined frame there, with no pre-split to reconcile."""
        categorical = set()
        for col in df.columns:
            series = df[col]
            n_unique = series.nunique()
            # is_string_dtype, not `dtype == object`: pandas >= 2.x can
            # give a string column its own "str"/StringDtype instead of
            # legacy "object" (confirmed on pandas 3.0.3, where `dtype ==
            # object` silently missed every string column and defeated
            # this whole heuristic -- caught by test_client.py's
            # test_categorical_column_survives_json_fit_path).
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_bool_dtype(series):
                if n_unique <= 50:
                    categorical.add(col)
            elif pd.api.types.is_integer_dtype(series):
                if n_unique <= 20:
                    categorical.add(col)
        return categorical

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
    ) -> Dict[str, Any]:
        """Fits (or re-fits) a project's baseline from a DataFrame. Every
        column is sent and classified server-side (continuous vs.
        categorical) -- the caller doesn't need to pre-split them."""
        if len(df) > self.large_frame_row_threshold:
            return self._fit_via_upload(project_id, df, calibration_config)
        return self._fit_via_json(project_id, df, calibration_config)

    def _fit_via_json(self, project_id, df, calibration_config):
        # The server classifies each column as continuous or categorical
        # from the combined frame, but only RETAINS a column under the
        # dict the caller submitted it in (main.py/fit_model_baseline: a
        # column submitted under reference_data that profiles as
        # categorical is silently dropped, and vice versa -- a pre-
        # existing server-side quirk, not something this client changes).
        # Pre-splitting with the same heuristic the server's profiler uses
        # (see _guess_categorical_columns) avoids that, without the
        # duplicate-column crash that submitting every column under BOTH
        # dicts causes (the server pd.concat()s them into one frame).
        categorical_cols = self._guess_categorical_columns(df)
        continuous_cols = [c for c in df.columns if c not in categorical_cols]
        payload: Dict[str, Any] = {
            "reference_data": {c: df[c].tolist() for c in continuous_cols},
            "categorical_data": {c: df[c].tolist() for c in categorical_cols},
        }
        if calibration_config is not None:
            payload["calibration_config"] = calibration_config
        resp = self.session.post(f"{self.base_url}/fit/{project_id}", json=payload, timeout=self.timeout)
        self._raise_for_status(resp)
        return resp.json()

    def _fit_via_upload(self, project_id, df, calibration_config):
        files = {"file": (f"{project_id}.parquet", self._to_parquet_bytes(df), "application/octet-stream")}
        data = {}
        if calibration_config is not None:
            data["calibration_config"] = json.dumps(calibration_config)
        resp = self.session.post(f"{self.base_url}/fit/{project_id}/upload", files=files, data=data,
                                  timeout=self.timeout)
        self._raise_for_status(resp)
        return resp.json()

    # ---------------------------------------------------------
    # analyze
    # ---------------------------------------------------------
    def analyze(self, project_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes a production batch against project_id's baseline."""
        if len(df) > self.large_frame_row_threshold:
            return self._analyze_via_upload(project_id, df)
        return self._analyze_via_json(project_id, df)

    def _analyze_via_json(self, project_id, df):
        payload = {"production_data": self._to_column_dict(df)}
        resp = self.session.post(f"{self.base_url}/analyze/{project_id}", json=payload, timeout=self.timeout)
        self._raise_for_status(resp)
        return resp.json()

    def _analyze_via_upload(self, project_id, df):
        files = {"file": (f"{project_id}_batch.parquet", self._to_parquet_bytes(df), "application/octet-stream")}
        resp = self.session.post(f"{self.base_url}/analyze/{project_id}/upload", files=files, timeout=self.timeout)
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
