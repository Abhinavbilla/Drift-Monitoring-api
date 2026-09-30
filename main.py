import sqlite3
import jwt
import binascii
import hashlib
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, status, UploadFile, File, Form, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json
import ingest.readers as ingest_readers
from contextlib import asynccontextmanager
from pydantic import BaseModel
from PIL import UnidentifiedImageError
from drift.alerts import send_drift_email
from auth.tokens import parse_prefix, verify_token_hash
# Importing custom modules
from models import (
    FitBaselineRequest, FitBaselineResponse, FeatureCalibrationInfo,
    PredictRequest, PredictResponse,
    AnalyzeBatchRequest, AnalyzeBatchResponse,
    HealthCheckResponse,
    FitTextBaselineRequest, AnalyzeTextBatchRequest,
    FitImageBaselineRequest, AnalyzeImageBatchRequest,
    EmbeddingFitResponse,
    FitJointBaselineRequest, AnalyzeJointBatchRequest,
)
from db import crud
from drift.detector import compute_iqr_anomalies, DistributionDetector
from drift.embedding_detector import EmbeddingDriftDetector, HARD_MIN_SAMPLES, RECOMMENDED_MIN_SAMPLES
from drift.calibration import (
    CalibrationConfig, minimum_detectable_d_at_fit_time, recommended_batch_size,
    min_batch_size_at_floor, NEW_PROJECT_DEFAULT_DECISION_MODE,
)
from adapters.tabular import TabularAdapter
from adapters.text import TextAdapter
from adapters.image import ImageAdapter
from adapters.joint import JointAdapter, build_joint_classifier
from utils.profiler import profile_columns
from utils.validation import (
    ValidationError,
    validate_tabular_columns,
    warn_if_below_recommended_samples,
    validate_min_samples,
    validate_joint_records,
)
from drift.alerts import check_drift_alert
import os
from dotenv import load_dotenv

# Adapter-level exceptions that mean "the input was malformed," not "the
# server is broken" -- caught around every adapter .transform() call below
# and converted to a clean 400 instead of a raw 500. Verified, not assumed:
# PIL.UnidentifiedImageError is an OSError subclass (NOT a ValueError), so a
# bare `except ValueError` would miss the single most likely real-world
# failure (a corrupted or non-image file upload) -- see
# tests/test_ingestion_robustness.py.
BAD_INPUT_EXCEPTIONS = (ValueError, UnidentifiedImageError, binascii.Error, UnicodeDecodeError)

# Load environment variables from .env file
load_dotenv()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
if not GOOGLE_CLIENT_ID:
    raise ValueError("Missing GOOGLE_CLIENT_ID in environment variables")

COOKIE_KEY = os.getenv("COOKIE_KEY")
if not COOKIE_KEY:
    raise ValueError("Missing COOKIE_KEY in environment variables")

# ---------------------------------------------------------
# SECURITY: SESSION TOKENS DERIVED FROM GOOGLE LOGIN
# ---------------------------------------------------------
# The dashboard authenticates users via Google OAuth, then mints a
# short-lived session token locally (signed with this same COOKIE_KEY,
# shared between both services) rather than provisioning a separate
# long-lived API key. No Google API calls happen per-request here — we
# only verify the signature/expiry of a token our own frontend minted.
bearer_scheme = HTTPBearer(auto_error=True)

# ---------------------------------------------------------
# SECURITY: PERSONAL ACCESS TOKENS (Step 3a, 2026-09-30)
# ---------------------------------------------------------
# Lets a script call this API without a browser/Google login. Presented
# tokens are matched by their (non-secret) prefix, then verified with a
# constant-time hash comparison -- see auth/tokens.py. The plaintext
# token is never logged; only its SHA-256 hash is ever persisted (see
# db/crud.py's api_tokens table). Minted via scripts/create_token.py --
# no UI exists or is planned until the React frontend (see HANDOFF).
def _verify_pat(token: str) -> dict:
    prefix = parse_prefix(token)
    row = crud.get_api_token_by_prefix(prefix)
    if row is None or not verify_token_hash(token, row["token_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access Denied: Invalid access token."
        )
    if row["revoked"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access Denied: This access token has been revoked."
        )
    if row["expires_at"]:
        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) >= expires_at:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Access Denied: This access token has expired."
            )
    crud.touch_api_token_last_used(row["id"], datetime.now(timezone.utc).isoformat())
    return {
        "name": f"PAT:{row['name']}", "email": row["user_email"],
        "auth_type": "pat", "pat_scope": row["project_scope"],
    }


def verify_access(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> dict:
    """Validates either the session token minted by the dashboard after
    Google login, or a personal access token (dm_<prefix>_<secret>)."""
    token = credentials.credentials
    if parse_prefix(token) is not None:
        return _verify_pat(token)

    try:
        payload = jwt.decode(token, COOKIE_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access Denied: Invalid or expired session token."
        )

    email = payload.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access Denied: Malformed session token."
        )

    return {"name": payload.get("name", "User"), "email": email, "auth_type": "session", "pat_scope": None}


PROJECT_NAMESPACE_SEP = "::"


def _project_owner(project_id: str) -> Optional[str]:
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    cursor.execute("SELECT owner_email FROM projects WHERE id = ?", (project_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def _row_exists(internal_id: str) -> bool:
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM projects WHERE id = ?", (internal_id,))
    row = cursor.fetchone()
    conn.close()
    return row is not None


def _baseline_row_exists(project_id: str) -> bool:
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM baselines WHERE project_id = ?", (project_id,))
    row = cursor.fetchone()
    conn.close()
    return row is not None


def _internal_project_key(public_id: str, owner_email: str) -> str:
    return f"{owner_email}{PROJECT_NAMESPACE_SEP}{public_id}"


def _display_project_id(internal_id: str, owner_email: str) -> str:
    prefix = f"{owner_email}{PROJECT_NAMESPACE_SEP}"
    return internal_id[len(prefix):] if internal_id.startswith(prefix) else internal_id


def _resolve_project_key(public_id: str, client: dict) -> str:
    """Step 5 item 1 (2026-10-01): project ids are namespaced per owner
    so two different users' same-named project ("demo") are always
    different rows -- never a collision, and so /fit on it can never
    reveal whether ANOTHER user already has one, by construction (not by
    404 wording, which was cleanup item 4's narrower fix).

    Resolution, always scoped to THIS caller, never another user's row:
    1. Does the namespaced key (owner::public_id) already exist? Use it.
    2. Else, does a LEGACY plain-key row (public_id, no namespace) exist
       AND belong to this exact caller? Use it as-is -- pre-migration
       projects are never force-migrated, they just keep their old key.
       "Belongs to this caller" covers two cases: (a) a `projects` row
       exists for it with a matching owner_email, or (b) NO `projects`
       row exists at all but a `baselines` row does -- true legacy data
       that predates ownership tracking entirely (e.g. inserted before
       the projects table existed). Ownerless legacy rows are adoptable
       by whichever caller references their exact plain id, same as
       before this namespacing change -- unchanged behavior for them.
    3. Else: brand new project for this caller -- namespaced from the
       start. (If public_id happens to be taken by ANOTHER user under
       the old plain-key scheme, this deliberately does NOT touch that
       row -- the caller gets their own fresh namespaced one instead,
       indistinguishable from any other new-project creation.)"""
    namespaced = _internal_project_key(public_id, client["email"])
    if _row_exists(namespaced):
        return namespaced
    owner = _project_owner(public_id)
    if owner == client["email"]:
        return public_id
    if owner is None and _baseline_row_exists(public_id):
        return public_id
    return namespaced


def _enforce_ownership(project_id: str, client: dict) -> None:
    """Cross-user isolation (2026-09-30 hardening pass): a project that
    ALREADY EXISTS with a different owner_email is invisible to every
    other user, on both session-JWT and PAT auth -- 404, not 403, so a
    stranger cannot even tell the project exists. A project_id with no
    existing owner (brand new, or a legacy row from before ownership was
    tracked) is unaffected -- creation-on-first-/fit and pre-existing
    ownerless rows both still work exactly as before. This was previously
    left as a known, undocumented gap; the 2026-09-30 hardening pass
    closes it as the highest-priority item (an explicit, approved
    exception to "additive only" -- this changes existing behavior for
    session-JWT auth, deliberately).

    2026-10-01 cleanup: the detail message is the bare, generic "Baseline
    not found" -- NOT an f-string echoing project_id back (as it
    originally did here) -- because most downstream "genuinely doesn't
    exist yet" checks across these same endpoints use that exact generic
    wording (see get_baseline, get_logs, etc.), and a caller comparing
    THIS message against THAT one for the same endpoint must not be able
    to tell a foreign-owned project apart from one that was never
    created. (POST /fit is a known, accepted exception: a genuinely free
    project_id succeeds there (200, creates it) rather than 404ing, so
    the status code alone still distinguishes "taken" from "free" for
    that one endpoint -- inherent to its create-or-overwrite semantics,
    not a wording leak, and not fixed here.)"""
    owner = _project_owner(project_id)
    if owner is not None and owner != client["email"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Baseline not found")


def _require_existing_project(internal_id: str) -> None:
    """Step 5 item 1 regression guard: _resolve_project_key always
    returns EITHER a row the caller already owns OR a brand-new,
    not-yet-created namespaced key (so /fit can create it) -- so
    ownership checks on the resolved key alone can no longer tell 'yours'
    apart from 'nothing here yet'. Fine for /fit (create-or-use is the
    point); wrong for any endpoint that only reads/deletes/lists, since
    those would otherwise silently succeed against a caller's own
    nonexistent row instead of 404ing -- exactly the same information
    leak the hardening pass closed, reopened as a side effect of
    namespacing a resolved-but-absent key. Call this in every
    non-/fit handler that doesn't already fail closed by needing the
    baseline's contents to do its job (get_baseline, /analyze, /predict,
    /health already do, implicitly, by requiring crud.get_baseline(...)
    to return something)."""
    if not _row_exists(internal_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Baseline not found")


def verify_project_access(project_id: str, client: dict = Depends(verify_access)) -> dict:
    """Wraps verify_access with PAT project-scope enforcement AND
    cross-user ownership isolation (both session-JWT and PAT). A PAT's
    project_scope is checked against the PUBLIC project_id (what the
    token was scoped to), not the internal key. Resolves and attaches
    client["internal_project_id"] (Step 5 item 1) -- handlers use that
    for all storage calls, never the raw path param, so two users'
    same-named projects are always different rows. _enforce_ownership on
    the resolved key is a redundant safety net (resolution already
    guarantees it belongs to this caller or is brand new) kept for
    defense in depth."""
    if client.get("auth_type") == "pat":
        scope = client.get("pat_scope")
        if scope is not None and "*" not in scope and project_id not in scope:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This access token is not scoped to project '{project_id}'."
            )
    internal_id = _resolve_project_key(project_id, client)
    _enforce_ownership(internal_id, client)
    client["internal_project_id"] = internal_id
    return client


def verify_model_access(model_id: str, client: dict = Depends(verify_access)) -> dict:
    """Same as verify_project_access, for the legacy model_id path param
    name (DELETE /models/{model_id})."""
    if client.get("auth_type") == "pat":
        scope = client.get("pat_scope")
        if scope is not None and "*" not in scope and model_id not in scope:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This access token is not scoped to project '{model_id}'."
            )
    internal_id = _resolve_project_key(model_id, client)
    _enforce_ownership(internal_id, client)
    client["internal_project_id"] = internal_id
    return client


# ---------------------------------------------------------
# LIFESPAN & APP BOOTSTRAP
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize main tables via your crud module
    crud.init_db()
    yield

app = FastAPI(
    title="Drift Monitoring API",
    description="Real-time and batch machine learning anomaly detection engine.",
    version="1.0.0",
    lifespan=lifespan
)


@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    """
    Central conversion of structural ingestion problems (utils/validation.py)
    into a clean 422 naming exactly what's wrong -- so every endpoint can
    just call validate_*() and let a ValidationError propagate, instead of
    repeating try/except boilerplate at each of the eight fit/analyze
    handlers.
    """
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/baseline/{project_id}", tags=["Management"])
def get_baseline(project_id: str, client: dict = Depends(verify_project_access)):
    """Returns the IQR fences, feature types, and modality for a project."""
    import sqlite3
    import json
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    cursor.execute("SELECT iqr_fences, feature_types, modality FROM baselines WHERE project_id = ?",
                    (client["internal_project_id"],))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Baseline not found")
    fences = json.loads(row[0]) if row[0] else []   # list of dicts with 'feature_name', 'q1', 'q3'
    feature_types = json.loads(row[1]) if row[1] else {}  # dict of {feature_name: "continuous"|"categorical"}
    modality = row[2] or "tabular"
    return {"fences": fences, "feature_types": feature_types, "modality": modality}

@app.get("/logs/{project_id}", tags=["Management"])
def get_logs(project_id: str, client: dict = Depends(verify_project_access)):
    """Returns the recent logs for a project (last 1000)."""
    import sqlite3
    import json
    _require_existing_project(client["internal_project_id"])
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT input_data, score, is_ood FROM logs WHERE project_id = ? ORDER BY rowid DESC LIMIT 1000",
        (client["internal_project_id"],)
    )
    rows = cursor.fetchall()
    conn.close()
    logs = []
    for row in rows:
        logs.append({
            "input_data": json.loads(row[0]), 
            "score": row[1],
            "is_ood": row[2]
        })
    return logs


@app.get("/history/{project_id}", tags=["Analytics"])
def get_analysis_history(
    project_id: str,
    since: Optional[str] = None,
    until: Optional[str] = None,
    feature: Optional[str] = None,
    alert_only: bool = False,
    limit: int = 50,
    offset: int = 0,
    client: dict = Depends(verify_project_access),
):
    """Step 5 item 2: history of /analyze calls for a project -- statistics
    only, never raw production rows (see analysis_runs.feature_results).
    since/until compare lexicographically against each run's UTC ISO8601
    timestamp (safe: ISO8601 sorts identically to chronological order).
    feature narrows to runs that measured that feature and trims each
    run's feature_metrics down to just it; alert_only keeps only runs
    where system_alert fired. Pagination applies after every other
    filter; feature_time_series regroups the returned page by feature,
    oldest first, as a convenience over reassembling it from `runs`."""
    _require_existing_project(client["internal_project_id"])
    if not (1 <= limit <= 500):
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500.")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be >= 0.")

    runs = crud.get_analysis_runs(client["internal_project_id"], since=since, until=until, alert_only=alert_only)
    if feature is not None:
        runs = [r for r in runs if feature in r["feature_results"]]

    total = len(runs)
    page = runs[offset: offset + limit]

    def _narrowed_metrics(run):
        if feature is not None:
            return {feature: run["feature_results"][feature]} if feature in run["feature_results"] else {}
        return run["feature_results"]

    feature_time_series: Dict[str, List[dict]] = {}
    for run in reversed(page):  # oldest first within the series
        for feat_name, metrics in _narrowed_metrics(run).items():
            feature_time_series.setdefault(feat_name, []).append({
                "ts": run["ts"],
                "statistic": metrics.get("statistic"),
                "effect_size": metrics.get("effect_size"),
                "p_value_adjusted": metrics.get("p_value_adjusted"),
                "significant": metrics.get("significant"),
                "material": metrics.get("material"),
            })

    runs_out = [{
        "id": run["id"],
        "ts": run["ts"],
        "baseline_version": run["baseline_version"],
        "batch_size": run["batch_size"],
        "decision_mode": run["decision_mode"],
        "system_alert": run["system_alert"],
        "sustained_alert": run["sustained_alert"],
        "feature_metrics": _narrowed_metrics(run),
        "schema_report": run["schema_report"],
    } for run in page]

    return {
        "project_id": project_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "runs": runs_out,
        "feature_time_series": feature_time_series,
    }


# 1. Define the Expected Request Data
class ProfileRequest(BaseModel):
    reference_data: Dict[str, List[Any]]

# 2. Create the Endpoint
@app.post("/profile", tags=["Machine Learning"])
def profile_dataset(request: ProfileRequest, client: dict = Depends(verify_access)):
    """
    Accepts a sample of the dataset, converts it to a DataFrame, 
    and returns the smart schema mapping.
    """
    try:
        # Convert the incoming JSON dictionary back into a Pandas DataFrame
        df = pd.DataFrame(request.reference_data)
        
        # Pass it to your AI Profiler engine
        profiles = profile_columns(df)
        
        return profiles
    except Exception as e:
        # If anything goes wrong, return a clean 500 error instead of crashing
        raise HTTPException(status_code=500, detail=str(e))
    

    #endpoint 1 fit:
    
@app.post("/fit/{project_id}", response_model=FitBaselineResponse, tags=["Machine Learning"])
def fit_model_baseline(project_id: str, request: FitBaselineRequest, client: dict = Depends(verify_project_access)):
    """
    Upload historical training data. The system will profile it,
    calculate the IQR boundaries, and lock the baseline in the database.
    """
    # 0. Structural validation, before any DataFrame construction --
    # validated separately per dict (not combined) because reference_data
    # and categorical_data are legitimately allowed to differ in length
    # (the dashboard drops NaN rows independently per split; step 3 below
    # already handles that intentionally). What's NOT legitimate is
    # mismatched lengths *within* one dict -- that's what crashes
    # pd.DataFrame() below with a generic pandas error instead of a
    # specific one; see tests/test_ingestion_robustness.py.
    if request.reference_data:
        validate_tabular_columns(request.reference_data)
    if request.categorical_data:
        validate_tabular_columns(request.categorical_data)
    if not request.reference_data and not request.categorical_data:
        raise ValidationError("At least one of reference_data or categorical_data must be provided.")

    # 1. Create DataFrames from the request data
    continuous_df = pd.DataFrame(request.reference_data) if request.reference_data else pd.DataFrame()

    # 2. Get categorical data (may be None)
    categorical_dict = request.categorical_data or {}
    categorical_df = pd.DataFrame(categorical_dict) if categorical_dict else pd.DataFrame()
    
    # 3. Align lengths for profiling -- ONLY when BOTH sides actually have
    # data. Trimming to min(len(a), len(b)) is for the case where a
    # caller submits a genuinely shorter categorical split (the dashboard
    # does this when it drops NaN rows independently per split); it must
    # NOT apply when one side is simply absent (a fit with only
    # categorical_data and no reference_data, or vice versa) -- min(0, 60)
    # = 0 would silently truncate the present side to zero rows too, an
    # even more severe silent-drop than the column-level one (2026-09-30
    # hardening pass item 3 caught this while testing that fix: a
    # request.reference_data == {} fit with all-categorical data lost
    # every row, not just a column, and reported "0 continuous and 0
    # categorical features" with no error).
    if len(continuous_df) > 0 and len(categorical_df) > 0:
        min_len = min(len(continuous_df), len(categorical_df))
        continuous_df = continuous_df.iloc[:min_len]
        categorical_df = categorical_df.iloc[:min_len]
    
    # 4. Combine for profiling
    combined_df = pd.concat([continuous_df, categorical_df], axis=1) if len(categorical_df) > 0 else continuous_df
    
    # 5. Get detailed profiles from the generalised engine
    detailed_profiles = profile_columns(combined_df)

    # 6. ADAPTER: Route each column to the correct monitoring engine
    inferred_feature_types = {}
    for p in detailed_profiles:
        if p["monitor"] is True:
            inferred_feature_types[p["name"]] = "continuous"
        elif p["monitor"] == "Categorical":
            inferred_feature_types[p["name"]] = "categorical"

    # 6b. Explicit caller overrides (2026-09-30 hardening pass, item 3):
    # feature_types={col: "continuous"|"categorical"} lets a caller pin a
    # column's classification instead of relying on the profiler -- e.g.
    # a low-cardinality integer column the caller specifically wants
    # monitored as continuous. Columns not named here are unaffected.
    if request.feature_types:
        for col, ftype in request.feature_types.items():
            if ftype not in ("continuous", "categorical"):
                raise ValidationError(
                    f"feature_types['{col}'] must be 'continuous' or 'categorical', got '{ftype}'."
                )
            if col not in combined_df.columns:
                raise ValidationError(
                    f"feature_types names column '{col}', which is not present in reference_data "
                    f"or categorical_data."
                )
            inferred_feature_types[col] = ftype

    # 7. Split reference data by type. Looked up from a dict merging BOTH
    # reference_data and categorical_data (2026-09-30 hardening pass,
    # item 3) -- previously this only checked reference_data for a
    # "continuous"-classified column and categorical_data for a
    # "categorical"-classified one, so a column submitted under the
    # "wrong" dict for what the profiler (or an explicit feature_types
    # override) decided was silently dropped from the stored baseline
    # entirely, with no error. Merging first means a column is found
    # regardless of which dict the caller put it in.
    raw_values_by_col = {**dict(request.reference_data), **categorical_dict}
    continuous_features = {
        k: raw_values_by_col[k] for k in inferred_feature_types
        if inferred_feature_types[k] == "continuous" and k in raw_values_by_col
    }
    categorical_features = {
        k: raw_values_by_col[k] for k in inferred_feature_types
        if inferred_feature_types[k] == "categorical" and k in raw_values_by_col
    }

    # Safety net, not expected to ever trigger given the merge above: a
    # column the profiler/override classified but that isn't present in
    # EITHER submitted dict at all is a clear 422, never a silent drop.
    lost_cols = sorted(set(inferred_feature_types) - set(continuous_features) - set(categorical_features))
    if lost_cols:
        raise ValidationError(
            f"Column(s) {lost_cols} were classified but have no submitted values in either "
            f"reference_data or categorical_data -- this should not happen; please report it."
        )

    # 8. Resolve the calibration config.
    # - Caller provided one explicitly (even {}): resolve and store it,
    #   whether this is a new or existing project.
    # - Caller omitted it AND the project already exists (a re-fit): leave
    #   whatever it already has untouched (crud.insert_baseline's __UNSET__
    #   sentinel) -- an existing project's decision_mode never changes just
    #   because someone re-fit it without mentioning calibration.
    # - Caller omitted it AND this is a brand-new project: apply the
    #   new-project default explicitly (NEW_PROJECT_DEFAULT_DECISION_MODE =
    #   "calibrated", decided 2026-09-30) -- existing projects fit before
    #   this change keep calibration_config=NULL in the DB and so still
    #   resolve to CalibrationConfig's own "legacy" default; only a project
    #   created from this point on gets an explicit "calibrated" config
    #   written at creation time.
    return _resolve_and_persist_fit(
        project_id, inferred_feature_types, continuous_features, categorical_features,
        combined_df, request.calibration_config, client,
    )


def _resolve_and_persist_fit(
    project_id: str, inferred_feature_types: dict, continuous_features: dict,
    categorical_features: dict, combined_df: pd.DataFrame,
    calibration_config_request: Any, client: dict,
) -> FitBaselineResponse:
    """Shared tail of /fit/{project_id} (JSON body) and
    /fit/{project_id}/upload (multipart file) -- both endpoints build
    inferred_feature_types/continuous_features/categorical_features their
    own way (see each caller), then converge here: resolve the
    calibration config, persist the baseline, and build the response.

    project_id here is the PUBLIC id (used only in the response message);
    all storage calls use client["internal_project_id"] (Step 5 item 1:
    owner-namespaced, so two users' same-named projects never collide)."""
    internal_id = client["internal_project_id"]
    # 8. Resolve the calibration config.
    # - Caller provided one explicitly (even {}): resolve and store it,
    #   whether this is a new or existing project.
    # - Caller omitted it AND the project already exists (a re-fit): leave
    #   whatever it already has untouched (crud.insert_baseline's __UNSET__
    #   sentinel) -- an existing project's decision_mode never changes just
    #   because someone re-fit it without mentioning calibration.
    # - Caller omitted it AND this is a brand-new project: apply the
    #   new-project default explicitly (NEW_PROJECT_DEFAULT_DECISION_MODE =
    #   "calibrated", decided 2026-09-30) -- existing projects fit before
    #   this change keep calibration_config=NULL in the DB and so still
    #   resolve to CalibrationConfig's own "legacy" default; only a project
    #   created from this point on gets an explicit "calibrated" config
    #   written at creation time.
    is_new_project = crud.get_baseline(internal_id) is None
    if calibration_config_request is not None:
        resolved_calibration_config = CalibrationConfig.from_dict(calibration_config_request).to_dict()
    elif is_new_project:
        resolved_calibration_config = CalibrationConfig(decision_mode=NEW_PROJECT_DEFAULT_DECISION_MODE).to_dict()
    else:
        resolved_calibration_config = "__UNSET__"

    # 9. Persist baselines
    insert_kwargs = dict(
        project_id=internal_id,
        feature_types=inferred_feature_types,
        reference_data=continuous_features,
        categorical_data=categorical_features,
    )
    if resolved_calibration_config != "__UNSET__":
        insert_kwargs["calibration_config"] = resolved_calibration_config
    cleaning_summary = crud.insert_baseline(**insert_kwargs)

    crud.create_project(internal_id, f"Project {project_id}", client["email"])

    # 10. Minimum-detectable-D + configured floor per continuous feature, at
    # this reference's size -- API response field only, shown regardless of
    # decision_mode so a caller can see what this reference size can and
    # cannot detect before choosing floors (see docs/PROGRESS.md HANDOFF).
    active_config = CalibrationConfig.from_dict(
        resolved_calibration_config if resolved_calibration_config != "__UNSET__"
        else (crud.get_baseline(internal_id) or {}).get("calibration_config")
    )
    calibration_info = {}
    for feature, ftype in inferred_feature_types.items():
        if ftype == "continuous":
            m = len(continuous_features.get(feature, []))
            if m > 0:
                min_d = minimum_detectable_d_at_fit_time(m, active_config.alpha)
                floor = active_config.effect_floor_for(feature, "ks_d")
                too_small = bool(min_d > floor)
                calibration_info[feature] = FeatureCalibrationInfo(
                    minimum_detectable_d=min_d,
                    effect_floor=floor,
                    reference_too_small_for_floor=too_small,
                    recommended_batch_size=None if too_small else recommended_batch_size(m, floor, active_config.alpha),
                    min_batch_size_at_floor=None if too_small else min_batch_size_at_floor(m, floor, active_config.alpha),
                )
        else:
            floor = active_config.effect_floor_for(feature, "psi")
            calibration_info[feature] = FeatureCalibrationInfo(effect_floor=floor)

    message = (
        f"Baseline locked for project '{project_id}' by {client['name']}. "
        f"Monitoring {len(continuous_features)} continuous and "
        f"{len(categorical_features)} categorical features."
    )
    sample_warning = warn_if_below_recommended_samples(len(combined_df), label="rows")
    if sample_warning:
        message += f" Warning: {sample_warning}"
    if cleaning_summary:
        dropped_note = ", ".join(f"{col}: {info['dropped_non_numeric']} dropped" for col, info in cleaning_summary.items())
        message += f" Note: non-numeric values were dropped during cleaning ({dropped_note})."
    too_small = [f for f, info in calibration_info.items() if info.reference_too_small_for_floor]
    if too_small:
        message += (
            f" Warning: this reference is too small to reliably detect effects as small as the "
            f"configured floor for: {', '.join(too_small)} (see calibration_info)."
        )

    return FitBaselineResponse(
        status="success",
        message=message,
        inferred_feature_types=inferred_feature_types,
        cleaning_summary=cleaning_summary,
        calibration_info=calibration_info,
    )
MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024  # 200MB


@app.post("/fit/{project_id}/upload", response_model=FitBaselineResponse, tags=["Machine Learning"])
async def fit_model_baseline_upload(
    project_id: str,
    file: UploadFile = File(..., description="CSV or Parquet file of reference data."),
    calibration_config: Optional[str] = Form(None, description="Optional calibration_config, JSON-encoded."),
    feature_types: Optional[str] = Form(
        None, description="Optional {column: 'continuous'|'categorical'} override, JSON-encoded."
    ),
    client: dict = Depends(verify_project_access),
):
    """
    Multipart-upload counterpart to /fit/{project_id} for large reference
    sets that are awkward to inline as JSON -- CSV or Parquet, up to
    MAX_UPLOAD_SIZE_BYTES. Every column is profiled and classified as
    continuous or categorical from the uploaded frame itself (there is no
    caller-provided pre-split to reconcile against, unlike the JSON
    endpoint), then persisted via the same shared tail as /fit/{project_id}.
    """
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB upload limit.",
        )
    if not raw_bytes:
        raise ValidationError("Uploaded file is empty.")

    try:
        combined_df = ingest_readers.read_uploaded_file(file.filename or "upload.csv", raw_bytes)
    except (ValueError, ImportError) as e:
        raise ValidationError(f"Could not parse uploaded file: {e}")

    if combined_df.empty:
        raise ValidationError("Uploaded file parsed to zero rows.")

    detailed_profiles = profile_columns(combined_df)
    inferred_feature_types = {}
    for p in detailed_profiles:
        if p["monitor"] is True:
            inferred_feature_types[p["name"]] = "continuous"
        elif p["monitor"] == "Categorical":
            inferred_feature_types[p["name"]] = "categorical"

    if feature_types:
        try:
            parsed_feature_types = json.loads(feature_types)
        except json.JSONDecodeError as e:
            raise ValidationError(f"feature_types is not valid JSON: {e}")
        for col, ftype in parsed_feature_types.items():
            if ftype not in ("continuous", "categorical"):
                raise ValidationError(
                    f"feature_types['{col}'] must be 'continuous' or 'categorical', got '{ftype}'."
                )
            if col not in combined_df.columns:
                raise ValidationError(f"feature_types names column '{col}', which is not present in the uploaded file.")
            inferred_feature_types[col] = ftype

    # Split from the (possibly overridden) classification of the combined
    # frame itself -- so every classified column lands somewhere,
    # continuous or categorical; there is no caller-provided pre-split to
    # reconcile against here, unlike the JSON endpoint.
    continuous_features = {
        col: combined_df[col].tolist() for col, ftype in inferred_feature_types.items() if ftype == "continuous"
    }
    categorical_features = {
        col: [str(v) for v in combined_df[col].tolist()]
        for col, ftype in inferred_feature_types.items() if ftype == "categorical"
    }

    parsed_calibration_config = None
    if calibration_config:
        try:
            parsed_calibration_config = json.loads(calibration_config)
        except json.JSONDecodeError as e:
            raise ValidationError(f"calibration_config is not valid JSON: {e}")

    return _resolve_and_persist_fit(
        project_id, inferred_feature_types, continuous_features, categorical_features,
        combined_df, parsed_calibration_config, client,
    )


# ---------------------------------------------------------
# ENDPOINT 2: REAL-TIME ANOMALY TRIPWIRE
# ---------------------------------------------------------
@app.post("/predict/{project_id}", response_model=PredictResponse, tags=["Machine Learning"])
def predict_realtime_anomaly(project_id: str, request: PredictRequest, background_tasks: BackgroundTasks, client: dict = Depends(verify_project_access)):
    """
    Check a single incoming data point against the locked IQR boundaries.
    """
    internal_id = client["internal_project_id"]
    state = crud.get_baseline(internal_id)
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit first.")

    adapter = TabularAdapter()
    clean_data = adapter.clean_data(request.features)

    score, is_ood, feature_results = compute_iqr_anomalies(
        input_data=clean_data,
        baselines=state["iqr_fences"]
    )

    background_tasks.add_task(crud.insert_log, internal_id, clean_data, score, is_ood)
    
    return PredictResponse(
        is_anomaly=bool(is_ood),
        anomaly_score=score,
        feature_deviations=feature_results
    )


# ---------------------------------------------------------
# ENDPOINT 3: BATCH DRIFT DETECTION
# ---------------------------------------------------------
IDEMPOTENCY_KEY_TTL_DAYS = 7


@app.post("/analyze/{project_id}", response_model=AnalyzeBatchResponse, tags=["Analytics"])
def analyze_production_batch(
    project_id: str,
    request: AnalyzeBatchRequest,
    background_tasks: BackgroundTasks, # <-- 1. Inject BackgroundTasks
    client: dict = Depends(verify_project_access), # <-- 2. Fixed dependency
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """
    Analyze a large batch of recent production data using KS Tests and TVD
    to detect long-term mathematical drift.
    """
    return _run_tabular_analysis(project_id, request.production_data, client, background_tasks, idempotency_key)


def _run_tabular_analysis(project_id: str, production_data: dict, client: dict,
                           background_tasks: BackgroundTasks,
                           idempotency_key: Optional[str] = None) -> AnalyzeBatchResponse:
    """Shared tail of /analyze/{project_id} (JSON body) and
    /analyze/{project_id}/upload (multipart file) -- both converge on the
    same flat {feature_name: [values...]} shape. project_id is the
    PUBLIC id (used for display/email only); storage uses
    client["internal_project_id"] (Step 5 item 1)."""
    internal_id = client["internal_project_id"]
    state = crud.get_baseline(internal_id)
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit first.")

    # Step 5 item 3: idempotency, scoped per (internal, i.e. owner-
    # namespaced) project. Checked BEFORE running the detector, so a
    # replay costs nothing beyond the lookup. Same key + same payload ->
    # return the stored result, no new row. Same key + different payload
    # -> 409 (a caller reusing a key for new data is almost certainly a
    # bug, not an intentional replay). An expired (>7 day old) key is
    # treated as never having been used.
    payload_hash = hashlib.sha256(
        json.dumps(production_data, sort_keys=True, default=str).encode()
    ).hexdigest()
    if idempotency_key:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=IDEMPOTENCY_KEY_TTL_DAYS)).isoformat()
        existing = crud.find_analysis_run_by_idempotency_key(internal_id, idempotency_key, cutoff)
        if existing:
            if existing["payload_hash"] != payload_hash:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Idempotency-Key was already used with a different payload.",
                )
            return AnalyzeBatchResponse(
                system_alert_triggered=existing["system_alert"],
                feature_metrics=existing["feature_results"],
            )

    # None (every project fit before Step 2, or never given a config) ->
    # CalibrationConfig.from_dict(None) -> legacy -> DistributionDetector's
    # calibrated branch never runs -- byte-identical to pre-Step-2 behavior.
    calibration_config = CalibrationConfig.from_dict(state.get("calibration_config"))
    detector = DistributionDetector(p_value_threshold=0.05, calibration_config=calibration_config)

    detector.fit_baseline(
        reference_features=state["reference_data"],
        feature_types=state["feature_types"]
    )

    report = detector.analyze_production_window(production_data)

    # Step 5 item 2: one history row per /analyze call, statistics only
    # (never the raw production_data itself -- payload_hash is a one-way
    # digest of it, used above for item 3's idempotency replay detection,
    # not for recovering the data). baseline_version is hardcoded to 1
    # until item 5 adds real versioning; schema_report/sustained_alert
    # are NULL until items 4/6 populate them for real.
    batch_size = len(next(iter(production_data.values()), []))
    crud.insert_analysis_run(
        project=internal_id,
        baseline_version=1,
        ts=datetime.now(timezone.utc).isoformat(),
        batch_size=batch_size,
        idempotency_key=idempotency_key,
        payload_hash=payload_hash,
        decision_mode=calibration_config.decision_mode,
        system_alert=report["system_alert_triggered"],
        sustained_alert=None,
        feature_results=report["feature_metrics"],
        schema_report=None,
    )

    # ==========================================
    # NEW: ASYNCHRONOUS ALERT TRIGGER
    # ==========================================
    if report["system_alert_triggered"]:
        # We extract the features that actually drifted to include in the email
        drifted_features = [f for f, metrics in report["feature_metrics"].items() if metrics["drift_detected"]]

        # Add the email dispatch to the background queue so the API responds instantly
        background_tasks.add_task(
            send_drift_email,
            project_id=project_id,
            owner_email=client["email"],
            flagged_features=drifted_features
        )
    # ==========================================

    return AnalyzeBatchResponse(
        system_alert_triggered=report["system_alert_triggered"],
        feature_metrics=report["feature_metrics"]
    )


@app.post("/analyze/{project_id}/upload", response_model=AnalyzeBatchResponse, tags=["Analytics"])
async def analyze_production_batch_upload(
    project_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV or Parquet file of production data."),
    client: dict = Depends(verify_project_access),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """Multipart-upload counterpart to /analyze/{project_id} -- CSV or
    Parquet, up to MAX_UPLOAD_SIZE_BYTES."""
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB upload limit.",
        )
    if not raw_bytes:
        raise ValidationError("Uploaded file is empty.")

    try:
        df = ingest_readers.read_uploaded_file(file.filename or "upload.csv", raw_bytes)
    except (ValueError, ImportError) as e:
        raise ValidationError(f"Could not parse uploaded file: {e}")

    if df.empty:
        raise ValidationError("Uploaded file parsed to zero rows.")

    production_data = {col: df[col].tolist() for col in df.columns}
    return _run_tabular_analysis(project_id, production_data, client, background_tasks, idempotency_key)


# ---------------------------------------------------------
# ENDPOINTS: TEXT DRIFT MONITORING (v2.0 — Domain Classifier Test)
# ---------------------------------------------------------
@app.post("/fit/{project_id}/text", response_model=EmbeddingFitResponse, tags=["Machine Learning"])
def fit_text_baseline(project_id: str, request: FitTextBaselineRequest, client: dict = Depends(verify_project_access)):
    """Embeds a baseline batch of text and locks it as the reference distribution."""
    sample_warning = validate_min_samples(
        len(request.reference_texts), HARD_MIN_SAMPLES, RECOMMENDED_MIN_SAMPLES, "reference text"
    )

    try:
        embeddings = TextAdapter().transform(request.reference_texts)
    except BAD_INPUT_EXCEPTIONS as e:
        raise HTTPException(status_code=400, detail=f"Could not embed reference_texts: {e}")

    crud.insert_embedding_baseline(
        project_id=client["internal_project_id"],
        modality="text",
        embeddings=embeddings,
        model_name=TextAdapter.model_name,
    )
    crud.create_project(client["internal_project_id"], f"Project {project_id}", client["email"])

    message = f"Text baseline locked for project '{project_id}' with {len(embeddings)} reference samples."
    if sample_warning:
        message += f" Warning: {sample_warning}"

    return EmbeddingFitResponse(status="success", message=message)


@app.post("/analyze/{project_id}/text", response_model=AnalyzeBatchResponse, tags=["Analytics"])
def analyze_text_batch(
    project_id: str,
    request: AnalyzeTextBatchRequest,
    background_tasks: BackgroundTasks,
    client: dict = Depends(verify_project_access)
):
    """Compares a production text batch against the locked text baseline via the Domain Classifier Test."""
    state = crud.get_baseline(client["internal_project_id"])
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit/{project_id}/text first.")
    if state["modality"] != "text":
        raise HTTPException(status_code=400, detail=f"Project '{project_id}' has a '{state['modality']}' baseline, not 'text'.")

    validate_min_samples(len(request.production_texts), HARD_MIN_SAMPLES, RECOMMENDED_MIN_SAMPLES, "production text")

    try:
        cur_embeddings = TextAdapter().transform(request.production_texts)
        result = EmbeddingDriftDetector().analyze(state["embedding_reference"], cur_embeddings)
    except BAD_INPUT_EXCEPTIONS as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result["drift_detected"]:
        background_tasks.add_task(
            send_drift_email,
            project_id=project_id,
            owner_email=client["email"],
            flagged_features=["embedding_drift"]
        )

    return AnalyzeBatchResponse(
        system_alert_triggered=result["drift_detected"],
        feature_metrics={"embedding_drift": result}
    )


# ---------------------------------------------------------
# ENDPOINTS: IMAGE DRIFT MONITORING (v2.0 — Domain Classifier Test)
# ---------------------------------------------------------
@app.post("/fit/{project_id}/image", response_model=EmbeddingFitResponse, tags=["Machine Learning"])
def fit_image_baseline(project_id: str, request: FitImageBaselineRequest, client: dict = Depends(verify_project_access)):
    """Embeds a baseline batch of images and locks it as the reference distribution."""
    sample_warning = validate_min_samples(
        len(request.reference_images), HARD_MIN_SAMPLES, RECOMMENDED_MIN_SAMPLES, "reference image"
    )

    try:
        embeddings = ImageAdapter().transform(request.reference_images)
    except BAD_INPUT_EXCEPTIONS as e:
        raise HTTPException(status_code=400, detail=f"Could not decode reference_images: {e}")

    crud.insert_embedding_baseline(
        project_id=client["internal_project_id"],
        modality="image",
        embeddings=embeddings,
        model_name=ImageAdapter.model_name,
    )
    crud.create_project(client["internal_project_id"], f"Project {project_id}", client["email"])

    message = f"Image baseline locked for project '{project_id}' with {len(embeddings)} reference samples."
    if sample_warning:
        message += f" Warning: {sample_warning}"

    return EmbeddingFitResponse(status="success", message=message)


@app.post("/analyze/{project_id}/image", response_model=AnalyzeBatchResponse, tags=["Analytics"])
def analyze_image_batch(
    project_id: str,
    request: AnalyzeImageBatchRequest,
    background_tasks: BackgroundTasks,
    client: dict = Depends(verify_project_access)
):
    """Compares a production image batch against the locked image baseline via the Domain Classifier Test."""
    state = crud.get_baseline(client["internal_project_id"])
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit/{project_id}/image first.")
    if state["modality"] != "image":
        raise HTTPException(status_code=400, detail=f"Project '{project_id}' has a '{state['modality']}' baseline, not 'image'.")

    validate_min_samples(len(request.production_images), HARD_MIN_SAMPLES, RECOMMENDED_MIN_SAMPLES, "production image")

    try:
        cur_embeddings = ImageAdapter().transform(request.production_images)
        result = EmbeddingDriftDetector().analyze(state["embedding_reference"], cur_embeddings)
    except BAD_INPUT_EXCEPTIONS as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result["drift_detected"]:
        background_tasks.add_task(
            send_drift_email,
            project_id=project_id,
            owner_email=client["email"],
            flagged_features=["embedding_drift"]
        )

    return AnalyzeBatchResponse(
        system_alert_triggered=result["drift_detected"],
        feature_metrics={"embedding_drift": result}
    )


# ---------------------------------------------------------
# ENDPOINTS: JOINT MULTIMODAL CONTEXT DRIFT MONITORING
# ---------------------------------------------------------
@app.post("/fit/{project_id}/joint", response_model=EmbeddingFitResponse, tags=["Machine Learning"])
def fit_joint_baseline(project_id: str, request: FitJointBaselineRequest, client: dict = Depends(verify_project_access)):
    """Embeds a baseline batch of joint records (tabular + text + image) and locks it as the reference distribution."""
    records = [r.model_dump() for r in request.reference_records]
    validate_joint_records(records)
    sample_warning = validate_min_samples(len(records), HARD_MIN_SAMPLES, RECOMMENDED_MIN_SAMPLES, "reference record")

    adapter = JointAdapter()
    try:
        tabular_stats = adapter.fit_tabular_schema(records)
        embeddings = adapter.transform(records, tabular_stats)
    except BAD_INPUT_EXCEPTIONS as e:
        raise HTTPException(status_code=400, detail=str(e))

    crud.insert_joint_baseline(
        project_id=client["internal_project_id"],
        embeddings=embeddings,
        tabular_stats=tabular_stats,
        model_name="joint-v1",
    )
    crud.create_project(client["internal_project_id"], f"Project {project_id}", client["email"])

    return EmbeddingFitResponse(
        status="success",
        message=(
            f"Joint baseline locked for project '{project_id}' with {len(embeddings)} reference records."
            + (f" Warning: {sample_warning}" if sample_warning else "")
        )
    )


@app.post("/analyze/{project_id}/joint", response_model=AnalyzeBatchResponse, tags=["Analytics"])
def analyze_joint_batch(
    project_id: str,
    request: AnalyzeJointBatchRequest,
    background_tasks: BackgroundTasks,
    client: dict = Depends(verify_project_access)
):
    """Compares a production batch of joint records against the locked joint baseline via the Domain Classifier Test."""
    state = crud.get_baseline(client["internal_project_id"])
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit/{project_id}/joint first.")
    if state["modality"] != "joint":
        raise HTTPException(status_code=400, detail=f"Project '{project_id}' has a '{state['modality']}' baseline, not 'joint'.")

    records = [r.model_dump() for r in request.production_records]
    validate_joint_records(records)
    validate_min_samples(len(records), HARD_MIN_SAMPLES, RECOMMENDED_MIN_SAMPLES, "production record")
    tabular_stats = state["feature_types"]

    try:
        cur_embeddings = JointAdapter().transform(records, tabular_stats)
        result = EmbeddingDriftDetector(classifier=build_joint_classifier()).analyze(state["embedding_reference"], cur_embeddings)
    except BAD_INPUT_EXCEPTIONS as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result["drift_detected"]:
        background_tasks.add_task(
            send_drift_email,
            project_id=project_id,
            owner_email=client["email"],
            flagged_features=["embedding_drift"]
        )

    return AnalyzeBatchResponse(
        system_alert_triggered=result["drift_detected"],
        feature_metrics={"embedding_drift": result}
    )

# ---------------------------------------------------------
# ENDPOINT 4: SYSTEM HEALTH CHECK (BURST ALERTS)
# ---------------------------------------------------------
@app.get("/health/{project_id}", response_model=HealthCheckResponse, tags=["Analytics"])
def check_system_health(project_id: str, client_name: str = Depends(verify_project_access)):
    """
    Ping this endpoint (e.g., every 60 seconds via a cron job or dashboard) 
    to see if the system is currently experiencing a wave of real-time anomalies.
    """
    internal_id = client_name["internal_project_id"]
    state = crud.get_baseline(internal_id)
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit first.")

    is_alert, ratio = check_drift_alert(internal_id, window_size=10, threshold=0.3)
    status_message = "Degraded" if is_alert else "Healthy"
    
    return HealthCheckResponse(
        system_status=status_message,
        is_burst_alert=is_alert,
        drift_ratio=ratio
    )
    

@app.get("/projects", tags=["Management"])
def list_projects(client: dict = Depends(verify_access)):
    """
    Returns a list of project IDs for the authenticated user. Internally
    stored ids may be owner-namespaced (Step 5 item 1, "email::name") --
    displayed here with that prefix stripped, so a user always sees just
    the name they created it with, never their own email echoed back.
    """
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM projects WHERE owner_email = ?", (client["email"],))
    rows = cursor.fetchall()
    conn.close()

    projects = [_display_project_id(row[0], client["email"]) for row in rows]
    return {"projects": projects}


# ---------------------------------------------------------
# ENDPOINT 5: HARD DELETE PROJECT
# ---------------------------------------------------------
def _delete_project_data(project_id: str) -> str:
    import urllib.parse
    import sqlite3

    clean_project_id = urllib.parse.unquote(project_id)

    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM projects WHERE id = ?", (clean_project_id,))
        cursor.execute("DELETE FROM baselines WHERE project_id = ?", (clean_project_id,))
        cursor.execute("DELETE FROM logs WHERE project_id = ?", (clean_project_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()
    return clean_project_id


@app.delete("/projects/{project_id}", tags=["Management"])
def delete_project(project_id: str, client: dict = Depends(verify_project_access)):
    """Permanently deletes a project and all its associated baseline/log data."""
    _require_existing_project(client["internal_project_id"])
    _delete_project_data(client["internal_project_id"])
    return {"status": "success", "message": f"Project '{project_id}' completely wiped."}


@app.delete("/models/{model_id}", tags=["Management"], deprecated=True)
def delete_model(model_id: str, client: dict = Depends(verify_model_access)):
    """Deprecated alias for DELETE /projects/{project_id} -- kept for
    backward compatibility, same behavior, not removed."""
    _require_existing_project(client["internal_project_id"])
    _delete_project_data(client["internal_project_id"])
    return {"status": "success", "message": f"Model '{model_id}' completely wiped."}