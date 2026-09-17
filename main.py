import sqlite3
import jwt
import pandas as pd
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from pydantic import BaseModel
from drift.alerts import send_drift_email
# Importing custom modules
from models import (
    FitBaselineRequest, FitBaselineResponse,
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
from drift.embedding_detector import EmbeddingDriftDetector
from adapters.tabular import TabularAdapter
from adapters.text import TextAdapter
from adapters.image import ImageAdapter
from adapters.joint import JointAdapter, build_joint_classifier
from utils.profiler import profile_columns
from drift.alerts import check_drift_alert
import os
from dotenv import load_dotenv

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

def verify_access(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> dict:
    """Validates the session token minted by the dashboard after Google login."""
    try:
        payload = jwt.decode(credentials.credentials, COOKIE_KEY, algorithms=["HS256"])
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

    return {"name": payload.get("name", "User"), "email": email}


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

@app.get("/baseline/{project_id}", tags=["Management"])
def get_baseline(project_id: str, client: dict = Depends(verify_access)):
    """Returns the IQR fences, feature types, and modality for a project."""
    import sqlite3
    import json
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    cursor.execute("SELECT iqr_fences, feature_types, modality FROM baselines WHERE project_id = ?", (project_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Baseline not found")
    fences = json.loads(row[0]) if row[0] else []   # list of dicts with 'feature_name', 'q1', 'q3'
    feature_types = json.loads(row[1]) if row[1] else {}  # dict of {feature_name: "continuous"|"categorical"}
    modality = row[2] or "tabular"
    return {"fences": fences, "feature_types": feature_types, "modality": modality}

@app.get("/logs/{project_id}", tags=["Management"])
def get_logs(project_id: str, client: dict = Depends(verify_access)):
    """Returns the recent logs for a project (last 1000)."""
    import sqlite3
    import json
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT input_data, score, is_ood FROM logs WHERE project_id = ? ORDER BY rowid DESC LIMIT 1000",
        (project_id,)
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
def fit_model_baseline(project_id: str, request: FitBaselineRequest, client: dict = Depends(verify_access)):
    """
    Upload historical training data. The system will profile it, 
    calculate the IQR boundaries, and lock the baseline in the database.
    """
    
    # 1. Create DataFrames from the request data
    continuous_df = pd.DataFrame(request.reference_data) if request.reference_data else pd.DataFrame()
    
    # 2. Get categorical data (may be None)
    categorical_dict = request.categorical_data or {}
    categorical_df = pd.DataFrame(categorical_dict) if categorical_dict else pd.DataFrame()
    
    # 3. FIX: Align lengths for profiling
    # Get the minimum length between continuous and categorical data
    min_len = min(len(continuous_df), len(categorical_df)) if len(categorical_df) > 0 else len(continuous_df)
    
    # Trim both DataFrames to the same length
    continuous_df = continuous_df.iloc[:min_len]
    categorical_df = categorical_df.iloc[:min_len] if len(categorical_df) > 0 else categorical_df
    
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
    
    # 7. Split reference data by type
    continuous_features = {
        k: v for k, v in request.reference_data.items()
        if inferred_feature_types.get(k) == "continuous"
    }
    categorical_features = {
        k: v for k, v in categorical_dict.items()
        if inferred_feature_types.get(k) == "categorical"
    }
    
    # 8. Persist baselines
    crud.insert_baseline(
        project_id=project_id,
        feature_types=inferred_feature_types,
        reference_data=continuous_features,
        categorical_data=categorical_features
    )
    
    crud.create_project(project_id, f"Project {project_id}", client["email"])
    
    return FitBaselineResponse(
        status="success",
        message=f"Baseline locked for project '{project_id}' by {client['name']}. "
                f"Monitoring {len(continuous_features)} continuous and "
                f"{len(categorical_features)} categorical features.",
        inferred_feature_types=inferred_feature_types
    )
# ---------------------------------------------------------
# ENDPOINT 2: REAL-TIME ANOMALY TRIPWIRE
# ---------------------------------------------------------
@app.post("/predict/{project_id}", response_model=PredictResponse, tags=["Machine Learning"])
def predict_realtime_anomaly(project_id: str, request: PredictRequest, background_tasks: BackgroundTasks, client: dict = Depends(verify_access)):
    """
    Check a single incoming data point against the locked IQR boundaries.
    """
    state = crud.get_baseline(project_id)
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit first.")
        
    adapter = TabularAdapter()
    clean_data = adapter.clean_data(request.features)
    
    score, is_ood, feature_results = compute_iqr_anomalies(
        input_data=clean_data, 
        baselines=state["iqr_fences"]
    )
    
    background_tasks.add_task(crud.insert_log, project_id, clean_data, score, is_ood)
    
    return PredictResponse(
        is_anomaly=bool(is_ood),
        anomaly_score=score,
        feature_deviations=feature_results
    )


# ---------------------------------------------------------
# ENDPOINT 3: BATCH DRIFT DETECTION
# ---------------------------------------------------------
@app.post("/analyze/{project_id}", response_model=AnalyzeBatchResponse, tags=["Analytics"])
def analyze_production_batch(
    project_id: str, 
    request: AnalyzeBatchRequest, 
    background_tasks: BackgroundTasks, # <-- 1. Inject BackgroundTasks
    client: dict = Depends(verify_access) # <-- 2. Fixed dependency
):
    """
    Analyze a large batch of recent production data using KS Tests and TVD 
    to detect long-term mathematical drift.
    """
    state = crud.get_baseline(project_id)
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit first.")
        
    detector = DistributionDetector(p_value_threshold=0.05)
    
    detector.fit_baseline(
        reference_features=state["reference_data"], 
        feature_types=state["feature_types"]
    )
    
    report = detector.analyze_production_window(request.production_data)
    
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

# ---------------------------------------------------------
# ENDPOINTS: TEXT DRIFT MONITORING (v2.0 — Domain Classifier Test)
# ---------------------------------------------------------
@app.post("/fit/{project_id}/text", response_model=EmbeddingFitResponse, tags=["Machine Learning"])
def fit_text_baseline(project_id: str, request: FitTextBaselineRequest, client: dict = Depends(verify_access)):
    """Embeds a baseline batch of text and locks it as the reference distribution."""
    embeddings = TextAdapter().transform(request.reference_texts)

    crud.insert_embedding_baseline(
        project_id=project_id,
        modality="text",
        embeddings=embeddings,
        model_name=TextAdapter.model_name,
    )
    crud.create_project(project_id, f"Project {project_id}", client["email"])

    return EmbeddingFitResponse(
        status="success",
        message=f"Text baseline locked for project '{project_id}' with {len(embeddings)} reference samples."
    )


@app.post("/analyze/{project_id}/text", response_model=AnalyzeBatchResponse, tags=["Analytics"])
def analyze_text_batch(
    project_id: str,
    request: AnalyzeTextBatchRequest,
    background_tasks: BackgroundTasks,
    client: dict = Depends(verify_access)
):
    """Compares a production text batch against the locked text baseline via the Domain Classifier Test."""
    state = crud.get_baseline(project_id)
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit/{project_id}/text first.")
    if state["modality"] != "text":
        raise HTTPException(status_code=400, detail=f"Project '{project_id}' has a '{state['modality']}' baseline, not 'text'.")

    cur_embeddings = TextAdapter().transform(request.production_texts)
    try:
        result = EmbeddingDriftDetector().analyze(state["embedding_reference"], cur_embeddings)
    except ValueError as e:
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
def fit_image_baseline(project_id: str, request: FitImageBaselineRequest, client: dict = Depends(verify_access)):
    """Embeds a baseline batch of images and locks it as the reference distribution."""
    embeddings = ImageAdapter().transform(request.reference_images)

    crud.insert_embedding_baseline(
        project_id=project_id,
        modality="image",
        embeddings=embeddings,
        model_name=ImageAdapter.model_name,
    )
    crud.create_project(project_id, f"Project {project_id}", client["email"])

    return EmbeddingFitResponse(
        status="success",
        message=f"Image baseline locked for project '{project_id}' with {len(embeddings)} reference samples."
    )


@app.post("/analyze/{project_id}/image", response_model=AnalyzeBatchResponse, tags=["Analytics"])
def analyze_image_batch(
    project_id: str,
    request: AnalyzeImageBatchRequest,
    background_tasks: BackgroundTasks,
    client: dict = Depends(verify_access)
):
    """Compares a production image batch against the locked image baseline via the Domain Classifier Test."""
    state = crud.get_baseline(project_id)
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit/{project_id}/image first.")
    if state["modality"] != "image":
        raise HTTPException(status_code=400, detail=f"Project '{project_id}' has a '{state['modality']}' baseline, not 'image'.")

    cur_embeddings = ImageAdapter().transform(request.production_images)
    try:
        result = EmbeddingDriftDetector().analyze(state["embedding_reference"], cur_embeddings)
    except ValueError as e:
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
def fit_joint_baseline(project_id: str, request: FitJointBaselineRequest, client: dict = Depends(verify_access)):
    """Embeds a baseline batch of joint records (tabular + text + image) and locks it as the reference distribution."""
    records = [r.model_dump() for r in request.reference_records]
    adapter = JointAdapter()

    try:
        tabular_stats = adapter.fit_tabular_schema(records)
        embeddings = adapter.transform(records, tabular_stats)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    crud.insert_joint_baseline(
        project_id=project_id,
        embeddings=embeddings,
        tabular_stats=tabular_stats,
        model_name="joint-v1",
    )
    crud.create_project(project_id, f"Project {project_id}", client["email"])

    return EmbeddingFitResponse(
        status="success",
        message=f"Joint baseline locked for project '{project_id}' with {len(embeddings)} reference records."
    )


@app.post("/analyze/{project_id}/joint", response_model=AnalyzeBatchResponse, tags=["Analytics"])
def analyze_joint_batch(
    project_id: str,
    request: AnalyzeJointBatchRequest,
    background_tasks: BackgroundTasks,
    client: dict = Depends(verify_access)
):
    """Compares a production batch of joint records against the locked joint baseline via the Domain Classifier Test."""
    state = crud.get_baseline(project_id)
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit/{project_id}/joint first.")
    if state["modality"] != "joint":
        raise HTTPException(status_code=400, detail=f"Project '{project_id}' has a '{state['modality']}' baseline, not 'joint'.")

    records = [r.model_dump() for r in request.production_records]
    tabular_stats = state["feature_types"]

    try:
        cur_embeddings = JointAdapter().transform(records, tabular_stats)
        result = EmbeddingDriftDetector(classifier=build_joint_classifier()).analyze(state["embedding_reference"], cur_embeddings)
    except ValueError as e:
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
def check_system_health(project_id: str, client_name: str = Depends(verify_access)):
    """
    Ping this endpoint (e.g., every 60 seconds via a cron job or dashboard) 
    to see if the system is currently experiencing a wave of real-time anomalies.
    """
    state = crud.get_baseline(project_id)
    if not state:
        raise HTTPException(status_code=404, detail="Baseline not found. Call /fit first.")
        
    is_alert, ratio = check_drift_alert(project_id, window_size=10, threshold=0.3)
    status_message = "Degraded" if is_alert else "Healthy"
    
    return HealthCheckResponse(
        system_status=status_message,
        is_burst_alert=is_alert,
        drift_ratio=ratio
    )
    

@app.get("/projects", tags=["Management"])
def list_projects(client: dict = Depends(verify_access)):
    """
    Returns a list of project IDs for the authenticated user.
    """
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM projects WHERE owner_email = ?", (client["email"],))
    rows = cursor.fetchall()
    conn.close()
    
    projects = [row[0] for row in rows]
    return {"projects": projects}


# ---------------------------------------------------------
# ENDPOINT 5: HARD DELETE MODEL
# ---------------------------------------------------------
@app.delete("/models/{model_id}", tags=["Management"])
def delete_model(model_id: str, client: dict = Depends(verify_access)):
    """Permanently deletes a model and all its associated baseline/log data."""
    import urllib.parse
    import sqlite3
    
    clean_model_id = urllib.parse.unquote(model_id)
    
    # Connect to the database and wipe the ghost data
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    
    try:
        
        cursor.execute("DELETE FROM projects WHERE id = ?", (clean_model_id,))
        
        # These tables correctly use 'project_id'
        cursor.execute("DELETE FROM baselines WHERE project_id = ?", (clean_model_id,))
        cursor.execute("DELETE FROM logs WHERE project_id = ?", (clean_model_id,))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()
        
    return {"status": "success", "message": f"Model '{clean_model_id}' completely wiped."}