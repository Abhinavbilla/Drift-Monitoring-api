import sqlite3
import pandas as pd
import secrets
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, status
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
from pydantic import BaseModel
from drift.alerts import send_drift_email
# Importing custom modules
from models import (
    FitBaselineRequest, FitBaselineResponse, 
    PredictRequest, PredictResponse, 
    AnalyzeBatchRequest, AnalyzeBatchResponse,
    HealthCheckResponse  
)
from db import crud
from drift.detector import compute_iqr_anomalies, DistributionDetector
from adapters.tabular import TabularAdapter
from utils.profiler import profile_columns
from drift.alerts import check_drift_alert  
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import os
from dotenv import load_dotenv
from fastapi import Request

# Load environment variables from .env file
load_dotenv()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
if not GOOGLE_CLIENT_ID:
    raise ValueError("Missing GOOGLE_CLIENT_ID in environment variables")


def verify_google_token(token: str) -> dict:
    """Verify the Google ID token and return the user info."""
    try:
        info = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            audience=GOOGLE_CLIENT_ID
        )
        if info.get('iss') not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')
        return info
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid Google token: {str(e)}"
        )

# ---------------------------------------------------------
# SECURITY VAULT CONFIGURATION
# ---------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_access(api_key: str = Security(api_key_header)):
    """Validates the incoming API Key against the SQLite database."""
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    
    # UPDATED: Select owner_email as well!
    cursor.execute("SELECT owner_name, owner_email, is_active FROM api_keys WHERE key = ?", (api_key,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not row[2]: # row[2] is now is_active
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access Denied: Invalid, missing, or revoked Security Key."
        )
        
    # Return a dictionary with both the name and email
    return {"name": row[0], "email": row[1]}


# ---------------------------------------------------------
# LIFESPAN & APP BOOTSTRAP
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize main tables via your crud module
    crud.init_db()
    
    # Dynamically ensure the security table exists without needing to edit crud.py
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    # Updated CREATE TABLE command to include owner_email
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY, 
            owner_name TEXT, 
            owner_email TEXT, 
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()
    
    yield

app = FastAPI(
    title="Drift Monitoring API",
    description="Real-time and batch machine learning anomaly detection engine.",
    version="1.0.0",
    lifespan=lifespan
)


# ---------------------------------------------------------
# ENDPOINT 0: GENERATE SECURE KEYS (ADMIN)
# ---------------------------------------------------------
@app.post("/admin/generate_key", tags=["Security"])
def generate_key(owner_name: str):
    """Generates a highly secure, unique API key for a new user or team."""
    new_key = f"sk-drift-{secrets.token_hex(16)}"
    
    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO api_keys (key, owner_name) VALUES (?, ?)", (new_key, owner_name))
    conn.commit()
    conn.close()
    
    return {
        "owner": owner_name, 
        "api_key": new_key, 
        "message": "Save this key securely! It will not be shown again."
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
    


class RegisterRequest(BaseModel):
    owner_name: str
    owner_email: str

@app.post("/register", tags=["Security"])
def register_user(request: Request, payload: RegisterRequest):
    # Only allow requests from localhost
    client_host = request.client.host
    if client_host not in ["127.0.0.1", "localhost", "::1"]:
        raise HTTPException(status_code=403, detail="Registration only allowed from localhost")
    
    owner_email = payload.owner_email
    owner_name = payload.owner_name

    conn = sqlite3.connect("drift.db")
    cursor = conn.cursor()
    cursor.execute("SELECT key FROM api_keys WHERE owner_email = ?", (owner_email,))
    existing = cursor.fetchone()

    if existing:
        api_key = existing[0]
    else:
        new_key = f"sk-drift-{secrets.token_hex(16)}"
        cursor.execute(
            "INSERT INTO api_keys (key, owner_name, owner_email) VALUES (?, ?, ?)",
            (new_key, owner_name, owner_email)
        )
        api_key = new_key

    conn.commit()
    conn.close()
    return {"message": "Registration successful", "api_key": api_key}


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