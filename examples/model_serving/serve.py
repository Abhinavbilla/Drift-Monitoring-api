"""
Step 3e end-to-end example, part 2: serve the trained model with FastAPI,
logging POST-PREPROCESSING input features asynchronously so logging never
blocks the prediction response. This is a separate small app from the
Drift Monitoring API itself (its own port) -- it represents a team's own
model-serving service, which happens to also log the features it fed the
model, for a separate process (scheduled_job.py) to pick up later and
send to the Drift Monitoring API.

"Post-preprocessing" here means: whatever the model actually received as
input (the assembled feature vector), not the raw request body -- in this
demo preprocessing is a no-op (the request fields already match the
model's input schema), but the logging call is placed after feature
assembly, not before, so it generalizes to a real pipeline with actual
preprocessing (imputation, encoding, scaling, ...) in between.

Run:
    python -m uvicorn examples.model_serving.serve:app --port 8001
"""

import json
import os
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "artifacts", "model.pkl")
LOG_PATH = os.path.join(HERE, "logs", "served_features.jsonl")

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(MODEL_PATH)
    _state["model"] = bundle["model"]
    _state["input_features"] = bundle["input_features"]
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    yield


app = FastAPI(title="Model Serving Example", lifespan=lifespan)


class PredictionRequest(BaseModel):
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    gender_id: int


class PredictionResponse(BaseModel):
    predicted_trip_duration_seconds: float


def _log_features(features: dict):
    """Runs in the background (after the response is already sent) --
    appends one JSON line per prediction. A real deployment would ship
    this to a log aggregator instead of a local file; the shape (one
    record per prediction, post-preprocessing feature values) is what
    matters for this example."""
    record = {**features, "logged_at": time.time()}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, background_tasks: BackgroundTasks):
    features = req.model_dump()
    X = pd.DataFrame([features])[_state["input_features"]]
    pred = float(_state["model"].predict(X)[0])

    # Logging is scheduled as a background task -- it runs AFTER this
    # function returns its response, so it never adds latency to the
    # prediction itself.
    background_tasks.add_task(_log_features, features)

    return PredictionResponse(predicted_trip_duration_seconds=pred)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": "model" in _state}
