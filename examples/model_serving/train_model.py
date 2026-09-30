"""
Step 3e end-to-end example, part 1: train a small scikit-learn model on
the Jan-Mar Citi Bike reference split (tests/splits/reference_n5000.csv,
the same 5,000-row sample used throughout this project's Step 1/2
validation work).

Target: trip_duration (regression). Inputs: pickup_longitude,
pickup_latitude, dropoff_longitude, dropoff_latitude, gender_id -- these
five are the model's INPUT features, and so are exactly what gets
monitored for drift in this example (trip_duration is the prediction
target, never an input; "month" is a split artifact, not a real
predictive feature, and is excluded).

Run:
    python examples/model_serving/train_model.py
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..")
REFERENCE_CSV = os.path.join(REPO_ROOT, "tests", "splits", "reference_n5000.csv")
MODEL_PATH = os.path.join(HERE, "artifacts", "model.pkl")

INPUT_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "gender_id"]
TARGET = "trip_duration"


def main():
    df = pd.read_csv(REFERENCE_CSV)
    X = df[INPUT_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Trained on {len(X_train)} rows, held out {len(X_test)}. Test MAE: {mae:.1f} seconds "
          f"(trip_duration mean: {y.mean():.1f}s, median: {y.median():.1f}s -- a small demo model, "
          f"not tuned for accuracy).")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "input_features": INPUT_FEATURES, "target": TARGET}, MODEL_PATH)
    print(f"Wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
