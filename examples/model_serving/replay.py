"""
Step 3e end-to-end example, part 3: replay the Apr-Jun 2016 Citi Bike
production data through the serving app (serve.py) in fixed windows,
simulating live traffic. Each request triggers serve.py's async
post-prediction logging, which scheduled_job.py later reads.

Samples REPLAY_ROWS_PER_MONTH rows per month (fixed seed, reproducible),
sorted by pickup_datetime within the sample, and POSTs each row to
/predict in that order. A demo-scale sample, not the literal full
~800K-1.15M rows/month -- impractical to send one HTTP call per row for
a demo, and unnecessary to make the point.

Run (with serve.py already running, default port 8001):
    python examples/model_serving/replay.py
"""

import os
import time

import pandas as pd
import requests

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..")
SERVE_URL = os.environ.get("MODEL_SERVE_URL", "http://127.0.0.1:8001")
REPLAY_ROWS_PER_MONTH = 3000
SEED = 42
MONTHS = [4, 5, 6]
INPUT_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "gender_id"]


def main():
    session = requests.Session()
    total = 0
    t0 = time.time()
    for m in MONTHS:
        path = os.path.join(REPO_ROOT, "tests", "splits", f"production_month_{m:02d}.csv")
        df = pd.read_csv(path)
        sample = df.sample(n=min(REPLAY_ROWS_PER_MONTH, len(df)), random_state=SEED).sort_values("pickup_datetime")
        for _, row in sample.iterrows():
            payload = {
                "pickup_longitude": float(row["pickup_longitude"]),
                "pickup_latitude": float(row["pickup_latitude"]),
                "dropoff_longitude": float(row["dropoff_longitude"]),
                "dropoff_latitude": float(row["dropoff_latitude"]),
                "gender_id": int(row["gender_id"]),
            }
            resp = session.post(f"{SERVE_URL}/predict", json=payload, timeout=30)
            resp.raise_for_status()
            total += 1
        print(f"Month {m:02d}: replayed {len(sample)} rows.")
    print(f"Total replayed: {total} rows in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
