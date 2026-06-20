import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("❌ API_KEY not found in environment. Please set it in .env file.")
API_BASE = "http://api:8000"
MODEL_ID = "citi_bike_v1"

HEADERS = {"X-API-Key": API_KEY}

prod_df = pd.read_csv("tests/citi_bike_production.csv")
sample = prod_df.sample(n=20000, random_state=42)

payload = {"production_data": sample.to_dict(orient="list")}

resp = requests.post(
    f"{API_BASE}/analyze/{MODEL_ID}",
    json=payload,
    headers=HEADERS
)

if resp.status_code == 200:
    result = resp.json()
    print("System alert triggered:", result["system_alert_triggered"])
    for feature, metrics in result["feature_metrics"].items():
        print(f"  {feature}: drift_detected={metrics['drift_detected']}")
else:
    print(f"Error: {resp.status_code}")