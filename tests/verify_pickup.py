"""
Quick verification: pickup_latitude detection rate at 15,000 rows
with fresh random seeds (20-35, different from original 0-14).
"""

import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("❌ API_KEY not found in environment. Please set it in .env file.")

API_BASE = "http://127.0.0.1:8000"
MODEL_ID = "citi_bike_v1"
HEADERS = {"X-API-Key": API_KEY}

print("Loading production data...")
prod_df = pd.read_csv("tests/citi_bike_production.csv")
print(f"Loaded {len(prod_df):,} rows\n")

print("Testing pickup_latitude detection at 15,000 rows (fresh seeds 20-35)...")
print("-" * 50)

results = []
trial_count = 15

for trial in range(20, 20 + trial_count):
    sample = prod_df.sample(n=15000, random_state=trial)
    
    payload = {
        "production_data": {
            "pickup_latitude": [float(v) if not pd.isna(v) else None for v in sample["pickup_latitude"].values]
        }
    }
    
    try:
        resp = requests.post(
            f"{API_BASE}/analyze/{MODEL_ID}",
            json=payload,
            headers=HEADERS,
            timeout=30
        )
        
        if resp.status_code == 200:
            detected = resp.json()["feature_metrics"]["pickup_latitude"]["drift_detected"]
            results.append(detected)
            status = "✅ DETECTED" if detected else "❌ MISSED"
            print(f"  Trial {trial}: {status}")
        else:
            print(f"  Trial {trial}: ❌ API Error {resp.status_code}")
    except Exception as e:
        print(f"  Trial {trial}: ❌ Exception: {e}")
    
    time.sleep(0.2)

rate = sum(results) / len(results) * 100 if results else 0
print("-" * 50)
print(f"\n📊 pickup_latitude detection rate at 15,000 rows: {rate:.0f}% ({sum(results)}/{len(results)})")
print(f"   Previous rate: 53% (8/15)")
print(f"   Difference: {rate - 53:.1f}%")

if 45 <= rate <= 65:
    print("✅ Stable – pickup_latitude is genuinely the hardest feature (smallest KS-stat 0.0189)")
else:
    print("⚠️  Significant variance detected – consider using larger batch size for this feature")