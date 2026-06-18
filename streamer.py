import requests
import pandas as pd
import time
import random
import urllib.parse
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("❌ API_KEY not found in environment. Please set it in .env file.")
# ==========================================
# ⚙️ CONFIGURATION (UPDATE THESE!)
# ==========================================
 
PROJECT_ID = "credit card fraud detector"
CSV_PATH = "creditcard.csv"         # <-- Path to your Kaggle dataset
BASE_URL = "http://127.0.0.1:8000"

HEADERS = {"X-API-Key": API_KEY}
SAFE_PROJECT_ID = urllib.parse.quote(PROJECT_ID)

def run_simulation():
    print(f"\n🚀 Starting Chaos Monkey Simulator for: '{PROJECT_ID}'")
    print("-" * 50)
    
    # 1. Load data
    try:
        df = pd.read_csv(CSV_PATH)
        # Drop columns we set to "Ignore" in the schema UI
        df = df.drop(columns=['Class', 'Time'], errors='ignore')
        print(f"✅ Loaded dataset: {len(df)} rows.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{CSV_PATH}'.")
        sys.exit(1)

    # ---------------------------------------------------------
    # PHASE 1: NORMAL TRAFFIC
    # ---------------------------------------------------------
    print("\n🌊 PHASE 1: Streaming Normal Traffic (10 requests)...")
    normal_sample = df.sample(10).to_dict(orient="records")
    
    for i, row in enumerate(normal_sample):
        payload = {"features": row}
        res = requests.post(f"{BASE_URL}/predict/{SAFE_PROJECT_ID}", json=payload, headers=HEADERS)
        
        if res.status_code == 200:
            data = res.json()
            status = "🔴 ANOMALY" if data.get('is_anomaly') else "🟢 Normal "
            print(f"  [Req {i+1}] {status} | IQR Score: {data.get('anomaly_score', 0):.2f}")
        else:
            print(f"  [Error] {res.text}")
        time.sleep(0.3) # Simulate real-time delay

    # ---------------------------------------------------------
    # PHASE 2: ANOMALY SPIKE
    # ---------------------------------------------------------
    print("\n⚠️ PHASE 2: Simulating Cyber Attack (Data Mutation)...")
    time.sleep(1)
    
    drifted_sample = df.sample(10).copy()
    # Mutate the data to force anomalies
    drifted_sample['Amount'] = drifted_sample['Amount'] * random.uniform(50, 100)
    drifted_sample['V1'] = drifted_sample['V1'] + random.uniform(10, 20)
    drifted_records = drifted_sample.to_dict(orient="records")
    
    for i, row in enumerate(drifted_records):
        payload = {"features": row}
        res = requests.post(f"{BASE_URL}/predict/{SAFE_PROJECT_ID}", json=payload, headers=HEADERS)
        
        if res.status_code == 200:
            data = res.json()
            status = "🔴 ANOMALY" if data.get('is_anomaly') else "🟢 Normal "
            print(f"  [Req {i+1}] {status} | IQR Score: {data.get('anomaly_score', 0):.2f}")
        time.sleep(0.3)

    # ---------------------------------------------------------
    # PHASE 3: BATCH ALERT TRIGGER
    # ---------------------------------------------------------
    print("\n🚨 PHASE 3: Triggering Batch Distribution Analysis...")
    time.sleep(1)
    
    # Create a batch of 500 mutated rows to guarantee statistical drift
    batch_drift = df.sample(500).copy()
    batch_drift['Amount'] = batch_drift['Amount'] * random.uniform(10, 50)
    batch_drift['V2'] = batch_drift['V2'] - 15
    
    # FIXED: Changed orient="records" to orient="list" to match your API schema
    batch_payload = {"production_data": batch_drift.to_dict(orient="list")}
    print("  Sending massive drifted payload to /analyze... (Watch your API terminal!)")
    
    res = requests.post(f"{BASE_URL}/analyze/{SAFE_PROJECT_ID}", json=batch_payload, headers=HEADERS)
    
    if res.status_code == 200:
        report = res.json()
        if report.get("system_alert_triggered"):
            print("\n✅ SUCCESS: Statistical Drift Confirmed!")
            print("📩 If your SMTP is configured, an email was just dispatched.")
        else:
            print("\n❌ Math didn't trigger drift. You may need to mutate the data more severely.")
    else:
        print(f"\n[Error] {res.text}")

if __name__ == "__main__":
    run_simulation()