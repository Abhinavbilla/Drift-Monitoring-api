"""
Universal ML Telemetry Streamer
-------------------------------
A generalized tool to stream local .csv and .dat datasets to the Drift Sentinel API.
Automatically detects column headers, handles data typing, and simulates live production traffic.
"""

import argparse
import requests
import time
import csv
import urllib.parse
import glob
import os
import sys

def parse_value(val):
    """Automatically converts extracted string values into floats or ints for the JSON payload."""
    if not val:
        return None
    try:
        if '.' in val:
            return float(val)
        return int(val)
    except ValueError:
        return val  # Keep as string if it's categorical/text data

def stream_datasets(api_url, raw_model_id, api_key, data_dir, delay):
    """Discovers datasets in the target directory and streams them row-by-row."""
    
    # 1. System Setup & URL Encoding
    model_id = urllib.parse.quote(raw_model_id.strip())  
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    print(f"\n🚀 Starting Universal Data Streamer for model: '{raw_model_id}'")
    
    # 2. Multi-File Discovery
    search_paths = [
        os.path.join(data_dir, "*.csv"),
        os.path.join(data_dir, "*.dat")
    ]

    files_to_stream = []
    for path in search_paths:
        files_to_stream.extend(glob.glob(path))

    if not files_to_stream:
        print(f"❌ Error: No .csv or .dat files found in directory '{data_dir}'.")
        sys.exit(1)

    print(f"📂 Found {len(files_to_stream)} dataset(s). Preparing to stream...\n")
    request_count = 0

    # 3. The Streaming Engine
    try:
        for file_path in files_to_stream:
            print(f"--- 📖 Now reading: {os.path.basename(file_path)} ---")
            
            with open(file_path, mode='r', encoding='utf-8') as file:
                # Smart Sniffer: Detects commas, tabs, or custom delimiters
                sample_text = file.read(2048)
                file.seek(0) 
                
                try:
                    dialect = csv.Sniffer().sniff(sample_text)
                    csv_reader = csv.DictReader(file, dialect=dialect)
                except csv.Error:
                    # Fallback to standard CSV parsing if sniffer fails
                    csv_reader = csv.DictReader(file)
                
                for row in csv_reader:
                    # Clean the row (convert types, strip whitespace, ignore empty columns)
                    clean_features = {key.strip(): parse_value(value.strip()) for key, value in row.items() if key}
                    
                    payload = {
                        "features": clean_features
                    }

                    # Fire the Request
                    response = requests.post(f"{api_url}/predict/{model_id}", json=payload, headers=headers)
                    request_count += 1
                    
                    # Parse Response Status
                    if response.status_code == 200:
                        result = response.json()
                        status = "🔴 OOD/ANOMALY" if result.get("is_anomaly") else "🟢 NORMAL"
                        score = result.get('anomaly_score', 0)
                        print(f"Tx #{request_count:05d} | {status} | Score: {score:.2f} | File: {os.path.basename(file_path)}")
                    else:
                        print(f"❌ Error {response.status_code}: {response.text}")

                    # Simulate real-world delay between requests
                    time.sleep(delay)
                    
            print(f"✅ Finished streaming {os.path.basename(file_path)}.\n")

        print(f"🎉 All datasets successfully streamed! Total records sent: {request_count}")

    except KeyboardInterrupt:
        print("\n🛑 Streamer stopped safely by user.")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection failed. Ensure your API is running at {api_url}.")
        sys.exit(1)

if __name__ == "__main__":
    # Standard CLI Setup for GitHub Repos
    parser = argparse.ArgumentParser(description="Stream local CSV/DAT files to the Drift Sentinel API.")
    
    parser.add_argument("--url", type=str, default="http://api:8000", help="Base URL of the FastAPI server.")
    parser.add_argument("--model", type=str, required=True, help="The exact Model ID (e.g., 'credit card fraud detector').")
    parser.add_argument("--key", type=str, required=True, help="Your Developer API Key.")
    parser.add_argument("--dir", type=str, default="./data", help="Directory containing the .csv or .dat files.")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between sending rows (default: 1.0).")

    args = parser.parse_args()

    stream_datasets(
        api_url=args.url,
        raw_model_id=args.model,
        api_key=args.key,
        data_dir=args.dir,
        delay=args.delay
    )