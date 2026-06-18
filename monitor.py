import time
import requests
import schedule
from datetime import datetime

# The URL of your local FastAPI server
BASE_URL = "http://127.0.0.1:8000"
PROJECT_ID = "project_alpha"

def trigger_slack_alert(drift_ratio: float):
    """
    In a real production environment, this function would send an HTTP POST
    request to a Slack or Discord webhook. For now, it sounds the terminal alarm.
    """
    print("\n" + "="*50)
    print(f"🚨 CRITICAL ALERT: SYSTEM DEGRADED 🚨")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Project: {PROJECT_ID}")
    print(f"Anomaly Rate: {drift_ratio * 100}%")
    print("="*50 + "\n")

def check_system_health():
    """
    Pings the /health endpoint to check for sudden bursts of anomalies.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Pinging health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health/{PROJECT_ID}")
        
        # If the server is offline or the project doesn't exist, this catches it
        response.raise_for_status() 
        
        data = response.json()
        
        if data["is_burst_alert"]:
            trigger_slack_alert(data["drift_ratio"])
        else:
            print(f"   -> System Healthy. Anomaly rate: {data['drift_ratio'] * 100}%")
            
    except requests.exceptions.RequestException as e:
        print(f"   -> ⚠️ Failed to reach API: {e}")


# THE SCHEDULER (The Cron Job)

print("Starting Drift Sentinel Automated Monitor...")
print("Press Ctrl+C to exit.\n")

# For testing, we run it every 10 seconds. 
# In production, you would change this to: schedule.every(1).minutes.do(...)
schedule.every(100).seconds.do(check_system_health)

# This infinite loop keeps the script alive and checks the clock
while True:
    schedule.run_pending()
    time.sleep(1)