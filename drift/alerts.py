from db.crud import get_logs
import smtplib
from email.message import EmailMessage
import os
import datetime

# ==========================================
# 1. REAL-TIME BURST ALERTS (Existing)
# ==========================================
def check_drift_alert(project_id: str, window_size: int = 10, threshold: float = 0.3) -> tuple:
    """
    Checks the most recent real-time logs to see if a high percentage 
    of them were flagged as anomalies (The Burst Inspector).
    """
    logs = get_logs(project_id)

    # Take the last N logs (the most recent ones)
    recent_logs = logs[-window_size:]

    if not recent_logs:
        return False, 0.0

    total = len(recent_logs)
    
    # In SQLite, the 'logs' table has columns: (project_id, input_data, score, is_ood)
    # is_ood is the 4th column, which is accessed via index [3]
    ood_count = sum(row[3] for row in recent_logs)

    drift_ratio = ood_count / total

    # Trigger an alert if the ratio of anomalies exceeds our threshold
    alert = bool(drift_ratio > threshold)

    return alert, float(drift_ratio)


# ==========================================
# 2. ASYNCHRONOUS BATCH EMAIL ALERTS (New)
# ==========================================
def send_drift_email(project_id: str, owner_email: str, flagged_features: list):
    """
    Constructs and sends an email alert asynchronously.
    Ensure you have set ALERT_EMAIL and ALERT_PASSWORD in your environment variables.
    """
    # Fail gracefully if credentials aren't set yet (great for local testing)
    sender_email = os.getenv('ALERT_EMAIL')
    sender_password = os.getenv('ALERT_PASSWORD')
    
    if not sender_email or not sender_password:
        print(f"\n⚠️ WARNING: Drift detected in {project_id}!")
        print(f"⚠️ Flagged Features: {', '.join(flagged_features)}")
        print("⚠️ No SMTP credentials found in environment. Email alert suppressed.\n")
        return

    msg = EmailMessage()
    msg['Subject'] = f"🚨 Drift Sentinel Alert: Distribution Shift in {project_id}"
    msg['From'] = sender_email
    msg['To'] = owner_email
    
    features_str = ", ".join(flagged_features) if flagged_features else "Unknown"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    msg.set_content(f"""
    Drift Sentinel has detected a significant distribution shift in your production data.
    
    Project ID: {project_id}
    Time Detected: {current_time}
    Flagged Features: {features_str}
    
    Please log in to your dashboard to view the Kolomogorov-Smirnov and TVD metrics.
    
    - The Drift Sentinel Automated System
    """)
    
    try:
        # Connect to Gmail's SMTP server securely
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
            print(f"✅ Alert email successfully sent to {owner_email}")
    except Exception as e:
        print(f"❌ Failed to send alert email: {str(e)}")