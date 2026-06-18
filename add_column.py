import sqlite3

conn = sqlite3.connect('drift.db')
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE baselines ADD COLUMN categorical_baselines TEXT")
    print("✅ Column 'categorical_baselines' added successfully.")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        print("⚠️ Column already exists.")
    else:
        print(f"Error: {e}")

conn.commit()
conn.close()