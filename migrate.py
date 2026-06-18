import sqlite3

conn = sqlite3.connect("drift.db")
cursor = conn.cursor()

try:
    cursor.execute(
        "ALTER TABLE baselines ADD COLUMN categorical_baselines TEXT DEFAULT '{}'"
    )
    conn.commit()
    print("✅ Migration successful — categorical_baselines column added")
except Exception as e:
    print(f"ℹ️  Column may already exist: {e}")

conn.close()