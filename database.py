import sqlite3

DB_NAME = "drift.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT,
        owner_email TEXT, -- Added this missing column!
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS baselines (
        project_id TEXT,
        feature_name TEXT,
        mean REAL,
        std REAL,
        upper_limit REAL,
        lower_limit REAL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        input_data TEXT,
        score REAL,
        is_ood INTEGER
    )
    """)

    conn.commit()
    conn.close()
    
