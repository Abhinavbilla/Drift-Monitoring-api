import sqlite3
import json
import numpy as np
from typing import Dict, List, Any
from typing import Optional   
import pandas as pd 
DB_PATH = "drift.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes the database tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # UPDATED: Added owner_email TEXT to link projects to specific users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY, 
            name TEXT,
            owner_email TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS baselines (
            project_id TEXT PRIMARY KEY,
            feature_types TEXT,
            reference_data TEXT,
            iqr_fences TEXT,
            categorical_baselines TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            project_id TEXT,
            input_data TEXT,
            score REAL,
            is_ood INTEGER
        )
    ''')

    conn.commit()
    conn.close()

# UPDATED: Added owner_email as a parameter
def create_project(project_id: str, name: str, owner_email: str):
    conn = get_connection()
    cursor = conn.cursor()
    # UPDATED: Insert owner_email into the database
    cursor.execute(
        "INSERT OR REPLACE INTO projects (id, name, owner_email) VALUES (?, ?, ?)", 
        (project_id, name, owner_email)
    )
    conn.commit()
    conn.close()

def _calculate_boundaries(reference_data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Calculates Q1/Q3 for numbers, and Allowed Sets for categorical strings."""
    fences = []
    for feature, data in reference_data.items():
        clean_data = [x for x in data if x is not None]
        if not clean_data:
            continue
        
        # 1. CATEGORICAL DATA (Strings)
        if isinstance(clean_data[0], str):
            # Find all unique words in this column
            unique_values = list(set(clean_data))
            fences.append({
                "feature_name": feature,
                "type": "categorical",
                "allowed_values": unique_values
            })
            
        # 2. NUMERICAL DATA (Floats/Integers)
        else:
            q1 = float(np.percentile(clean_data, 25))
            q3 = float(np.percentile(clean_data, 75))
            fences.append({
                "feature_name": feature,
                "type": "continuous",
                "q1": q1,
                "q3": q3
            })
            
    return fences

def insert_baseline(
    project_id: str, 
    feature_types: dict, 
    reference_data: dict,
    categorical_data: Optional[dict] = None
):
    """
    Stores all raw reference data (continuous + categorical) in `reference_data` column.
    This ensures batch drift detection (PSI/KS) can access the full distribution.
    Real‑time fences (IQR + allowed values) are stored separately in `iqr_fences`.
    """
    # Merge continuous and categorical raw data into a single dictionary
    combined_raw_data = dict(reference_data)
    if categorical_data:
        combined_raw_data.update(categorical_data)
    
    # Calculate IQR fences (for continuous) AND allowed values (for categorical)
    # This uses the merged data so categorical features get their allowed_values.
    fences = _calculate_boundaries(combined_raw_data)
    
    # (Optional) Pre‑compute frequency baselines for categorical features
    # Not strictly needed because we now have raw data, but kept for backward compatibility.
    cat_freq_baselines = {}
    if categorical_data:
        cat_df = pd.DataFrame(categorical_data)
        for col in cat_df.columns:
            freq = cat_df[col].value_counts(normalize=True).to_dict()
            cat_freq_baselines[col] = freq

    conn = get_connection()
    cursor = conn.cursor()
    
    # Store the merged raw data in the `reference_data` column.
    # The `categorical_baselines` column is not used by the detector,
    # but we keep it to avoid migration issues.
    cursor.execute('''
        INSERT OR REPLACE INTO baselines 
            (project_id, feature_types, reference_data, iqr_fences, categorical_baselines)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        project_id,
        json.dumps(feature_types),
        json.dumps(combined_raw_data),      # <-- now includes categorical raw values
        json.dumps(fences),
        json.dumps(cat_freq_baselines)
    ))
    
    conn.commit()
    conn.close()

def get_baseline(project_id: str) -> dict:
    """Retrieves the model state and parses the JSON back into Python dictionaries."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT feature_types, reference_data, iqr_fences FROM baselines WHERE project_id = ?', (project_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    return {
        "feature_types": json.loads(row[0]),
        "reference_data": json.loads(row[1]),
        "iqr_fences": json.loads(row[2])
    }

def insert_log(project_id: str, input_data: dict, score: float, is_ood: int):
    conn = get_connection()
    cursor = conn.cursor()
    input_json = json.dumps(input_data)
    
    cursor.execute('''
        INSERT INTO logs (project_id, input_data, score, is_ood)
        VALUES (?, ?, ?, ?)
    ''', (project_id, input_json, score, is_ood))
    
    conn.commit()
    conn.close()

def get_logs(project_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs WHERE project_id = ?", (project_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows