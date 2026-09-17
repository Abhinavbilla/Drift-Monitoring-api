import sqlite3
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from typing import Optional
import pandas as pd

from utils.profiler import coerce_numeric_column

DB_PATH = "drift.db"

# Matches utils/profiler.py's own "Low-cardinality text (Suitable for PSI)"
# threshold -- a storage-layer safety net, not a duplicate classification
# decision, for whatever reaches here without having gone through the
# profiler (e.g. a caller other than /fit/{project_id}).
MAX_CATEGORICAL_CARDINALITY = 50

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
            categorical_baselines TEXT,
            modality TEXT DEFAULT 'tabular',
            embedding_reference TEXT,
            embedding_model TEXT
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

    # Self-healing migration for DBs created before the multimodal (v2.0)
    # columns existed — avoids requiring a separate manual migration step.
    for _column, ddl in [
        ("modality", "ALTER TABLE baselines ADD COLUMN modality TEXT DEFAULT 'tabular'"),
        ("embedding_reference", "ALTER TABLE baselines ADD COLUMN embedding_reference TEXT"),
        ("embedding_model", "ALTER TABLE baselines ADD COLUMN embedding_model TEXT"),
    ]:
        try:
            cursor.execute(ddl)
        except sqlite3.OperationalError:
            pass  # column already exists

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

def _calculate_boundaries(reference_data: Dict[str, List[Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """
    Calculates Q1/Q3 for numbers, and Allowed Sets for categorical strings.

    Classification used to be "check the first non-null value's type" --
    a single stray non-numeric cell (a real-world "N/A"/typo placeholder)
    either crashed np.percentile on a mixed-type list (if a number came
    first) or silently miscategorized an entire numeric column as
    categorical (if the bad value came first). Both reproduced in
    tests/test_ingestion_robustness.py. Now uses the same coercion rule
    utils/profiler.py's own classification already relies on
    (coerce_numeric_column, shared threshold) so a column's type decision
    and its actual cleaned values can never disagree, and unconvertible
    cells are dropped with a reported count instead of crashing.

    Returns (fences, cleaning_summary) -- cleaning_summary reports how many
    values were dropped per column, so silent data loss stays visible.
    """
    fences = []
    cleaning_summary: Dict[str, Dict[str, int]] = {}

    for feature, data in reference_data.items():
        clean_data = [x for x in data if x is not None]
        if not clean_data:
            continue

        numeric_values, dropped = coerce_numeric_column(clean_data)

        # 1. NUMERICAL DATA (coercion succeeded above threshold)
        if numeric_values is not None:
            if not numeric_values:
                continue
            q1 = float(np.percentile(numeric_values, 25))
            q3 = float(np.percentile(numeric_values, 75))
            fences.append({
                "feature_name": feature,
                "type": "continuous",
                "q1": q1,
                "q3": q3
            })
            if dropped:
                cleaning_summary[feature] = {"dropped_non_numeric": dropped}

        # 2. CATEGORICAL DATA (everything else)
        else:
            unique_values = list(set(clean_data))
            if len(unique_values) > MAX_CATEGORICAL_CARDINALITY:
                raise ValueError(
                    f"Column '{feature}' has {len(unique_values)} unique values, exceeding the "
                    f"{MAX_CATEGORICAL_CARDINALITY}-value cap for categorical fields — likely a "
                    f"high-cardinality/free-text/ID field that shouldn't be monitored as categorical."
                )
            fences.append({
                "feature_name": feature,
                "type": "categorical",
                "allowed_values": unique_values
            })

    return fences, cleaning_summary

def insert_baseline(
    project_id: str,
    feature_types: dict,
    reference_data: dict,
    categorical_data: Optional[dict] = None
) -> Dict[str, Dict[str, int]]:
    """
    Stores all raw reference data (continuous + categorical) in `reference_data` column.
    This ensures batch drift detection (PSI/KS) can access the full distribution.
    Real‑time fences (IQR + allowed values) are stored separately in `iqr_fences`.

    Returns a cleaning_summary ({field: {"dropped_non_numeric": n}}) for any
    continuous column that had unconvertible cells dropped, so callers can
    surface that data loss instead of it being silent.
    """
    # Merge continuous and categorical raw data into a single dictionary
    combined_raw_data = dict(reference_data)
    if categorical_data:
        combined_raw_data.update(categorical_data)

    # Clean continuous columns BEFORE they're stored, not just when computing
    # fences below -- this same reference_data blob is separately consumed by
    # DistributionDetector's batch KS-test (drift/detector.py), which doesn't
    # crash on a mixed numeric/string column, it silently produces WRONG
    # statistics (numpy upcasts the array to string dtype, so ks_2samp
    # compares strings lexicographically) -- verified, not assumed; see
    # tests/test_ingestion_robustness.py.
    cleaning_summary: Dict[str, Dict[str, int]] = {}
    for feature, ftype in feature_types.items():
        if ftype == "continuous" and feature in combined_raw_data:
            numeric_values, dropped = coerce_numeric_column(combined_raw_data[feature])
            if numeric_values is not None:
                combined_raw_data[feature] = numeric_values
                if dropped:
                    cleaning_summary[feature] = {"dropped_non_numeric": dropped}

    # Calculate IQR fences (for continuous) AND allowed values (for categorical)
    # This uses the now-cleaned merged data so fences and the stored blob agree.
    fences, fence_cleaning_summary = _calculate_boundaries(combined_raw_data)
    for feature, info in fence_cleaning_summary.items():
        cleaning_summary.setdefault(feature, {}).update(info)

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

    return cleaning_summary

def get_baseline(project_id: str) -> dict:
    """Retrieves the model state and parses the JSON back into Python dictionaries."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        'SELECT feature_types, reference_data, iqr_fences, modality, embedding_reference, embedding_model '
        'FROM baselines WHERE project_id = ?',
        (project_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "feature_types": json.loads(row[0]) if row[0] else {},
        "reference_data": json.loads(row[1]) if row[1] else {},
        "iqr_fences": json.loads(row[2]) if row[2] else [],
        "modality": row[3] or "tabular",
        "embedding_reference": json.loads(row[4]) if row[4] else None,
        "embedding_model": row[5],
    }


def insert_embedding_baseline(project_id: str, modality: str, embeddings, model_name: str, max_reference_samples: int = 3000):
    """
    Stores a text/image baseline as raw reference embeddings, capped at
    max_reference_samples, so the Domain Classifier Test has real vectors
    to retrain against on every /analyze call (mirrors how DistributionDetector
    re-fits from raw arrays for tabular baselines).
    """
    embeddings_list = np.asarray(embeddings)
    if len(embeddings_list) > max_reference_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings_list), size=max_reference_samples, replace=False)
        embeddings_list = embeddings_list[idx]

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO baselines
            (project_id, feature_types, reference_data, iqr_fences, categorical_baselines,
             modality, embedding_reference, embedding_model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        project_id,
        json.dumps({}),
        json.dumps({}),
        json.dumps([]),
        json.dumps({}),
        modality,
        json.dumps(embeddings_list.tolist()),
        model_name,
    ))
    conn.commit()
    conn.close()

def insert_joint_baseline(project_id: str, embeddings, tabular_stats: dict, model_name: str, max_reference_samples: int = 3000):
    """
    Stores a joint-modality baseline: raw reference embeddings (same
    capping/sampling as insert_embedding_baseline) plus the tabular
    sub-schema statistics (per-field mean/std or category frequencies)
    needed to vectorize tabular fields consistently at /analyze time.

    Deliberately a separate function rather than extending
    insert_embedding_baseline — it repurposes the feature_types column
    (normally hardcoded to '{}' for text/image rows) to hold tabular_stats
    instead, and keeping it separate means the text/image code path is
    provably untouched.
    """
    embeddings_arr = np.asarray(embeddings)
    if len(embeddings_arr) > max_reference_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings_arr), size=max_reference_samples, replace=False)
        embeddings_arr = embeddings_arr[idx]

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO baselines
            (project_id, feature_types, reference_data, iqr_fences, categorical_baselines,
             modality, embedding_reference, embedding_model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        project_id,
        json.dumps(tabular_stats),
        json.dumps({}),
        json.dumps([]),
        json.dumps({}),
        "joint",
        json.dumps(embeddings_arr.tolist()),
        model_name,
    ))
    conn.commit()
    conn.close()


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