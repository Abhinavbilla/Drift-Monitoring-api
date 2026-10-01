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

    # Personal access tokens (Step 3a). user_email, not user_id -- this
    # codebase has no separate users table; email is the identity used
    # throughout (projects.owner_email, session-JWT's "email" claim).
    # project_scope is a JSON list of project_ids the token may access,
    # or the single-element list ["*"] for unrestricted (still bound to
    # this user_email, never a cross-user escalation). Only token_hash is
    # ever persisted -- the plaintext token exists only in the response
    # to the create call.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_tokens (
            id TEXT PRIMARY KEY,
            user_email TEXT NOT NULL,
            name TEXT,
            prefix TEXT UNIQUE NOT NULL,
            token_hash TEXT NOT NULL,
            project_scope TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            last_used_at TEXT,
            revoked INTEGER NOT NULL DEFAULT 0
        )
    ''')

    # Step 5 item 2: one row per /analyze (and /analyze/upload) call --
    # statistics only, never raw production rows. "id" is a plain
    # autoincrement surrogate key so history entries never need a natural
    # key; project stores the INTERNAL (owner-namespaced) project key, same
    # as baselines/logs. baseline_version, idempotency_key, schema_report
    # and sustained_alert are columns items 3/4/5/6 will populate for real
    # -- item 2 writes them as NULL (or baseline_version=1, the only
    # version that exists before item 5) since those features don't exist
    # yet; this way item 3/4/5/6 only ever need to START writing real
    # values into an already-existing column, no further schema change.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            baseline_version INTEGER,
            ts TEXT NOT NULL,
            batch_size INTEGER,
            idempotency_key TEXT,
            payload_hash TEXT,
            decision_mode TEXT,
            system_alert INTEGER,
            sustained_alert INTEGER,
            feature_results TEXT,
            schema_report TEXT
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_runs_project_ts ON analysis_runs (project, ts)')

    # Self-healing migration for DBs created before the multimodal (v2.0)
    # columns existed — avoids requiring a separate manual migration step.
    for _column, ddl in [
        ("modality", "ALTER TABLE baselines ADD COLUMN modality TEXT DEFAULT 'tabular'"),
        ("embedding_reference", "ALTER TABLE baselines ADD COLUMN embedding_reference TEXT"),
        ("embedding_model", "ALTER TABLE baselines ADD COLUMN embedding_model TEXT"),
        # Step 2: NULL here (the default for every pre-existing row, and for
        # any row inserted without specifying it) means
        # CalibrationConfig.from_dict(None) resolves to the legacy default --
        # existing projects stay legacy permanently with no data migration.
        ("calibration_config", "ALTER TABLE baselines ADD COLUMN calibration_config TEXT"),
        # Step 5 item 4: reference_null_rates is recomputed on every /fit
        # (reflects the CURRENT reference data); schema_policy persists
        # across re-fits like calibration_config, via the same
        # "__UNSET__" preserve-existing-value sentinel.
        ("reference_null_rates", "ALTER TABLE baselines ADD COLUMN reference_null_rates TEXT"),
        ("schema_policy", "ALTER TABLE baselines ADD COLUMN schema_policy TEXT"),
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
    categorical_data: Optional[dict] = None,
    calibration_config: Optional[dict] = "__UNSET__",
    schema_policy: Optional[dict] = "__UNSET__",
) -> Dict[str, Dict[str, int]]:
    """
    Stores all raw reference data (continuous + categorical) in `reference_data` column.
    This ensures batch drift detection (PSI/KS) can access the full distribution.
    Real‑time fences (IQR + allowed values) are stored separately in `iqr_fences`.

    calibration_config: the sentinel default ("__UNSET__", distinct from an
    explicit None) means "preserve whatever this project already has" --
    this uses `INSERT OR REPLACE`, which in SQLite deletes and re-inserts
    the row, silently resetting any column not named in the INSERT back to
    its default (verified: a bare INSERT OR REPLACE would wipe an existing
    project's calibration_config to NULL on every re-fit). Pass an explicit
    dict (or None, to deliberately clear it) to actually change the config
    as part of this call.

    Returns a cleaning_summary ({field: {"dropped_non_numeric": n}}) for any
    continuous column that had unconvertible cells dropped, so callers can
    surface that data loss instead of it being silent.
    """
    if calibration_config == "__UNSET__":
        existing = get_baseline(project_id)
        calibration_config = existing["calibration_config"] if existing else None
    if schema_policy == "__UNSET__":
        existing = get_baseline(project_id)
        schema_policy = existing["schema_policy"] if existing else None
    # Merge continuous and categorical raw data into a single dictionary
    combined_raw_data = dict(reference_data)
    if categorical_data:
        combined_raw_data.update(categorical_data)

    # Step 5 item 4: null rate of the RAW reference data, before any
    # cleaning below -- /analyze compares a production batch's null rate
    # against this to flag a meaningful increase.
    reference_null_rates: Dict[str, float] = {}
    for feature, values in combined_raw_data.items():
        if not values:
            reference_null_rates[feature] = 0.0
            continue
        null_count = sum(1 for v in values if v is None or (isinstance(v, float) and np.isnan(v)))
        reference_null_rates[feature] = null_count / len(values)

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
            (project_id, feature_types, reference_data, iqr_fences, categorical_baselines,
             calibration_config, reference_null_rates, schema_policy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        project_id,
        json.dumps(feature_types),
        json.dumps(combined_raw_data),      # <-- now includes categorical raw values
        json.dumps(fences),
        json.dumps(cat_freq_baselines),
        json.dumps(calibration_config) if calibration_config is not None else None,
        json.dumps(reference_null_rates),
        json.dumps(schema_policy) if schema_policy is not None else None,
    ))

    conn.commit()
    conn.close()

    return cleaning_summary

def get_baseline(project_id: str) -> dict:
    """Retrieves the model state and parses the JSON back into Python dictionaries."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        'SELECT feature_types, reference_data, iqr_fences, modality, embedding_reference, embedding_model, '
        'calibration_config, reference_null_rates, schema_policy '
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
        # None (the default for every pre-Step-2 row) -> legacy, via
        # CalibrationConfig.from_dict(None) -- callers pass this straight
        # through, not resolved here, so this module stays independent of
        # drift/calibration.py.
        "calibration_config": json.loads(row[6]) if row[6] else None,
        # Step 5 item 4: {} for every pre-item-4 row (no null-rate
        # baseline recorded yet) -- /analyze treats a missing entry as
        # "no baseline null rate known", not as 0.
        "reference_null_rates": json.loads(row[7]) if row[7] else {},
        # None -> the default policy (alert on missing columns, warn on
        # everything else) -- resolved in main.py, not here.
        "schema_policy": json.loads(row[8]) if row[8] else None,
    }


def set_calibration_config(project_id: str, config: Optional[dict]) -> None:
    """Stores (or clears, if config=None) a project's calibration config,
    independent of any /fit call -- lets a project's decision_mode/floors be
    changed without re-fitting its baseline. Row must already exist (created
    by a prior /fit call); this only updates the calibration_config column."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE baselines SET calibration_config = ? WHERE project_id = ?',
        (json.dumps(config) if config is not None else None, project_id)
    )
    conn.commit()
    conn.close()


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

    # Preserve any existing calibration_config -- INSERT OR REPLACE deletes
    # and re-inserts the row, which would otherwise silently wipe it back to
    # NULL on every re-fit (verified in db/crud.py's insert_baseline).
    existing = get_baseline(project_id)
    calibration_config = existing["calibration_config"] if existing else None

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO baselines
            (project_id, feature_types, reference_data, iqr_fences, categorical_baselines,
             modality, embedding_reference, embedding_model, calibration_config)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        project_id,
        json.dumps({}),
        json.dumps({}),
        json.dumps([]),
        json.dumps({}),
        modality,
        json.dumps(embeddings_list.tolist()),
        model_name,
        json.dumps(calibration_config) if calibration_config is not None else None,
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

    # Preserve any existing calibration_config -- see insert_embedding_baseline.
    existing = get_baseline(project_id)
    calibration_config = existing["calibration_config"] if existing else None

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO baselines
            (project_id, feature_types, reference_data, iqr_fences, categorical_baselines,
             modality, embedding_reference, embedding_model, calibration_config)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        project_id,
        json.dumps(tabular_stats),
        json.dumps({}),
        json.dumps([]),
        json.dumps({}),
        "joint",
        json.dumps(embeddings_arr.tolist()),
        model_name,
        json.dumps(calibration_config) if calibration_config is not None else None,
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


# ---------------------------------------------------------
# PERSONAL ACCESS TOKENS (Step 3a)
# ---------------------------------------------------------

def _row_to_token_dict(row) -> Dict[str, Any]:
    (token_id, user_email, name, prefix, token_hash, project_scope,
     created_at, expires_at, last_used_at, revoked) = row
    return {
        "id": token_id, "user_email": user_email, "name": name, "prefix": prefix,
        "token_hash": token_hash,
        "project_scope": json.loads(project_scope) if project_scope else None,
        "created_at": created_at, "expires_at": expires_at, "last_used_at": last_used_at,
        "revoked": bool(revoked),
    }


def create_api_token(
    token_id: str, user_email: str, name: str, prefix: str, token_hash: str,
    project_scope: Optional[List[str]], created_at: str, expires_at: Optional[str],
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO api_tokens (id, user_email, name, prefix, token_hash, project_scope, "
        "created_at, expires_at, last_used_at, revoked) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)",
        (token_id, user_email, name, prefix, token_hash,
         json.dumps(project_scope) if project_scope is not None else None,
         created_at, expires_at),
    )
    conn.commit()
    conn.close()


def get_api_token_by_prefix(prefix: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, user_email, name, prefix, token_hash, project_scope, created_at, "
        "expires_at, last_used_at, revoked FROM api_tokens WHERE prefix = ?", (prefix,)
    )
    row = cursor.fetchone()
    conn.close()
    return _row_to_token_dict(row) if row else None


def list_api_tokens(user_email: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, user_email, name, prefix, token_hash, project_scope, created_at, "
        "expires_at, last_used_at, revoked FROM api_tokens WHERE user_email = ? "
        "ORDER BY created_at DESC", (user_email,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_token_dict(r) for r in rows]


def revoke_api_token(token_id: str, user_email: str) -> bool:
    """Revokes a token, scoped to the requesting user (cannot revoke
    another user's token). Returns True iff a row was actually changed."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE api_tokens SET revoked = 1 WHERE id = ? AND user_email = ? AND revoked = 0",
        (token_id, user_email),
    )
    changed = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return changed


def touch_api_token_last_used(token_id: str, when_iso: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE api_tokens SET last_used_at = ? WHERE id = ?", (when_iso, token_id))
    conn.commit()
    conn.close()


def insert_analysis_run(
    project: str, baseline_version: Optional[int], ts: str, batch_size: int,
    idempotency_key: Optional[str], payload_hash: Optional[str], decision_mode: Optional[str],
    system_alert: bool, sustained_alert: Optional[bool], feature_results: dict,
    schema_report: Optional[dict],
) -> int:
    """Step 5 item 2: one row per /analyze call. feature_results/schema_report
    are the already-aggregated statistics dicts the response itself returns
    -- never raw production rows, per instruction."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO analysis_runs (project, baseline_version, ts, batch_size, idempotency_key, "
        "payload_hash, decision_mode, system_alert, sustained_alert, feature_results, schema_report) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            project, baseline_version, ts, batch_size, idempotency_key, payload_hash, decision_mode,
            1 if system_alert else 0,
            None if sustained_alert is None else (1 if sustained_alert else 0),
            json.dumps(feature_results) if feature_results is not None else None,
            json.dumps(schema_report) if schema_report is not None else None,
        ),
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def _row_to_analysis_run(row) -> Dict[str, Any]:
    return {
        "id": row[0], "project": row[1], "baseline_version": row[2], "ts": row[3],
        "batch_size": row[4], "idempotency_key": row[5], "payload_hash": row[6],
        "decision_mode": row[7], "system_alert": bool(row[8]),
        "sustained_alert": None if row[9] is None else bool(row[9]),
        "feature_results": json.loads(row[10]) if row[10] else {},
        "schema_report": json.loads(row[11]) if row[11] else None,
    }


def get_analysis_runs(
    project: str, since: Optional[str] = None, until: Optional[str] = None,
    alert_only: bool = False,
) -> List[Dict[str, Any]]:
    """Newest first. since/until/alert_only are applied here in SQL; a
    `feature` filter and pagination are the caller's job (main.py) since
    they operate on the decoded feature_results JSON, not raw columns."""
    conn = get_connection()
    cursor = conn.cursor()
    query = (
        "SELECT id, project, baseline_version, ts, batch_size, idempotency_key, payload_hash, "
        "decision_mode, system_alert, sustained_alert, feature_results, schema_report "
        "FROM analysis_runs WHERE project = ?"
    )
    params: List[Any] = [project]
    if since:
        query += " AND ts >= ?"
        params.append(since)
    if until:
        query += " AND ts <= ?"
        params.append(until)
    if alert_only:
        query += " AND system_alert = 1"
    query += " ORDER BY ts DESC, id DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_analysis_run(r) for r in rows]


def find_analysis_run_by_idempotency_key(
    project: str, idempotency_key: str, not_before_ts: str
) -> Optional[Dict[str, Any]]:
    """Step 5 item 3: the most recent run stored under this project+key,
    if any, and not older than not_before_ts (the 7-day expiry cutoff --
    an expired key is treated the same as one that was never used)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, project, baseline_version, ts, batch_size, idempotency_key, payload_hash, "
        "decision_mode, system_alert, sustained_alert, feature_results, schema_report "
        "FROM analysis_runs WHERE project = ? AND idempotency_key = ? AND ts >= ? "
        "ORDER BY ts DESC, id DESC LIMIT 1",
        (project, idempotency_key, not_before_ts),
    )
    row = cursor.fetchone()
    conn.close()
    return _row_to_analysis_run(row) if row else None