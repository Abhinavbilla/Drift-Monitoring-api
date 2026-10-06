import sqlite3
import json
import numpy as np
from datetime import datetime, timezone
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
    # timeout: the table-job worker thread writes concurrently with
    # request handlers, so wait for a lock instead of failing immediately.
    return sqlite3.connect(DB_PATH, timeout=30)

def init_db():
    """Initializes the database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    # WAL lets readers proceed while the job worker writes. Persistent on
    # the DB file once set, so this is a no-op after the first startup.
    cursor.execute("PRAGMA journal_mode=WAL")

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

    # Step 5 Part 2: webhooks for drift alerts. `secret` is stored in
    # PLAINTEXT (unlike api_tokens' hash-only storage) -- it's needed at
    # delivery time to compute each outgoing HMAC signature, not just to
    # verify a presented credential, so there's no hash-and-compare
    # option here. Still only ever returned to the caller once, at
    # creation (see main.py) -- GET /webhooks never echoes it back.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS webhooks (
            id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            url TEXT NOT NULL,
            secret TEXT NOT NULL,
            event_filter TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_webhooks_project ON webhooks (project)')

    # One row per delivery TASK (one event, one webhook) -- updated in
    # place on every attempt (attempt_number increments, status_code/
    # response_snippet reflect the MOST RECENT try), not one row per
    # attempt. next_retry_at is only meaningful while status='pending';
    # NULL once status is 'success' or 'failed' (gave up after 5 tries).
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS webhook_deliveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            webhook_id TEXT NOT NULL,
            event_id INTEGER,
            event_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            attempt_number INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'pending',
            status_code INTEGER,
            success INTEGER,
            response_snippet TEXT,
            next_retry_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_pending '
        "ON webhook_deliveries (status, next_retry_at)"
    )

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

    # Step 5 item 5: every /fit archives a full snapshot here, old
    # versions never overwritten (unlike `baselines`, which stays a
    # materialized view of whichever version is currently ACTIVE -- every
    # existing read path, get_baseline() included, keeps working
    # unchanged). baseline_active_version is a one-row-per-project pointer
    # into this table; activating an old version copies its snapshot back
    # into `baselines`.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS baseline_versions (
            project_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            model_version_label TEXT,
            feature_types TEXT,
            reference_data TEXT,
            iqr_fences TEXT,
            categorical_baselines TEXT,
            modality TEXT,
            embedding_reference TEXT,
            embedding_model TEXT,
            calibration_config TEXT,
            reference_null_rates TEXT,
            schema_policy TEXT,
            PRIMARY KEY (project_id, version)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS baseline_active_version (
            project_id TEXT PRIMARY KEY,
            version INTEGER NOT NULL
        )
    ''')

    # Step 5 item 6: one row per STATE TRANSITION only (opened/resolved/
    # still_open) -- never one row per /analyze call, so a long stable
    # "ok" or steady "open" streak doesn't grow this table. The current
    # alert_state for a (project, baseline_version) is always just "is
    # the latest row's transition opened/still_open (-> open) or resolved
    # (-> ok)", or "ok" if no row exists yet.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alert_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            baseline_version INTEGER NOT NULL,
            ts TEXT NOT NULL,
            transition TEXT NOT NULL,
            sustained_alert INTEGER NOT NULL,
            windows_considered INTEGER NOT NULL
        )
    ''')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_alert_events_project_version_ts '
        'ON alert_events (project, baseline_version, ts)'
    )

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
        # Step 5 item 6: per-project alert policy -- NOT part of a
        # baseline version (unlike calibration_config/schema_policy), so
        # it lives on `projects`, never archived/activated with a
        # baseline. DEFAULT 1 for both -- SQLite backfills it onto every
        # existing row, matching today's un-sustained-alert behavior
        # exactly (k=1 of last m=1 means "this analysis alerted", the
        # only check that existed before this item).
        ("alert_k", "ALTER TABLE projects ADD COLUMN alert_k INTEGER DEFAULT 1"),
        ("alert_m", "ALTER TABLE projects ADD COLUMN alert_m INTEGER DEFAULT 1"),
        ("windows_considered", "ALTER TABLE analysis_runs ADD COLUMN windows_considered INTEGER"),
        # Unified table path (M1): which workflow produced the run, and the job that ran it.
        ("report_kind", "ALTER TABLE analysis_runs ADD COLUMN report_kind TEXT DEFAULT 'tabular'"),
        ("job_id", "ALTER TABLE analysis_runs ADD COLUMN job_id TEXT"),
    ]:
        try:
            cursor.execute(ddl)
        except sqlite3.OperationalError:
            pass  # column already exists

    # ---- Unified table path (docs/unified_table_plan.md, section F) ----
    # Large artifacts (embeddings, staged uploads, row-aligned reference
    # rows) live in db/blob_store.py; these tables hold only keys/metadata.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS staged_uploads (
            id TEXT PRIMARY KEY,
            owner_email TEXT NOT NULL,
            project_id TEXT NOT NULL,
            purpose TEXT NOT NULL,
            status TEXT NOT NULL,
            table_blob TEXT, zip_blob TEXT, table_filename TEXT,
            table_bytes INTEGER, zip_bytes INTEGER,
            profile TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            owner_email TEXT NOT NULL,
            owner_name TEXT,
            project_id TEXT NOT NULL,
            public_project_id TEXT,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER DEFAULT 0,
            progress_message TEXT,
            stage_id TEXT,
            payload TEXT,
            idempotency_key TEXT,
            payload_hash TEXT,
            result TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs (status, created_at)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_schemas (
            project_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            column_name TEXT NOT NULL,
            ordinal INTEGER,
            proposed_type TEXT, proposed_monitor INTEGER, confidence REAL,
            evidence TEXT, reason TEXT, alternative_type TEXT,
            final_type TEXT NOT NULL, final_monitor INTEGER NOT NULL,
            decided_by TEXT, decided_at TEXT,
            PRIMARY KEY (project_id, version, column_name)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_column_baselines (
            project_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            column_name TEXT NOT NULL,
            col_type TEXT NOT NULL,
            state TEXT,
            embeddings_blob TEXT,
            null_blob TEXT,
            PRIMARY KEY (project_id, version, column_name)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_reference_rows (
            project_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            rows_blob TEXT,
            n_rows INTEGER,
            sample_seed INTEGER,
            dedup_dropped INTEGER,
            PRIMARY KEY (project_id, version)
        )
    ''')
    # Populated from M2 (relationship engine); created now so the version
    # layout is final and M2 needs no further schema change.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS table_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL, version INTEGER NOT NULL,
            col_a TEXT NOT NULL, col_b TEXT NOT NULL, kind TEXT NOT NULL,
            proposed INTEGER, proposal_reason TEXT, proposal_strength REAL,
            final_monitor INTEGER, decided_by TEXT,
            reference_state TEXT, materiality_floor REAL,
            null_blob TEXT, probe_blob TEXT, created_at TEXT
        )
    ''')

    conn.commit()
    conn.close()

# UPDATED: Added owner_email as a parameter
def create_project(project_id: str, name: str, owner_email: str):
    """Called on every /fit (new project or re-fit). Step 5 item 6 found
    a real bug here: the old `INSERT OR REPLACE` listed only (id, name,
    owner_email), so SQLite reset every OTHER column -- including
    alert_k/alert_m -- back to their DEFAULT on every re-fit, silently
    discarding a project's alert_policy the next time it was fit (same
    bug class insert_baseline's calibration_config sentinel already
    guards against). Upserting instead touches only name/owner_email;
    every other column on an existing row -- alert_k, alert_m, and any
    future project-level column -- is left completely alone."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO projects (id, name, owner_email) VALUES (?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET name = excluded.name, owner_email = excluded.owner_email",
        (project_id, name, owner_email)
    )
    conn.commit()
    conn.close()

def _calculate_boundaries(
    reference_data: Dict[str, List[Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]], Dict[str, str]]:
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

    Categorical values are stripped of leading/trailing whitespace before
    counting uniques -- " USA" and "USA" are almost never meant to be
    different categories, just inconsistent export formatting. Case is
    left alone (not lowercased): unlike whitespace, a case difference can
    be a real, intended distinction, so this isn't collapsed by default.

    A column that exceeds the categorical cardinality cap (likely a
    free-text or ID field) no longer aborts the WHOLE fit -- it's
    excluded, like an empty or all-dropped column already was, and
    reported in the third return value instead of raised as an error
    that would block every other column too.

    Returns (fences, cleaning_summary, excluded_columns).
    cleaning_summary reports how many values were dropped per column, so
    silent data loss stays visible. excluded_columns maps a column name
    to why it isn't being monitored at all (empty, fully non-numeric
    after coercion, or over the categorical cardinality cap).
    """
    fences = []
    cleaning_summary: Dict[str, Dict[str, int]] = {}
    excluded_columns: Dict[str, str] = {}

    for feature, data in reference_data.items():
        clean_data = [x for x in data if x is not None]
        if not clean_data:
            excluded_columns[feature] = "Column is empty (every value was null)."
            continue

        numeric_values, dropped = coerce_numeric_column(clean_data)

        # 1. NUMERICAL DATA (coercion succeeded above threshold)
        if numeric_values is not None:
            if not numeric_values:
                excluded_columns[feature] = "Every value in this numeric column was dropped during cleaning."
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
            normalized = [v.strip() if isinstance(v, str) else v for v in clean_data]
            unique_values = list(set(normalized))
            if len(unique_values) > MAX_CATEGORICAL_CARDINALITY:
                excluded_columns[feature] = (
                    f"Has {len(unique_values)} unique values, over the {MAX_CATEGORICAL_CARDINALITY}-value "
                    f"cap for categorical fields -- likely a free-text or ID field that shouldn't be "
                    f"monitored as categorical."
                )
                continue
            fences.append({
                "feature_name": feature,
                "type": "categorical",
                "allowed_values": unique_values
            })

    return fences, cleaning_summary, excluded_columns

def insert_baseline(
    project_id: str,
    feature_types: dict,
    reference_data: dict,
    categorical_data: Optional[dict] = None,
    calibration_config: Optional[dict] = "__UNSET__",
    schema_policy: Optional[dict] = "__UNSET__",
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, str]]:
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

    Returns (cleaning_summary, excluded_columns). cleaning_summary is
    {field: {"dropped_non_numeric": n}} for any continuous column that had
    unconvertible cells dropped. excluded_columns maps a column name to a
    human-readable reason it isn't being monitored at all (empty, fully
    non-numeric, or over the categorical cardinality cap) -- see
    _calculate_boundaries. Both exist so callers can surface this instead
    of it being silent.
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
        elif ftype == "categorical" and feature in combined_raw_data:
            # Strip whitespace here too, in the STORED values -- not just
            # when computing the allowed_values fence below -- so " USA"
            # and "USA" agree as one category everywhere, not just in the
            # fence. Case is deliberately left alone (see
            # _calculate_boundaries' docstring).
            combined_raw_data[feature] = [
                v.strip() if isinstance(v, str) else v for v in combined_raw_data[feature]
            ]

    # Calculate IQR fences (for continuous) AND allowed values (for categorical)
    # This uses the now-cleaned merged data so fences and the stored blob agree.
    fences, fence_cleaning_summary, excluded_columns = _calculate_boundaries(combined_raw_data)
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

    return cleaning_summary, excluded_columns

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


_BASELINE_VERSION_COLUMNS = (
    "feature_types, reference_data, iqr_fences, categorical_baselines, modality, "
    "embedding_reference, embedding_model, calibration_config, reference_null_rates, schema_policy"
)


def _backfill_version_one_if_needed(project_id: str) -> None:
    """Step 5 item 5: a project fit before this feature existed has a
    `baselines` row but no `baseline_versions` history. The first time any
    version-aware call touches it, archive its CURRENT state as version 1
    -- so GET /baselines and activate work for every pre-existing project
    without a separate migration step. created_at reflects this backfill
    moment, not the project's true original fit time (never recorded)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM baseline_versions WHERE project_id = ? LIMIT 1", (project_id,))
    if cursor.fetchone():
        conn.close()
        return
    cursor.execute(f"SELECT {_BASELINE_VERSION_COLUMNS} FROM baselines WHERE project_id = ?", (project_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        f"INSERT INTO baseline_versions (project_id, version, created_at, model_version_label, "
        f"{_BASELINE_VERSION_COLUMNS}) VALUES (?, 1, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (project_id, now, *row),
    )
    cursor.execute(
        "INSERT OR REPLACE INTO baseline_active_version (project_id, version) VALUES (?, 1)",
        (project_id,),
    )
    conn.commit()
    conn.close()


def archive_baseline_version(project_id: str, model_version_label: Optional[str] = None) -> int:
    """Step 5 item 5: snapshots the CURRENT `baselines` row (just written
    by insert_baseline/insert_embedding_baseline/insert_joint_baseline)
    into baseline_versions as the next version for this project, and
    makes it the active version. Returns the new version number. Old
    versions are never deleted (no retention cap enforced).

    Deliberately does NOT call _backfill_version_one_if_needed: this is
    always invoked right after a /fit call has just written the new state
    into `baselines`, so backfilling first would double-count that same
    write as both a synthesized version 1 AND a freshly archived version
    N -- the backfill path is for READ paths touching a project that has
    never been archived at all."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM baseline_versions WHERE project_id = ?", (project_id,))
    next_version = cursor.fetchone()[0]
    cursor.execute(f"SELECT {_BASELINE_VERSION_COLUMNS} FROM baselines WHERE project_id = ?", (project_id,))
    row = cursor.fetchone()
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        f"INSERT INTO baseline_versions (project_id, version, created_at, model_version_label, "
        f"{_BASELINE_VERSION_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (project_id, next_version, now, model_version_label, *row),
    )
    cursor.execute(
        "INSERT OR REPLACE INTO baseline_active_version (project_id, version) VALUES (?, ?)",
        (project_id, next_version),
    )
    conn.commit()
    conn.close()
    return next_version


def list_baseline_versions(project_id: str) -> List[Dict[str, Any]]:
    """Newest first; each entry flags whether it's the active version."""
    _backfill_version_one_if_needed(project_id)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT version, created_at, model_version_label, modality FROM baseline_versions "
        "WHERE project_id = ? ORDER BY version DESC",
        (project_id,),
    )
    rows = cursor.fetchall()
    active = get_active_baseline_version(project_id)
    conn.close()
    return [
        {
            "version": r[0], "created_at": r[1], "model_version_label": r[2],
            "modality": r[3] or "tabular", "active": r[0] == active,
        }
        for r in rows
    ]


def get_active_baseline_version(project_id: str) -> Optional[int]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT version FROM baseline_active_version WHERE project_id = ?", (project_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def get_baseline_version(project_id: str, version: int) -> Optional[Dict[str, Any]]:
    """Same shape as get_baseline(), but for one specific archived
    version -- used by /analyze's optional baseline_version override."""
    _backfill_version_one_if_needed(project_id)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT {_BASELINE_VERSION_COLUMNS} FROM baseline_versions WHERE project_id = ? AND version = ?",
        (project_id, version),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "feature_types": json.loads(row[0]) if row[0] else {},
        "reference_data": json.loads(row[1]) if row[1] else {},
        "iqr_fences": json.loads(row[2]) if row[2] else [],
        "modality": row[4] or "tabular",
        "embedding_reference": json.loads(row[5]) if row[5] else None,
        "embedding_model": row[6],
        "calibration_config": json.loads(row[7]) if row[7] else None,
        "reference_null_rates": json.loads(row[8]) if row[8] else {},
        "schema_policy": json.loads(row[9]) if row[9] else None,
    }


def set_active_baseline_version(project_id: str, version: int) -> bool:
    """Activates an existing version: points baseline_active_version at
    it AND refreshes the `baselines` materialized-view row to match, so
    every existing read path (get_baseline included) immediately reflects
    it. Returns False if that version doesn't exist for this project."""
    _backfill_version_one_if_needed(project_id)
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT {_BASELINE_VERSION_COLUMNS} FROM baseline_versions WHERE project_id = ? AND version = ?",
        (project_id, version),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False
    cursor.execute(
        f"INSERT OR REPLACE INTO baselines (project_id, {_BASELINE_VERSION_COLUMNS}) "
        f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (project_id, *row),
    )
    cursor.execute(
        "INSERT OR REPLACE INTO baseline_active_version (project_id, version) VALUES (?, ?)",
        (project_id, version),
    )
    conn.commit()
    conn.close()
    return True


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


def insert_embedding_baseline(
    project_id: str, modality: str, embeddings, model_name: str, max_reference_samples: int = 3000,
    calibration_config: Optional[dict] = "__UNSET__",
):
    """
    Stores a text/image baseline as raw reference embeddings, capped at
    max_reference_samples, so the Domain Classifier Test has real vectors
    to retrain against on every /analyze call (mirrors how DistributionDetector
    re-fits from raw arrays for tabular baselines).

    calibration_config: Step 2 (d), 2026-10-04 -- same "__UNSET__"
    preserve-existing-value sentinel as insert_baseline's. Explicit value
    (or None, to deliberately clear it) changes it; omitted leaves
    whatever the project already has untouched across a re-fit.
    """
    embeddings_list = np.asarray(embeddings)
    if len(embeddings_list) > max_reference_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings_list), size=max_reference_samples, replace=False)
        embeddings_list = embeddings_list[idx]

    # INSERT OR REPLACE deletes and re-inserts the row, which would
    # otherwise silently wipe calibration_config back to NULL on every
    # re-fit (verified in db/crud.py's insert_baseline) unless preserved.
    if calibration_config == "__UNSET__":
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
    schema_report: Optional[dict], report_kind: str = "tabular", job_id: Optional[str] = None,
) -> int:
    """Step 5 item 2: one row per /analyze call. feature_results/schema_report
    are the already-aggregated statistics dicts the response itself returns
    -- never raw production rows, per instruction."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO analysis_runs (project, baseline_version, ts, batch_size, idempotency_key, "
        "payload_hash, decision_mode, system_alert, sustained_alert, feature_results, schema_report, "
        "report_kind, job_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            project, baseline_version, ts, batch_size, idempotency_key, payload_hash, decision_mode,
            1 if system_alert else 0,
            None if sustained_alert is None else (1 if sustained_alert else 0),
            json.dumps(feature_results) if feature_results is not None else None,
            json.dumps(schema_report) if schema_report is not None else None,
            report_kind, job_id,
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
        "windows_considered": row[12],
    }


def update_analysis_run_alert_fields(run_id: int, sustained_alert: bool, windows_considered: int) -> None:
    """Step 5 item 6: filled in right after insert_analysis_run, once the
    alert state machine has evaluated the last `m` runs on this baseline
    version (which requires this row to already exist, to be countable)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE analysis_runs SET sustained_alert = ?, windows_considered = ? WHERE id = ?",
        (1 if sustained_alert else 0, windows_considered, run_id),
    )
    conn.commit()
    conn.close()


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
        "decision_mode, system_alert, sustained_alert, feature_results, schema_report, windows_considered "
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
        "decision_mode, system_alert, sustained_alert, feature_results, schema_report, windows_considered "
        "FROM analysis_runs WHERE project = ? AND idempotency_key = ? AND ts >= ? "
        "ORDER BY ts DESC, id DESC LIMIT 1",
        (project, idempotency_key, not_before_ts),
    )
    row = cursor.fetchone()
    conn.close()
    return _row_to_analysis_run(row) if row else None


def get_alert_policy(project_id: str) -> Tuple[int, int]:
    """Step 5 item 6: per-project {k, m}, defaulting to (1, 1) -- today's
    behavior (sustained_alert is true iff this single analysis alerted).
    Lives on `projects`, not `baselines`/`baseline_versions`: it governs
    alerting behavior over time, independent of which baseline is active."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT alert_k, alert_m FROM projects WHERE id = ?", (project_id,))
    row = cursor.fetchone()
    conn.close()
    if not row or row[0] is None or row[1] is None:
        return (1, 1)
    return (row[0], row[1])


def set_alert_policy(project_id: str, k: int, m: int) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE projects SET alert_k = ?, alert_m = ? WHERE id = ?", (k, m, project_id))
    conn.commit()
    conn.close()


def recent_system_alerts_for_version(project_id: str, baseline_version: int, m: int) -> List[bool]:
    """The last up-to-`m` /analyze calls' system_alert values for this
    (project, baseline_version), newest first -- "last m analyses on the
    SAME baseline version", per instruction. Fewer than m are returned if
    fewer exist yet (e.g. just after activating a different version)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT system_alert FROM analysis_runs WHERE project = ? AND baseline_version = ? "
        "ORDER BY ts DESC, id DESC LIMIT ?",
        (project_id, baseline_version, m),
    )
    rows = cursor.fetchall()
    conn.close()
    return [bool(r[0]) for r in rows]


def get_last_alert_event(project_id: str, baseline_version: int) -> Optional[Dict[str, Any]]:
    """The most recent transition for this (project, baseline_version), if
    any -- its transition alone tells us the CURRENT state: "opened" or
    "still_open" means open, "resolved" means ok. No row at all means ok
    (never alerted on this baseline version)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT transition FROM alert_events WHERE project = ? AND baseline_version = ? "
        "ORDER BY ts DESC, id DESC LIMIT 1",
        (project_id, baseline_version),
    )
    row = cursor.fetchone()
    conn.close()
    return {"transition": row[0]} if row else None


def insert_alert_event(
    project_id: str, baseline_version: int, ts: str, transition: str,
    sustained_alert: bool, windows_considered: int,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO alert_events (project, baseline_version, ts, transition, sustained_alert, "
        "windows_considered) VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, baseline_version, ts, transition, 1 if sustained_alert else 0, windows_considered),
    )
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id


def get_alert_events(project_id: str, baseline_version: Optional[int] = None) -> List[Dict[str, Any]]:
    """Newest first; optionally scoped to one baseline version."""
    conn = get_connection()
    cursor = conn.cursor()
    query = (
        "SELECT id, project, baseline_version, ts, transition, sustained_alert, windows_considered "
        "FROM alert_events WHERE project = ?"
    )
    params: List[Any] = [project_id]
    if baseline_version is not None:
        query += " AND baseline_version = ?"
        params.append(baseline_version)
    query += " ORDER BY ts DESC, id DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": r[0], "project": r[1], "baseline_version": r[2], "ts": r[3],
            "transition": r[4], "sustained_alert": bool(r[5]), "windows_considered": r[6],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Step 5 Part 2: webhooks
# ---------------------------------------------------------------------------

def create_webhook(
    webhook_id: str, project: str, url: str, secret: str, event_filter: List[str], created_at: str,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO webhooks (id, project, url, secret, event_filter, enabled, created_at) "
        "VALUES (?, ?, ?, ?, ?, 1, ?)",
        (webhook_id, project, url, secret, json.dumps(event_filter), created_at),
    )
    conn.commit()
    conn.close()


def _row_to_webhook(row) -> Dict[str, Any]:
    return {
        "id": row[0], "project": row[1], "url": row[2], "secret": row[3],
        "event_filter": json.loads(row[4]), "enabled": bool(row[5]), "created_at": row[6],
    }


def get_webhook(webhook_id: str) -> Optional[Dict[str, Any]]:
    """Includes the secret -- for internal delivery use only. Callers
    exposing this to an HTTP response must strip it themselves."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, project, url, secret, event_filter, enabled, created_at FROM webhooks WHERE id = ?",
        (webhook_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return _row_to_webhook(row) if row else None


def list_webhooks(project: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, project, url, secret, event_filter, enabled, created_at FROM webhooks "
        "WHERE project = ? ORDER BY created_at DESC",
        (project,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_webhook(r) for r in rows]


def delete_webhook(webhook_id: str, project: str) -> bool:
    """Scoped to the project the caller owns -- returns False (no-op)
    if the webhook doesn't exist or belongs to a different project,
    same 404-not-403 isolation posture as every other project resource."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM webhooks WHERE id = ? AND project = ?", (webhook_id, project))
    changed = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return changed


def enqueue_webhook_delivery(
    webhook_id: str, event_id: Optional[int], event_type: str, payload: dict,
    next_retry_at: str, now: str,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO webhook_deliveries (webhook_id, event_id, event_type, payload, attempt_number, "
        "status, next_retry_at, created_at, updated_at) VALUES (?, ?, ?, ?, 1, 'pending', ?, ?, ?)",
        (webhook_id, event_id, event_type, json.dumps(payload), next_retry_at, now, now),
    )
    delivery_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return delivery_id


def _row_to_delivery(row) -> Dict[str, Any]:
    return {
        "id": row[0], "webhook_id": row[1], "event_id": row[2], "event_type": row[3],
        "payload": json.loads(row[4]), "attempt_number": row[5], "status": row[6],
        "status_code": row[7], "success": None if row[8] is None else bool(row[8]),
        "response_snippet": row[9], "next_retry_at": row[10],
        "created_at": row[11], "updated_at": row[12],
    }


def get_due_webhook_deliveries(now: str) -> List[Dict[str, Any]]:
    """Pending deliveries whose next_retry_at has arrived -- what the
    sweep loop processes on each tick."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, webhook_id, event_id, event_type, payload, attempt_number, status, "
        "status_code, success, response_snippet, next_retry_at, created_at, updated_at "
        "FROM webhook_deliveries WHERE status = 'pending' AND next_retry_at <= ?",
        (now,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_delivery(r) for r in rows]


def record_delivery_attempt(
    delivery_id: int, attempt_number: int, success: bool, status_code: Optional[int],
    response_snippet: Optional[str], now: str, next_retry_at: Optional[str] = None,
) -> None:
    """Updates a delivery task in place after an attempt. attempt_number
    is the count of attempts made so far (including this one).
    - success=True: status -> 'success', next_retry_at cleared.
    - success=False and next_retry_at given: status stays 'pending',
      another retry is scheduled.
    - success=False and next_retry_at is None: gave up -- status ->
      'failed'."""
    conn = get_connection()
    cursor = conn.cursor()
    if success:
        status = "success"
        next_retry_at = None
    elif next_retry_at is not None:
        status = "pending"
    else:
        status = "failed"
    cursor.execute(
        "UPDATE webhook_deliveries SET status = ?, status_code = ?, success = ?, "
        "response_snippet = ?, attempt_number = ?, next_retry_at = ?, updated_at = ? WHERE id = ?",
        (status, status_code, 1 if success else 0, response_snippet, attempt_number, next_retry_at, now, delivery_id),
    )
    conn.commit()
    conn.close()


def list_webhook_deliveries(webhook_id: str) -> List[Dict[str, Any]]:
    """Newest first. Not yet exposed via an endpoint (noted as a
    follow-up in docs/PROGRESS.md) -- used directly by tests for now."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, webhook_id, event_id, event_type, payload, attempt_number, status, "
        "status_code, success, response_snippet, next_retry_at, created_at, updated_at "
        "FROM webhook_deliveries WHERE webhook_id = ? ORDER BY created_at DESC, id DESC",
        (webhook_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_delivery(r) for r in rows]


# ---------------------------------------------------------
# Unified table path (docs/unified_table_plan.md): staged uploads, jobs,
# and per-version table artifacts. Large payloads live in db/blob_store.py;
# rows here hold only keys and metadata.
# ---------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows_as_dicts(cursor) -> List[Dict[str, Any]]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def _update_row(table: str, key_col: str, key: str, allowed: set, fields: Dict[str, Any]) -> None:
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Cannot update {table} column(s) {sorted(unknown)}")
    if not fields:
        return
    conn = get_connection()
    conn.execute(
        f"UPDATE {table} SET {', '.join(f'{c} = ?' for c in fields)} WHERE {key_col} = ?",
        (*fields.values(), key),
    )
    conn.commit()
    conn.close()


def create_stage(stage_id: str, owner_email: str, project_id: str, purpose: str, table_blob: str,
                 zip_blob: Optional[str], table_filename: str, table_bytes: int, zip_bytes: int,
                 expires_at: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO staged_uploads (id, owner_email, project_id, purpose, status, table_blob, zip_blob, "
        "table_filename, table_bytes, zip_bytes, created_at, expires_at) "
        "VALUES (?, ?, ?, ?, 'uploaded', ?, ?, ?, ?, ?, ?, ?)",
        (stage_id, owner_email, project_id, purpose, table_blob, zip_blob, table_filename,
         table_bytes, zip_bytes, _now_iso(), expires_at),
    )
    conn.commit()
    conn.close()


def get_stage(stage_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    rows = _rows_as_dicts(conn.execute("SELECT * FROM staged_uploads WHERE id = ?", (stage_id,)))
    conn.close()
    if not rows:
        return None
    row = rows[0]
    row["profile"] = json.loads(row["profile"]) if row["profile"] else None
    return row


def update_stage(stage_id: str, **fields) -> None:
    if fields.get("profile") is not None:
        fields["profile"] = json.dumps(fields["profile"])
    _update_row("staged_uploads", "id", stage_id, {"status", "profile", "table_blob", "zip_blob"}, fields)


def list_expired_stages(now_iso: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = _rows_as_dicts(conn.execute(
        "SELECT id, table_blob, zip_blob FROM staged_uploads "
        "WHERE expires_at < ? AND status NOT IN ('expired', 'deleted', 'consumed')",
        (now_iso,),
    ))
    conn.close()
    return rows


def create_job(job_id: str, owner_email: str, owner_name: str, project_id: str, public_project_id: str,
               kind: str, stage_id: Optional[str], payload: Optional[dict],
               idempotency_key: Optional[str] = None, payload_hash: Optional[str] = None) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO jobs (id, owner_email, owner_name, project_id, public_project_id, kind, status, "
        "stage_id, payload, idempotency_key, payload_hash, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)",
        (job_id, owner_email, owner_name, project_id, public_project_id, kind, stage_id,
         json.dumps(payload) if payload is not None else None, idempotency_key, payload_hash, _now_iso()),
    )
    conn.commit()
    conn.close()


def _decode_job(row: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("payload", "result"):
        row[k] = json.loads(row[k]) if row.get(k) else None
    return row


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    rows = _rows_as_dicts(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)))
    conn.close()
    return _decode_job(rows[0]) if rows else None


def claim_next_job() -> Optional[Dict[str, Any]]:
    """Atomically moves the oldest queued job to 'running' and returns it --
    the conditional UPDATE guarantees two claimers never take the same job."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1").fetchone()
        if not row:
            return None
        cur = conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ? AND status = 'queued'",
            (_now_iso(), row[0]),
        )
        conn.commit()
        if cur.rowcount != 1:
            return None
    finally:
        conn.close()
    return get_job(row[0])


def update_job(job_id: str, **fields) -> None:
    if fields.get("result") is not None:
        fields["result"] = json.dumps(fields["result"])
    _update_row("jobs", "id", job_id,
                {"status", "progress", "progress_message", "result", "error", "finished_at"}, fields)


def clear_profile_job_results(stage_id: str) -> None:
    """A profile job's result holds sample values; they must not outlive the stage."""
    conn = get_connection()
    conn.execute("UPDATE jobs SET result = NULL WHERE stage_id = ? AND kind = 'profile'", (stage_id,))
    conn.commit()
    conn.close()


def mark_running_jobs_interrupted() -> int:
    """Startup recovery: a job left 'running' belonged to a process that
    died (restart, spin-down) -- it will never finish, so say so."""
    conn = get_connection()
    cur = conn.execute(
        "UPDATE jobs SET status = 'interrupted', finished_at = ?, "
        "error = 'Interrupted by a server restart before it finished. Please resubmit.' "
        "WHERE status = 'running'",
        (_now_iso(),),
    )
    conn.commit()
    conn.close()
    return cur.rowcount


def find_job_by_idempotency_key(project_id: str, kind: str, key: str, cutoff_iso: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    rows = _rows_as_dicts(conn.execute(
        "SELECT * FROM jobs WHERE project_id = ? AND kind = ? AND idempotency_key = ? AND created_at >= ? "
        "AND status NOT IN ('failed', 'interrupted') ORDER BY created_at DESC LIMIT 1",
        (project_id, kind, key, cutoff_iso),
    ))
    conn.close()
    return _decode_job(rows[0]) if rows else None


def set_baseline_modality(project_id: str, version: int, modality: str) -> None:
    """Marks both the active `baselines` row and its archived version, so
    activating that version later restores the same kind."""
    conn = get_connection()
    conn.execute("UPDATE baselines SET modality = ? WHERE project_id = ?", (modality, project_id))
    conn.execute("UPDATE baseline_versions SET modality = ? WHERE project_id = ? AND version = ?",
                 (modality, project_id, version))
    conn.commit()
    conn.close()


def insert_table_version_artifacts(project_id: str, version: int, schema_rows: List[Dict[str, Any]],
                                   column_baselines: List[Dict[str, Any]],
                                   reference_rows: Optional[Dict[str, Any]]) -> None:
    """All per-version table artifacts in one transaction, so a version is
    never left with a schema but no column baselines (or vice versa)."""
    now = _now_iso()
    conn = get_connection()
    try:
        conn.executemany(
            "INSERT INTO table_schemas (project_id, version, column_name, ordinal, proposed_type, "
            "proposed_monitor, confidence, evidence, reason, alternative_type, final_type, final_monitor, "
            "decided_by, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(project_id, version, r["column_name"], r["ordinal"], r.get("proposed_type"),
              None if r.get("proposed_monitor") is None else int(r["proposed_monitor"]),
              r.get("confidence"), json.dumps(r.get("evidence") or {}), r.get("reason"),
              r.get("alternative_type"), r["final_type"], int(r["final_monitor"]),
              r.get("decided_by"), now) for r in schema_rows],
        )
        conn.executemany(
            "INSERT INTO table_column_baselines (project_id, version, column_name, col_type, state, "
            "embeddings_blob) VALUES (?, ?, ?, ?, ?, ?)",
            [(project_id, version, c["column_name"], c["col_type"], json.dumps(c.get("state") or {}),
              c.get("embeddings_blob")) for c in column_baselines],
        )
        if reference_rows is not None:
            conn.execute(
                "INSERT INTO table_reference_rows (project_id, version, rows_blob, n_rows, sample_seed, "
                "dedup_dropped) VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, version, reference_rows["rows_blob"], reference_rows["n_rows"],
                 reference_rows["sample_seed"], reference_rows["dedup_dropped"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_table_version(project_id: str, version: int) -> Optional[Dict[str, Any]]:
    """Everything stored for one table baseline version, or None if that
    version isn't a table version."""
    conn = get_connection()
    schema = _rows_as_dicts(conn.execute(
        "SELECT * FROM table_schemas WHERE project_id = ? AND version = ? ORDER BY ordinal", (project_id, version)))
    if not schema:
        conn.close()
        return None
    columns = _rows_as_dicts(conn.execute(
        "SELECT * FROM table_column_baselines WHERE project_id = ? AND version = ?", (project_id, version)))
    ref_rows = _rows_as_dicts(conn.execute(
        "SELECT * FROM table_reference_rows WHERE project_id = ? AND version = ?", (project_id, version)))
    relationships = _rows_as_dicts(conn.execute(
        "SELECT * FROM table_relationships WHERE project_id = ? AND version = ?", (project_id, version)))
    conn.close()
    for r in schema:
        r["evidence"] = json.loads(r["evidence"]) if r["evidence"] else {}
        r["proposed_monitor"] = None if r["proposed_monitor"] is None else bool(r["proposed_monitor"])
        r["final_monitor"] = bool(r["final_monitor"])
    for c in columns:
        c["state"] = json.loads(c["state"]) if c["state"] else {}
    return {"schema": schema, "columns": columns,
            "reference_rows": ref_rows[0] if ref_rows else None, "relationships": relationships}


def delete_table_project_data(project_id: str) -> None:
    """Rows only -- the caller deletes the project's blobs via BlobStore.delete_prefix."""
    conn = get_connection()
    for table in ("table_schemas", "table_column_baselines", "table_reference_rows",
                  "table_relationships", "staged_uploads", "jobs"):
        conn.execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))
    conn.commit()
    conn.close()