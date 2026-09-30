# Step 5 recon (2026-10-01, no code)

**NULL/NaN — inconsistent, no explicit policy.**
- `/fit`: `coerce_numeric_column` drops `None`/uncoercible values from
  continuous columns before storage, reports a count. Stored data is
  NaN-free.
- `/analyze`: **zero cleaning.** Verified: a NaN in a continuous batch
  makes `scipy.stats.ks_2samp` silently return `statistic=nan,
  pvalue=nan` (no crash) -> `drift_detected=False` from `nan < alpha`, a
  meaningless result reported as valid. `inf` doesn't crash (treated as
  an extreme finite value).

**Missing/extra column, dtype change — all silent today.**
- Extra production column: `if feature_name not in self.reference_data:
  continue` -- silently skipped, no trace in the response.
- Missing production column: loop only iterates `production_features`,
  so an absent feature never appears in `feature_metrics` -- no warning.
- Dtype change (continuous feature arrives as strings): verified,
  doesn't crash -- `ks_2samp` returns `statistic=1.0, p~0` (spurious
  "total drift"), silently.

**`/logs`** (`insert_log`/`get_logs`, `/predict`-only, unrelated to
`/analyze`): stores the FULL raw `input_data` dict per request as JSON.
Pre-existing pattern -- new `analysis_runs`/`alert_events` tables must
NOT replicate this (statistics only, per instruction).

**`/fit` on an existing project: full overwrite, no versioning.**
`baselines` PK is `project_id` (one row/project); `INSERT OR REPLACE`
destroys the prior reference on every re-fit (preserves
`calibration_config` via the `__UNSET__` sentinel when unmentioned).
Confirms item 5's premise directly.

**SQLite: no WAL, no explicit busy_timeout PRAGMA.**
`get_connection()` is bare `sqlite3.connect(DB_PATH)` -- default
rollback journal, default Python `sqlite3` `timeout=5.0s` implicit wait,
nothing explicit configured. Relevant to whether Part 1's new tables
need write-contention handling now (fuller WAL+busy_timeout work is
item 9, not in this Part).

No proposal here changes a default -- continuing to items 1-7.
