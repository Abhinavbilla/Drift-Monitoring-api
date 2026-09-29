"""
Step 1 tabular validation reconciliation, scoped to Apr-Jun 2016 (the only
production data actually available locally -- see
results/citi_bike_provenance_forensics.md). Runs against the live backend
via HTTP, matching the existing tests/test_drift_engine.py methodology
(never re-implements the detector's own math -- that would test a
reimplementation, not the real system).

ALL config values below are fixed and recorded verbatim in every output
file. Seeds are fixed. Nothing here is tuned against its own output.

Inputs (produced by scripts/split_citi_bike.py, already run for both sizes):
    tests/splits/reference_n{5000,50000}.csv
    tests/splits/holdout_n25000.csv          (disjoint from both references)
    tests/splits/production_month_{04,05,06}.csv

Batch design:
  - Real per-month production batches (sweep + classification): DISJOINT
    non-overlapping row slices within each month -- no row is reused across
    trials for the same (month, batch_size). Number of trials per
    (month, size) = min(TRIALS_TARGET, floor(month_rows / size)); this cap
    is recorded per row in the output, never silently applied.
  - A/A (iid test-calibration check): independent random draws (WITH
    replacement ACROSS trials, WITHOUT replacement within a single trial's
    rows) from the fixed 25,000-row holdout pool -- the ">=100 batches per
    size" requirement is mathematically impossible as strictly disjoint
    batches from a 25,000-row pool at most sizes, so trials necessarily
    overlap each other (never within one trial). Sizes exceeding the
    holdout pool itself (50,000) are excluded, not silently downgraded.
  - Synthetic severity injection: independent random draws (with
    replacement across trials, without replacement within a trial) from
    the same holdout pool, at a fixed batch size, injected via the existing
    inject_synthetic_drift-equivalent (mean shift = severity * baseline_std).

Run:
    python scripts/step1_validation.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from _session_auth import mint_session_token  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG -- every value here is recorded verbatim in the output manifest.
# ---------------------------------------------------------------------------
BACKEND = "http://127.0.0.1:8000"
SPLIT_DIR = "tests/splits"
RESULTS_DIR = "results"
SEED = 42
ALPHA = 0.05  # the engine's actual KS alpha (drift/detector.py p_value_threshold default)
PSI_THRESHOLD = 0.2  # drift/detector.py hardcoded
REFERENCE_SIZES = [5000, 50000]
HOLDOUT_SIZE = 25000
MONTHS = [4, 5, 6]  # Apr, May, Jun 2016 -- the only production data available locally
SWEEP_SIZES = [1000, 3000, 5000, 10000, 15000, 20000, 50000]
TRIALS_TARGET_PER_MONTH = 8  # capped by disjointness; actual count recorded per row
AA_TRIALS_PER_SIZE = 100
AA_SIZES = [s for s in SWEEP_SIZES if s <= HOLDOUT_SIZE]  # 50000 excluded -- exceeds holdout pool
SEVERITIES = [0.02, 0.05, 0.1, 0.2, 0.5, 1.5, 3.0]
SEVERITY_BATCH_SIZE = 5000
SEVERITY_TRIALS = 5  # disjoint within the 25,000-row holdout at this size (floor(25000/5000)=5)
EFFECT_SIZE_GT_THRESHOLDS = [0.01, 0.02, 0.05]  # population D thresholds for the alt. ground truth
MIN_DRIFT_FRACTION_BATCH_SIZE = 100
MIN_DRIFT_FRACTION_STEP = 10
MIN_DRIFT_FRACTION_SEVERITY = "moderate"  # 1.5 sigma, matches legacy default

CONTINUOUS_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_duration"]
CATEGORICAL_FEATURES = ["gender_id", "month"]
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

TOKEN = mint_session_token("step1-validation@example.com")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

run_log = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    run_log.append(line)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits():
    refs = {}
    for size in REFERENCE_SIZES:
        path = os.path.join(SPLIT_DIR, f"reference_n{size}.csv")
        refs[size] = pd.read_csv(path)
    holdout = pd.read_csv(os.path.join(SPLIT_DIR, f"holdout_n{HOLDOUT_SIZE}.csv"))
    months = {}
    for m in MONTHS:
        months[m] = pd.read_csv(os.path.join(SPLIT_DIR, f"production_month_{m:02d}.csv"))
    return refs, holdout, months


def to_payload(df):
    ref = {c: df[c].tolist() for c in CONTINUOUS_FEATURES}
    cat = {c: [str(v) for v in df[c].tolist()] for c in CATEGORICAL_FEATURES}
    return ref, cat


# ---------------------------------------------------------------------------
# Backend interaction
# ---------------------------------------------------------------------------

def fit_baseline(project_id, ref_df):
    ref, cat = to_payload(ref_df)
    resp = requests.post(f"{BACKEND}/fit/{project_id}", json={"reference_data": ref, "categorical_data": cat},
                          headers=HEADERS, timeout=120)
    resp.raise_for_status()
    return resp.json()


def analyze_batch(project_id, batch_df):
    ref, cat = to_payload(batch_df)
    payload = {"production_data": {**ref, **cat}}
    resp = requests.post(f"{BACKEND}/analyze/{project_id}", json=payload, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Ground truth (independent of the API -- scipy/PSI computed directly)
# ---------------------------------------------------------------------------

def compute_psi(ref_series, cur_series, epsilon=0.0001):
    cats = sorted(set(ref_series.astype(str).unique()) | set(cur_series.astype(str).unique()))
    ref_p = ref_series.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0) + 0
    cur_p = cur_series.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0) + 0
    ref_p = ref_p.replace(0, epsilon)
    cur_p = cur_p.replace(0, epsilon)
    return float(((cur_p - ref_p) * np.log(cur_p / ref_p)).sum())


def population_ground_truth(ref_df, pooled_prod_df):
    """Ground truth using the FULL reference vs FULL pooled Apr-Jun production
    -- independent of the API, using scipy (continuous) / direct PSI
    (categorical). KEEPS THE CURRENT (legacy) DEFINITION EXACTLY:
    continuous = p_value < ALPHA (matches tests/test_drift_engine.py's
    compute_ground_truth -- p-value based, NOT an effect-size threshold);
    categorical = PSI > 0.2. The effect-size-threshold variant is a
    SEPARATE, additional ground truth computed by
    ground_truth_at_effect_floor(), not a replacement for this one."""
    gt = {}
    details = {}
    for col in CONTINUOUS_FEATURES:
        d, p = stats.ks_2samp(ref_df[col].values, pooled_prod_df[col].values)
        gt[col] = bool(p < ALPHA)
        details[col] = {"population_D": float(d), "ks_pvalue": float(p)}
    for col in CATEGORICAL_FEATURES:
        psi = compute_psi(ref_df[col], pooled_prod_df[col])
        gt[col] = bool(psi > PSI_THRESHOLD)
        details[col] = {"population_PSI": psi}
    return gt, details


def ground_truth_at_effect_floor(details, floor):
    """Alternate ground truth: continuous features flagged iff population D >= floor.
    Categorical unchanged (PSI > 0.2, no effect-size variant requested for it)."""
    gt = {}
    for col in CONTINUOUS_FEATURES:
        gt[col] = bool(details[col]["population_D"] >= floor)
    for col in CATEGORICAL_FEATURES:
        gt[col] = bool(details[col]["population_PSI"] > PSI_THRESHOLD)
    return gt


# ---------------------------------------------------------------------------
# Disjoint batch drawing
# ---------------------------------------------------------------------------

def disjoint_batches(df, size, trials_target, seed):
    """Non-overlapping row slices from a shuffled copy of df. Returns up to
    min(trials_target, floor(len(df)/size)) batches; never more than fit
    without reuse. Returns (batches, actual_trial_count, capped_by_disjointness)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    max_trials = len(df) // size
    n_trials = min(trials_target, max_trials)
    batches = [df.iloc[idx[i * size:(i + 1) * size]].reset_index(drop=True) for i in range(n_trials)]
    return batches, n_trials, n_trials < trials_target


def with_replacement_batches(df, size, n_trials, seed):
    """n_trials independent draws of `size` rows WITHOUT replacement within
    a trial, but trials may overlap each other (only used where the pool is
    too small to support the requested trial count disjointly)."""
    rng = np.random.default_rng(seed)
    batches = []
    for t in range(n_trials):
        batches.append(df.sample(n=size, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True))
    return batches


def inject_severity(df, col, severity, baseline_std):
    """Deterministic mean-shift injection: severity * baseline_std, added to
    every row's value for `col`. baseline_std is passed explicitly (computed
    once from the reference/holdout pool) rather than recomputed per-batch,
    so a small batch's own sample std doesn't distort the intended shift."""
    out = df.copy()
    if len(out) > 0:
        out[col] = out[col] + severity * baseline_std
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log("Loading splits...")
    refs, holdout, months = load_splits()
    pooled_prod = pd.concat(months.values(), ignore_index=True)
    log(f"Reference sizes available: {list(refs.keys())}, holdout={len(holdout)}, "
        f"months={{m: len(df) for m,df in months.items()}}".replace("{m: len(df) for m,df in months.items()}",
                                                                      str({m: len(d) for m, d in months.items()})))

    all_results = {
        "config": {
            "backend": BACKEND, "seed": SEED, "alpha": ALPHA, "psi_threshold": PSI_THRESHOLD,
            "reference_sizes": REFERENCE_SIZES, "holdout_size": HOLDOUT_SIZE, "months": MONTHS,
            "sweep_sizes": SWEEP_SIZES, "trials_target_per_month": TRIALS_TARGET_PER_MONTH,
            "aa_trials_per_size": AA_TRIALS_PER_SIZE, "aa_sizes": AA_SIZES,
            "severities": SEVERITIES, "severity_batch_size": SEVERITY_BATCH_SIZE,
            "severity_trials": SEVERITY_TRIALS, "effect_size_gt_thresholds": EFFECT_SIZE_GT_THRESHOLDS,
            "min_drift_fraction_batch_size": MIN_DRIFT_FRACTION_BATCH_SIZE,
            "min_drift_fraction_step": MIN_DRIFT_FRACTION_STEP,
            "min_drift_fraction_severity": MIN_DRIFT_FRACTION_SEVERITY,
            "data_scope": "Apr-Jun 2016 (production); Jan-Mar 2016 (reference/holdout)",
        },
        "by_reference_size": {},
    }

    for ref_size in REFERENCE_SIZES:
        log(f"===== REFERENCE SIZE {ref_size} =====")
        ref_df = refs[ref_size]
        project_id = f"step1_val_ref{ref_size}"

        log(f"Fitting baseline (project={project_id}, n={len(ref_df)})...")
        fit_baseline(project_id, ref_df)

        log("Computing population ground truth (pooled Apr-Jun)...")
        gt_default, gt_details = population_ground_truth(ref_df, pooled_prod)
        gt_variants = {str(f): ground_truth_at_effect_floor(gt_details, f) for f in EFFECT_SIZE_GT_THRESHOLDS}

        log("Computing per-month population ground truth...")
        per_month_gt = {}
        for m, mdf in months.items():
            gt_m, details_m = population_ground_truth(ref_df, mdf)
            per_month_gt[m] = {"ground_truth": gt_m, "details": details_m}

        ref_results = {
            "reference_size": ref_size,
            "ground_truth_pooled": {"ground_truth": gt_default, "details": gt_details},
            "ground_truth_effect_size_variants": gt_variants,
            "ground_truth_per_month": per_month_gt,
            "sweep": {},
            "aa_test": {},
            "severity": {},
            "min_drift_fraction": {},
        }

        # ---------------- SWEEP (disjoint per-month batches) ----------------
        log("Running batch-size sweep (disjoint per-month batches)...")
        for size in SWEEP_SIZES:
            size_result = {"per_month": {}, "pooled": None}
            pooled_raw = []
            for m, mdf in months.items():
                batches, n_trials, capped = disjoint_batches(mdf, size, TRIALS_TARGET_PER_MONTH, seed=SEED + size + m)
                month_raw = []
                for batch in batches:
                    resp = analyze_batch(project_id, batch)
                    month_raw.append(resp["feature_metrics"])
                size_result["per_month"][m] = {
                    "n_trials": n_trials, "capped_by_disjointness": capped,
                    "max_possible_disjoint_trials": len(mdf) // size,
                    "raw_responses": month_raw,
                }
                pooled_raw.extend(month_raw)
            size_result["pooled"] = {"n_trials_total": len(pooled_raw), "raw_responses": pooled_raw}
            log(f"  size={size}: " + ", ".join(f"m{m}={size_result['per_month'][m]['n_trials']}t" for m in months))
            ref_results["sweep"][size] = size_result

        # ---------------- A/A TEST (iid test-calibration check) ----------------
        log("Running A/A test (iid test-calibration check, holdout pool)...")
        for size in AA_SIZES:
            batches = with_replacement_batches(holdout, size, AA_TRIALS_PER_SIZE, seed=SEED + 1000 + size)
            raw = [analyze_batch(project_id, b)["feature_metrics"] for b in batches]
            ref_results["aa_test"][size] = {"n_trials": AA_TRIALS_PER_SIZE, "raw_responses": raw}
            log(f"  A/A size={size}: {AA_TRIALS_PER_SIZE} trials done")

        # ---------------- SEVERITY (extended) ----------------
        log("Running extended severity sweep...")
        holdout_std = {col: holdout[col].std() for col in CONTINUOUS_FEATURES}
        for sev in SEVERITIES:
            sev_result = {}
            for col in CONTINUOUS_FEATURES:
                batches, n_trials, capped = disjoint_batches(holdout, SEVERITY_BATCH_SIZE, SEVERITY_TRIALS,
                                                               seed=SEED + 2000 + int(sev * 1000))
                detections = 0
                for batch in batches:
                    injected = inject_severity(batch, col, sev, holdout_std[col])
                    resp = analyze_batch(project_id, injected)
                    if resp["feature_metrics"][col]["drift_detected"]:
                        detections += 1
                sev_result[col] = {"n_trials": n_trials, "detections": detections}
            ref_results["severity"][sev] = sev_result
            log(f"  severity={sev}: " + ", ".join(f"{c}={v['detections']}/{v['n_trials']}" for c, v in sev_result.items()))

        # ---------------- MINIMUM DRIFT FRACTION (renamed from "latency") ----------------
        log("Running minimum-drift-fraction sweep...")
        base_batch, _, _ = disjoint_batches(holdout, MIN_DRIFT_FRACTION_BATCH_SIZE, 1, seed=SEED + 3000)
        base_batch = base_batch[0]
        for col in CONTINUOUS_FEATURES:
            fractions_tested = []
            min_fraction = None
            for n_drifted in range(0, MIN_DRIFT_FRACTION_BATCH_SIZE + 1, MIN_DRIFT_FRACTION_STEP):
                n_clean = MIN_DRIFT_FRACTION_BATCH_SIZE - n_drifted
                drifted_part = inject_severity(
                    holdout.sample(n=n_drifted, random_state=SEED) if n_drifted > 0 else holdout.iloc[0:0],
                    col, 1.5, holdout_std[col]
                )
                clean_part = base_batch.iloc[:n_clean]
                mixed = pd.concat([drifted_part, clean_part], ignore_index=True)
                resp = analyze_batch(project_id, mixed)
                detected = resp["feature_metrics"][col]["drift_detected"]
                fractions_tested.append({"n_drifted": n_drifted, "detected": detected})
                if detected and min_fraction is None:
                    min_fraction = n_drifted / MIN_DRIFT_FRACTION_BATCH_SIZE
            ref_results["min_drift_fraction"][col] = {
                "batch_size": MIN_DRIFT_FRACTION_BATCH_SIZE,
                "drift_source": "holdout pool, mean-shift injection",
                "severity": MIN_DRIFT_FRACTION_SEVERITY,
                "min_fraction": min_fraction,
                "curve": fractions_tested,
            }
            log(f"  {col}: min_drift_fraction={min_fraction}")

        all_results["by_reference_size"][ref_size] = ref_results

    out_path = os.path.join(RESULTS_DIR, "tabular_validation_legacy_raw.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Wrote raw results -> {out_path}")

    log_path = os.path.join(RESULTS_DIR, "step1_run_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(run_log))
    log(f"Wrote run log -> {log_path}")


if __name__ == "__main__":
    main()
