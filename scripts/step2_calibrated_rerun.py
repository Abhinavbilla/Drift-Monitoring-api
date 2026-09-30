"""
Step 2 (c): calibrated re-run of the Step 1 sweep + A/A test, reusing the
EXACT SAME batch-drawing seeds as scripts/step1_validation.py, so the
legacy (already collected in results/tabular_validation_legacy_raw.json)
and calibrated runs are compared on byte-identical batches -- only the
decision logic differs.

Scope, deliberately: sweep + A/A only (the two comparisons explicitly
requested for the side-by-side). Severity/min-drift-fraction are not
re-run in calibrated mode here -- not asked for in this step, and legacy's
synthetic-injection method was already fixed and is unaffected by
decision_mode (severity detection uses the same two-gate engine, already
unit-tested in tests/test_detector_calibration.py).

Run:
    python scripts/step2_calibrated_rerun.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from _session_auth import mint_session_token  # noqa: E402

BACKEND = "http://127.0.0.1:8000"
SPLIT_DIR = "tests/splits"
RESULTS_DIR = "results"
SEED = 42  # matches step1_validation.py exactly
REFERENCE_SIZES = [5000, 50000]
HOLDOUT_SIZE = 25000
MONTHS = [4, 5, 6]
SWEEP_SIZES = [1000, 3000, 5000, 10000, 15000, 20000, 50000]
TRIALS_TARGET_PER_MONTH = 8
AA_TRIALS_PER_SIZE = 100
AA_SIZES = [s for s in SWEEP_SIZES if s <= HOLDOUT_SIZE]

CONTINUOUS_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_duration"]
CATEGORICAL_FEATURES = ["gender_id", "month"]
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

TOKEN = mint_session_token("step2-calibrated-rerun@example.com")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_payload(df):
    ref = {c: df[c].tolist() for c in CONTINUOUS_FEATURES}
    cat = {c: [str(v) for v in df[c].tolist()] for c in CATEGORICAL_FEATURES}
    return ref, cat


def fit_calibrated_baseline(project_id, ref_df):
    ref, cat = to_payload(ref_df)
    resp = requests.post(
        f"{BACKEND}/fit/{project_id}",
        json={"reference_data": ref, "categorical_data": cat,
              "calibration_config": {"decision_mode": "calibrated"}},  # everything else = locked defaults
        headers=HEADERS, timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def analyze_batch(project_id, batch_df):
    ref, cat = to_payload(batch_df)
    resp = requests.post(f"{BACKEND}/analyze/{project_id}", json={"production_data": {**ref, **cat}},
                          headers=HEADERS, timeout=120)
    resp.raise_for_status()
    return resp.json()


def disjoint_batches(df, size, trials_target, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    max_trials = len(df) // size
    n_trials = min(trials_target, max_trials)
    batches = [df.iloc[idx[i * size:(i + 1) * size]].reset_index(drop=True) for i in range(n_trials)]
    return batches, n_trials, n_trials < trials_target


def with_replacement_batches(df, size, n_trials, seed):
    rng = np.random.default_rng(seed)
    batches = []
    for t in range(n_trials):
        batches.append(df.sample(n=size, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True))
    return batches


def main():
    log("Loading splits (identical to step1_validation.py)...")
    refs = {size: pd.read_csv(os.path.join(SPLIT_DIR, f"reference_n{size}.csv")) for size in REFERENCE_SIZES}
    holdout = pd.read_csv(os.path.join(SPLIT_DIR, f"holdout_n{HOLDOUT_SIZE}.csv"))
    months = {m: pd.read_csv(os.path.join(SPLIT_DIR, f"production_month_{m:02d}.csv")) for m in MONTHS}

    all_results = {"config": {"decision_mode": "calibrated", "reference_sizes": REFERENCE_SIZES,
                               "sweep_sizes": SWEEP_SIZES, "aa_sizes": AA_SIZES,
                               "trials_target_per_month": TRIALS_TARGET_PER_MONTH,
                               "aa_trials_per_size": AA_TRIALS_PER_SIZE, "seed": SEED,
                               "note": "Same batch-drawing seeds as step1_validation.py -- byte-identical "
                                       "batches to the legacy run, only calibrated_config differs."},
                   "by_reference_size": {}}

    for ref_size in REFERENCE_SIZES:
        project_id = f"step2_val_ref{ref_size}_calibrated"
        log(f"===== Reference size {ref_size} (calibrated) =====")
        log(f"Fitting calibrated baseline (project={project_id}, n={len(refs[ref_size])})...")
        fit_calibrated_baseline(project_id, refs[ref_size])

        rs = {"sweep": {}, "aa_test": {}}

        log("Running batch-size sweep (identical batches to legacy)...")
        for size in SWEEP_SIZES:
            size_result = {"per_month": {}, "pooled": None}
            pooled_raw = []
            for m, mdf in months.items():
                batches, n_trials, capped = disjoint_batches(mdf, size, TRIALS_TARGET_PER_MONTH, seed=SEED + size + m)
                month_raw = [analyze_batch(project_id, b)["feature_metrics"] for b in batches]
                size_result["per_month"][m] = {"n_trials": n_trials, "capped_by_disjointness": capped,
                                                "raw_responses": month_raw}
                pooled_raw.extend(month_raw)
            size_result["pooled"] = {"n_trials_total": len(pooled_raw), "raw_responses": pooled_raw}
            log(f"  size={size}: " + ", ".join(f"m{m}={size_result['per_month'][m]['n_trials']}t" for m in months))
            rs["sweep"][size] = size_result

        log("Running A/A test (identical batches to legacy)...")
        for size in AA_SIZES:
            batches = with_replacement_batches(holdout, size, AA_TRIALS_PER_SIZE, seed=SEED + 1000 + size)
            raw = [analyze_batch(project_id, b)["feature_metrics"] for b in batches]
            rs["aa_test"][size] = {"n_trials": AA_TRIALS_PER_SIZE, "raw_responses": raw}
            log(f"  A/A size={size}: {AA_TRIALS_PER_SIZE} trials done")

        all_results["by_reference_size"][ref_size] = rs

    out_path = os.path.join(RESULTS_DIR, "tabular_validation_calibrated_raw.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
