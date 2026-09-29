"""
Replaces the additive-shift synthetic drift injection (step1_validation.py's
inject_severity) with a support-preserving method, after confirming the
additive method was producing an artifact on the discrete/near-discrete
features (pickup_longitude/latitude, dropoff_longitude/latitude have only
~475 unique values out of 1.5M+ rows -- station coordinates; trip_duration
is integer seconds).

CONFIRMED (see results/synthetic_drift_realism_check.json, produced by this
script): adding a shift of 1e-9 to pickup_longitude nearly doubled its KS
statistic (0.0143 -> 0.0217) and pushed p from 0.36 to 0.039 -- a supposedly
negligible shift "detected" as drift. This happens because additive shifts
move every value off its exact discrete lattice position, making the
reference and shifted empirical distributions locally disjoint regardless of
shift magnitude -- KS's sup|F1-F2| is then dominated by point-mass
misalignment, not by the true effect size.

FIX: exponential tilting. Resample (WITH replacement) rows from the holdout
pool with weights proportional to exp(lambda * z), z = the feature's
standardized value, lambda solved numerically so the tilted sample's mean
standardized value equals the target severity. This only ever produces
values that already exist in the pool -- support-preserving by
construction, no off-lattice artifact.

The old additive-shift severity/min-drift-fraction results (already
collected in results/tabular_validation_legacy_raw.json) are NOT discarded
-- they're preserved and clearly labeled in the output as
"additive_shift_DIAGNOSTIC_ONLY", per instruction, since they demonstrate
the artifact rather than real detection sensitivity.

Run:
    python scripts/step1_severity_v2.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy import optimize, stats
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from _session_auth import mint_session_token  # noqa: E402

BACKEND = "http://127.0.0.1:8000"
SPLIT_DIR = "tests/splits"
RESULTS_DIR = "results"
SEED = 42
REFERENCE_SIZES = [5000, 50000]
CONTINUOUS_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_duration"]
CATEGORICAL_FEATURES = ["gender_id", "month"]
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
SEVERITIES = [0.02, 0.05, 0.1, 0.2, 0.5, 1.5, 3.0]
SEVERITY_BATCH_SIZE = 5000
SEVERITY_TRIALS = 5
MIN_DRIFT_FRACTION_BATCH_SIZE = 100
MIN_DRIFT_FRACTION_STEP = 10

TOKEN = mint_session_token("step1-severity-v2@example.com")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_payload(df):
    ref = {c: df[c].tolist() for c in CONTINUOUS_FEATURES}
    cat = {c: [str(v) for v in df[c].tolist()] for c in CATEGORICAL_FEATURES}
    return ref, cat


def analyze_batch(project_id, batch_df):
    ref, cat = to_payload(batch_df)
    resp = requests.post(f"{BACKEND}/analyze/{project_id}", json={"production_data": {**ref, **cat}},
                          headers=HEADERS, timeout=120)
    resp.raise_for_status()
    return resp.json()


def solve_lambda(z, target, bracket=1000):
    """Returns (lambda, achieved_target, was_capped). The tilted mean is
    bounded above by max(z) and below by min(z) as |lambda|->infinity --
    resampling can only reweight EXISTING values, never manufacture values
    beyond the observed range. If the requested target exceeds what's
    achievable, it's honestly capped (not silently ignored, not crashed)."""
    z_max, z_min = float(np.max(z)), float(np.min(z))
    safe_target = target
    was_capped = False
    if target >= z_max * 0.995:
        safe_target = z_max * 0.995
        was_capped = True
    elif target <= z_min * 0.995:
        safe_target = z_min * 0.995
        was_capped = True

    if safe_target == 0:
        return 0.0, 0.0, was_capped

    def tilted_mean(lam):
        logw = lam * z
        logw = logw - logsumexp(logw)
        return np.sum(z * np.exp(logw))

    lam = optimize.brentq(lambda l: tilted_mean(l) - safe_target, -bracket, bracket, xtol=1e-10)
    return lam, safe_target, was_capped


def tilted_resample(pool_df, col, severity, size, rng):
    """Weighted resample WITH replacement from pool_df, weights proportional
    to exp(lambda*z) chosen so the resample's mean standardized value of
    `col` equals `severity` (or the closest achievable value -- see
    solve_lambda). Resamples whole rows (not just the column), so other
    columns keep their natural joint relationship to the shifted one.
    Returns (resampled_df, achieved_severity, was_capped)."""
    x = pool_df[col].values
    mean, std = x.mean(), x.std()
    z = (x - mean) / std
    lam, achieved, was_capped = solve_lambda(z, severity)
    logw = lam * z
    logw = logw - logsumexp(logw)
    w = np.exp(logw)
    idx = rng.choice(len(pool_df), size=size, replace=True, p=w)
    return pool_df.iloc[idx].reset_index(drop=True), achieved, was_capped


def realism_check(holdout):
    """Reproduces the negligible-shift artifact check, saved for the record."""
    log("Running synthetic-drift realism check (additive vs tilted, negligible shift)...")
    rng = np.random.default_rng(SEED)
    batch = holdout.sample(n=5000, random_state=1)
    results = {}
    for col in CONTINUOUS_FEATURES:
        n_unique = holdout[col].nunique()
        ref_vals = holdout[col].values

        d_zero, p_zero = stats.ks_2samp(ref_vals, batch[col].values)
        d_tiny, p_tiny = stats.ks_2samp(ref_vals, batch[col].values + 1e-9)

        tilted_zero, _, _ = tilted_resample(holdout, col, 0.0, 5000, rng)
        d_tilt_zero, p_tilt_zero = stats.ks_2samp(ref_vals, tilted_zero[col].values)

        results[col] = {
            "n_unique_in_holdout": int(n_unique),
            "n_holdout_rows": int(len(holdout)),
            "additive_shift_0": {"D": float(d_zero), "p": float(p_zero)},
            "additive_shift_1e-9": {"D": float(d_tiny), "p": float(p_tiny)},
            "additive_artifact_confirmed": bool(d_tiny > 1.3 * d_zero or p_tiny < 0.05 <= p_zero),
            "tilted_resample_severity_0": {"D": float(d_tilt_zero), "p": float(p_tilt_zero)},
        }
        log(f"  {col}: additive D(0)={d_zero:.4f} p={p_zero:.3f} -> D(1e-9)={d_tiny:.4f} p={p_tiny:.2e} "
            f"[artifact={results[col]['additive_artifact_confirmed']}]  "
            f"tilted D(sev=0)={d_tilt_zero:.4f} p={p_tilt_zero:.3f}")
    return results


def main():
    holdout = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_n25000.csv"))
    old_raw = json.load(open(os.path.join(RESULTS_DIR, "tabular_validation_legacy_raw.json")))

    realism = realism_check(holdout)

    out = {
        "method": "exponential tilting -- resample WITH replacement from the holdout pool, weights "
                  "proportional to exp(lambda*z), lambda solved so the resample's mean standardized "
                  "value equals the target severity. Support-preserving by construction.",
        "realism_check": realism,
        "by_reference_size": {},
    }

    for ref_size in REFERENCE_SIZES:
        project_id = f"step1_val_ref{ref_size}"
        log(f"===== Reference size {ref_size}: tilted severity + min-drift-fraction =====")

        rs = {
            "additive_shift_DIAGNOSTIC_ONLY": {
                "severity": old_raw["by_reference_size"][str(ref_size)]["severity"],
                "min_drift_fraction": old_raw["by_reference_size"][str(ref_size)]["min_drift_fraction"],
                "note": "Kept only as a diagnostic of the additive-shift artifact -- NOT used for any "
                        "'smallest reliably detected severity' claim. See realism_check above.",
            },
            "tilted_resample": {"severity": {}, "min_drift_fraction": {}},
        }

        rng = np.random.default_rng(SEED)
        for sev in SEVERITIES:
            sev_result = {}
            for col in CONTINUOUS_FEATURES:
                detections = 0
                achieved_vals = []
                for t in range(SEVERITY_TRIALS):
                    batch, achieved, was_capped = tilted_resample(holdout, col, sev, SEVERITY_BATCH_SIZE, rng)
                    achieved_vals.append(achieved)
                    resp = analyze_batch(project_id, batch)
                    if resp["feature_metrics"][col]["drift_detected"]:
                        detections += 1
                sev_result[col] = {"n_trials": SEVERITY_TRIALS, "detections": detections,
                                    "achieved_severity": achieved_vals[0], "was_capped": was_capped}
            rs["tilted_resample"]["severity"][sev] = sev_result
            log(f"  severity={sev}: " + ", ".join(
                f"{c}={v['detections']}/{v['n_trials']}" + ("*capped*" if v["was_capped"] else "")
                for c, v in sev_result.items()))

        for col in CONTINUOUS_FEATURES:
            base_batch, _, _ = tilted_resample(holdout, col, 0.0, MIN_DRIFT_FRACTION_BATCH_SIZE, rng)
            fractions_tested = []
            min_fraction = None
            for n_drifted in range(0, MIN_DRIFT_FRACTION_BATCH_SIZE + 1, MIN_DRIFT_FRACTION_STEP):
                n_clean = MIN_DRIFT_FRACTION_BATCH_SIZE - n_drifted
                if n_drifted > 0:
                    drifted_part, _, _ = tilted_resample(holdout, col, 1.5, n_drifted, rng)
                else:
                    drifted_part = holdout.iloc[0:0]
                clean_part = base_batch.iloc[:n_clean]
                mixed = pd.concat([drifted_part, clean_part], ignore_index=True)
                resp = analyze_batch(project_id, mixed)
                detected = resp["feature_metrics"][col]["drift_detected"]
                fractions_tested.append({"n_drifted": n_drifted, "detected": detected})
                if detected and min_fraction is None:
                    min_fraction = n_drifted / MIN_DRIFT_FRACTION_BATCH_SIZE
            rs["tilted_resample"]["min_drift_fraction"][col] = {
                "batch_size": MIN_DRIFT_FRACTION_BATCH_SIZE,
                "drift_source": "holdout pool, exponential-tilting resample (support-preserving)",
                "severity": 1.5,
                "min_fraction": min_fraction,
                "curve": fractions_tested,
            }
            log(f"  {col}: min_drift_fraction={min_fraction}")

        out["by_reference_size"][ref_size] = rs

    out_path = os.path.join(RESULTS_DIR, "tabular_validation_severity_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
