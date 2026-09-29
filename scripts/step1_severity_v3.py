"""
Severity scale fix (Step 2 prerequisite, per instruction: "run this as part
of the legacy re-run at the start of Step 2, so legacy and calibrated use
identical synthetic data").

CONFIRMED (see results/severity_scale_investigation.json): v2's exponential
tilting on raw standardized values (z) let a few extreme trip_duration
values dominate the importance weights -- at target severity=3.0 sigma, the
effective sample size collapsed to 1,136 of 25,000 rows (4.5%), so the MEAN
shifted 14.6x but the KS statistic (which measures where the BULK of the
distribution sits, not the mean) only reached D=0.065 -- far short of what a
genuine 3-sigma-equivalent shift should produce. trip_duration is heavily
right-skewed (mean=880, median=550, max=234,243), so raw-z tilting is
dominated by its tail.

FIX: tilt on normal scores of ranks -- Phi^-1(rank/(n+1)) -- instead of raw
z. This forces the tilting variable itself to be (by construction)
approximately standard normal regardless of the underlying feature's shape,
so a given tilting severity produces a comparable, bulk-of-distribution
shift across heavy-tailed and light-tailed features alike. Verified: at
severity=0.05 (normal-score), trip_duration's D jumps from ~0.01 (raw-z,
undetectable) to ~0.03 (normal-score, detectable) -- now consistent with
the coordinate features' behavior at the same nominal severity.

Severity is now reported PRIMARILY as the achieved POPULATION D of the
tilted distribution (computed exactly via a weighted-KS statistic against
the full pool -- no resampling noise), with the normal-score sigma as a
secondary column, per instruction.

Run:
    python scripts/step1_severity_v3.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy import optimize
from scipy.special import logsumexp, ndtri

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from _session_auth import mint_session_token  # noqa: E402

BACKEND = "http://127.0.0.1:8000"
SPLIT_DIR = "tests/splits"
RESULTS_DIR = "results"
SEED = 42
REFERENCE_SIZES = [5000, 50000]
CONTINUOUS_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_duration"]
CATEGORICAL_FEATURES = ["gender_id", "month"]
SEVERITIES_NORMAL_SCORE = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
SEVERITY_BATCH_SIZE = 5000
SEVERITY_TRIALS = 5
ALPHA = 0.05
C_ALPHA = 1.36
MIN_DRIFT_FRACTION_BATCH_SIZE = 100
MIN_DRIFT_FRACTION_STEP = 10
MIN_DRIFT_FRACTION_SEVERITY = 1.5  # normal-score sigma, matches the "moderate" legacy default

TOKEN = mint_session_token("step1-severity-v3@example.com")
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


def normal_scores(x):
    n = len(x)
    ranks = pd.Series(x).rank(method="average").values
    return ndtri(ranks / (n + 1))


def solve_lambda(scores, target, bracket=1000):
    s_max, s_min = float(np.max(scores)), float(np.min(scores))
    safe_target = target
    was_capped = False
    if target >= s_max * 0.995:
        safe_target = s_max * 0.995
        was_capped = True
    elif target <= s_min * 0.995:
        safe_target = s_min * 0.995
        was_capped = True
    if safe_target == 0:
        return 0.0, 0.0, was_capped

    def tilted_mean(lam):
        logw = lam * scores
        logw = logw - logsumexp(logw)
        return np.sum(scores * np.exp(logw))

    lam = optimize.brentq(lambda l: tilted_mean(l) - safe_target, -bracket, bracket, xtol=1e-10)
    return lam, safe_target, was_capped


def tilt_weights(scores, severity):
    lam, achieved, was_capped = solve_lambda(scores, severity)
    logw = lam * scores
    logw = logw - logsumexp(logw)
    w = np.exp(logw)
    return w, achieved, was_capped


def weighted_ks(x, weights):
    """Exact population-level KS statistic between the unweighted empirical
    distribution of x and the weighted (tilted) distribution of the SAME
    values -- no resampling noise, a direct computation of what the tilted
    population's D would be at infinite batch size."""
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = weights[order]
    f_unweighted = np.arange(1, len(x) + 1) / len(x)
    f_weighted = np.cumsum(w_sorted)
    return float(np.max(np.abs(f_unweighted - f_weighted)))


def tilted_resample(pool_df, col, scores, weights, size, rng):
    idx = rng.choice(len(pool_df), size=size, replace=True, p=weights)
    return pool_df.iloc[idx].reset_index(drop=True)


def main():
    holdout = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_n25000.csv"))

    # --- Investigation record: raw-z vs normal-score tilting for trip_duration ---
    log("Recording the raw-z vs normal-score investigation for trip_duration...")
    investigation = {}
    x = holdout["trip_duration"].values
    z = (x - x.mean()) / x.std()
    ns = normal_scores(x)
    for method_name, scores in [("raw_z", z), ("normal_score", ns)]:
        rows = []
        for sev in [0.02, 0.05, 0.1, 0.2, 0.5, 1.5, 3.0]:
            w, achieved, capped = tilt_weights(scores, sev)
            ess = float(1.0 / np.sum(w ** 2))
            pop_d = weighted_ks(x, w)
            rows.append({"target_severity": sev, "achieved_severity": achieved, "was_capped": capped,
                         "effective_sample_size": ess, "pool_size": len(holdout), "population_D": pop_d})
        investigation[method_name] = rows
    with open(os.path.join(RESULTS_DIR, "severity_scale_investigation.json"), "w", encoding="utf-8") as f:
        json.dump({"feature": "trip_duration", "methods": investigation}, f, indent=2)
    log("Wrote results/severity_scale_investigation.json")
    for row in investigation["raw_z"]:
        log(f"  raw_z sev={row['target_severity']:.2f}: ESS={row['effective_sample_size']:.0f}/{len(holdout)} pop_D={row['population_D']:.4f}")
    for row in investigation["normal_score"]:
        log(f"  normal_score sev={row['target_severity']:.2f}: ESS={row['effective_sample_size']:.0f}/{len(holdout)} pop_D={row['population_D']:.4f}")

    # --- Full severity re-run with normal-score tilting, all continuous features ---
    old_raw = json.load(open(os.path.join(RESULTS_DIR, "tabular_validation_legacy_raw.json")))
    out = {
        "method": "Normal-score tilting: tilt on Phi^-1(rank/(n+1)) instead of raw standardized z, "
                  "so tilting severity is comparable across heavy-tailed and light-tailed features. "
                  "Severity reported primarily as the achieved POPULATION D (exact weighted-KS "
                  "computation, no resampling noise), normal-score sigma kept as a secondary column.",
        "by_reference_size": {},
    }

    for ref_size in REFERENCE_SIZES:
        project_id = f"step1_val_ref{ref_size}"
        log(f"===== Reference size {ref_size}: normal-score severity re-run =====")
        rng = np.random.default_rng(SEED)
        rs = {"severity": {}}

        for col in CONTINUOUS_FEATURES:
            scores = normal_scores(holdout[col].values)
            col_results = []
            for sev in SEVERITIES_NORMAL_SCORE:
                w, achieved, capped = tilt_weights(scores, sev)
                pop_d = weighted_ks(holdout[col].values, w)
                detections = 0
                for t in range(SEVERITY_TRIALS):
                    batch = tilted_resample(holdout, col, scores, w, SEVERITY_BATCH_SIZE, rng)
                    resp = analyze_batch(project_id, batch)
                    if resp["feature_metrics"][col]["drift_detected"]:
                        detections += 1
                col_results.append({
                    "target_normal_score_severity": sev, "achieved_normal_score_severity": achieved,
                    "was_capped": capped, "population_D": pop_d,
                    "n_trials": SEVERITY_TRIALS, "detections": detections,
                })
            rs["severity"][col] = col_results
            log(f"  {col}: " + ", ".join(f"D={r['population_D']:.3f}->{r['detections']}/{r['n_trials']}" for r in col_results))

        # --- Minimum drift fraction, same normal-score tilting method ---
        rs["min_drift_fraction"] = {}
        for col in CONTINUOUS_FEATURES:
            scores = normal_scores(holdout[col].values)
            w_zero, _, _ = tilt_weights(scores, 0.0)
            w_drift, achieved_drift, capped_drift = tilt_weights(scores, MIN_DRIFT_FRACTION_SEVERITY)
            pop_d_drift = weighted_ks(holdout[col].values, w_drift)
            base_batch = tilted_resample(holdout, col, scores, w_zero, MIN_DRIFT_FRACTION_BATCH_SIZE, rng)
            fractions_tested = []
            min_fraction = None
            for n_drifted in range(0, MIN_DRIFT_FRACTION_BATCH_SIZE + 1, MIN_DRIFT_FRACTION_STEP):
                n_clean = MIN_DRIFT_FRACTION_BATCH_SIZE - n_drifted
                if n_drifted > 0:
                    drifted_part = tilted_resample(holdout, col, scores, w_drift, n_drifted, rng)
                else:
                    drifted_part = holdout.iloc[0:0]
                clean_part = base_batch.iloc[:n_clean]
                mixed = pd.concat([drifted_part, clean_part], ignore_index=True)
                resp = analyze_batch(project_id, mixed)
                detected = resp["feature_metrics"][col]["drift_detected"]
                fractions_tested.append({"n_drifted": n_drifted, "detected": detected})
                if detected and min_fraction is None:
                    min_fraction = n_drifted / MIN_DRIFT_FRACTION_BATCH_SIZE
            rs["min_drift_fraction"][col] = {
                "batch_size": MIN_DRIFT_FRACTION_BATCH_SIZE,
                "drift_source": "holdout pool, normal-score exponential-tilting resample",
                "normal_score_severity": MIN_DRIFT_FRACTION_SEVERITY,
                "achieved_normal_score_severity": achieved_drift,
                "was_capped": capped_drift,
                "population_D_of_drifted_slice": pop_d_drift,
                "min_fraction": min_fraction,
                "curve": fractions_tested,
            }
            log(f"  min_drift_fraction {col}: D_drifted_slice={pop_d_drift:.4f} min_fraction={min_fraction}")

        out["by_reference_size"][ref_size] = rs

    # --- Consistency check: smallest reliably-detected D vs the KS critical-value floor ---
    log("Checking consistency of smallest reliably-detected D against the theoretical floor...")
    consistency = {}
    for ref_size in REFERENCE_SIZES:
        floor = C_ALPHA / np.sqrt(ref_size)
        per_feature_min_d = {}
        for col in CONTINUOUS_FEATURES:
            rows = out["by_reference_size"][ref_size]["severity"][col]
            reliable = [r for r in rows if r["detections"] == r["n_trials"]]
            min_d = min((r["population_D"] for r in reliable), default=None)
            per_feature_min_d[col] = min_d
        consistency[ref_size] = {"theoretical_floor": float(floor), "smallest_reliable_D_per_feature": per_feature_min_d}
        log(f"  m={ref_size}: theoretical floor={floor:.4f}, smallest reliable D per feature={per_feature_min_d}")
    out["consistency_check"] = consistency

    out_path = os.path.join(RESULTS_DIR, "tabular_validation_severity_v3.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
