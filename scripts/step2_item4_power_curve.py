"""
Step 2 review item 4 (2026-09-30 instruction): near-floor power curve +
reference-draw variability, calibrated mode only.

Design:
  - >=10 independent, mutually disjoint reference draws per reference size
    (5000, 50000), sampled from the Jan-Mar pool EXCLUDING the fixed
    25,000-row holdout (tests/splits/holdout_n25000.csv) -- same holdout
    construction as scripts/split_citi_bike.py (SEED=42, holdout drawn
    first from a shuffle of the full pool, everything after it is the
    remaining pool reference draws are sampled from).
  - Each draw is /fit as its OWN calibrated project
    (step2_item4_ref{size}_draw{i}) -- decision_mode=calibrated, all other
    config at locked defaults (alpha=0.05, Holm, KS floor=0.05).
  - Synthetic drift: normal-score exponential tilting (Phi^-1(rank/(n+1)),
    per the severity-scale fix from Step 1 -- see step1_severity_v3.py,
    whose tilt_weights/weighted_ks/normal_scores are reused verbatim here)
    on the SAME fixed holdout pool, one feature tilted at a time (the other
    4 continuous + 2 categorical columns ride along at whatever value
    happens to co-occur with the selected holdout row -- consistent with
    every other synthetic-injection script in this project, e.g.
    step1_severity_v3.py's severity sweep).
  - achieved population D in {0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10} per
    continuous feature (computed once per (feature, target) via exact
    weighted-KS against the whole holdout -- no resampling noise -- then
    reused for every draw/trial at that target, since weights depend only
    on the fixed holdout pool, not on which reference draw is being
    tested).
  - >=20 independent trial batches PER REFERENCE DRAW (not shared across
    draws -- "at least 20 disjoint batches per reference draw" is read as
    each draw getting its own independently-sampled set), at n in
    {1000, 5000, 20000}.
  - Only the TARGET feature's metric is stored per trial (statistic,
    p_value, p_value_adjusted, effect_size, effect_floor, significant,
    material, drift_detected) -- the other 6 features' responses are
    incidental nulls, not the object of this experiment, and storing all 7
    per trial across 42,000 trials would bloat the raw JSON ~7x for no
    analytical benefit here.

Checkpointed to a JSONL file so a long run can be safely resumed: on
restart, already-completed (ref_size, draw, feature, target_d, n, trial)
keys are skipped.

Run:
    python scripts/step2_item4_power_curve.py
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from scipy import optimize
from scipy.special import logsumexp, ndtri
from scipy.stats import beta as beta_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from _session_auth import mint_session_token  # noqa: E402

BACKEND = "http://127.0.0.1:8000"
SPLIT_DIR = "tests/splits"
RESULTS_DIR = "results"
BASELINE_CSV = "tests/citi_bike_baseline.csv"
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "step2_item4_power_curve_checkpoint.jsonl")
OUT_JSON = os.path.join(RESULTS_DIR, "tabular_validation_item4_power_curve_raw.json")

SEED = 42  # matches split_citi_bike.py's holdout construction, so the holdout stays identical
HOLDOUT_SIZE = 25000
REFERENCE_SIZES = [5000, 50000]
N_REFERENCE_DRAWS = 10
D_POP_TARGETS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
BATCH_SIZES = [1000, 5000, 20000]
N_TRIALS = 20
ALPHA = 0.05
MAX_WORKERS = 1  # sequential: concurrent requests reset the dev uvicorn server's connections on Windows
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2

CONTINUOUS_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "trip_duration"]
CATEGORICAL_FEATURES = ["gender_id", "month"]

# mint_session_token()'s tokens expire after a fixed 1 hour (see tests/_session_auth.py) --
# too short for this multi-hour run. Rather than change that shared auth helper, HEADERS is
# refreshed in place (re-minting a fresh token) whenever a request comes back 401.
HEADERS = {"Authorization": f"Bearer {mint_session_token('step2-item4-power-curve@example.com')}"}


def refresh_token():
    HEADERS["Authorization"] = f"Bearer {mint_session_token('step2-item4-power-curve@example.com')}"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------- normal-score tilting (verbatim from step1_severity_v3.py) ----------------

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
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = weights[order]
    f_unweighted = np.arange(1, len(x) + 1) / len(x)
    f_weighted = np.cumsum(w_sorted)
    return float(np.max(np.abs(f_unweighted - f_weighted)))


def solve_weights_for_target_d(x, scores, target_d, tol=1e-4, max_iter=40):
    """D_pop targets are specified directly (not normal-score sigma), so
    binary-search the normal-score severity that achieves the requested
    population D via the exact weighted-KS statistic."""
    lo, hi = 0.0, float(np.max(np.abs(scores))) * 0.99
    best = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        w, achieved_sev, capped = tilt_weights(scores, mid)
        d = weighted_ks(x, w)
        best = (w, d, mid, capped)
        if abs(d - target_d) < tol:
            break
        if d < target_d:
            lo = mid
        else:
            hi = mid
    return best  # (weights, achieved_D, normal_score_severity, was_capped)


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 1.0)
    lower = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lower), float(upper))


# ---------------- reference draws ----------------

def build_reference_draws(baseline_df):
    """Disjoint holdout construction identical to split_citi_bike.py, then
    N_REFERENCE_DRAWS disjoint chunks of each reference size drawn from
    what's left, per reference size (draws for size A and size B may
    overlap each other, but draws WITHIN a given size are disjoint)."""
    n_total = len(baseline_df)
    rng_holdout = np.random.default_rng(SEED)
    pool_idx = np.arange(n_total)
    rng_holdout.shuffle(pool_idx)
    holdout_idx = pool_idx[:HOLDOUT_SIZE]
    remaining_idx = pool_idx[HOLDOUT_SIZE:]

    draws = {}
    for ref_size in REFERENCE_SIZES:
        need = ref_size * N_REFERENCE_DRAWS
        if need > len(remaining_idx):
            raise ValueError(f"Not enough remaining rows ({len(remaining_idx)}) for "
                              f"{N_REFERENCE_DRAWS} disjoint draws of size {ref_size}.")
        rng = np.random.default_rng(SEED + ref_size)
        shuffled = remaining_idx.copy()
        rng.shuffle(shuffled)
        draws[ref_size] = [shuffled[i * ref_size:(i + 1) * ref_size] for i in range(N_REFERENCE_DRAWS)]

    return holdout_idx, draws


def to_payload(df):
    ref = {c: df[c].tolist() for c in CONTINUOUS_FEATURES}
    cat = {c: [str(v) for v in df[c].tolist()] for c in CATEGORICAL_FEATURES}
    return ref, cat


def fit_calibrated_baseline(session, project_id, ref_df):
    ref, cat = to_payload(ref_df)
    payload = {"reference_data": ref, "categorical_data": cat,
               "calibration_config": {"decision_mode": "calibrated"}}
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.post(f"{BACKEND}/fit/{project_id}", json=payload, headers=HEADERS, timeout=120)
            if resp.status_code == 401:
                refresh_token()
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_exc


def analyze_batch(session, project_id, batch_df):
    ref, cat = to_payload(batch_df)
    payload = {"production_data": {**ref, **cat}}
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.post(f"{BACKEND}/analyze/{project_id}", json=payload,
                                 headers=HEADERS, timeout=120)
            if resp.status_code == 401:
                refresh_token()
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_exc


def load_checkpoint():
    done = set()
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done.add((rec["ref_size"], rec["draw"], rec["feature"], rec["target_d"], rec["n"], rec["trial"]))
    return done


def main():
    log("Loading Jan-Mar baseline pool...")
    baseline_df = pd.read_csv(BASELINE_CSV)
    holdout_idx, ref_draws_idx = build_reference_draws(baseline_df)
    holdout = baseline_df.iloc[holdout_idx].reset_index(drop=True)
    log(f"Holdout: {len(holdout)} rows (identical construction to split_citi_bike.py).")
    for ref_size in REFERENCE_SIZES:
        log(f"Reference size {ref_size}: {N_REFERENCE_DRAWS} disjoint draws built.")

    log("Solving tilt weights for every (feature, target D) pair against the holdout...")
    weight_cache = {}
    for feat in CONTINUOUS_FEATURES:
        x = holdout[feat].values
        scores = normal_scores(x)
        for target_d in D_POP_TARGETS:
            w, achieved_d, achieved_sev, capped = solve_weights_for_target_d(x, scores, target_d)
            weight_cache[(feat, target_d)] = {"weights": w, "achieved_d": achieved_d,
                                               "normal_score_severity": achieved_sev, "was_capped": capped}
            log(f"  {feat} target_D={target_d}: achieved_D={achieved_d:.4f} "
                f"sev={achieved_sev:.4f} capped={capped}")

    done_keys = load_checkpoint()
    if done_keys:
        log(f"Resuming: {len(done_keys)} trials already recorded in checkpoint.")

    session = requests.Session()
    checkpoint_f = open(CHECKPOINT_PATH, "a", encoding="utf-8")

    total_jobs = len(REFERENCE_SIZES) * N_REFERENCE_DRAWS * len(CONTINUOUS_FEATURES) * len(D_POP_TARGETS) * len(BATCH_SIZES) * N_TRIALS
    log(f"Total planned trials: {total_jobs}")
    completed = len(done_keys)

    for ref_size in REFERENCE_SIZES:
        for draw_i in range(N_REFERENCE_DRAWS):
            project_id = f"step2_item4_ref{ref_size}_draw{draw_i}"
            ref_df = baseline_df.iloc[ref_draws_idx[ref_size][draw_i]].reset_index(drop=True)
            log(f"===== Fitting {project_id} (n={len(ref_df)}) =====")
            fit_calibrated_baseline(session, project_id, ref_df)

            for feat in CONTINUOUS_FEATURES:
                for target_d in D_POP_TARGETS:
                    wc = weight_cache[(feat, target_d)]
                    weights = wc["weights"]
                    # Deterministic per (ref_size, draw, feature, target_d) trial-batch RNG stream.
                    # CONTINUOUS_FEATURES.index(), not hash(), since Python's string hash is
                    # randomized per-process by default and would break reproducibility.
                    feat_idx = CONTINUOUS_FEATURES.index(feat)
                    seed = (SEED * 1_000_003 + ref_size * 7919 + draw_i * 104729
                            + feat_idx * 100000 + int(target_d * 1000) * 13)
                    rng = np.random.default_rng(seed & 0xFFFFFFFF)

                    jobs = []
                    for n in BATCH_SIZES:
                        for trial in range(N_TRIALS):
                            key = (ref_size, draw_i, feat, target_d, n, trial)
                            if key in done_keys:
                                continue
                            idx = rng.choice(len(holdout), size=n, replace=True, p=weights)
                            batch = holdout.iloc[idx].reset_index(drop=True)
                            jobs.append((key, batch))

                    if not jobs:
                        continue

                    def run_job(key, batch):
                        ref_size_, draw_i_, feat_, target_d_, n_, trial_ = key
                        resp = analyze_batch(session, project_id, batch)
                        m = resp["feature_metrics"][feat_]
                        return {
                            "ref_size": ref_size_, "draw": draw_i_, "feature": feat_,
                            "target_d": target_d_, "achieved_d": wc["achieved_d"], "n": n_, "trial": trial_,
                            "statistic": m["statistic"], "p_value": m["p_value"],
                            "p_value_adjusted": m.get("p_value_adjusted"), "effect_size": m.get("effect_size"),
                            "effect_floor": m.get("effect_floor"), "significant": m.get("significant"),
                            "material": m.get("material"), "drift_detected": m["drift_detected"],
                        }

                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                        futures = [pool.submit(run_job, key, batch) for key, batch in jobs]
                        for fut in as_completed(futures):
                            rec = fut.result()
                            checkpoint_f.write(json.dumps(rec) + "\n")
                            completed += 1
                    checkpoint_f.flush()
                    log(f"  {project_id} {feat} target_D={target_d} (achieved={wc['achieved_d']:.4f}): "
                        f"{len(jobs)} trials done ({completed}/{total_jobs} total)")

    checkpoint_f.close()
    log("All trials complete. Aggregating checkpoint into final JSON...")

    all_records = []
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    out = {
        "config": {
            "reference_sizes": REFERENCE_SIZES, "n_reference_draws": N_REFERENCE_DRAWS,
            "d_pop_targets": D_POP_TARGETS, "batch_sizes": BATCH_SIZES, "n_trials": N_TRIALS,
            "alpha": ALPHA, "seed": SEED, "holdout_size": HOLDOUT_SIZE,
            "continuous_features": CONTINUOUS_FEATURES,
            "note": "Reference draws disjoint within a reference size; tilt weights computed once "
                    "per (feature, target_d) against the fixed holdout, reused for every draw; trial "
                    "batches independently resampled per (ref_size, draw, feature, target_d, n, trial).",
        },
        "achieved_d_by_feature_target": {
            feat: {str(td): weight_cache[(feat, td)]["achieved_d"] for td in D_POP_TARGETS}
            for feat in CONTINUOUS_FEATURES
        },
        "raw_records": all_records,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"Wrote {OUT_JSON} ({len(all_records)} records).")


if __name__ == "__main__":
    main()
