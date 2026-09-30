"""
Step 3e end-to-end example, labeled synthetic scenario (2026-09-30
hardening pass item 5): the real Apr-Jun replay's monitored coordinate
features sit at population D ~0.02 (see README.md), well below the 0.05
floor -- so it does NOT demonstrate the system catching real drift, only
that it correctly stays quiet on noise. This script generates a SEPARATE,
clearly-labeled synthetic batch with a real, substantial injected drift
(population D ~0.10 on pickup_longitude, normal-score exponential tilting
-- the same method and machinery as scripts/step2_item4_power_curve.py,
verified there via exact weighted-KS, no resampling noise) and replays it
through a second serve.py instance (its own port, its own log file), so
the SAME end-to-end pipeline (log -> scheduled job -> analyze) is
exercised for a batch that should, and does, alert in essentially every
window.

Run (with a second serve.py instance already running, logging to
logs/synthetic_features.jsonl -- see README.md):
    python examples/model_serving/replay_synthetic.py
"""

import os
import time

import numpy as np
import pandas as pd
import requests
from scipy import optimize
from scipy.special import logsumexp, ndtri

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..")
SYNTHETIC_SERVE_URL = os.environ.get("SYNTHETIC_MODEL_SERVE_URL", "http://127.0.0.1:8002")
HOLDOUT_CSV = os.path.join(REPO_ROOT, "tests", "splits", "holdout_n25000.csv")
SEED = 42
TARGET_D = 0.10
TILT_FEATURE = "pickup_longitude"
N_TRIALS = 4
WINDOW_SIZE = 3146  # recommended_batch_size(50000, 0.05) -- see README.md


def normal_scores(x):
    n = len(x)
    ranks = pd.Series(x).rank(method="average").values
    return ndtri(ranks / (n + 1))


def solve_lambda(scores, target, bracket=1000):
    s_max, s_min = float(np.max(scores)), float(np.min(scores))
    safe_target = target
    if target >= s_max * 0.995:
        safe_target = s_max * 0.995
    elif target <= s_min * 0.995:
        safe_target = s_min * 0.995

    def tilted_mean(lam):
        logw = lam * scores
        logw = logw - logsumexp(logw)
        return np.sum(scores * np.exp(logw))

    lam = optimize.brentq(lambda l: tilted_mean(l) - safe_target, -bracket, bracket, xtol=1e-10)
    return lam


def tilt_weights(scores, severity):
    lam = solve_lambda(scores, severity)
    logw = lam * scores
    logw = logw - logsumexp(logw)
    return np.exp(logw)


def weighted_ks(x, weights):
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = weights[order]
    f_unweighted = np.arange(1, len(x) + 1) / len(x)
    f_weighted = np.cumsum(w_sorted)
    return float(np.max(np.abs(f_unweighted - f_weighted)))


def solve_weights_for_target_d(x, scores, target_d, tol=1e-4, max_iter=40):
    lo, hi = 0.0, float(np.max(np.abs(scores))) * 0.99
    best = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        w = tilt_weights(scores, mid)
        d = weighted_ks(x, w)
        best = (w, d)
        if abs(d - target_d) < tol:
            break
        if d < target_d:
            lo = mid
        else:
            hi = mid
    return best


def main():
    holdout = pd.read_csv(HOLDOUT_CSV)
    x = holdout[TILT_FEATURE].values
    scores = normal_scores(x)
    weights, achieved_d = solve_weights_for_target_d(x, scores, TARGET_D)
    print(f"Tilting {TILT_FEATURE}: target D={TARGET_D}, achieved D={achieved_d:.4f} "
          f"(exact weighted-KS against the holdout pool).")

    session = requests.Session()
    rng = np.random.default_rng(SEED)
    total = 0
    t0 = time.time()
    for trial in range(N_TRIALS):
        idx = rng.choice(len(holdout), size=WINDOW_SIZE, replace=True, p=weights)
        batch = holdout.iloc[idx].reset_index(drop=True)
        for _, row in batch.iterrows():
            payload = {
                "pickup_longitude": float(row["pickup_longitude"]),
                "pickup_latitude": float(row["pickup_latitude"]),
                "dropoff_longitude": float(row["dropoff_longitude"]),
                "dropoff_latitude": float(row["dropoff_latitude"]),
                "gender_id": int(row["gender_id"]),
            }
            resp = session.post(f"{SYNTHETIC_SERVE_URL}/predict", json=payload, timeout=30)
            resp.raise_for_status()
            total += 1
        print(f"Trial {trial}: replayed {len(batch)} tilted rows.")
    print(f"Total replayed: {total} rows in {time.time() - t0:.1f}s. Achieved D={achieved_d:.4f} on {TILT_FEATURE}.")


if __name__ == "__main__":
    main()
