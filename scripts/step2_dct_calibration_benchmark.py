"""
Step 2 prep: benchmarks the two candidate DCT calibration strategies at
40/200/1000/5000 samples per side, per the original Step 2 spec. This is a
runtime MEASUREMENT to inform the "precomputed vs permutation" decision --
it does not choose a default; that's an explicit decision point for the
user per standing instructions ("stop and ask before... finalizing... DCT
calibration default").

Precomputed: at fit time, split the reference into pseudo-reference/
pseudo-batch pairs and run the CV pipeline B times to build a null AUC
distribution, stored once and reused at every future /analyze call.

Permutation: at analyze time, shuffle labels on the pooled (reference +
current) data B times and rerun the CV pipeline each time, per call.

Uses synthetic Gaussian embeddings (384-dim, matching all-MiniLM-L6-v2) --
this is a pure runtime benchmark, not a correctness check, so real model
inference isn't needed; only the embedding count/dimensionality matters for
timing the CV pipeline itself.

Run:
    python scripts/step2_dct_calibration_benchmark.py
"""

import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

SEED = 42
EMBEDDING_DIM = 384
SIZES_PER_SIDE = [40, 200, 1000, 5000]
B_DRAWS = 100  # matches the original spec's dct_permutations=100, used symmetrically for precomputed's null-quantile draw count


def run_cv_auc(X, y, seed):
    y = y.astype(int)
    n_splits = min(5, np.bincount(y).min())
    if n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=1000)
    probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, probs)


def benchmark_size(n, rng):
    # Reference pool: 2n samples (so it can be split into pseudo-ref/pseudo-batch of n each for "precomputed")
    ref_pool = rng.standard_normal((2 * n, EMBEDDING_DIM))
    current = rng.standard_normal((n, EMBEDDING_DIM))

    # --- Precomputed: fit-time cost (one-time, amortized) ---
    t0 = time.time()
    null_aucs = []
    for b in range(B_DRAWS):
        perm = rng.permutation(2 * n)
        pseudo_ref = ref_pool[perm[:n]]
        pseudo_batch = ref_pool[perm[n:2 * n]]
        X = np.vstack([pseudo_ref, pseudo_batch])
        y = np.concatenate([np.zeros(n), np.ones(n)])
        auc = run_cv_auc(X, y, seed=SEED + b)
        if auc is not None:
            null_aucs.append(auc)
    precomputed_fit_time = time.time() - t0
    precomputed_quantiles = {
        "p50": float(np.percentile(null_aucs, 50)),
        "p95": float(np.percentile(null_aucs, 95)),
        "p99": float(np.percentile(null_aucs, 99)),
    }

    # --- Permutation: per-analyze-call cost ---
    X_real = np.vstack([ref_pool[:n], current])
    y_real = np.concatenate([np.zeros(n), np.ones(n)])
    t0 = time.time()
    perm_aucs = []
    for b in range(B_DRAWS):
        y_shuffled = rng.permutation(y_real)
        auc = run_cv_auc(X_real, y_shuffled, seed=SEED + b)
        if auc is not None:
            perm_aucs.append(auc)
    permutation_analyze_time = time.time() - t0

    return {
        "n_per_side": n,
        "b_draws": B_DRAWS,
        "precomputed_fit_time_sec": precomputed_fit_time,
        "precomputed_fit_time_per_draw_ms": 1000 * precomputed_fit_time / B_DRAWS,
        "precomputed_null_quantiles": precomputed_quantiles,
        "permutation_analyze_time_sec": permutation_analyze_time,
        "permutation_time_per_draw_ms": 1000 * permutation_analyze_time / B_DRAWS,
    }


def main():
    rng = np.random.default_rng(SEED)
    results = []
    for n in SIZES_PER_SIDE:
        print(f"Benchmarking n={n} per side...", flush=True)
        r = benchmark_size(n, rng)
        results.append(r)
        print(f"  precomputed (one-time, at /fit): {r['precomputed_fit_time_sec']:.2f}s total "
              f"({r['precomputed_fit_time_per_draw_ms']:.1f}ms/draw)")
        print(f"  permutation (EVERY /analyze call): {r['permutation_analyze_time_sec']:.2f}s total "
              f"({r['permutation_time_per_draw_ms']:.1f}ms/draw)")

    out = {"config": {"seed": SEED, "embedding_dim": EMBEDDING_DIM, "b_draws": B_DRAWS,
                       "sizes_per_side": SIZES_PER_SIDE, "embedding_source": "synthetic Gaussian, runtime-only benchmark"},
           "results": results}
    with open("results/dct_calibration_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results/dct_calibration_benchmark.json")


if __name__ == "__main__":
    main()
