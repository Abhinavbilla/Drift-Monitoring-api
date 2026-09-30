"""
Hardening pass item 4: verify the recommended_batch_size redefinition by
simulation. Uses Uniform(0,1) vs Uniform(D, 1+D) for the reference and
production batch -- a closed-form construction where the population KS
distance between the two distributions is EXACTLY D (for D < 1), so
"true D" is controlled exactly, not approximated via importance
weighting.

For each reference size m in {5,000, 50,000} and each true D in
{0, 0.02, 0.05}, draws N_TRIALS independent (reference, batch) pairs at
BOTH the new recommended_batch_size(m, 0.05) and the old
min_batch_size_at_floor(m, 0.05), computes the two-sample KS statistic
each time, and reports:
  - the null (D_true=0) D_obs distribution (mean, std, quantiles)
  - P(D_obs > floor) at each true D, with a Clopper-Pearson 95% CI

Run:
    python scripts/step2_hardening_batch_size_simulation.py
"""

import json
import time

import numpy as np
from scipy.stats import ks_2samp, beta as beta_dist

from drift.calibration import recommended_batch_size, min_batch_size_at_floor

SEED = 42
FLOOR = 0.05
N_TRIALS = 2000
REFERENCE_SIZES = [5000, 50000]
TRUE_DS = [0.0, 0.02, 0.05]
OUT_JSON = "results/step2_hardening_batch_size_simulation.json"
OUT_MD = "results/step2_hardening_batch_size_simulation.md"


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 1.0)
    lower = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lower), float(upper))


def simulate(rng, m, n, true_d, n_trials):
    """reference ~ U(0,1), batch ~ U(true_d, 1+true_d) -- population KS
    distance is exactly true_d for true_d < 1 (a uniform shift's CDFs are
    parallel ramps offset by true_d, with sup|F-G| = true_d)."""
    d_obs_values = np.empty(n_trials)
    for t in range(n_trials):
        ref = rng.uniform(0.0, 1.0, size=m)
        batch = rng.uniform(true_d, 1.0 + true_d, size=n)
        d_obs_values[t] = ks_2samp(ref, batch).statistic
    return d_obs_values


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    results = {"config": {"floor": FLOOR, "n_trials": N_TRIALS, "seed": SEED,
                           "reference_sizes": REFERENCE_SIZES, "true_ds": TRUE_DS,
                           "method": "Uniform(0,1) vs Uniform(D,1+D) -- exact closed-form population D"},
               "by_reference_size": {}}

    notes = []
    notes.append("# Hardening pass item 4: recommended_batch_size redefinition, simulation verification\n")
    notes.append(
        f"Method: reference ~ Uniform(0,1) (m samples), batch ~ Uniform(D, 1+D) (n samples) -- the "
        f"population KS distance between these two is EXACTLY D (for D<1, a closed-form result, not "
        f"an approximation), so 'true D' is controlled exactly. {N_TRIALS} independent trials per cell, "
        f"floor={FLOOR}, seed={SEED}.\n"
    )

    for m in REFERENCE_SIZES:
        n_new = recommended_batch_size(m, FLOOR)
        n_old = min_batch_size_at_floor(m, FLOOR)
        notes.append(f"\n## m={m}: new n (recommended_batch_size) = {n_new}, old n (min_batch_size_at_floor) = {n_old}\n")
        results["by_reference_size"][m] = {"n_new": n_new, "n_old": n_old, "by_n": {}}

        for label, n in [("new", n_new), ("old", n_old)]:
            notes.append(f"\n### n={n} ({label} definition)\n")
            cell = {}
            for true_d in TRUE_DS:
                d_obs = simulate(rng, m, n, true_d, N_TRIALS)
                k = int(np.sum(d_obs > FLOOR))
                rate = k / N_TRIALS
                ci = clopper_pearson(k, N_TRIALS)
                cell[str(true_d)] = {
                    "mean_d_obs": float(np.mean(d_obs)), "std_d_obs": float(np.std(d_obs, ddof=1)),
                    "quantiles": {"p05": float(np.percentile(d_obs, 5)), "p50": float(np.percentile(d_obs, 50)),
                                  "p95": float(np.percentile(d_obs, 95))},
                    "p_dobs_gt_floor": rate, "k": k, "n_trials": N_TRIALS, "ci95": ci,
                }
                notes.append(
                    f"- true D={true_d}: mean(D_obs)={np.mean(d_obs):.4f}, std={np.std(d_obs, ddof=1):.4f}, "
                    f"median={np.percentile(d_obs, 50):.4f}, P(D_obs>{FLOOR})={rate:.4f} "
                    f"({k}/{N_TRIALS}, 95% CI {ci[0]:.4f}-{ci[1]:.4f})"
                )
            results["by_reference_size"][m]["by_n"][label] = {"n": n, "results": cell}

    notes.append(
        f"\n**Runtime**: {time.time()-t0:.1f}s for "
        f"{len(REFERENCE_SIZES)*2*len(TRUE_DS)*N_TRIALS} total ks_2samp calls.\n"
    )

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(notes))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
