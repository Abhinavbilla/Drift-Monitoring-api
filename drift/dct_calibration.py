"""
Step 2 (d), 2026-10-04: p-values for the Domain Classifier Test's AUC
statistic, via a precomputed null-distribution grid.

The grid (results/dct_null_distribution_grid.json, built once by
scripts/build_dct_calibration_grid.py) holds, for each (embedding_dim,
n_ref, n_batch) combination it was calibrated at, a set of AUC values
observed when the classifier is trained to distinguish two batches drawn
from the SAME synthetic Gaussian distribution -- the null hypothesis.
p_value = (1 + #{null_auc >= observed_auc}) / (draws + 1), the same
"+1 correction" convention drift/calibration.py's psi_bootstrap_pvalue
already uses (avoids a p-value of exactly 0).

A combination outside the grid's calibrated sizes is NOT silently
extrapolated or clamped to a different dimension's grid -- that would
be fabricating a number. Instead it falls back to computing the null
on-the-fly via a smaller live permutation (fewer draws than the
precomputed grid, since this happens inline during a real /analyze
call, not as a one-time offline cost).
"""

import functools
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np

GRID_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "dct_null_distribution_grid.json")
LIVE_FALLBACK_DRAWS = 30


@functools.lru_cache(maxsize=1)
def _load_grid() -> Optional[dict]:
    if not os.path.exists(GRID_PATH):
        return None
    with open(GRID_PATH, "r") as f:
        return json.load(f)


def _find_grid_cell(grid: dict, embedding_dim: int, n_ref: int, n_batch: int) -> Optional[List[float]]:
    """Nearest-neighbor match in log-size space, not an exact match --
    an arbitrary project's reference size will almost never equal one of
    the handful of sizes the grid was actually calibrated at. Distance is
    measured on log(n) since DCT/AUC sampling noise scales roughly
    logarithmically with sample size (same reasoning the DKW/KS
    asymptotic formulas already rely on an approximation at this
    resolution for). A documented approximation -- not an exact p-value
    for this precise (n_ref, n_batch), but the closest calibrated one."""
    cells = grid.get("grid", {}).get(str(embedding_dim))
    if not cells:
        return None
    target = (np.log(n_ref), np.log(n_batch))
    best = min(cells, key=lambda c: (np.log(c["n_ref"]) - target[0]) ** 2 + (np.log(c["n_batch"]) - target[1]) ** 2)
    return best["null_auc_draws"]


def _empirical_pvalue(observed_auc: float, null_draws: List[float]) -> float:
    exceed = sum(1 for d in null_draws if d >= observed_auc)
    return (1 + exceed) / (len(null_draws) + 1)


def _live_permutation_pvalue(
    observed_auc: float, embedding_dim: int, n_ref: int, n_batch: int,
    auc_fn: Callable[[np.ndarray, np.ndarray], float], seed: int = 42,
) -> float:
    """Fallback when (embedding_dim, n_ref, n_batch) isn't in the
    precomputed grid: draws LIVE_FALLBACK_DRAWS synthetic null samples at
    this EXACT size/dimension and calls the real detector's own AUC
    function on each -- fewer draws than the offline grid (speed, since
    this runs inline during a real request), but a genuinely computed
    p-value for this exact combination, not a number borrowed from a
    different one."""
    rng = np.random.default_rng(seed)
    null_draws = []
    for _ in range(LIVE_FALLBACK_DRAWS):
        synthetic_ref = rng.standard_normal((n_ref, embedding_dim))
        synthetic_batch = rng.standard_normal((n_batch, embedding_dim))
        null_draws.append(auc_fn(synthetic_ref, synthetic_batch))
    return _empirical_pvalue(observed_auc, null_draws)


def dct_pvalue(
    observed_auc: float, embedding_dim: int, n_ref: int, n_batch: int,
    auc_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """p-value for an observed DCT AUC at this (embedding_dim, n_ref,
    n_batch). Looks up the precomputed grid first; falls back to a live
    permutation (see above) if this exact combination was never
    calibrated. auc_fn must be the SAME AUC computation the real
    detector uses (EmbeddingDriftDetector._compute_auc), so the
    fallback's null is computed identically to the observed statistic."""
    grid = _load_grid()
    if grid is not None:
        null_draws = _find_grid_cell(grid, embedding_dim, n_ref, n_batch)
        if null_draws is not None:
            return _empirical_pvalue(observed_auc, null_draws)
    return _live_permutation_pvalue(observed_auc, embedding_dim, n_ref, n_batch, auc_fn)
