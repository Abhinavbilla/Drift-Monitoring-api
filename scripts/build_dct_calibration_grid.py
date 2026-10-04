"""
Step 2 (d), 2026-10-04: builds results/dct_null_distribution_grid.json --
the precomputed null-distribution grid drift/dct_calibration.py looks up
at /analyze time.

For each embedding dimension (384 = TextAdapter's all-MiniLM-L6-v2, 512 =
ImageAdapter's resnet18) and each (n_ref, n_batch) pair in SIZES x SIZES,
draws B_DRAWS pairs of synthetic batches from the SAME standard-normal
distribution (the null: no true difference) and records the AUC
EmbeddingDriftDetector's own _compute_auc() produces for each draw -- the
exact same code path used at real analyze time, so there's no risk of
the calibration drifting out of sync with runtime behavior.

Run (from the repo root):
    python scripts/build_dct_calibration_grid.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift.embedding_detector import EmbeddingDriftDetector  # noqa: E402

SIZES = [50, 200, 1000, 5000]
EMBEDDING_DIMS = {"text (all-MiniLM-L6-v2)": 384, "image (resnet18)": 512}
B_DRAWS = 200
SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
                           "dct_null_distribution_grid.json")


def build_grid():
    detector = EmbeddingDriftDetector()
    grid = {}
    config = {"seed": SEED, "b_draws": B_DRAWS, "sizes": SIZES, "dims": EMBEDDING_DIMS,
              "embedding_source": "synthetic standard-normal, null-hypothesis draws"}
    started = time.time()

    for label, dim in EMBEDDING_DIMS.items():
        rng = np.random.default_rng(SEED)
        cells = []
        for n_ref in SIZES:
            for n_batch in SIZES:
                t0 = time.time()
                null_aucs = []
                for _ in range(B_DRAWS):
                    synthetic_ref = rng.standard_normal((n_ref, dim))
                    synthetic_batch = rng.standard_normal((n_batch, dim))
                    null_aucs.append(detector._compute_auc(synthetic_ref, synthetic_batch))
                cells.append({"n_ref": n_ref, "n_batch": n_batch, "null_auc_draws": null_aucs})
                elapsed = time.time() - t0
                print(f"[{label}] n_ref={n_ref} n_batch={n_batch}: "
                      f"p50={np.median(null_aucs):.4f} p95={np.percentile(null_aucs, 95):.4f} "
                      f"({elapsed:.1f}s)", flush=True)
        grid[str(dim)] = cells

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"config": config, "grid": grid}, f, indent=2)
    print(f"\nWrote {OUTPUT_PATH} in {time.time() - started:.1f}s total.")


if __name__ == "__main__":
    build_grid()
