"""
Step 2 (e) supplement: the ~40-sample borderline case flagged in
docs/step2_proposal.md (two same-domain text batches, 20 vs 20, produced a
borderline AUC=0.69). step2e_validate.py's smallest cell is a 40-row batch
against a 100+ row reference, which is NOT that regime -- this covers it
directly: reference and batch BOTH tiny.

  aa   dogs vs dogs                      (no drift; false-alarm rate)
  age  young dogs vs adult dogs          (different-but-same-domain; exploratory,
                                          same caveat as step2e_validate.py)

The original case's exact texts are not recorded anywhere, so this does NOT
claim to reproduce it -- it measures the same regime on real data.

Run:
    python scripts/step2e_small_batch.py
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json

import numpy as np
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import step2e_validate as v  # noqa: E402

SIZES = [(20, 20), (40, 40)]
DRAWS = {"aa": 200, "age": 100}
OUT = os.path.join(ROOT, "results", "step2e_small_batch_raw.json")


def main():
    rows = []
    for modality in ["text", "image"]:
        p = v.pools(modality)
        for n_ref, n_batch in SIZES:
            for scenario, nd in DRAWS.items():
                res = Parallel(n_jobs=8)(delayed(v.one)(p, modality, scenario, n_ref, n_batch, i) for i in range(nd))
                rows.extend(r for r in res if r is not None)
    json.dump({"sizes": SIZES, "draws": DRAWS, "rows": rows}, open(OUT, "w"))

    print("modality scenario n_ref n_batch draws meanAUC | detection rate: legacy, cal0.55, cal0.60, cal0.65 | P(p<0.05)")
    for modality in ["text", "image"]:
        for scenario in DRAWS:
            for n_ref, n_batch in SIZES:
                g = [r for r in rows if (r["modality"], r["scenario"], r["n_ref"], r["n_batch"]) == (modality, scenario, n_ref, n_batch)]
                auc = np.array([r["auc"] for r in g])
                pv = np.array([r["p"] for r in g])
                rates = [float((auc > 0.65).mean())] + [float(((pv < 0.05) & (auc >= f)).mean()) for f in v.FLOORS]
                print(f"{modality:5s} {scenario:4s} {n_ref:3d} {n_batch:3d} {len(g):3d} {auc.mean():.3f} | "
                      + ", ".join(f"{100*x:5.1f}%" for x in rates) + f" | {100*float((pv<0.05).mean()):5.1f}%"
                      + f" | AUC max {auc.max():.3f} P(AUC>0.65)={100*float((auc>0.65).mean()):.1f}%")


if __name__ == "__main__":
    main()
