"""
Step 2 (e): validate the text/image Domain Classifier Test on real PetFinder
data, against ground truth that is defined BY CONSTRUCTION (see below).

PRE-REGISTERED DESIGN (written before any result was seen, so the labels
below cannot be tuned after the fact):

  Population: dogs (Type==1) are the "reference" population; cats (Type==2)
  are the "different" population. Reference and batch draws are always
  disjoint rows.

  Scenarios (per modality x n_ref x n_batch):
    aa        reference dogs vs batch dogs, no mixing.           LABEL: no drift
    mix05/10  batch = 5% / 10% cats, rest dogs.                  power curve only
    mix25/50  batch = 25% / 50% cats, rest dogs.                 LABEL: drift
    mix100    batch = 100% cats.                                 LABEL: drift
    age       ref = young dogs, batch = adult dogs.              EXPLORATORY only
    adopt     ref = fast-adopted dogs, batch = not-adopted dogs. EXPLORATORY only

  Why mix05/mix10 and the within-dog scenarios are NOT scored as positives:
  a 5% cat contamination is below what any finite-batch test can reliably
  see (that is a power question, not a correctness one), and whether young vs
  adult dogs' descriptions/photos genuinely differ is exactly what is being
  measured, so labelling them "drift" would be circular. They are reported as
  detection rates, never in the confusion matrix.

  Decision rules compared on the SAME draws:
    legacy            AUC > 0.65                                (bare cutoff)
    calibrated(floor) p_value < alpha=0.05 AND AUC >= floor, floor in
                      {0.55, 0.60, 0.65}; p from the project's own
                      precomputed grid via drift.dct_calibration.dct_pvalue.

  The AUC itself comes from EmbeddingDriftDetector._compute_auc -- the exact
  function /analyze calls. A random subset of draws is additionally pushed
  through the real EmbeddingDriftDetector.analyze() and asserted identical,
  so this script cannot drift from the shipped code path.

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/step2e_validate.py [--quick]
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drift.calibration import CalibrationConfig  # noqa: E402
from drift.dct_calibration import dct_pvalue  # noqa: E402
from drift.embedding_detector import EmbeddingDriftDetector  # noqa: E402

DATA = os.path.join(ROOT, "datasets", "petfinder")
OUT = os.path.join(ROOT, "results", "step2e_text_image_raw.json")

N_REFS = [100, 500]
N_BATCHES = [40, 100, 300, 1000]
DRAWS = {"aa": 200, "mix05": 100, "mix10": 100, "mix25": 100, "mix50": 100, "mix100": 100, "age": 50, "adopt": 50}
MIX_FRAC = {"mix05": 0.05, "mix10": 0.10, "mix25": 0.25, "mix50": 0.50, "mix100": 1.0}
FLOORS = [0.55, 0.60, 0.65]
ALPHA = 0.05


def load(modality):
    z = np.load(os.path.join(DATA, f"_step2e_{modality}_emb.npz"), allow_pickle=True)
    meta = pd.read_csv(os.path.join(DATA, "_step2e_meta.csv")).set_index("PetID")
    m = meta.loc[z["ids"]]
    return z["emb"].astype(np.float32), m["Type"].to_numpy(), m["Age"].to_numpy(), m["AdoptionSpeed"].to_numpy()


def pools(modality):
    emb, typ, age, speed = load(modality)
    dog, cat = typ == 1, typ == 2
    return {
        "emb": emb,
        "dog": np.flatnonzero(dog),
        "cat": np.flatnonzero(cat),
        "young": np.flatnonzero(dog & (age <= 6)),
        "adult": np.flatnonzero(dog & (age >= 24)),
        "fast": np.flatnonzero(dog & (speed <= 1)),
        "slow": np.flatnonzero(dog & (speed == 4)),
    }


def seed_for(modality, scenario, n_ref, n_batch, i):
    return zlib.crc32(f"{modality}|{scenario}|{n_ref}|{n_batch}|{i}".encode()) & 0xFFFFFFFF


def draw(p, modality, scenario, n_ref, n_batch, i):
    rng = np.random.default_rng(seed_for(modality, scenario, n_ref, n_batch, i))
    if scenario in MIX_FRAC or scenario == "aa":
        frac = MIX_FRAC.get(scenario, 0.0)
        n_cat = int(round(frac * n_batch))
        n_dog = n_batch - n_cat
        dogs = rng.choice(p["dog"], size=n_ref + n_dog, replace=False)
        ref_idx, batch_idx = dogs[:n_ref], dogs[n_ref:]
        if n_cat:
            batch_idx = np.concatenate([batch_idx, rng.choice(p["cat"], size=n_cat, replace=False)])
    else:
        a, b = ("young", "adult") if scenario == "age" else ("fast", "slow")
        if len(p[a]) < n_ref or len(p[b]) < n_batch:
            return None
        ref_idx = rng.choice(p[a], size=n_ref, replace=False)
        batch_idx = rng.choice(p[b], size=n_batch, replace=False)
    return p["emb"][ref_idx], p["emb"][batch_idx]


def one(p, modality, scenario, n_ref, n_batch, i):
    pair = draw(p, modality, scenario, n_ref, n_batch, i)
    if pair is None:
        return None
    ref, cur = pair
    det = EmbeddingDriftDetector()
    auc = det._compute_auc(ref, cur)
    pv = dct_pvalue(auc, ref.shape[1], len(ref), len(cur), auc_fn=det._compute_auc)
    return {"modality": modality, "scenario": scenario, "n_ref": n_ref, "n_batch": n_batch, "i": i, "auc": auc, "p": pv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (few draws), writes no results file")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    draws = {k: (3 if args.quick else v) for k, v in DRAWS.items()}
    rows, skipped = [], []
    t0 = time.time()
    for modality in ["text", "image"]:
        p = pools(modality)
        print(f"[{modality}] dogs={len(p['dog'])} cats={len(p['cat'])} young={len(p['young'])} "
              f"adult={len(p['adult'])} fast={len(p['fast'])} slow={len(p['slow'])}", flush=True)
        for n_ref in N_REFS:
            for n_batch in N_BATCHES:
                for scenario, nd in draws.items():
                    res = Parallel(n_jobs=args.jobs)(
                        delayed(one)(p, modality, scenario, n_ref, n_batch, i) for i in range(nd)
                    )
                    got = [r for r in res if r is not None]
                    if len(got) < nd:
                        skipped.append({"modality": modality, "scenario": scenario, "n_ref": n_ref, "n_batch": n_batch,
                                        "reason": "group too small for this size"})
                    rows.extend(got)
                    print(f"  {modality} ref={n_ref} batch={n_batch} {scenario:7s} done "
                          f"({len(got)} draws, {time.time()-t0:.0f}s)", flush=True)

    # ---- integrity check: re-run a sample through the REAL detector.analyze() ----
    rng = np.random.default_rng(0)
    check_idx = rng.choice(len(rows), size=min(20, len(rows)), replace=False)
    cfg_cal = CalibrationConfig(decision_mode="calibrated", alpha=ALPHA)
    mismatches = 0
    pcache = {m: pools(m) for m in ["text", "image"]}
    for k in check_idx:
        r = rows[int(k)]
        ref, cur = draw(pcache[r["modality"]], r["modality"], r["scenario"], r["n_ref"], r["n_batch"], r["i"])
        det = EmbeddingDriftDetector()
        legacy = det.analyze(ref, cur)
        cal = det.analyze(ref, cur, cfg_cal)
        ok = (abs(legacy["statistic"] - r["auc"]) < 1e-9
              and legacy["drift_detected"] == bool(r["auc"] > 0.65)
              and abs(cal["p_value"] - r["p"]) < 1e-9
              and cal["drift_detected"] == bool(r["p"] < ALPHA and r["auc"] >= 0.65))
        mismatches += (not ok)
    print(f"integrity check vs real analyze(): {len(check_idx) - mismatches}/{len(check_idx)} identical", flush=True)

    if args.quick:
        print("--quick: not writing results file")
        return
    with open(OUT, "w") as f:
        json.dump({
            "config": {"n_refs": N_REFS, "n_batches": N_BATCHES, "draws": draws, "floors": FLOORS, "alpha": ALPHA,
                       "mix_frac": MIX_FRAC},
            "integrity_check": {"checked": len(check_idx), "mismatches": mismatches},
            "skipped": skipped,
            "rows": rows,
        }, f)
    print(f"wrote {OUT} ({len(rows)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
