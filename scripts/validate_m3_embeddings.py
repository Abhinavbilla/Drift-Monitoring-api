"""
M3 gate experiments (docs/unified_table_plan.md, sections J and L) on the
cached PetFinder embeddings from scripts/step2e_embed_petfinder.py.

PRE-REGISTERED (written before running):
Part 1 -- representation. Reference 1000 dogs, batch 200 with cat fraction
  f in {0, 0.25, 0.5}, 50 draws; DCT AUC on raw embeddings vs PCA-64 (fit on
  the reference), detection = AUC > 0.65. Use PCA-64 if, for both modalities,
  its detection at f=0.25 and f=0.5 is within 5 points of raw (or better) and
  its f=0 false-alarm rate is at most 2 points higher. Otherwise keep raw.
Part 2 -- calibration of the real-embedding split-null p (PCA-64, K=200,
  Gaussian tail). A/A: reference 1000 dogs, batch n in {50, 200}, 100 draws.
  PASS if P(p<0.05) is within the 95% Wilson interval around 5% in every
  cell. Also power: batch with 25% cats, 50 draws, detection p < 0.05.
  This decides whether text/image column tests may join the Holm family.
Part 3 -- probes and matching (PCA-64, K=200), reference 1000 mixed pets,
  batch 200, 50 draws per scenario:
  probes text->Type, image->Type (categorical, balanced accuracy) and
  text->Age (numeric, Spearman); matching text<->image.
    aa       same population             -> no alarm expected
    mix      batch 80% cats (Type prior shifts, pairing intact) -> no alarm expected
    shuffle  target (or text) shuffled across batch rows -> alarm expected
  Alarm = p < 0.05 and drop >= 0.10 (no family correction here).

Run:  python scripts/validate_m3_embeddings.py   (about 30-40 min on 8 cores)
"""

import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drift import embedding_tests as et  # noqa: E402
from drift.embedding_detector import EmbeddingDriftDetector  # noqa: E402

DATA = os.path.join(ROOT, "datasets", "petfinder")
OUT = os.path.join(ROOT, "results", "m3_embedding_validation.json")
AUC = EmbeddingDriftDetector()._compute_auc
JOBS = 8


def load():
    meta = pd.read_csv(os.path.join(DATA, "_step2e_meta.csv")).set_index("PetID")
    mods = {}
    for m in ("text", "image"):
        z = np.load(os.path.join(DATA, f"_step2e_{m}_emb.npz"), allow_pickle=True)
        mods[m] = pd.DataFrame({"emb": list(z["emb"].astype(np.float32))}, index=z["ids"])
    both = mods["text"].index.intersection(mods["image"].index)
    return meta, mods, both


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def draw_dog_cat(meta, emb_df, rng, n_ref, n_bat, cat_frac):
    ids = emb_df.index
    dogs, cats = ids[meta.loc[ids, "Type"].to_numpy() == 1], ids[meta.loc[ids, "Type"].to_numpy() == 2]
    n_cat = int(round(cat_frac * n_bat))
    d = rng.choice(dogs, size=n_ref + n_bat - n_cat, replace=False)
    bat = list(d[n_ref:]) + list(rng.choice(cats, size=n_cat, replace=False))
    stack = lambda keys: np.stack(emb_df.loc[keys, "emb"].to_numpy())
    return stack(d[:n_ref]), stack(bat)


def part1(meta, mods, m, f, i):
    rng = np.random.default_rng(et_seed("p1", m, f, i))
    ref, bat = draw_dog_cat(meta, mods[m], rng, 1000, 200, f)
    pca = et.fit_pca(ref)
    return {"modality": m, "f": f, "raw": AUC(ref, bat), "pca": AUC(et.project(ref, pca), et.project(bat, pca))}


def part2(meta, mods, m, n, f, i):
    rng = np.random.default_rng(et_seed("p2", m, n, f, i))
    ref, bat = draw_dog_cat(meta, mods[m], rng, 1000, n, f)
    pca = et.fit_pca(ref)
    r = et.dct_test(AUC, et.project(ref, pca), et.project(bat, pca), et.NULL_DRAWS, et_seed("null", m, n, f, i))
    return {"modality": m, "n": n, "f": f, "p": r["p_value"], "auc": r["auc"], "tail": r["tail_extrapolated"]}


def part3(meta, mods, both, test, scenario, i):
    rng = np.random.default_rng(et_seed("p3", test, scenario, i))
    ids = np.array(both)
    typ = meta.loc[ids, "Type"].to_numpy()
    ref_ids = rng.choice(ids, size=1000, replace=False)
    rest = np.setdiff1d(ids, ref_ids)
    if scenario == "mix":
        rest_typ = meta.loc[rest, "Type"].to_numpy()
        w = np.where(rest_typ == 2, 4.0, 1.0)
        bat_ids = rng.choice(rest, size=200, replace=False, p=w / w.sum())
    else:
        bat_ids = rng.choice(rest, size=200, replace=False)
    emb = lambda m, keys: np.stack(mods[m].loc[keys, "emb"].to_numpy())
    if test == "text<->image":
        pt, pi = et.fit_pca(emb("text", ref_ids)), et.fit_pca(emb("image", ref_ids))
        tr, ir = et.project(emb("text", ref_ids), pt), et.project(emb("image", ref_ids), pi)
        tb, ib = et.project(emb("text", bat_ids), pt), et.project(emb("image", bat_ids), pi)
        if scenario == "shuffle":
            tb = tb[rng.permutation(len(tb))]
        score = et.cross_fitted_matching(ir, tr)
        r = et.matching_test(ir, tr, ib, tb, et.fit_matching(ir, tr), score, et.NULL_DRAWS, et_seed("m", scenario, i))
    else:
        source, target_col = test.split("->")
        target = "categorical" if target_col == "Type" else "numeric"
        pca = et.fit_pca(emb(source, ref_ids))
        xr, xb = et.project(emb(source, ref_ids), pca), et.project(emb(source, bat_ids), pca)
        col = meta[target_col]
        yr = col.loc[ref_ids].to_numpy().astype(str if target == "categorical" else float)
        yb = col.loc[bat_ids].to_numpy().astype(str if target == "categorical" else float)
        if scenario == "shuffle":
            yb = yb[rng.permutation(len(yb))]
        from drift.embedding_tests import _fit
        score = et.cross_fitted_score(xr, yr, target)
        params = et.probe_params(_fit(xr, yr, target), target)
        r = et.probe_test(xr, yr, xb, yb, target, params, score, et.NULL_DRAWS, et_seed("pr", test, scenario, i))
    r = {k: v for k, v in r.items() if k in ("p_value", "observed", "reference_value", "current_value", "tail_extrapolated")}
    return {"test": test, "scenario": scenario, **r,
            "alarm": bool(r.get("p_value", 1) < 0.05 and r.get("observed", 0) >= 0.10)}


def et_seed(*parts):
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode())


def main():
    meta, mods, both = load()
    t0 = time.time()
    out = {}

    p1 = Parallel(n_jobs=JOBS)(delayed(part1)(meta, mods, m, f, i)
                               for m in ("text", "image") for f in (0.0, 0.25, 0.5) for i in range(50))
    summary1, use_pca = [], True
    for m in ("text", "image"):
        rates = {}
        for f in (0.0, 0.25, 0.5):
            g = [r for r in p1 if r["modality"] == m and r["f"] == f]
            rates[f] = {k: float(np.mean([r[k] > 0.65 for r in g])) for k in ("raw", "pca")}
            summary1.append({"modality": m, "f": f, "detect_raw": rates[f]["raw"], "detect_pca": rates[f]["pca"],
                             "mean_auc_raw": float(np.mean([r["raw"] for r in g])),
                             "mean_auc_pca": float(np.mean([r["pca"] for r in g]))})
        ok = all(rates[f]["pca"] >= rates[f]["raw"] - 0.05 for f in (0.25, 0.5)) and \
            rates[0.0]["pca"] <= rates[0.0]["raw"] + 0.02
        use_pca = use_pca and ok
    out["part1"] = {"summary": summary1, "decision_use_pca": use_pca}
    print("part1", json.dumps(out["part1"], indent=1), f"{time.time() - t0:.0f}s", flush=True)

    p2 = Parallel(n_jobs=JOBS)(delayed(part2)(meta, mods, m, n, f, i) for m in ("text", "image")
                               for n, f, k in ((50, 0.0, 100), (200, 0.0, 100), (200, 0.25, 50)) for i in range(k))
    summary2, passed = [], True
    for m in ("text", "image"):
        for n, f in ((50, 0.0), (200, 0.0), (200, 0.25)):
            g = [r for r in p2 if r["modality"] == m and r["n"] == n and r["f"] == f]
            k = sum(r["p"] < 0.05 for r in g)
            lo, hi = wilson(k, len(g))
            row = {"modality": m, "n": n, "cat_fraction": f, "draws": len(g), "p_lt_005": k / len(g),
                   "wilson": [round(lo, 3), round(hi, 3)], "tail_used": float(np.mean([r["tail"] for r in g]))}
            if f == 0.0:
                row["calibrated"] = lo <= 0.05 <= hi
                passed = passed and row["calibrated"]
            summary2.append(row)
    out["part2"] = {"summary": summary2, "calibration_gate_passed": passed}
    print("part2", json.dumps(out["part2"], indent=1), f"{time.time() - t0:.0f}s", flush=True)

    tests = ["text->Type", "image->Type", "text->Age", "text<->image"]
    p3 = Parallel(n_jobs=JOBS)(delayed(part3)(meta, mods, both, t, s, i)
                               for t in tests for s in ("aa", "mix", "shuffle") for i in range(50))
    summary3 = []
    for t in tests:
        for s in ("aa", "mix", "shuffle"):
            g = [r for r in p3 if r["test"] == t and r["scenario"] == s and "p_value" in r]
            summary3.append({"test": t, "scenario": s, "draws": len(g), "alarm_rate": float(np.mean([r["alarm"] for r in g])),
                             "mean_reference": float(np.mean([r["reference_value"] for r in g])),
                             "mean_current": float(np.mean([r["current_value"] for r in g]))})
    out["part3"] = {"summary": summary3}
    print("part3", json.dumps(out["part3"], indent=1), f"{time.time() - t0:.0f}s", flush=True)
    json.dump({**out, "raw": {"part1": p1, "part2": p2, "part3": p3}}, open(OUT, "w"), default=float)


if __name__ == "__main__":
    main()
