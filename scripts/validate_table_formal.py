"""
M4 formal validation of unified table monitoring (docs/unified_table_plan.md,
section L, experiments A-E) on two real datasets, treated as one table each.

PRE-REGISTERED (written before running):
  Datasets
    petfinder  PetFinder.my (local): Type, Age, Breed1, Gender, Fee, Quantity,
               MaturitySize, State, Description (text), Photo (image; cached
               resnet18 embeddings from scripts/step2e_embed_petfinder.py).
    airbnb     Inside Airbnb Edinburgh listings (CC BY 4.0): room_type,
               property_type, neighbourhood_cleansed, accommodates, bedrooms,
               price, minimum_nights, review_scores_rating, description (text).
               No images: photos are remote URLs, which the product never
               fetches, so image results come from PetFinder only.
  Column types and relationships come from the real profiler and proposal code
  (profile_table, propose_relationships, table_monitor's embedding proposals)
  run on 1,500 held-out rows, excluded from all draws.
  Each draw: disjoint reference (1,500) and batch (300), seeded. Decisions use
  the production building blocks and ONE Holm family (alpha 0.05): KS/PSI
  column tests, text/image DCT with own-reference null (floor 0.55),
  numeric/categorical relationships, categorical probes, text<->image matching.
  Probes into numeric columns are report-only and recorded separately.
  AMENDED before any result existed (the first scenario had not finished): for
  compute budget, probes into numeric columns are left out of this run (they
  never affect decisions, and their behaviour was measured in the M3 gate),
  airbnb runs 50 draws per scenario instead of 100, and 14 workers are used.

  Scenarios (100 draws each):
    A   aa             no change
    B1  monotone       numeric columns in relationships -> 1.3x + 20
    B2  mix            batch resampled with one category level weighted 3x
                       (petfinder: Type=2, airbnb: room_type=Private room)
    C1  shuffle_num    the numeric column in the most relationships, shuffled within the batch
    C2  shuffle_cat    the categorical column in the most relationships, shuffled
    C3  shuffle_text   the text column shuffled (each text kept, pairing broken)
    C4  shuffle_image  the image column shuffled (petfinder only)
    D   subgroup       C1's column shuffled only among rows where C2's column equals its mode
  Expected: A no alarm; B column alarms only (no relationship alarm); C/D
  relationship alarms on pairs involving the changed column, no column alarm on it.
  Metrics: P(any alarm), P(any relationship alarm), P(column alarm on the changed
  column), attribution = flagged relationship tests involving the changed column /
  all flagged relationship tests; batch-level precision/recall/F1 for relationship
  drift (positives C1-C4, D; negatives A, B1, B2); detection latency derived from
  the per-batch detection rate.

Run (a few hours on 8 cores; caches Airbnb text embeddings on first run):
    HF_HUB_OFFLINE=1 python scripts/validate_table_formal.py
"""

import json
import os
import sys
import time
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drift import embedding_tests as et  # noqa: E402
from drift import relationship_detector as rel  # noqa: E402
from drift import table_monitor as tm  # noqa: E402
from drift.calibration import CalibrationConfig  # noqa: E402
from drift.detector import DistributionDetector  # noqa: E402
from drift.embedding_detector import EmbeddingDriftDetector  # noqa: E402
from utils.profiler import profile_table  # noqa: E402

N_REF, N_BAT, N_PROFILE, ALPHA = 1500, 300, 1500, 0.05
DRAWS = {"petfinder": 100, "airbnb": 50}
JOBS = 14
OUT = os.path.join(ROOT, "results", "m4_formal_validation.json")
AUC = EmbeddingDriftDetector()._compute_auc


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
def load_petfinder():
    d = pd.read_csv(os.path.join(ROOT, "datasets", "petfinder", "train.csv"),
                    usecols=["PetID", "Type", "Age", "Breed1", "Gender", "Fee", "Quantity", "MaturitySize", "State",
                             "Description"])
    emb = {}
    for col, m in (("Description", "text"), ("Photo", "image")):
        z = np.load(os.path.join(ROOT, "datasets", "petfinder", f"_step2e_{m}_emb.npz"), allow_pickle=True)
        emb[col] = dict(zip(z["ids"], z["emb"].astype(np.float32)))
    d = d[d.PetID.isin(emb["Description"]) & d.PetID.isin(emb["Photo"])].reset_index(drop=True)
    vectors = {c: np.stack([emb[c][p] for p in d.PetID]) for c in emb}
    d["Photo"] = d.PetID + "-1.jpg"
    return d.drop(columns=["PetID"]), vectors, {"Description": "text", "Photo": "image"}, ("Type", "2")


def load_airbnb():
    path = os.path.join(ROOT, "datasets", "airbnb")
    d = pd.read_csv(os.path.join(path, "edinburgh_listings.csv.gz"),
                    usecols=["room_type", "property_type", "neighbourhood_cleansed", "accommodates", "bedrooms",
                             "price", "minimum_nights", "review_scores_rating", "description"])
    d = d.dropna(subset=["description"]).reset_index(drop=True)
    d["price"] = pd.to_numeric(d["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    cache = os.path.join(path, "_description_emb.npy")
    if not os.path.exists(cache):
        from adapters.text import TextAdapter
        texts = d["description"].astype(str).tolist()
        np.save(cache, np.vstack([TextAdapter().transform(texts[i:i + 64]) for i in range(0, len(texts), 64)]))
    return d, {"description": np.load(cache).astype(np.float32)}, {"description": "text"}, ("room_type", "Private room")


# ---------------------------------------------------------
# Setup: profiler types + proposed relationships on held-out rows
# ---------------------------------------------------------
def setup(name):
    df, vectors, emb_types, mix = {"petfinder": load_petfinder, "airbnb": load_airbnb}[name]()
    order = np.random.default_rng(2026).permutation(len(df))
    prof_idx, pool_idx = order[:N_PROFILE], order[N_PROFILE:]
    prof = df.iloc[prof_idx].reset_index(drop=True)
    types = {p["name"]: p["proposed_type"] for p in profile_table(prof.drop(columns=list(emb_types)))
             if p["proposed_type"] in ("numeric", "categorical")}
    types.update(emb_types)
    relationships = [(r["col_a"], r["col_b"], r["kind"]) for r in rel.propose_relationships(
        {c: (t, rel.clean_column(prof[c].tolist(), t)) for c, t in types.items() if t in tm._NUMCAT}) if r["proposed"]]
    columns = [{"name": c, "proposed_type": t, "proposed_monitor": True} for c, t in types.items()]
    cache = {c: {"emb": vectors[c][prof_idx], "status": np.array(["ok"] * len(prof)), "type": t}
             for c, t in emb_types.items()}
    relationships += [(r["col_a"], r["col_b"], r["kind"]) for r in tm._propose_embedding_relationships(prof, columns, cache)
                      if r["proposed"] and not (r["kind"] == "probe" and types[r["col_b"]] == "numeric")]
    counts = {}
    for a, b, _ in relationships:
        for c in (a, b):
            counts[c] = counts.get(c, 0) + 1
    pick = lambda kind: max((c for c in counts if types[c] == kind), key=counts.get, default=None)
    targets = {"shuffle_num": pick("numeric"), "shuffle_cat": pick("categorical"),
               "shuffle_text": next(c for c, t in emb_types.items() if t == "text"),
               "shuffle_image": next((c for c, t in emb_types.items() if t == "image"), None)}
    return {"df": df, "vectors": vectors, "types": types, "relationships": relationships, "pool": pool_idx,
            "mix": mix, "targets": targets}


# ---------------------------------------------------------
# One draw: production building blocks, one Holm family
# ---------------------------------------------------------
def one_draw(ctx, scenario, i):
    df, vec, types, pool = ctx["df"], ctx["vectors"], ctx["types"], ctx["pool"]
    rng = np.random.default_rng(zlib.crc32(f"{scenario}|{i}".encode()))
    perm = rng.permutation(pool)
    ref_idx, rest = perm[:N_REF], perm[N_REF:]
    if scenario == "mix":
        col, level = ctx["mix"]
        w = np.where(df[col].iloc[rest].astype(str).to_numpy() == level, 3.0, 1.0)
        bat_idx = rng.choice(rest, size=N_BAT, replace=False, p=w / w.sum())
    else:
        bat_idx = rest[:N_BAT]
    ref, bat = df.iloc[ref_idx].reset_index(drop=True), df.iloc[bat_idx].reset_index(drop=True).copy()
    bvec = {c: v[bat_idx].copy() for c, v in vec.items()}

    changed = None
    if scenario == "monotone":
        for c in sorted({c for a, b, k in ctx["relationships"] for c in (a, b) if types[c] == "numeric"}):
            bat[c] = pd.to_numeric(bat[c], errors="coerce") * 1.3 + 20
    elif scenario.startswith("shuffle_") or scenario == "subgroup":
        changed = ctx["targets"]["shuffle_num" if scenario == "subgroup" else scenario]
        if scenario == "subgroup":
            gcol = ctx["targets"]["shuffle_cat"]
            rows = np.flatnonzero(bat[gcol].astype(str).to_numpy() == str(ref[gcol].mode()[0]))
        else:
            rows = np.arange(N_BAT)
        p = rng.permutation(rows)
        if changed in vec:
            bvec[changed][rows] = bvec[changed][p]
        else:
            bat.loc[rows, changed] = bat.loc[p, changed].to_numpy()

    numcat = [c for c, t in types.items() if t in tm._NUMCAT]
    clean = lambda frame: {c: rel.clean_column(frame[c].tolist(), types[c]) for c in numcat}
    rc, bc = clean(ref), clean(bat)
    family, report_only = [], {}

    proj = {}
    for c in vec:
        pca = et.fit_pca(vec[c][ref_idx])
        proj[c] = (et.project(vec[c][ref_idx], pca), et.project(bvec[c], pca))
        r = et.dct_test(AUC, proj[c][0], proj[c][1], et.NULL_DRAWS, zlib.crc32(f"dct|{scenario}|{i}|{c}".encode()))
        family.append({"name": c, "group": "column", "p_value": r["p_value"], "effect_size": r["auc"],
                       "effect_floor": tm.DCT_AUC_FLOOR, "label": "dct"})

    for a, b, kind in ctx["relationships"]:
        name, seed = rel.pair_name(a, b), zlib.crc32(f"{scenario}|{i}|{a}|{b}".encode())
        if kind == "probe":
            t_type = types[b]
            raw = rc[b]
            levels = rel.build_levels(raw[rel.complete_rows(raw)]) if t_type == "categorical" else None
            to_y = lambda vals: (np.array([None if v is None else v for v in rel.map_levels(vals.astype(str), levels)],
                                          dtype=object) if t_type == "categorical" else vals)
            yr, yb = to_y(rc[b]), to_y(bc[b])
            mr, mb = rel.complete_rows(yr), rel.complete_rows(yb)
            cast = str if t_type == "categorical" else float
            xr, ycr = proj[a][0][mr], yr[mr].astype(cast)
            score = et.cross_fitted_score(xr, ycr, t_type)
            if score is None:
                continue
            res = et.probe_test(xr, ycr, proj[a][1][mb], yb[mb].astype(cast), t_type, et.fit_probe(xr, ycr, t_type),
                                score, et.NULL_DRAWS, seed)
        elif kind == "text_image":
            xi, xt = proj[b], proj[a]
            score = et.cross_fitted_matching(xi[0], xt[0])
            res = et.matching_test(xi[0], xt[0], xi[1], xt[1], et.fit_matching(xi[0], xt[0]), score, et.NULL_DRAWS, seed)
        else:
            res = None
        if res is not None:
            if not res["testable"]:
                continue
            if kind == "probe" and types[b] == "numeric":
                report_only[name] = bool(res["p_value"] < ALPHA and res["effect"] >= et.FLOORS["probe"])
            else:
                family.append({"name": name, "group": "relationship", "p_value": res["p_value"],
                               "effect_size": res["effect"], "effect_floor": et.FLOORS[kind], "label": kind})

    numcat_rels = [(a, b, k) for a, b, k in ctx["relationships"] if k in rel.MIN_ROWS]
    k_draws = rel.null_draws_for_family(len(numcat) + len(family) + len(numcat_rels), ALPHA)
    for a, b, kind in numcat_rels:
        prep = rel.prepare_test(kind, rel.reference_state(kind, rc[a], rc[b]), (rc[a], rc[b]), (bc[a], bc[b]))
        if prep["testable"]:
            done = rel.finish_test(prep, k_draws, zlib.crc32(f"{scenario}|{i}|{a}|{b}".encode()))
            family.append({"name": rel.pair_name(a, b), "group": "relationship", "p_value": done["p_value"],
                           "effect_size": done["effect"], "effect_floor": rel.DEFAULT_FLOORS[kind], "label": kind})

    cfg = CalibrationConfig(decision_mode="calibrated")
    cfg.psi_null_draws = max(cfg.psi_null_draws, k_draws)
    det = DistributionDetector(calibration_config=cfg)
    to_list = lambda d: {c: [v for v in d[c] if v is not None and not (isinstance(v, float) and v != v)] for c in d}
    det.fit_baseline(to_list(rc), {c: "continuous" if types[c] == "numeric" else "categorical" for c in numcat})
    out = det.analyze_production_window(to_list(bc), family)
    flagged = [c for c, m in out["feature_metrics"].items() if m["drift_detected"]]
    flagged += [n for n, m in out.get("extra_metrics", {}).items() if m["drift_detected"]]
    return {"scenario": scenario, "i": i, "changed": changed, "flagged": flagged,
            "report_only_flagged": [n for n, v in report_only.items() if v], "family_size": len(numcat) + len(family)}


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
def summarize(rows, scenarios):
    lines, pos, neg = [], [], []
    for s in scenarios:
        g = [r for r in rows if r["scenario"] == s]
        is_rel = lambda n: "<->" in n
        rel_alarm = [any(is_rel(n) for n in r["flagged"]) for r in g]
        changed = g[0]["changed"]
        line = {"scenario": s, "draws": len(g), "changed_column": changed,
                "p_any_alarm": float(np.mean([bool(r["flagged"]) for r in g])),
                "p_any_relationship_alarm": float(np.mean(rel_alarm)),
                "mean_family_size": float(np.mean([r["family_size"] for r in g]))}
        if changed:
            flagged_rels = [n for r in g for n in r["flagged"] if is_rel(n)]
            involving = [n for n in flagged_rels if changed in n.split("<->")]
            line["p_column_alarm_on_changed"] = float(np.mean([changed in r["flagged"] for r in g]))
            line["attribution_precision"] = len(involving) / len(flagged_rels) if flagged_rels else None
            line["p_detected_with_correct_pair"] = float(np.mean(
                [any(is_rel(n) and changed in n.split("<->") for n in r["flagged"]) for r in g]))
            p = line["p_detected_with_correct_pair"]
            line["latency_batches_k1m1"] = round(1 / p, 2) if p > 0 else None
            pos += rel_alarm
        else:
            neg += rel_alarm
        lines.append(line)
    tp, fn, fp = sum(pos), len(pos) - sum(pos), sum(neg)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return lines, {"relationship_drift_precision": prec, "recall": rec,
                   "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "tp": tp, "fp": fp, "fn": fn}


def main():
    results = {}
    t0 = time.time()
    for name in ("petfinder", "airbnb"):
        ctx = setup(name)
        scenarios = ["aa", "monotone", "mix", "shuffle_num", "shuffle_cat", "shuffle_text", "subgroup"]
        if ctx["targets"]["shuffle_image"]:
            scenarios.insert(6, "shuffle_image")
        scenarios = [s for s in scenarios if not s.startswith("shuffle_") or ctx["targets"][s]]
        print(f"[{name}] types={ctx['types']}\n  relationships={ctx['relationships']}\n  targets={ctx['targets']}",
              flush=True)
        rows = []
        for s in scenarios:
            rows += Parallel(n_jobs=JOBS)(delayed(one_draw)(ctx, s, i) for i in range(DRAWS[name]))
            print(f"  {name} {s} done ({time.time() - t0:.0f}s)", flush=True)
        lines, batch_metrics = summarize(rows, scenarios)
        results[name] = {"types": ctx["types"], "relationships": ctx["relationships"], "targets": ctx["targets"],
                         "summary": lines, "batch_level": batch_metrics, "rows": rows}
        for line in lines:
            print("   ", line, flush=True)
        print("   batch-level:", batch_metrics, flush=True)
        json.dump(results, open(OUT, "w"), default=str)


if __name__ == "__main__":
    main()
