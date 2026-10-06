"""
M2 early validation slice (docs/unified_table_plan.md, section L, experiments
A/B/C) for numeric/categorical relationship drift on PetFinder.my.

PRE-REGISTERED (written before running):
  Pool: PetFinder train.csv, columns below. Types and the relationship set come
  from the real profiler (profile_table + propose_relationships) run once on
  the first 3,000 shuffled rows; those rows are then excluded from all draws.
  Each draw: disjoint reference (N=2000) and batch (n in {300, 1000}) from the
  remaining rows, seeded. Decisions use the production code path: calibrated
  DistributionDetector with the relationship tests as extra family members
  (Holm, alpha=0.05), relationship p-values from split nulls with
  K = null_draws_for_family(m, 0.05).

  A   aa        no change                      -> nothing should alarm
  B1  monotone  every numeric column in a proposed relationship -> 1.3 * x + 20 in batch
                                               -> those columns alarm; NO relationship should
                (AMENDED after the setup step printed the relationship set and BEFORE any B result
                existed: the original B1 transformed Fee, which ended up in no proposed relationship,
                so it could not test invariance.)
  B2  mix       batch rows resampled with Type=2 (cats) weighted 3x -> category mix shifts,
                within-row structure intact    -> column alarms allowed; relationship alarms
                are measured (P(B|A) is preserved for Type pairs only, so this is reported, not
                scored as a pure negative)
  C1  shuffle   the numeric column in the most proposed relationships is shuffled within
                the batch (100% of rows)       -> its pairs alarm; its column must NOT alarm
  C2  shuffle25 same, 25% of rows shuffled     -> power at a smaller break

  100 draws per (scenario, n). Metrics: P(any relationship alarm), P(column alarm on
  the modified column), attribution = flagged relationship tests involving the
  shuffled column / all flagged relationship tests.

Run (about 10-15 min on 8 cores):
    python scripts/validate_table_relationships.py
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

from drift import relationship_detector as rel  # noqa: E402
from drift.calibration import CalibrationConfig  # noqa: E402
from drift.detector import DistributionDetector  # noqa: E402
from utils.profiler import profile_table  # noqa: E402

COLUMNS = ["Type", "Age", "Breed1", "Gender", "Color1", "MaturitySize", "FurLength", "Health",
           "Quantity", "Fee", "State", "PhotoAmt"]
N_REF, BATCHES, DRAWS = 2000, [300, 1000], 100
SCENARIOS = ["aa", "monotone", "mix", "shuffle", "shuffle25"]
OUT = os.path.join(ROOT, "results", "m2_relationship_validation_raw.json")


def setup():
    pool = pd.read_csv(os.path.join(ROOT, "datasets", "petfinder", "train.csv"), usecols=COLUMNS)
    pool = pool.sample(frac=1.0, random_state=2026).reset_index(drop=True)
    profile_rows, pool = pool.iloc[:3000], pool.iloc[3000:].reset_index(drop=True)
    types = {p["name"]: p["proposed_type"] for p in profile_table(profile_rows)
             if p["proposed_type"] in ("numeric", "categorical")}
    proposals = rel.propose_relationships({c: (t, rel.clean_column(profile_rows[c].tolist(), t)) for c, t in types.items()})
    relationships = [(r["col_a"], r["col_b"], r["kind"]) for r in proposals if r["proposed"]]
    counts = {}
    for a, b, _ in relationships:
        for c in (a, b):
            if types[c] == "numeric":
                counts[c] = counts.get(c, 0) + 1
    shuffle_col = max(counts, key=counts.get)
    return pool, types, relationships, shuffle_col


def one_draw(pool, types, relationships, shuffle_col, scenario, n, i):
    monotone_cols = sorted({c for a, b, _ in relationships for c in (a, b) if types[c] == "numeric"})
    rng = np.random.default_rng(zlib.crc32(f"{scenario}|{n}|{i}".encode()))
    perm = rng.permutation(len(pool))
    ref = pool.iloc[perm[:N_REF]].reset_index(drop=True)
    rest = pool.iloc[perm[N_REF:]].reset_index(drop=True)
    if scenario == "mix":
        w = np.where(rest["Type"] == 2, 3.0, 1.0)
        bat = rest.iloc[rng.choice(len(rest), size=n, replace=False, p=w / w.sum())].reset_index(drop=True)
    else:
        bat = rest.iloc[:n].copy().reset_index(drop=True)
    if scenario == "monotone":
        for c in monotone_cols:
            bat[c] = bat[c] * 1.3 + 20
    if scenario in ("shuffle", "shuffle25"):
        k = n if scenario == "shuffle" else n // 4
        idx = rng.choice(n, size=k, replace=False)
        bat.loc[idx, shuffle_col] = bat.loc[rng.permutation(idx), shuffle_col].to_numpy()

    clean = lambda df: {c: rel.clean_column(df[c].tolist(), t) for c, t in types.items()}
    rc, bc = clean(ref), clean(bat)
    k_draws = rel.null_draws_for_family(len(types) + len(relationships), 0.05)
    extras = []
    for a, b, kind in relationships:
        prep = rel.prepare_test(kind, rel.reference_state(kind, rc[a], rc[b]), (rc[a], rc[b]), (bc[a], bc[b]))
        if not prep["testable"]:
            continue
        done = rel.finish_test(prep, k_draws, seed=zlib.crc32(f"{scenario}|{n}|{i}|{a}|{b}".encode()))
        extras.append({"name": rel.pair_name(a, b), "p_value": done["p_value"], "effect_size": done["effect"],
                       "effect_floor": rel.DEFAULT_FLOORS[kind], "label": kind})
    cfg = CalibrationConfig(decision_mode="calibrated")
    cfg.psi_null_draws = max(cfg.psi_null_draws, k_draws)
    det = DistributionDetector(calibration_config=cfg)
    to_list = lambda d: {c: [v for v in d[c] if v is not None and not (isinstance(v, float) and v != v)] for c in d}
    ftypes = {c: "continuous" if t == "numeric" else "categorical" for c, t in types.items()}
    det.fit_baseline(to_list(rc), ftypes)
    out = det.analyze_production_window(to_list(bc), extras)
    return {"scenario": scenario, "n": n, "i": i,
            "columns_flagged": [c for c, m in out["feature_metrics"].items() if m["drift_detected"]],
            "relationships_flagged": [r for r, m in out.get("extra_metrics", {}).items() if m["drift_detected"]],
            "relationships_tested": len(extras)}


def summarize(rows, shuffle_col, monotone_cols):
    lines = []
    for scenario in SCENARIOS:
        for n in BATCHES:
            g = [r for r in rows if r["scenario"] == scenario and r["n"] == n]
            any_rel = np.mean([bool(r["relationships_flagged"]) for r in g])
            targets = monotone_cols if scenario == "monotone" else [shuffle_col]
            target = "+".join(targets)
            col_alarm = np.mean([any(t in r["columns_flagged"] for t in targets) for r in g])
            flagged = [p for r in g for p in r["relationships_flagged"]]
            involving = [p for p in flagged if shuffle_col in p.split("<->")]
            attribution = len(involving) / len(flagged) if flagged else float("nan")
            any_alarm = np.mean([bool(r["relationships_flagged"] or r["columns_flagged"]) for r in g])
            lines.append({"scenario": scenario, "n": n, "draws": len(g), "p_any_relationship_alarm": any_rel,
                          f"p_column_alarm_{target}": col_alarm, "p_any_alarm": any_alarm,
                          "attribution_to_shuffled_column": attribution})
    return lines


def main():
    pool, types, relationships, shuffle_col = setup()
    print(f"types: {types}\nrelationships ({len(relationships)}): {relationships}\nshuffled column: {shuffle_col}",
          flush=True)
    t0 = time.time()
    rows = []
    for scenario in SCENARIOS:
        for n in BATCHES:
            rows += Parallel(n_jobs=8)(delayed(one_draw)(pool, types, relationships, shuffle_col, scenario, n, i)
                                       for i in range(DRAWS))
            print(f"  {scenario} n={n} done ({time.time() - t0:.0f}s)", flush=True)
    monotone_cols = sorted({c for a, b, _ in relationships for c in (a, b) if types[c] == "numeric"})
    summary = summarize(rows, shuffle_col, monotone_cols)
    json.dump({"types": types, "relationships": relationships, "shuffle_col": shuffle_col,
               "summary": summary, "rows": rows}, open(OUT, "w"), default=float)
    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
