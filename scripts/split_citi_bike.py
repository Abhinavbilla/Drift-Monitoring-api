"""
Documented, reproducible replacement for the missing split_citi_bike.py
referenced by README.md (see docs/recon.md sections 4/9 and
results/citi_bike_provenance_forensics.md — the original script that produced
the live citi_bike_v1 baseline doesn't exist anywhere in this repo, and that
baseline's exact provenance is unreproducible; this script supersedes it
rather than trying to reconstruct it).

Produces three disjoint pieces from the Citi Bike 2016 source files, all
drawn with a fixed seed so every run is reproducible:

  1. REFERENCE  -- a sample of `--reference-size` rows from the Jan-Mar pool,
     used to /fit a baseline. This is the one thing that varies between runs
     (see the reference-size-vs-batch-size analysis in Step 1).
  2. HOLDOUT    -- a sample of `--holdout-size` rows from the Jan-Mar pool,
     DISJOINT from the reference by construction, and IDENTICAL across every
     run regardless of --reference-size (same seed, drawn first, before the
     reference sample is drawn from what's left). This is the clean
     out-of-sample pool for A/A (false-alarm-rate) testing -- rows that were
     never part of any fitted baseline.
  3. PRODUCTION -- the full Apr-Dec pool, split into one file per calendar
     month. Independent of both reference-size and holdout; identical across
     every run.

Every output file's SHA256 hash, every parameter, and every row count is
written to a JSON manifest under results/, so any later analysis can cite
exactly which file (by hash) it used.

Source files (must already exist locally -- not committed, too large for
git; see README's Installation section for how the original Citi Bike 2016
dataset was obtained):
    tests/citi_bike_baseline.csv    (Jan-Mar pool; 'month' in {1,2,3})
    tests/citi_bike_production.csv  (Apr-Dec pool; 'month' in {4..12})

Usage:
    python scripts/split_citi_bike.py --reference-size 5000
    python scripts/split_citi_bike.py --reference-size 50000
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SEED = 42
BASELINE_CSV = "tests/citi_bike_baseline.csv"
PRODUCTION_CSV = "tests/citi_bike_production.csv"
OUTPUT_DIR = "tests/splits"
RESULTS_DIR = "results"


def sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_of_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-size", type=int, required=True,
                         help="Rows to sample from the Jan-Mar pool as the /fit reference.")
    parser.add_argument("--holdout-size", type=int, default=25000,
                         help="Rows to sample from the Jan-Mar pool as the disjoint A/A holdout pool "
                              "(default 25000; fixed across all --reference-size runs).")
    args = parser.parse_args()

    for path in (BASELINE_CSV, PRODUCTION_CSV):
        if not os.path.exists(path):
            print(f"ERROR: required source file not found: {path}", file=sys.stderr)
            print("This script does not download data. See README.md's Installation section "
                  "for how the Citi Bike 2016 dataset was obtained.", file=sys.stderr)
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Hashing source files (integrity record)...")
    baseline_hash = sha256_of_file(BASELINE_CSV)
    production_hash = sha256_of_file(PRODUCTION_CSV)

    print(f"Loading {BASELINE_CSV} ...")
    t0 = time.time()
    baseline_df = pd.read_csv(BASELINE_CSV)
    print(f"  {len(baseline_df):,} rows in {time.time() - t0:.1f}s")

    bad_months = sorted(set(baseline_df["month"].unique()) - {1, 2, 3})
    if bad_months:
        print(f"ERROR: expected only Jan-Mar (1,2,3) in {BASELINE_CSV}, "
              f"found unexpected month values: {bad_months}", file=sys.stderr)
        sys.exit(1)

    n_total = len(baseline_df)
    if args.reference_size + args.holdout_size > n_total:
        print(f"ERROR: reference_size ({args.reference_size}) + holdout_size ({args.holdout_size}) "
              f"exceeds available Jan-Mar rows ({n_total}).", file=sys.stderr)
        sys.exit(1)

    # Holdout is drawn FIRST and depends only on SEED + holdout_size, never on
    # reference_size -- so it's byte-identical across every --reference-size
    # run, which is what makes the reference-size-vs-batch-size comparison
    # (Step 1) clean: only the reference changes between runs.
    rng_holdout = np.random.default_rng(SEED)
    pool_idx = np.arange(n_total)
    rng_holdout.shuffle(pool_idx)
    holdout_idx = pool_idx[: args.holdout_size]
    remaining_idx = pool_idx[args.holdout_size:]

    # Reference is drawn from what's left after removing the holdout rows --
    # disjoint from holdout by construction, not just by chance.
    rng_ref = np.random.default_rng(SEED)
    reference_idx = rng_ref.choice(remaining_idx, size=args.reference_size, replace=False)

    assert len(set(reference_idx.tolist()) & set(holdout_idx.tolist())) == 0, \
        "BUG: reference and holdout overlap -- this should be impossible by construction."

    reference_df = baseline_df.iloc[reference_idx].reset_index(drop=True)
    holdout_df = baseline_df.iloc[holdout_idx].reset_index(drop=True)

    ref_path = os.path.join(OUTPUT_DIR, f"reference_n{args.reference_size}.csv")
    holdout_path = os.path.join(OUTPUT_DIR, f"holdout_n{args.holdout_size}.csv")
    reference_df.to_csv(ref_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)
    print(f"Wrote reference ({len(reference_df):,} rows) -> {ref_path}")
    print(f"Wrote holdout ({len(holdout_df):,} rows) -> {holdout_path}")

    print(f"Loading {PRODUCTION_CSV} ...")
    t0 = time.time()
    production_df = pd.read_csv(PRODUCTION_CSV)
    print(f"  {len(production_df):,} rows in {time.time() - t0:.1f}s")

    bad_months = sorted(set(production_df["month"].unique()) - set(range(4, 13)))
    if bad_months:
        print(f"ERROR: expected only Apr-Dec (4-12) in {PRODUCTION_CSV}, "
              f"found unexpected month values: {bad_months}", file=sys.stderr)
        sys.exit(1)

    present_months = sorted(production_df["month"].unique().tolist())
    missing_months = sorted(set(range(4, 13)) - set(present_months))
    if missing_months:
        print(f"WARNING: {PRODUCTION_CSV} does not cover the full Apr-Dec range.", file=sys.stderr)
        print(f"  Present months: {present_months}", file=sys.stderr)
        print(f"  Missing months: {missing_months}", file=sys.stderr)
        print(f"  The README's 'Apr-Dec 2016' claim is not reproducible with this file -- "
              f"see results/citi_bike_provenance_forensics.md.", file=sys.stderr)

    production_files = {}
    for m in present_months:
        month_df = production_df[production_df["month"] == m].reset_index(drop=True)
        month_path = os.path.join(OUTPUT_DIR, f"production_month_{m:02d}.csv")
        month_df.to_csv(month_path, index=False)
        production_files[m] = {
            "path": month_path,
            "rows": len(month_df),
            "sha256": sha256_of_file(month_path),
        }
        print(f"Wrote month {m:02d} ({len(month_df):,} rows) -> {month_path}")

    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "parameters": {
            "reference_size": args.reference_size,
            "holdout_size": args.holdout_size,
        },
        "source_files": {
            "baseline_csv": {"path": BASELINE_CSV, "rows": n_total, "sha256": baseline_hash},
            "production_csv": {"path": PRODUCTION_CSV, "rows": len(production_df), "sha256": production_hash},
        },
        "outputs": {
            "reference": {"path": ref_path, "rows": len(reference_df), "sha256": sha256_of_file(ref_path),
                          "content_hash": sha256_of_df(reference_df)},
            "holdout": {"path": holdout_path, "rows": len(holdout_df), "sha256": sha256_of_file(holdout_path),
                        "content_hash": sha256_of_df(holdout_df)},
            "production_by_month": production_files,
        },
        "production_months_present": present_months,
        "production_months_missing": missing_months,
        "production_apr_dec_claim_reproducible": len(missing_months) == 0,
        "disjointness_check": "reference and holdout row indices verified non-overlapping by assertion",
    }

    manifest_path = os.path.join(RESULTS_DIR, f"split_citi_bike_manifest_ref{args.reference_size}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
