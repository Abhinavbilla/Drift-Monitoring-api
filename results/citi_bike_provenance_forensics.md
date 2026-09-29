# Citi Bike `citi_bike_v1` Baseline — Provenance Forensics

Time-boxed (~30 min) investigation into how the live `citi_bike_v1` baseline
in `drift.db` was actually constructed, since `split_citi_bike.py` (which the
README says to run) doesn't exist anywhere in the repo (see `docs/recon.md`
§4/§9). Documented, not blocking — per instruction, no attempt was made to
reconstruct the exact original split; this is what could be determined from
the artifacts that do exist.

## What exists locally

- `tests/citi_bike_baseline.csv` — 1,577,611 rows, columns
  `id, gender_id, pickup_datetime, dropoff_datetime, pickup_longitude,
  pickup_latitude, dropoff_longitude, dropoff_latitude, trip_duration, month`.
- `tests/citi_bike_production.csv` — 2,922,389 rows (not re-inspected in this
  check; not needed to answer the reference-provenance question).
- The live SQLite baseline row for `project_id='citi_bike_v1'`
  (`baselines.reference_data`, measured in `docs/recon.md` §1): 5,000 rows
  for each continuous feature, 2,000 rows for each categorical feature.

## Method

Matched every stored reference value back to its first-occurrence row index
in `tests/citi_bike_baseline.csv`, per column, to characterize which rows (by
position) the old baseline was drawn from.

- `pickup_longitude` (high-precision float, GPS coordinate — reliable
  discriminator since duplicate values are rare): all 5,000 stored values
  found in the CSV. Matched row indices range from **0 to 16,774** (out of
  1,577,611 total rows) — i.e. confined to roughly the **first 1.06%** of the
  file. Mean matched index ≈ 470; not a contiguous prefix (exact list-order
  comparison against `df.iloc[:5000]` failed), but also not spread across the
  full Jan–Mar period — the sample is heavily front-loaded into an early
  slice of the file.
- `trip_duration` (integer seconds — **unreliable** discriminator: many
  repeated values across the full file mean "first occurrence" matches can
  land anywhere): matched index range 0 to 1,554,859, i.e. nearly the whole
  file. This is a false signal from value duplication, not evidence the
  sample spans the whole file — superseded by the `pickup_longitude` result
  above for any column with enough precision to avoid collisions.
- `month` (categorical, 2,000 stored rows): distribution `{1: 502, 2: 581,
  3: 917}` (25%/29%/46%). The first 20,000 rows of the CSV have distribution
  `{1: 5071, 2: 5641, 3: 9288}` (25%/28%/46%) — closely proportional, which is
  consistent with the categorical reference also being drawn from
  approximately that same early slice of the file, not a uniform sample
  across the full 1.58M rows.

## Conclusion

The old `citi_bike_v1` baseline's reference sample was **not** a uniform
random sample across the full Jan–Mar baseline file, and was **not** a
literal contiguous head of it either — the evidence points to a sample drawn
from roughly the first ~17,000 rows (~1%) of `tests/citi_bike_baseline.csv`,
with month proportions matching that early slice rather than the file's
overall Jan–Mar composition. The exact mechanism (random `random_state`,
which library call, whether it went through the dashboard's own
`combined_df.sample(n=min(10000, len(df)), random_state=42)`-style preview
step, or something else) **cannot be determined from the artifacts present in
this repo** — no script or log records the original `/fit` call's arguments.

**This does not block Step 1.** Per instruction, the old baseline and every
number derived from it are treated as superseded and unreproducible. A new,
documented `scripts/split_citi_bike.py` (see `docs/PROGRESS.md` and the
script itself) replaces this going forward, with a fixed seed, an explicit
disjoint Jan–Mar holdout pool for A/A testing, and reference size as an
explicit parameter — closing exactly the gap this forensic check found.

## Second finding (discovered while building the replacement script): the local production file doesn't cover Apr–Dec

Running `scripts/split_citi_bike.py` against the actual local
`tests/citi_bike_production.csv` surfaced a second, independent discrepancy:
the file's `month` column only contains **{4, 5, 6}** (802,939 / 961,369 /
1,158,081 rows respectively, totaling exactly its 2,922,389 rows) — **not**
Apr–Dec as the README's Validation Results section claims ("Real production
batches (Apr–Dec 2016) were compared against a baseline locked on Jan–Mar
2016 data"). Verified directly: `sorted(df['month'].unique())` on the full
file returns `[4, 5, 6]` only.

This means **the README's existing "Apr–Dec" framing is not reproducible
with any data currently available locally** — regardless of the baseline
provenance question above. Any regenerated validation numbers in Step 1 will
necessarily be scoped to Apr–Jun unless additional months are sourced (which,
per standing instructions, requires asking first if it means downloading
>~50MB of new data). `scripts/split_citi_bike.py` does not silently claim
Jul–Dec coverage it doesn't have — it detects and logs the missing months and
records `production_apr_dec_claim_reproducible: false` in its manifest.
