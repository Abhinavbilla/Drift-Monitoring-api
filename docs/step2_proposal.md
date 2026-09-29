# Step 2 Proposal — Two-Gate Calibrated Decisions

Design proposal only. Per standing instructions, this stops before
finalizing alpha, effect floors, the default decision mode, and the DCT
calibration default — those four are presented as explicit decision points
at the end, grounded in Step 1's actual measured data, not guessed.

---

## Config schema (per-project, stored with the baseline)

```
decision_mode: "legacy" | "calibrated"   # existing projects default "legacy"; see Decision Point 3
alpha: float                              # per-test significance level; see Decision Point 1
multiple_testing: "holm" | "bh" | "none"  # default "holm" (per original spec: system alert = "any
                                           # feature flagged", so FWER control bounds the system-level
                                           # false-alarm rate -- matches Step 1's A/A finding that
                                           # uncorrected per-feature tests inflate the system rate well
                                           # above alpha)
effect_floors: {ks_d: float, psi: float, dct_auc: float}   # see Decision Point 2
psi_null_draws: 1000
dct_calibration: "precomputed" | "permutation"   # see Decision Point 4
dct_permutations: 100
```

## Gate 1 — significance (per detector)

**KS (continuous, tabular)**: unchanged mechanism — `scipy.stats.ks_2samp`'s
p-value against the stored reference sample. **IID assumption, stated
explicitly**: this assumes both the reference and the production batch are
independent draws from their respective populations. Step 1's forensic work
found the source CSVs are already row-shuffled (not time-ordered), which
supports this for the Citi Bike data specifically — but it's a real
assumption the engine doesn't verify, and a caller with genuinely
autocorrelated data (e.g., true time-series structure) would violate it
silently.

**PSI (categorical, tabular)**: replace the flat `PSI > 0.2` cutoff with a
proper significance test — parametric bootstrap at the actual batch size
`n`: draw `psi_null_draws` (1000) multinomial samples of size `n` from the
stored reference frequencies (same binning, same epsilon as the existing
`_check_categorical_drift`), compute PSI for each null draw, and set
`p_value_adjusted = (1 + #{null_PSI >= observed_PSI}) / (psi_null_draws + 1)`.
This directly answers "how surprising is this PSI at this batch size,"
which the current flat threshold doesn't — a PSI of 0.15 is unremarkable at
n=100 but could be very significant at n=50,000.

**DCT (text/image/joint)**: two candidate strategies, benchmarked below
(`results/dct_calibration_benchmark.json`, synthetic 384-dim embeddings,
`B=100` draws, real backend-independent timing):

| n per side | Precomputed (one-time, at `/fit`) | Permutation (EVERY `/analyze` call) |
|---|---|---|
| 40 | 1.95s total (19.5ms/draw) | 1.73s total (17.3ms/draw) |
| 200 | 6.19s total (61.9ms/draw) | 6.68s total (66.8ms/draw) |
| 1,000 | 11.50s total (115.0ms/draw) | 10.90s total (109.0ms/draw) |
| 5,000 | 29.28s total (292.8ms/draw) | **23.16s total (231.6ms/draw)** |

**Per-draw cost is roughly the same between strategies** (both run the
identical CV pipeline `B` times) — the difference is *when* that cost is
paid. Precomputed pays it once at `/fit` time and reuses stored null
quantiles at every subsequent `/analyze` call. Permutation pays the full
cost on **every single `/analyze` call** — at n=5,000 per side, that's
**23 seconds added to one API call**, which is not viable for an
interactive endpoint at that batch size.

**Precomputed's documented bias, stated explicitly**: it splits the stored
reference in half (pseudo-reference vs. pseudo-batch) to build the null
distribution, so the null is calibrated against a *smaller* reference than
what's actually used at analyze time. This could make the calibration
slightly miscalibrated relative to the true null at full reference size —
a real tradeoff against the latency win, not a free lunch. Interpolating in
`log(n)` (per the original spec) at analyze time doesn't fix this bias, only
smooths across the pseudo-batch-size grid.

**Correction applied across**: tabular features (Holm/BH over the set of
monitored features) and joint modality pairs (Holm/BH over whichever
pairs are present) — both bounded by the same family-wise logic Step 1's
A/A test measured empirically (`docs/PROGRESS.md`: observed system
false-alarm rates were at or below the uncorrected alpha-predicted rate of
~0.30 for 7 tests — correction should bring this down toward alpha itself).

## Gate 2 — materiality

Flag only if significant after correction (Gate 1) **AND** effect size ≥
the configured floor. Response gains: `effect_size`, `effect_floor`,
`p_value_adjusted`, `significant`, `material`, `decision_mode`,
`threshold_used` — every existing field stays, these are additive.

## Legacy mode

`decision_mode="legacy"` must reproduce today's behavior exactly: KS
`p < alpha` alone (no correction, no effect floor), PSI `> 0.2` alone, DCT
`AUC > 0.65` alone. Existing projects default here per instruction. Tests:
run the exact fixtures from `tests/test_ingestion_robustness.py`,
`tests/test_embedding_adapters.py`, and `tests/test_joint_adapter.py`
through the new code path in legacy mode and assert byte-identical
`drift_detected`/`statistic`/`p_value` outputs to today's — this is the
literal regression gate before anything else in Step 2 ships.

---

## Four decision points (stopping here, per instruction)

**1. Alpha.** Recommend keeping **0.05** as the per-test rate, unchanged
from today — the family-wise correction (Holm/BH) is what controls the
system-level rate, not alpha itself, so there's no statistical reason to
also shrink alpha. Changing it would be a second, independent lever doing
the same job the correction already does.

**2. Effect floors.** This is the one Step 1's data most directly informs,
and where I'd push back on the most "obvious" choice:
- **KS D floor**: Step 1 found a knife-edge cluster of four features with
  population D between 0.0189 and 0.0219. **A floor of exactly 0.02 is
  arguably the worst choice available** — it splits this cluster
  down the middle, making the materiality gate's outcome hypersensitive to
  noise in the population D estimate itself. I'd recommend either **0.03**
  (clear of the whole cluster, a deliberately conservative "this is a real,
  unambiguous shift" floor) or **0.015** (a more liberal floor that still
  excludes the smallest, most float-noise-prone effects but keeps the
  cluster's features as positives) — not 0.02. Needs your call on which
  side of the tradeoff you want.
- **PSI floor**: propose keeping **0.2**, the existing industry-standard
  threshold, now used as materiality rather than the sole criterion —
  Gate 1's bootstrap significance test does the "is this surprising" job,
  Gate 2's 0.2 floor does the "is this big enough to matter" job.
- **DCT AUC floor**: propose keeping **0.65**, the existing default,
  same reasoning as PSI above.

**3. Default decision mode for NEW projects.** (Existing projects are
already settled as `legacy`, per instruction — this is only about what a
freshly-created project gets absent an explicit choice.) Two honest
options: **legacy** (safest, zero behavior surprise, but inherits every
false-positive/knife-edge issue Step 1 documented) or **calibrated**
(statistically sounder, but new — less battle-tested, and its own numeric
defaults are exactly what's being decided in points 1-2 and 4 right now).
I'd lean toward **legacy-by-default even for new projects** until
calibrated mode has run through Step 2's own test suite and the Step 1
suite has been re-run in calibrated mode for a side-by-side comparison
(both still pending) — but this is your call, not a technical constraint.

**4. DCT calibration default.** Given the benchmark above, **precomputed**
is the only viable default for interactive use once batch sizes reach
the low thousands — permutation's per-call cost becomes prohibitive
exactly where DCT calibration matters most (larger, more reliable batches).
Recommend **precomputed** as the default, with **permutation** available as
an explicit opt-in for offline/batch analysis where the bias concern above
outweighs the latency cost. This is the one recommendation I'm fairly
confident about; the others are more genuinely open trade-offs.

---

Once these four are settled, next steps are: implement the config schema
and both gates, write the legacy-identity regression tests, rerun the Step 1
suite in calibrated mode for `results/tabular_validation_calibrated.json`,
rerun the A/A test to measure the corrected false-alarm rate, build the
floor-sensitivity table across the KS D / PSI / AUC candidates above, and
rerun the text/image smoke tests including the ~40-sample borderline case.
