# Progress Log

Resume-from-here document for the multi-step drift-monitoring calibration/
hardening plan. Update this after every logical change, not just at step
boundaries.

---

## Step 0 — Recon

**Status: DONE.**

- `docs/recon.md` written, answering all 9 recon questions with file/line
  references and measured (not estimated) numbers.
- Commits: `bd76e48` (debug-print fix, done alongside Step 0 per instruction),
  `9a4c334` (recon.md).
- Tests: full existing suite re-run before/after the debug-print fix —
  39/39 pass, no regressions.

**Key findings (see `docs/recon.md` for detail):**
1. Tabular baseline stores a full, uncapped reference sample (not just IQR
   fences) — ~5ms load+fit cost at the current Citi Bike size (5,000/2,000
   rows, 0.31MB).
2. No multiple-testing correction exists anywhere today; `system_alert_triggered`
   is a plain OR across features.
3. `month`'s PSI=16.27 is the epsilon-substitution mechanism operating as
   intended on categories absent from the reference by construction — not a
   bug.
4. `tests/test_drift_engine.py`'s headline numbers are all at
   `PRODUCTION_BATCH_SIZE=25000`, a size the sample-size sweep never tests —
   the sweep is a separate experiment, not a source for the headline number.
5. Auth is `streamlit_google_auth.Authenticate` (pinned `1.1.8`, matches
   installed version exactly), not `st.login()`. **Corrected 2026-09-29**:
   `patched_init.py`'s diff against the exact pinned version is whitespace-
   only — it's a no-op today, not load-bearing. Scheduled for removal in
   Step 7 (together with the overwrite step) with a local Google-login smoke
   test as the acceptance check.
6. **Resolved 2026-09-29**: nothing is deployed anywhere. Target going
   forward is the single-container main `Dockerfile` topology (nginx +
   supervisord + backend + dashboard). Step 7 backlog (not started):
   compose runs that one image once with named volumes for DB + model cache;
   delete `Dockerfile.dashboard` + `runtime.txt` (approved in principle, diff
   first); multi-arch build (amd64 + arm64, target platform TBD); bump base
   image to `python:3.12-slim` (3.10 EOLs Oct 2026), full test suite as the
   verification gate; CI targets whatever the image actually uses.
7. `runtime.txt` (3.12.4) doesn't match either Dockerfile (3.10-slim) —
   confirmed stale, approved for deletion in Step 7.
8. **`split_citi_bike.py` did not exist anywhere in the repo** — resolved
   2026-09-29 (see items below): replaced with a new, documented
   `scripts/split_citi_bike.py` rather than reconstructing the original.
9. **New finding (found while building the replacement script)**: the local
   `tests/citi_bike_production.csv` only contains **Apr/May/Jun** data
   (verified: `month` ∈ {4,5,6} only, 2,922,389 rows total) — **not**
   Apr–Dec as the README's Validation Results section claims. Any Step 1
   numbers will be scoped to Apr–Jun unless more months are sourced (which
   needs sign-off first, per the >~50MB download gate). See
   `results/citi_bike_provenance_forensics.md`.

**Citi Bike provenance work (done 2026-09-29, per user decision):**
- Time-boxed forensic check written to
  `results/citi_bike_provenance_forensics.md`: the old `citi_bike_v1`
  reference sample (5,000 continuous / 2,000 categorical rows) was drawn
  non-uniformly from roughly the first ~17,000 rows (~1%) of
  `tests/citi_bike_baseline.csv` — not a contiguous prefix, not a uniform
  sample across the full Jan–Mar file. Exact original method unrecoverable;
  not blocking, per instruction.
- New `scripts/split_citi_bike.py` written: fixed seed (42), disjoint
  Jan–Mar holdout pool (25,000 rows, byte-identical across reference-size
  runs — verified), reference size as a CLI parameter, per-month production
  splits, full JSON manifest with source/output SHA256 hashes written to
  `results/`.
- Run twice: `--reference-size 5000` (legacy-equivalent) and
  `--reference-size 50000`. Manifests: `results/split_citi_bike_manifest_ref5000.json`,
  `results/split_citi_bike_manifest_ref50000.json`. Holdout pool confirmed
  byte-identical across both runs (hash-verified); only the reference differs.
- **Old README validation numbers are superseded** — decision recorded: must
  not be quoted anywhere until regenerated against this new split.

**Both prior open questions resolved 2026-09-29** (deployment topology,
Citi Bike split strategy) — see items 5-9 above and PROGRESS's decision log.

**Resolved 2026-09-29: proceed with Step 1 scoped to Apr–Jun 2016, don't
source Jul–Dec now.** Every result/table/note states "Apr–Jun 2016"
explicitly. Added to the Step 9 README-correction list: *"README claims
Apr–Dec; data is Apr–Jun."*

**A full Jan–Dec rebuild is explicitly deferred, not part of this pass** —
extending with Jul–Dec data prepared differently from the existing CSVs
would inject artificial drift. **Since no original preprocessing script
exists anywhere in the repo or its git history (confirmed below), that
deferred rebuild is not "add the missing months to what's here" — it means
writing preprocessing from scratch against the raw Citi Bike 2016 trip files
and regenerating ALL twelve months, including Jan–Jun, from that one new
script, not just the missing Jul–Dec.** The current baseline/production CSVs
cannot be partially extended; whenever this rebuild happens, everything gets
regenerated together so the whole year is produced by one consistent,
documented process.

**Pre-Step-1 verification, done 2026-09-29** (full detail in
`results/citi_bike_provenance_forensics.md`):
- Row counts: baseline 1,577,611 + production 2,922,389 = **4,500,000 —
  matches the README's "4.5 million rows" claim exactly.** Only the Apr–Dec
  month-range breakdown attached to that number is wrong.
- **The "early January only" hypothesis is WRONG — checked and rejected.**
  Both CSVs are row-shuffled, not time-ordered. The old reference's row-
  position range (0–16,774) maps to date range **2016-01-01 to 2016-03-31**
  — the full Jan–Mar period, not early January. The old results did NOT
  compare early-January behavior against spring.
- **"15 trials" is not "3 months × 5 batches"** — checked in the actual code
  (`run_classification_evaluation`, `tests/test_drift_engine.py:276-403`):
  no month-based stratification exists; Case B samples from the whole pooled
  `prod_df` regardless of month. `15 = 3×5` is numerically coincidental.
- **Original raw-data preprocessing script does not exist anywhere in the
  repo or its git history** (searched both). Confirms the Jan–Dec rebuild
  above has to start from scratch, not from a recoverable script.

---

## Step 1 — Tabular validation reconciliation

**Status: DONE.**

- `scripts/step1_validation.py`: ran against the live backend, both
  reference sizes (5000, 50000), disjoint per-month batches (Apr/May/Jun,
  8 trials/month/size), A/A test (100 trials/size from the holdout pool,
  6 sizes ≤ holdout capacity — 50000 excluded, disclosed), extended severity
  sweep (0.02σ–3.0σ, 5 trials each), minimum-drift-fraction sweep. ~2,000
  HTTP calls total, ~4.5 minutes wall time, zero errors.
- `scripts/step1_analyze.py`: consumes the raw output, produces
  `results/tabular_validation_legacy.json` (structured summary) and
  `results/tabular_validation_notes.md` (full write-up with every table).
- Raw output: `results/tabular_validation_legacy_raw.json` (2.6MB, every
  individual `/analyze` response preserved).
- Test projects (`step1_val_ref5000`, `step1_val_ref50000`) cleaned up from
  `drift.db` after the run.
- Commits: `4f01412` (initial run), `0a1c604` (ground-truth fix), and one
  more (this entry) for the threshold-wording/synthetic-realism fixes below.

**Synthetic drift injection bug found and fixed (2026-09-29, same day, user
caught it before Step 2).** The additive mean-shift method (`x +
severity*std`) was checked against a hypothesis: pickup/dropoff coordinates
are station locations with only ~475 unique values in a 25,000-row pool
(confirmed), and trip_duration is integer seconds. **Confirmed empirically**
(`scripts/step1_severity_v2.py`'s realism check, saved in
`results/tabular_validation_severity_v2.json`): adding a shift of `1e-9` to
`pickup_longitude` nearly doubled its KS D (0.0143→0.0217) and flipped
p from 0.36 to 0.039 — a negligible shift "detected" as drift, purely from
moving every value off its exact discrete lattice position, unrelated to
true effect size. **Fixed**: replaced additive shift with exponential
tilting (resample WITH replacement from the holdout pool, weights ∝
`exp(lambda*z)`, lambda solved so the resample's mean standardized value
hits the target severity) — support-preserving by construction, verified to
show no artifact at severity=0. One real limitation surfaced and handled
honestly, not silently: `pickup_latitude`/`dropoff_latitude`'s holdout pool
tops out around z≈2.3, so 3.0σ isn't actually achievable for them via
resampling — the script detects and caps this rather than fabricating a
result. The old additive results are kept only as a labeled diagnostic of
the artifact, both in the raw JSON and in the notes, never as a real
detection-sensitivity claim. New finding from the corrected data: smallest
reliably-detected severity varies hugely by feature (coordinates: 0.05–0.1σ;
`trip_duration`: 1.5σ, since its heavy right skew means a small mean-shift
via resampling barely perturbs the bulk of the distribution) — there is no
single "smallest reliable severity" number, so none is reported.

**Threshold-dependent wording fixed.** `pickup_latitude` (D=0.0189) and
`dropoff_latitude` (D=0.0195) are ground-truth positive at `D_gt=0.01` but
negative at `D_gt=0.02` — every recall/precision/FP statement in the notes
now names its `D_gt` explicitly; none is stated as a bare, threshold-free
number anymore. Added: a per-feature detection-RATE table (the power curve)
that doesn't depend on any `D_gt` cut at all, so the knife-edge problem is
sidestepped entirely for that view; an explicit note that `D_gt=0.02` falls
inside a four-feature cluster (population D 0.0189–0.0219), making metrics
at that exact threshold highly sensitive to the cut; an A/A caveat that KS is
conservative on near-discrete data, so false-alarm rates below alpha there
may partly reflect discreteness, not calibration; and a caveat that the
minimum-detectable-D floor formula is the continuous-case approximation,
not validated for near-discrete features.

**Takeaway, stated plainly (motivates Step 2's materiality gate)**: a larger
reference makes the p-value test detect population shifts the effect-size
ground truth calls immaterial — the power curve shows this is a smooth
curve, not a discontinuity, so a bare significance test has no way to
threshold it correctly on its own. Gate 2 (materiality, on top of Gate 1
significance) is the mechanism designed to fix exactly this.

**Ground truth bug found and fixed (2026-09-29, same day, before user
review).** The first pass computed "population ground truth" from the
SAMPLED reference (5,000 or 50,000 rows) vs. production — which meant the
label itself changed depending on which reference was drawn
(`dropoff_longitude` flipped between stable and drifted across the two
reference sizes on identical production data). **That is not a ground
truth — it's a reference-dependent diagnostic**, per the user's own framing:
"If a label changes when only the reference size changes, it is being
computed partly from the reference sample." Fixed in `scripts/step1_analyze.py`:
population truth is now computed **once**, from the **full baseline CSV**
(1,577,611 rows — never any sampled reference) vs. full production data,
using effect-size criteria only (D≥0.01/0.02/0.05 continuous, PSI≥0.1/0.2
categorical — no p-values at this population scale). This label is now
identical across every reference size and batch size by construction.
**Verified**: `dropoff_longitude` now has a single population D (0.0206)
used for both reference sizes. The old reference-vs-production computation
is kept only as a renamed diagnostic (`reference_detectable_effect`),
reported but never used to compute TP/FP/FN/TN. Relabeling was done
**offline against the already-collected detections — no new HTTP calls**,
since the engine's outputs don't change, only which label they're graded
against. Bonus consistency check: the new fixed population D values
(0.0219, 0.0189, 0.0206, 0.0195, 0.0924, 0.0432, 16.2660) exactly match
`tests/test_drift_engine.py`'s own original ground-truth-verification
numbers from a much earlier session run — strong evidence the fix is
correct, since both now compute the same thing (full baseline vs. full
available production) the same way.

**Headline findings (full detail and every table in
`results/tabular_validation_notes.md`, now on the corrected ground truth):**

1. **Reference size, not batch size, explains the recall plateau —
   confirmed empirically, exactly as hypothesized, and now on a ground
   truth that doesn't move when the reference does.** The theoretical KS
   critical-value floor `c(0.05)/sqrt(m)` is ≈0.0192 at m=5,000 and ≈0.0061
   at m=50,000. `pickup_latitude`'s FIXED population D is 0.0189 — right at
   the m=5,000 floor. At m=5,000 (D_gt=0.01, since 0.0189<0.02), recall
   plateaus around 0.79–0.83 even at the largest batch sizes tested (20k,
   50k) — it never reaches 1.0. At m=50,000, recall reaches 1.000 by batch
   size 10,000 and stays there. No amount of extra production data fixes a
   reference that's too small to resolve an effect that close to its own
   noise floor.
2. A/A system false-alarm rates ranged 0.02–0.27 across sweep sizes and
   reference sizes, generally at or below the alpha-predicted ≈0.30 for 7
   uncorrected tests (`1-(1-0.05)^7`) — roughly consistent with prediction,
   somewhat lower, plausibly because the 7 features aren't fully independent
   (correlated coordinates). No correction is applied yet (that's Step 2).
3. Effect-size ground truth variants (D≥0.01/0.02/0.05) move the confusion
   matrix substantially now that they're computed on a stable population
   truth — e.g. at m=50,000, D_gt=0.05 keeps recall at 1.000 but collapses
   precision to 0.333, since most of Apr-Jun's real population differences
   are small-effect, not large. The engine's p<0.05 default flags effects
   the size-based ground truth would call practically negligible — the same
   oversensitivity-at-scale phenomenon PSI's ground truth was built to avoid
   for categoricals, now shown to apply to the continuous KS path too.
4. `month`'s per-batch PSI has std=0.0000 at every batch — expected, not a
   bug: every row in a given month's production file has that exact month
   value, so every batch's PSI for `month` compares against an identical
   100%-one-category distribution.
5. Smallest synthetic severity reliably detected (100% of trials, every
   continuous feature): **0.05σ at both reference sizes.**

**A/A test framing (per instruction):** labeled throughout as an **iid
test-calibration check** — both the holdout and reference samples are drawn
at random from the same Jan–Mar pool, so this measures whether the false-
alarm rate matches what alpha predicts under iid sampling. It does **not**
capture month-to-month variation within the baseline period (e.g. a
February-vs-March shift), since both come from the same pooled draw.

**Everything here is scoped to Apr-Jun 2016** — no number above or in
`results/tabular_validation_notes.md` should be read as "Apr-Dec."

---

## Step 2 — Two-gate calibrated decisions

**Status: NOT STARTED.**

**Scope additions from Step 1's findings (user, 2026-09-29), recorded now,
not implemented yet:**
- Every KS result should return the minimum detectable D for its actual
  `(n, m)` at the configured alpha — `c(alpha) * sqrt((n+m)/(n*m))` — so
  users can see what the test cannot detect, not just what it did detect.
- `/fit` should warn when the reference is small enough that this detection
  floor exceeds the project's configured KS effect floor (once effect floors
  exist as a per-project config, per the original Step 2 spec).
- Propose (don't set) a recommended minimum reference size, grounded in
  Step 1's actual results (e.g. the m=5,000 vs m=50,000 comparison in
  `results/tabular_validation_notes.md`) — a concrete number, not a guess,
  and explicitly a proposal for the user to approve, not a new default.

---

## Step 3 — API usable without the dashboard (PATs, file-parsing extraction, project deletion)

**Status: NOT STARTED.**

Design decision already recorded (per user instruction, see `docs/recon.md`
§6): PAT verification lives entirely in the backend auth dependency, as a
sibling to the existing JWT check — not integrated into the auth widget.
Dashboard-side PAT UI only needs `st.session_state["user_info"]`'s `email`/
`name`, already available today.

---

## Step 4 — Cross-modal dependence monitor (`drift/dependence.py`)

**Status: NOT STARTED.**

---

## Step 5 — History, sustained alerts, versioning

**Status: NOT STARTED.**

---

## Step 6 — Explanations

**Status: NOT STARTED.**

---

## Step 7 — Deployment and CI

**Status: NOT STARTED.**

Scope already reduced per user instruction: CPU-only torch/torchvision is
already in place in both Dockerfiles (confirmed in Step 0) — this step is now
just the image-size report for whichever topology turns out to be live,
lazy-loading verification, the ONNX investigation, CI (targeting Python 3.10,
not 3.12), and cleanup proposals (excluding `patched_init.py` and
`supervisord.conf`, which are not stale; `runtime.txt` is a real cleanup
candidate). Also owes: a proposed build-time guard that fails the build if
the upstream `streamlit_google_auth` file doesn't match the expected
version/hash before `patched_init.py` overwrites it (propose only, per
instruction).

---

## Step 8 — Run text/image validation to completion

**Status: NOT STARTED.**

---

## Step 9 — README final pass

**Status: NOT STARTED.**

---

## Standing rules for this whole effort (from the original instructions — not to be violated in any step)

- No fabricated/estimated numbers; everything traced to a script run saved
  under `results/`.
- No regressions: existing tests/endpoints/fields keep working; legacy
  behavior stays reproducible.
- Fix seeds; record every config value; never tune and report on the same
  seeds/scenarios.
- Stop and ask before: changing default behavior, finalizing alpha/effect
  floors/default decision mode, deleting files, changing Dockerfiles/Render/
  auth config, adding heavy dependencies, downloading datasets >~50MB.
- Commit per logical change.
