# Progress Log

Resume-from-here document for the multi-step drift-monitoring calibration/
hardening plan. Update this after every logical change, not just at step
boundaries.

---

## HANDOFF — read this first if starting a fresh session (2026-09-30)

**Project goal, for context on every decision below (recorded 2026-09-30):**
this API exists so ML teams can integrate it into their model-serving
pipelines to monitor **data drift of model inputs only** — no labels, no
accuracy/performance tracking, no ground-truth-outcome monitoring of any
kind. Every design choice (batch-based, not streaming; statistical
significance + materiality, not accuracy metrics; per-feature not
per-prediction) should be read against that scope.

**Next priority AFTER the checks below (2026-09-30): programmatic
integration** — personal access tokens, upload endpoints, a Python client,
and an end-to-end model-serving example, so a team can wire this into a
real pipeline without the dashboard. **Do NOT start this in this pass** —
it will be requested separately. This supersedes/absorbs what was
previously sketched as "Step 3" in `docs/step2_proposal.md`'s references;
treat that as directionally right but not yet scoped in detail.

**Where things stand (2026-09-30)**: Step 0, Step 1, and Step 2 (a)/(b)/(c)
are DONE — see their sections below for full detail. The user is now
running a rigorous methodological review of (c)'s side-by-side before
treating it as final (see "Step 2 (c) review checks" below for that work,
in progress). **(d) DCT calibration and (e) text/image smoke tests are not
started and should not be started until the review checks below are done
and reviewed.**

**Decisions locked (user, 2026-09-29/30 — do not re-litigate, do not
re-derive from data, just implement):**
- `alpha = 0.05`, corrected family-wise per batch via **Holm** across
  features by default; **BH kept as an option** (`multiple_testing: "bh"`).
- Effect floors: **KS D = 0.05** (a round, conservative anti-alert-fatigue
  default chosen *independent* of the Citi Bike knife-edge cluster —
  explicitly not tuned to this evaluation's data, per ground rule 6).
  Floors are **per-project, per-feature overridable**. At `/fit`, each
  feature's floor is shown next to the minimum detectable D for the
  current reference size — **API response field, not a dashboard UI
  element** (see UI decision below). **PSI floor = 0.2** (industry
  convention). **DCT AUC floor = 0.65**, labeled provisional/unvalidated
  until Step 8's text/image validation runs.
- **Default decision_mode for NEW projects = "calibrated"** (changed
  2026-09-30, after reviewing (c)'s side-by-side: calibrated held
  precision=recall=1.000 at every batch size ≥3,000 vs. legacy's precision
  collapsing to 0.33-0.51 at scale, and A/A system false-alarm rate
  0.000-0.010 vs. legacy's 0.02-0.27 — see the Step 2 (c) section for
  numbers). **Existing projects stay legacy PERMANENTLY, with no
  migration** — this is a default for newly-created projects only, never
  retroactive. Implemented with a test (`tests/test_calibration_db.py` or
  a new test — see Step 2 (a)/(b) section for exact location) confirming
  new-project-with-no-explicit-config resolves to calibrated while a
  project fit before this change stays legacy.
- **DCT calibration default = "precomputed"**; permutation is opt-in only.
  Document the half-size-pseudo-reference bias explicitly wherever this is
  surfaced (conservative: fewer false alarms, slightly less power). For a
  batch size outside the calibrated grid: **clamp to the nearest grid point
  and return a warning — never extrapolate silently.**

**UI decision (user, 2026-09-30): the Streamlit dashboard is being replaced
by a new React + TypeScript frontend after Step 3.**
- **Build no new Streamlit UI from this point forward, for any step.**
- All Step 2 response-field additions are **API response fields only** —
  no dashboard rendering work.
- When PAT/upload-endpoint/client work (see "Next priority" above) comes
  up, it targets the API and a Python client, not `dashboard.py`.

**Step 2 (a)/(b)/(c): DONE.** Full detail, numbers, and commit hashes in
the "Step 2 — Two-gate calibrated decisions" section below. Do not re-read
this HANDOFF as the source of truth for their status — it's in that section.

**Step 2 (c) review checks (user, 2026-09-30) — IN PROGRESS, before (c) is
treated as final:**
1. Rewrite `results/step2_side_by_side.md`'s claims: at D_gt=0.05,
   calibrated's gate and the ground truth share one threshold, so
   precision/recall=1.000 is nearly true by construction (only
   `trip_duration`, D=0.092, is a true positive, far above the floor) —
   state this shows the materiality gate removes legacy's false positives,
   not "perfect accuracy."
2. A/A decomposition per batch/reference size: Gate-1-only rate
   (`significant`), Gate-2-only/combined rate, raw counts, Clopper-Pearson
   95% CIs. State the combined 0-1% mostly reflects the materiality gate,
   and that A/A batches are iid (don't capture temporal variation).
3. 2x2 ablation {Holm on/off} x {floor on/off} at D_gt=0.05: precision,
   recall, A/A system rate, attributing each improvement to its gate —
   computable offline from already-stored `p_value`/`p_value_adjusted`/
   `effect_size`.
4. NEW RUN: near-floor power curve + reference-draw variability. Normal-
   score-tilted D_pop in {0.02,0.03,0.04,0.05,0.06,0.08,0.10}, every
   continuous feature, >=10 independent reference draws per reference size,
   >=20 disjoint batches per draw at n in {1000,5000,20000}. Report
   detection(material) rate vs. D_pop with 95% CIs and between-draw
   variance. Expect a soft ramp around 0.05, report what actually happens.
5. Redo the data-collapse analysis as a binomial GLM (probit/logit) of
   detection on x=sqrt(nm/(n+m))*D_pop, then add feature and m as
   covariates with likelihood-ratio tests — the earlier within-bin-std
   check conflated curve slope with noise; theory only guarantees
   *approximate* collapse (power depends on the shape of F-G, not just
   sup|F-G|). Use item 4's multi-reference-draw data if possible.
6. Verify `/fit` returns, per KS feature: the floor, `minimum_detectable_d`,
   and a new `recommended_batch_size` (smallest n with
   `c(alpha)*sqrt((n+m)/(nm)) <= floor`) — add if missing (additive field).
   **DONE (2026-09-30)**: `minimum_detectable_d`/`effect_floor` already
   existed; `recommended_batch_size` was missing, added.
   `drift/calibration.py`'s new `recommended_batch_size(m, floor, alpha)`
   solves `c^2*(n+m) <= floor^2*n*m` for the smallest integer n (returns
   `None`, not a misleading number, when `floor <= minimum_detectable_d_at_fit_time(m,alpha)`
   — no finite batch size would ever reach that floor with that reference).
   Verified: boundary crosses exactly (m=5000, floor=0.05 -> n=869; D at
   n=869 is 0.04998 <= 0.05, at n=868 is 0.05001 > 0.05). 6 new tests in
   `tests/test_calibration.py::TestRecommendedBatchSize`. Wired into
   `models.py`'s `FeatureCalibrationInfo` and `main.py`'s `/fit` handler;
   confirmed live via the actual running backend (not just unit tests):
   `/fit` now returns `{"minimum_detectable_d": 0.0192, "effect_floor": 0.05,
   "reference_too_small_for_floor": false, "recommended_batch_size": 869}`
   for a 5,000-row reference. Full suite: 99/99 passing.

Items 1-3 are offline from already-stored results; items 4-5 need new
work/runs, item 6 is done. Commit per logical change; don't push. Stop
after these checks and report the numbers — don't proceed to (d)/(e) or
the programmatic-integration work without a further go-ahead.

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

**Status (2026-09-30): (a), (b), (c) DONE. Default-mode switch for new
projects DONE (see below). (c) is now under a rigorous six-item
methodological review (see HANDOFF) before being treated as final. (d) DCT
calibration and (e) text/image smoke tests NOT started.**

**New-project default switched to "calibrated" — DONE.** Per the user's
2026-09-30 decision (after reviewing (c)'s numbers): `drift/calibration.py`
gained `NEW_PROJECT_DEFAULT_DECISION_MODE = "calibrated"`, deliberately
kept **separate** from `CalibrationConfig`'s own `DEFAULT_DECISION_MODE`
(which stays `"legacy"` forever — that's the fallback every pre-existing
NULL `calibration_config` row resolves to). `main.py`'s `/fit` handler now
checks `crud.get_baseline(project_id) is None` (genuinely new project) and,
only when true AND the caller didn't supply an explicit
`calibration_config`, stores an explicit `{"decision_mode": "calibrated",
...}` config at creation time. A project that already exists (re-fit with
no calibration mention) keeps whatever it has — the existing `"__UNSET__"`
preserve-on-refit behavior from (a) is untouched. 4 new tests in
`tests/test_new_project_default_mode.py`: brand-new project defaults to
calibrated; explicit `legacy` on a new project is still respected;
re-fitting a calibrated-by-default project doesn't revert it; a project
simulating pre-2026-09-30 state (`calibration_config IS NULL`, inserted
directly via `crud.insert_baseline` with no config argument, exactly how
every project before this change was created) stays legacy after a normal
API re-fit with no calibration_config mentioned. Full suite: 93/93 passing.

**(c) Calibrated re-run of the Step 1 suite + legacy/calibrated
side-by-side — DONE.** `scripts/step2_calibrated_rerun.py` re-ran the
sweep + A/A test against two new calibrated-mode projects, reusing
`step1_validation.py`'s EXACT batch-drawing seeds — every comparison is on
byte-identical batches, only the decision logic differs. `scripts/step2_analyze.py`
produced `results/step2_side_by_side.md`/`.json`. Full results in that file;
headlines:
- **Precision/recall at the locked floor (D_gt=0.05)**: legacy's precision
  collapses at large batch sizes (0.333-0.397 at n≥15,000, both reference
  sizes) since it flags every population effect the p-value finds
  significant, however small. Calibrated mode holds **precision=1.000 and
  recall=1.000 (F1=1.000) at every batch size ≥3,000**, both reference
  sizes — the materiality gate removes exactly the false positives legacy
  produces, without losing the one real drift (`trip_duration`).
- **A/A system false-alarm rate**: legacy ranged 0.02-0.27 across sizes
  (already documented in Step 1 as roughly alpha-predicted for uncorrected
  tests). **Calibrated stayed at 0.000-0.010 at every batch size, both
  reference sizes** — Holm correction across 7 features is working as
  designed.
- **Significant-but-not-material, confirmed on pooled data (not an
  anecdote)**: at m=50,000, all four coordinate features are significant in
  **100% of batches but material in 0%**. `trip_duration` (D=0.092, above
  the floor) is significant AND material in 100% of batches at both
  reference sizes. At m=5,000 significance rate is lower (50-83%, matching
  Step 1's own reduced-power finding) but material rate stays 0% regardless.
- **Floor sensitivity** (0.015/0.02/0.03/0.05, calibrated, recomputed from
  stored `effect_size`/`significant` — no new HTTP calls needed): precision
  is worst at the knife-edge floor 0.02 (as Step 1 already flagged), best at
  0.05 (locked default) and 0.015. Full table in the results file.
- **Data-collapse check (x-axis corrected to population D, not observed D,
  per the 2026-09-30 correction)**: computed on legacy's raw significance
  decision (`drift_detected` = plain `p<alpha`, since that's what asymptotic
  KS theory actually makes a claim about — calibrated's two-gate output
  would confound the check with an arbitrary materiality choice). 210
  `(feature, n, m, month)` cells. Clean collapse at the extremes (x<0.5 and
  x>1.7: std=0 in most bins). In the transition region (x≈0.6-1.6), largest
  within-bin spread was std=0.319 vs. a worst-case binomial-noise floor of
  ~0.177 for 8 trials/cell (**1.8x the noise floor — reported as
  suggestive of a real deviation, not just sampling noise, but not
  confirmable without more trials/cell than this run collected** — stated
  with that exact hedge in the results file, not overclaimed either way).

Test artifacts (`step1_val_ref*`, `step2_val_ref*_calibrated`) cleaned up
from `drift.db`. Full suite re-run: 89/89 passing throughout.

**(a) DB schema + migration — DONE** (see commit `cd2e367`): `calibration_config`
column on `baselines`, self-healing migration, `get_baseline`/
`set_calibration_config` in `db/crud.py`. Found and fixed a real risk:
`INSERT OR REPLACE` (used by all three `insert_*_baseline` functions)
deletes and re-inserts the row, which would have silently wiped
`calibration_config` back to NULL on every re-fit — fixed by reading and
carrying forward the existing config unless the caller explicitly passes a
new one. 10 tests in `tests/test_calibration_db.py`.

**(b) Wired into `drift/detector.py` and the tabular `/fit`+`/analyze`
endpoints — DONE.** `DistributionDetector` takes an optional
`calibration_config`; `None` (every pre-Step-2 caller) or
`decision_mode="legacy"` produces the exact pre-Step-2 code path and
response shape (`statistic`/`p_value`/`drift_detected` only) — **verified
byte-identical**, not assumed (`tests/test_detector_calibration.py`).
Calibrated mode: raw KS/PSI statistics computed via the SAME
`_check_continuous_drift`/`_check_categorical_drift` calls as legacy (so
D/PSI values never differ between modes for identical data), categorical
features get a real p-value via `psi_bootstrap_pvalue` (legacy PSI has
none), all features' p-values corrected together via Holm/BH, then
`apply_two_gate` per feature. `models.py`'s `FeatureDriftMetric` gained 7
new Optional fields (`effect_size`, `effect_floor`, `p_value_adjusted`,
`significant`, `material`, `decision_mode`, `threshold_used`) — found and
fixed a real gap here too: the original strict pydantic model would have
silently dropped all of these on serialization if left unextended.
`/fit`'s response gained `calibration_info` (`FeatureCalibrationInfo` per
feature: `minimum_detectable_d`, `effect_floor`, `reference_too_small_for_floor`)
via a new `minimum_detectable_d_at_fit_time(m, alpha)` helper — shown
**regardless of decision_mode**, and a warning is added to `/fit`'s message
when the reference is too small for its configured floor. API response
fields only, per the UI decision — no dashboard changes.
**End-to-end verified against the live backend** (not just unit tests):
fit a calibrated project, analyze the documented small-effect/large-batch
case — `drift_detected: false` (`significant: true`, `material: false`),
vs. `drift_detected: true` for the identical data under a legacy project.
Large-effect case: both gates pass, `drift_detected: true`. 11 new tests in
`tests/test_detector_calibration.py`. Full suite: 89/89 passing.

**Core statistical engine — `drift/calibration.py` (new module, purely
additive, nothing in `drift/detector.py`/`drift/embedding_detector.py`
touched yet):**
- `holm_adjust`/`bh_adjust`: multiple-testing correction, verified against
  hand-computed examples (evenly-spaced p-values `[0.01,0.02,0.03,0.04,0.05]`
  → Holm `[0.05,0.08,0.09,0.09,0.09]`, BH → all `0.05`, both cross-checked
  against known standard-software output) plus monotonicity/bounds
  invariant tests.
- `minimum_detectable_d`/`ks_c_alpha`: the `c(alpha)*sqrt((n+m)/(nm))` floor,
  cross-checked against Step 1's own reported values (m=5000→0.0192,
  m=50000→0.0061, n=20000/m=5000→0.0215).
- `psi_bootstrap_pvalue`/`compute_psi`: parametric bootstrap replacing the
  flat `PSI>0.2` cutoff. **Calibration check passed**: under the null
  (batches genuinely drawn from the reference), false-rejection rate across
  200 trials stayed well under the alpha=0.05 target (test asserts <0.15
  for stochastic-test margin — actual measured rate: 9/200 = 0.0450, seed 123).
  Also verified: same frequency gap is more significant (smaller p) at a
  larger batch size, which is the whole point of replacing a flat threshold.
- `apply_two_gate`/`GateResult`: Gate 1 (significance) + Gate 2
  (materiality) combination. **Legacy-mode identity verified**: in legacy
  mode, `drift_detected` equals Gate 1 alone regardless of effect size —
  exactly today's single-threshold behavior. Calibrated mode requires both
  gates — explicitly tested against the documented Citi Bike case
  (significant at large n, D~0.02 below the 0.05 default floor → not
  material → not flagged).
- `CalibrationConfig`: per-project config dataclass with per-feature floor
  overrides; `from_dict(None)` (no stored config, e.g. an existing project)
  resolves to the legacy default with no migration needed.
- Tests: `tests/test_calibration.py`, 29 tests, all passing. Full existing
  suite re-run alongside it: 68/68 passing, zero regressions.

**Explicitly still remaining**: see the ordered list (a)-(e) in the
**HANDOFF** section at the top of this file — that list supersedes this one
(kept only so older links into this section don't 404; don't duplicate
edits here going forward, edit the HANDOFF section instead). Key corrections
made there since this list was first written: no dashboard UI work (React
rewrite planned after Step 3 — API response fields only), legacy mode must
keep its exact current flat-threshold code path rather than being routed
through the new bootstrap/gate machinery at the same nominal cutoff, and the
calibrated Step 1 re-run (c) must happen *before* any DCT work (d), not
concurrently or after.

**Severity scale fix (prerequisite, done before any Step 2 code, so legacy
and calibrated share identical synthetic data):** user's hypothesis
confirmed empirically — raw-z exponential tilting let `trip_duration`'s
heavy right skew (mean=880s, median=550s, max=234,243s) dominate the
importance weights; at target severity=3σ the effective sample size
collapsed to 1,136/25,000 rows (4.5%), so the mean shifted 14.6x but the
population D only reached 0.063 (`results/severity_scale_investigation.json`).
Fixed with normal-score tilting (`Phi^-1(rank/(n+1))` instead of raw z) —
forces the tilting variable to be approximately standard normal regardless
of feature shape. Re-ran the full severity + minimum-drift-fraction sweep
(`scripts/step1_severity_v3.py` → `results/tabular_validation_severity_v3.json`):
smallest reliably-detected population D is now consistent across all five
continuous features (0.020–0.040 at m=50,000, vs. theoretical floor 0.0061)
instead of `trip_duration` being a 15-30x outlier. Severity now reported
primarily as achieved population D (exact weighted-KS, no resampling
noise), normal-score σ kept as a secondary column. This is now the
canonical synthetic-injection method going forward.

- `docs/step2_proposal.md`: full config schema, Gate 1 mechanics per
  detector (KS p-value + Holm/BH; PSI parametric bootstrap at actual batch
  size; DCT precomputed vs. permutation), Gate 2 materiality mechanics, and
  legacy-mode exact-reproduction requirement.
- `scripts/step2_dct_calibration_benchmark.py` /
  `results/dct_calibration_benchmark.json`: real timing benchmark (synthetic
  384-dim embeddings, B=100 draws) at 40/200/1,000/5,000 per side.
  **Per-draw cost is similar between strategies; the difference is when
  it's paid** — precomputed pays once at `/fit`, permutation pays on
  *every* `/analyze` call (23s/call at n=5,000 per side — not viable
  interactively at that size). Precomputed's tradeoff, stated plainly: it
  calibrates against a half-size pseudo-reference, a real bias against the
  latency win.
- **Four decision points RESOLVED 2026-09-29** (user decisions, full detail
  in the user's own message and reflected in the implementation below):
  1. **Alpha = 0.05**, applied family-wise per batch via Holm across
     features (default); BH kept as an option.
  2. **Effect floors**: KS D default = **0.05** — a round, conservative
     anti-alert-fatigue default chosen deliberately independent of the Citi
     Bike knife-edge cluster (not tuned to the evaluation data, per ground
     rule 6). Floors are per-project, overridable per-feature. PSI = 0.2
     (industry convention). DCT AUC = 0.65, labeled provisional/unvalidated
     until Step 8's text/image validation runs. At `/fit`, each feature's
     floor is shown next to the minimum detectable D for the current
     reference size, as part of human-in-the-loop schema confirmation.
  3. **Default decision mode = legacy**, implemented now; existing projects
     stay legacy permanently. After the calibrated evaluation, present a
     side-by-side (A/A system false-alarm rate per batch size; precision/
     recall at matched thresholds) — switching the NEW-project default to
     calibrated needs separate approval contingent on that comparison.
  4. **DCT calibration default = precomputed**; permutation kept as opt-in.
     Document the half-size-pseudo-reference bias explicitly (conservative:
     fewer false alarms, slightly less power). For batch sizes outside the
     calibrated grid: clamp to the nearest grid point and return a warning,
     never extrapolate silently.

**Scope additions from Step 1's findings, recorded, not implemented yet:**
- Every KS result should return the minimum detectable D for its actual
  `(n, m)` at the configured alpha — `c(alpha) * sqrt((n+m)/(n*m))` — so
  users can see what the test cannot detect, not just what it did detect.
- `/fit` should warn when the reference is small enough that this detection
  floor exceeds the project's configured KS effect floor.
- Propose (don't set) a recommended minimum reference size, grounded in
  Step 1's actual results — a concrete number, not a guess, and explicitly
  a proposal for the user to approve, not a new default.

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
