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
Apr–Dec; data is Apr–Jun."* A full Jan–Dec rebuild from raw source data with
one documented preprocessing script is explicitly **deferred, not part of
this pass** — extending with Jul–Dec data prepared differently from the
existing CSVs would inject artificial drift.

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
- Commits: (pending — see below).

**Headline findings (full detail and every table in
`results/tabular_validation_notes.md`):**

1. **Reference size, not batch size, explains the recall plateau —
   confirmed empirically, exactly as hypothesized.** The theoretical KS
   critical-value floor `c(0.05)/sqrt(m)` is ≈0.0192 at m=5,000 and ≈0.0061
   at m=50,000. `pickup_latitude`'s population D (≈0.021–0.025) sits right
   at the m=5,000 floor. At m=5,000, its recall plateaus around 0.79–0.83
   even at the largest batch sizes tested (20k, 50k) — it never reaches 1.0.
   At m=50,000, recall reaches 1.000 by batch size 10,000 and stays there.
   No amount of extra production data fixes a reference that's too small to
   resolve an effect that close to its own noise floor.
2. **New caveat this run surfaced (not asked for, found while computing the
   above): the "population ground truth" itself is not reference-size-
   invariant.** `dropoff_longitude` is labeled stable under the m=5,000
   reference (p=0.079) but drifted under the m=50,000 reference
   (p=1.5×10⁻²⁶) — same Apr-Jun production data, different reference
   samples. A small reference's own sampling noise leaks into what looks
   like an "independent" ground truth. Practical takeaway: for borderline-
   effect features, a ground truth computed from a small reference isn't a
   stable target to grade the engine against — the m=50,000 ground truth
   should be trusted over the m=5,000 one, not just the engine's detections.
3. A/A system false-alarm rates ranged 0.02–0.27 across sweep sizes and
   reference sizes, generally at or below the alpha-predicted ≈0.30 for 7
   uncorrected tests (`1-(1-0.05)^7`) — roughly consistent with prediction,
   somewhat lower, plausibly because the 7 features aren't fully independent
   (correlated coordinates). No correction is applied yet (that's Step 2).
4. Effect-size ground truth variants (D≥0.01/0.02/0.05) move the confusion
   matrix substantially — e.g. at m=50,000, D≥0.05 drops recall from 1.000
   to 1.000 but explodes false positives (precision 1.000→0.333), since most
   of Apr-Jun's real population differences are small-effect, not large.
5. `month`'s per-batch PSI has std=0.0000 at every batch — expected, not a
   bug: every row in a given month's production file has that exact month
   value, so every batch's PSI for `month` compares against an identical
   100%-one-category distribution.

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
