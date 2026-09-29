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

**Status: IN PROGRESS.**

Prerequisites done: forensic check, pre-flight verification (above), and
`scripts/split_citi_bike.py` run at both reference sizes (5000, 50000).
Proceeding now with the main Step 1 analysis: disjoint per-month batch
draws, raw TP/FP/FN/TN with an explicit decision unit, per-batch-type
breakdown, sweep with precision/raw-counts/15k point at both reference
sizes, A/A test (labeled as an iid test-calibration check, not a test of
month-to-month baseline variation — see below), extended severity levels,
effect-size ground truth variants, per-feature D mean/std, and the
"minimum drift fraction" rename. All Apr-Jun-scoped, all per-month AND
pooled.

**A/A test framing (per instruction):** labeled throughout as an **iid
test-calibration check** — both the holdout and reference samples are drawn
at random from the same Jan–Mar pool, so this measures whether the false-
alarm rate matches what alpha predicts under iid sampling. It does **not**
capture month-to-month variation within the baseline period (e.g. a
February-vs-March shift), since both come from the same pooled draw.

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
