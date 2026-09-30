# Progress Log

Resume-from-here document for the multi-step drift-monitoring calibration/
hardening plan. Update this after every logical change, not just at step
boundaries.

---

## HANDOFF — read this first if starting a fresh session (2026-09-30)

**Hardening pass (user, 2026-09-30) — required before Step 5, IN
PROGRESS.** The 2026-09-30 precondition check for Step 5 found this pass
had NOT actually been done (no matching commits, no cross-user isolation
tests, no clean-venv run, no client contract changes) despite being
described as a precondition -- confirmed by grepping the repo and git log
before starting anything. Scope, in priority order:
1. **DONE.** Security — cross-user isolation. Every project-scoped
   endpoint had NO ownership check at all -- any authenticated user
   (session JWT or unrestricted PAT) could read/overwrite/analyze/delete
   ANY other user's project just by knowing the project_id. Fixed:
   `verify_project_access`/`verify_model_access` (main.py) now check
   project ownership for both auth types; a project owned by someone else
   is 404, not 403 (a stranger can't even confirm it exists). Scope
   (what a PAT's own account can reach) and ownership (which account can
   reach it at all) are independent gates, verified explicitly. 36 tests,
   `tests/test_cross_user_isolation.py`. Also confirmed: PAT default
   expiry is never (unless set), revoked tokens rejected, last_used_at
   tracked, zero leaked `dm_` tokens anywhere in the repo. Commit
   `0cc4967`.
2. **PENDING APPROVAL — not started.** Reproducible environment.
   Installed versions (pandas 3.0.3, scikit-learn 1.9.0) don't match
   `requirements.txt` (2.2.2 / 1.5.2) -- environment drift already
   implicated in two client bugs (Step 3d/3e). Proposed: clean venv from
   `requirements.txt` on Python 3.12 (confirmed installed on this
   machine, alongside 3.10/3.14), full suite run, fix failures, OR
   propose new pins if moving to the newer already-installed versions is
   preferable. Waiting on user go-ahead before creating the venv.
3. **DONE.** Client/server contract. `FitBaselineRequest` gained an
   optional `feature_types={col: "continuous"|"categorical"}` override,
   server-honored (422 for an unknown column or invalid value). More
   importantly, fixed the actual root cause the client was working around:
   `/fit`'s column retention now looks up values from BOTH reference_data
   and categorical_data merged, not from whichever dict the caller
   happened to submit a column under -- no more silent drop regardless of
   feature_types. A second, more severe pre-existing bug found while
   testing the first fix: a fit with `reference_data={}` (all-categorical,
   no continuous columns) silently lost EVERY row, not just a column
   (`min(0, N)=0` truncated the present side to zero) -- fixed. Client's
   `_guess_categorical_columns` heuristic removed entirely (no longer
   needed); `fit()` gained `feature_types=` passthrough. 6 new server
   tests (`tests/test_fit_feature_types.py`) + 3 new client tests.
   Commit `738bdca`.
4. **DONE.** `recommended_batch_size` redefinition: now targets floor/2,
   not floor (kept as `min_batch_size_at_floor`, unchanged). Verified by
   simulation (`scripts/step2_hardening_batch_size_simulation.py`,
   `results/step2_hardening_batch_size_simulation.md`): Uniform(0,1) vs
   Uniform(D,1+D) gives an exact closed-form population D. Real finding
   (corrected from an initial wrong guess about "50% power at the floor,"
   caught before it shipped): the critical value is an alpha-level
   null-rejection threshold, not a power-at-floor point -- at the OLD n
   (m=5,000: 869), a batch with NO true drift already has ~alpha (5%)
   chance of exceeding the floor from noise alone (confirmed 5.45%/4.65%
   for m=5,000/50,000); the NEW n drives this to 0% in both cases. Power
   AT true D=floor was already ~99% at the OLD n -- not the bottleneck.
   Real benefit: much lower false-material rate at/below the floor, for
   substantially more required data (m=5,000: 869->7,252; m=50,000:
   751->3,146). Commit `ab5f94c`.
5. **DONE.** Fixed the model-serving example. The Apr-Jun replay's
   monitored coordinate features sit at population D 0.019-0.022, BELOW
   the 0.05 floor (verified from stored population truth, not assumed) --
   the "real drift" wording for two flagged windows was wrong and is
   corrected; framed as floor-adjacent noise instead. Rebuilt on the
   50,000-row reference at the new `recommended_batch_size` (3,146). Added
   a labeled synthetic scenario (`replay_synthetic.py`, D=0.10 on
   `pickup_longitude` via the same verified tilting method as
   `step2_item4_power_curve.py`) -- 4/4 windows alerted. README tables now
   generated from `reports/real|synthetic/*.json` by script
   (`generate_readme_tables.py`), never typed by hand. Commit `30e801b`.

Tests + isolation checks required on every new/changed endpoint. Commit
per logical change; don't push. **Do not start Step 5 until this is
reported and the user says go.**

**Project goal, for context on every decision below (recorded 2026-09-30):**
this API exists so ML teams can integrate it into their model-serving
pipelines to monitor **data drift of model inputs only** — no labels, no
accuracy/performance tracking, no ground-truth-outcome monitoring of any
kind. Every design choice (batch-based, not streaming; statistical
significance + materiality, not accuracy metrics; per-feature not
per-prediction) should be read against that scope.

**Project goal (reaffirmed 2026-09-30, second time): ML teams integrate
this API into their model pipelines to monitor DATA drift of model
inputs — input distribution shift only, no labels or accuracy tracking.**

**Programmatic integration is now the critical path (2026-09-30): today
no script can call the API without the dashboard's browser login.** This
supersedes/absorbs what was previously sketched as "Step 3" in
`docs/step2_proposal.md`'s references — that sketch is directionally
right but the scope below (from the user directly) is authoritative.

**New step order (user, 2026-09-30), from here:**
1. **Step 3 — programmatic access** (PATs, file-parsing refactor +
   multipart upload endpoints, `DELETE /projects/{project_id}`, Python
   client, end-to-end model-serving example). Full scope below.
   **DONE (2026-09-30)** — see the detailed writeup after the scope list.
2. **Step 5 — history, k-of-m sustained alerts, baseline versioning**,
   with **webhooks now newly in scope**: HMAC-SHA256 signed payloads,
   retries with backoff, delivered in the background (never blocking the
   request that triggered them).
3. **Step 2 (d)/(e)** — DCT calibration + text/image validation, run
   under calibrated mode.
4. **React + TypeScript frontend** (replaces Streamlit — build no new
   Streamlit UI in the meantime, per the standing UI decision below),
   **then** deployment to the Azure VM.

**Backlog (record only — do not start):**
- Per-input OOD scoring: kNN / Mahalanobis distance with conformal
  calibration, `POST /score/{project_id}`.
- An "LLM prompt drift" scenario for Step 8's text validation.
- (added 2026-09-30, evidence-fix caveat #4 below) `/fit` should warn
  when the configured KS floor is below the DKW bound
  `sqrt(ln(2/0.05)/(2m))` for the reference size, and report the minimum
  reference size that would support the configured floor.

**Ground rules, reaffirmed unchanged**: no fabricated numbers (every
figure from an actual run), no regressions, additive API changes only,
**ask before**: changing defaults, deleting files, touching Docker or
auth config, or adding a heavy new dependency. Commit per logical
change; don't push.

**Step 3 scope (user, 2026-09-30):**
- **(a) Personal access tokens.** `api_tokens` table: id, user_id, name,
  prefix, SHA-256 hash of the full token, project scope, created/expires/
  last_used/revoked. Token format `dm_<prefix>_<secrets.token_urlsafe(32)>`
  — store only the hash, look up by prefix, compare with
  `hmac.compare_digest`. Backend auth accepts EITHER the existing session
  JWT or a PAT; project scope enforced on every project endpoint; tokens
  never logged. No UI — `scripts/create_token.py` (create/list/revoke, by
  user email) is the way to mint tokens until the React frontend exists.
  Tests: valid, expired, revoked, wrong scope, no plaintext persisted
  anywhere.
- **(b) File parsing.** Move dashboard.py's file-parsing logic into
  `ingest/readers.py` with IDENTICAL behavior (dashboard imports it, not
  duplicates it). Add multipart upload endpoints for tabular fit/analyze
  (Parquet + CSV), with a size limit. Fixture tests per format asserting
  identical parsed frames.
- **(c)** Add `DELETE /projects/{project_id}`; keep
  `DELETE /models/{model_id}` as a deprecated alias (same behavior, not
  removed).
- **(d) Python client** at `clients/python/` (package
  `drift_monitor_client`): `DriftClient(base_url, token)` with
  `fit(project_id, df)` / `analyze(project_id, df)`; auth header,
  timeouts, retries with backoff, Parquet upload for large frames. Tests
  against a locally running server.
- **(e) End-to-end example** at `examples/model_serving/`: train a small
  scikit-learn model on the Jan-Mar Citi Bike split (monitored features =
  the model's own input features only); serve it with FastAPI, logging
  post-preprocessing input features asynchronously (never blocking the
  prediction path); a scheduled-job script that reads each window's
  logged features and calls `analyze()` via the client; replay the
  Apr-Jun production data through the server in fixed windows; save
  per-window drift reports. Use `decision_mode="calibrated"` and a window
  size at or above `/fit`'s `recommended_batch_size`.

**Stop after (e) and show**: the auth design, the client API, and the
example's per-window drift output (actual numbers, not illustrative
ones).

**Step 3 completion writeup (2026-09-30) — all five sub-parts DONE, full
suite 146/146 passing throughout, nothing pushed:**
- **(a)** `auth/tokens.py` (token gen/hash/verify) + `db/crud.py`'s new
  `api_tokens` table + `main.py`'s `verify_access` (session JWT or PAT,
  auto-detected) / `verify_project_access` / `verify_model_access`
  (PAT project-scope enforcement) + `scripts/create_token.py`. Real bug
  caught before shipping: the prefix was originally generated with
  `secrets.token_urlsafe`, whose alphabet includes `_` — a prefix
  containing `_` would break the `split("_", 2)` parsing used to look it
  up, silently failing auth for a fraction of minted tokens. Fixed by
  generating the prefix with `token_hex` instead (0-9a-f only, can never
  collide with the delimiter). 13 tests, `tests/test_api_tokens.py`.
- **(b)** `ingest/readers.py` (dashboard.py's file-parsing moved out
  verbatim, Streamlit dependency removed); two new endpoints,
  `POST /fit/{project_id}/upload` and `POST /analyze/{project_id}/upload`
  (CSV/Parquet, 200MB cap); both existing JSON endpoints refactored to
  share their tail logic with the new upload endpoints (`_resolve_and_
  persist_fit`, `_run_tabular_analysis`) rather than duplicating it.
  15 + 9 tests (`test_ingest_readers.py`, `test_upload_endpoints.py`).
- **(c)** `DELETE /projects/{project_id}`; `DELETE /models/{model_id}`
  kept as a `deprecated=True` alias, identical behavior. 2 tests
  (`test_delete_project.py`).
- **(d)** `clients/python/` (`drift_monitor_client`): `DriftClient(base_url,
  token).fit()/.analyze()`, urllib3 retry-with-backoff, automatic Parquet
  upload above `large_frame_row_threshold` (default 10,000 rows). Real
  bug caught while building (e): the JSON `fit()` path put every column
  under `reference_data` only, so any column the server profiled as
  categorical (e.g. `gender_id`) was silently dropped by a pre-existing
  server-side quirk (main.py only retains a column under the dict it
  arrived in — not touched, see (b)'s note below). Fixed by replicating
  `utils/profiler.py`'s own classification threshold client-side and
  pre-splitting columns before sending — NOT by submitting every column
  under both dicts, which seemed simpler but crashes with a 500 (the
  server `pd.concat()`s both dicts into one frame; a column in both
  produces a duplicate column name). A SECOND bug surfaced fixing the
  first: the heuristic's `series.dtype == object` check silently misses
  every string column on pandas 3.0.3 (the actually-installed version;
  requirements.txt still pins 2.2.2 — environment drift), which gives
  string columns their own `str` dtype. Fixed with
  `pd.api.types.is_string_dtype()`. 8 tests against a locally running
  server, `clients/python/tests/test_client.py`.
- **(e)** `examples/model_serving/`: full pipeline actually run, not just
  written. 5,000-row reference (`tests/splits/reference_n5000.csv`) ->
  `/fit` (`decision_mode="calibrated"`) -> `recommended_batch_size=869`
  for every continuous feature -> 9,000 replayed predictions (3,000 each
  from Apr/May/Jun 2016) through a real served model -> 10 complete
  869-row windows analyzed -> 2 flagged real drift (window 4:
  `dropoff_longitude`, D=0.0670; window 6: `pickup_latitude`, D=0.0659;
  both Holm+floor gates clear). All 10 window reports committed as real
  evidence (`examples/model_serving/reports/window_*.json`); see that
  directory's README.md for the full table and how to re-run.
- **Known gap, surfaced but deliberately NOT touched this pass**: neither
  session-JWT auth nor the existing JSON `/fit` endpoint's
  reference_data-vs-categorical_data column retention has any
  cross-user ownership check or silent-drop fix applied — both are
  pre-existing behavior, out of scope for "additive changes only," and
  are noted here rather than fixed quietly. The PAT system enforces
  project scope properly (new code, no existing behavior to preserve);
  the Python client works around the column-retention quirk client-side.

**Evidence-fix caveats (user, 2026-09-30 — record only, do NOT rerun
anything now; revisit if/when this matters for a real decision):**
1. **A/A multi-draw design caveat**: clean batches were drawn from the
   fixed 25,000-row holdout pool, so at n=20,000 the "independent"
   batches overlap heavily and all reference draws share the pool's own
   sampling error. The per-draw Gate-1 rates of 0.000-1.000 (mean 0.575)
   in `results/step2_aa_multidraw_report.md` should NOT be quoted as a
   property of Holm's correction itself. A proper follow-up needs two
   variants: a marginal A/A drawing BOTH reference and batch fresh from
   the full baseline pool each trial, and a fixed-reference conditional
   version (closer to what a real deployed project actually experiences).
2. **GLM cluster caveat**: only 20 clusters (10 draws x 2 reference
   sizes) — read `results/step2_item5_glm.md`'s cluster-robust results as
   descriptive, not a rigorously powered inference. The feature gaps at
   matched x (up to 33 points) mean any batch-size guidance derived from
   this GLM is approximate, not exact.
3. **Materiality floor applies to the OBSERVED D, not the true D.**
   Effective true-D thresholds (where 50% detection power is actually
   reached) are about 0.035-0.049 depending on n (see
   `results/step2_item4_power_curve_report.md`'s D50 table) — noticeably
   below the nominal 0.05 floor at small n, converging toward it as n
   grows. **This needs to be documented in the API docs** (not done yet)
   so a user configuring a 0.05 floor understands the floor is on the
   noisy observed statistic, not a guarantee about the true population D.
4. Backlog item added above: `/fit` DKW-bound floor warning +
   minimum-reference-size report.

**Where things stand (2026-09-30)**: Step 0, Step 1, Step 2 (a)/(b)/(c)
(plus its six-item review and the evidence fixes above), and **Step 3
(all five sub-parts)** are DONE — see their sections for full detail.
**Next up per the new step order: Step 5** (history, k-of-m alerts,
baseline versioning, webhooks). **Step 2 (d) DCT calibration and (e)
text/image smoke tests are NOT started** — ordered after Step 5, not
next.

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
1. **DONE (2026-09-30)**: Rewrote `results/step2_side_by_side.md`'s
   precision/recall section. States plainly that at D_gt=0.05, calibrated's
   Gate 2 floor and the ground-truth threshold are the SAME number, so only
   `trip_duration` (D=0.092) is a true positive and precision=recall=1.000
   is close to true by construction — it demonstrates the materiality gate
   removes legacy's false positives on the four sub-floor coordinate
   features, not general accuracy. Points to the floor-sensitivity table
   (unchanged, already in the doc) as the real generalization test.
2. **DONE (2026-09-30)**: Added `aa_gate_decomposition()` to
   `scripts/step2_analyze.py` — splits the calibrated A/A system rate into
   Gate-1-only (`significant`, any feature), Gate-2-only (`material`, any
   feature), and combined (`drift_detected`), each with raw k/n and
   Clopper-Pearson 95% CIs, per (reference_size, batch_size). Findings:
   Gate-1-only rates track close to legacy's own (uncorrected) rates,
   modulo Holm pulling them down somewhat; the combined rate collapses to
   0-1% almost entirely because of Gate 2, since a null batch's true effect
   is exactly zero and essentially never independently clears the floor at
   the same moment a p-value is significant. States plainly that A/A
   batches are iid draws from the reference pool and don't capture
   month-to-month temporal variation within the baseline period.
3. **DONE (2026-09-30)**: Added the 2x2 ablation {Holm on/off} x
   {floor on/off} at D_gt=0.05 to `scripts/step2_analyze.py`, recomputed
   entirely offline from calibrated_raw's stored `p_value` (raw),
   `significant` (Holm-adjusted), `effect_size`/`effect_floor` — no new
   HTTP calls. Finding: **the floor is the dominant lever for
   matched-threshold precision** — (Holm=off,Floor=on) already reaches
   1.000/1.000 identically to (Holm=on,Floor=on) at the largest sweep size,
   since sub-floor coordinate features are significant whether or not
   Holm-adjusted at that sample size. **Holm is the dominant lever for
   system-level false-alarm control under the null (A/A)** — with the
   floor off, turning Holm on cuts the A/A rate substantially at every
   batch size (e.g. ref=5000/batch=10000: 0.880 -> 0.430); adding the
   floor on top drives it to ~0. The two gates fix different failure
   modes; both are needed for a system good on both axes. Recall is 1.000
   in all four cells (`trip_duration`'s D=0.092 is far too large for
   either gate to cost recall). All three items in commit `04e3ffe`; full
   suite 99/99 passing.
4. **DONE (2026-09-30)**: NEW RUN — near-floor power curve + reference-draw
   variability. `scripts/step2_item4_power_curve.py`: 42,000 trials — 2
   reference sizes (5000, 50000) x 10 independent, mutually disjoint
   reference draws per size x 5 continuous features x 7 normal-score-tilted
   D_pop targets (achieved D verified via exact weighted-KS, no resampling
   noise) x 3 batch sizes (1000/5000/20000) x 20 independent trials each.
   Report + between-draw variance in `scripts/step2_item4_report.py` ->
   `results/step2_item4_power_curve_report.md`. Finding: 26/30
   (ref_size, feature, batch_size) power curves show a genuine soft ramp
   (>=2 D_pop points with 0.05<rate<0.95); the curve compresses toward a
   step at n=20000 because the transition band narrows relative to the
   fixed D_pop grid tested, not because detection is actually
   discontinuous. Between-draw variance is EXACTLY zero for 41.4% of
   cells (the extremes — every draw agrees when D_pop is far from the
   floor) and concentrates entirely in the transition region (top cell:
   ref=5000, pickup_latitude, n=20000, D=0.04 — mean rate 0.465, std=0.466
   across the 10 draws). Raw data (42,000 records) in
   `results/tabular_validation_item4_power_curve_raw.json`.
   Implementation notes: `mint_session_token`'s fixed 1-hour expiry is too
   short for this multi-hour run — fixed locally in the new script (401 ->
   remint -> retry) rather than touching the shared
   `tests/_session_auth.py` helper. Concurrent requests destabilized the
   dev uvicorn server on Windows (connection resets) — script runs
   sequentially with retries. Checkpointed to a JSONL file (not committed,
   redundant with the aggregated raw JSON) so the run survived one restart
   (a host memory-pressure kill mid-run) with zero data loss.
5. **DONE (2026-09-30, REVISED 2026-09-30 — see "Evidence fixes" below for
   the superseding version)**: Redo the data-collapse analysis as a
   binomial GLM. `scripts/step2_item5_glm.py`, using item 4's data. The
   version in commit `c3e63df` used `material` as the outcome — this was
   methodologically wrong (materiality is a fixed-floor threshold on the
   observed effect size, not a function of x, so collapse theory makes no
   prediction about it) and was corrected the same day; see "Evidence
   fixes" for the current (Gate-1-only, cluster-robust) version. Do not
   cite the `material`-based coefficients from this entry.
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

**All six items DONE (2026-09-30).**

**Evidence fixes (user, 2026-09-30, same day as the six-item review) — DONE:**
1. **A/A over multiple reference draws.** `scripts/step2_aa_multidraw.py`:
   the original A/A test used ONE reference draw per size, so its
   false-alarm rate was conditional on that one draw's own sampling
   error, not an unconditional estimate. Reuses item 4's 10 disjoint
   reference draws per size (already fit, not refit) x >=20 independent
   clean batches per draw x n in {1000, 5000, 20000} = 1,200 trials.
   Reports Gate-1-only/Gate-2-only/combined per draw and averaged, with
   Clopper-Pearson CIs and between-draw spread
   (`results/step2_aa_multidraw_report.md`). Finding: at ref=5000,
   n=20000, Gate-1-only ranges from 0.000 to 1.000 ACROSS THE 10 DRAWS
   (mean 0.575, std 0.419) — confirms the original single-draw estimate
   could have landed anywhere in that range. The combined (two-gate) rate
   stays <=0.05 in every single draw/cell tested, at both reference
   sizes — the materiality gate is what keeps the system stable across
   reference draws, not Gate 1 alone. Explains the original single-draw
   Holm=on/Floor=off figure (0.430 at ref=5000, n=10000): the elevated,
   n-growing false-alarm rate reflects the reference's own finite-sample
   deviation from the true population (typical size `c(alpha)/sqrt(m)`),
   which every trial against that one reference inherits — larger n gives
   the KS test more power to detect even this small, fixed,
   reference-specific artifact. **Design implication, stated explicitly**:
   the materiality floor must be at least around `c(alpha)/sqrt(m)`
   (0.0192 at m=5,000; 0.0061 at m=50,000 — exactly the
   `minimum_detectable_d_at_fit_time` quantity from item 6) for a stable
   false-alarm rate; the locked default floor (0.05) sits comfortably
   above both.
2. **GLM outcome and inference, corrected.** `scripts/step2_item5_glm.py`
   rewritten: outcome is now `significant` (Gate 1) ONLY — materiality
   depends on observed effect size vs. a FIXED floor, not on x, so
   collapse theory makes no prediction about it and the prior version's
   use of `material` as the primary target was a category error (superseded
   entry above). Refit with Huber-White cluster-robust SEs, clustered by
   the physical reference draw (20 clusters — flagged as below the usual
   30-50+ comfort zone, so treated as indicative). Headline results are
   predicted-probability GAPS, not p-values: max gap between features at
   matched x = 0.331 (33 points, at x=1.203, `pickup_longitude` vs.
   `dropoff_latitude`) — real and survives cluster-robust correction
   (all 4 feature-dummy p-values < 0.05 even clustered). Max gap between
   reference sizes at matched x = 0.068 (7 points, at x=1.032) — this is
   what the raw M2 coefficient (-0.274 for m=50000 vs. m=5000; NOTE: not
   the -0.849 from the superseded `material`-based fit) means in practice.
   Naive p-value on this coefficient was 1.07e-11; cluster-robust p-value
   is 0.124 — NOT significant once the 20-cluster structure is accounted
   for. Reported honestly as an upper bound on a plausible-but-unproven
   effect, not a confirmed one.
3. **Item 4 additions.** `scripts/step2_item4_report.py` extended: all 4
   non-soft-ramp curves are at ref_size=50000, n=20000 (the largest
   reference x largest batch combination) — `dropoff_latitude`,
   `pickup_latitude`, `pickup_longitude`, `trip_duration`; explained as a
   grid-resolution artifact (the transition band narrows at large n,
   narrow enough that the fixed 7-point D_pop grid sometimes straddles it
   between two adjacent tested points), not a real discontinuity — item
   5's GLM fits one smooth sigmoid across all n with no special-case
   discontinuity term needed. D50 (50%-detection D_pop, linearly
   interpolated) reported for all 30 curves relative to the 0.05 floor:
   D50 sits BELOW the floor for 29/30 curves (0.55x-0.98x; one exception
   at 1.02x) — reflects the two-sample KS statistic's known finite-sample
   upward bias (more pronounced at small n), shrinking toward the nominal
   floor as n grows (n=1000 average 0.70x -> n=20000 average 0.89x).
4. **Default decision_mode confirmed** (was already done earlier the same
   day, re-verified here per this instruction): new projects with no
   explicit `calibration_config` resolve to `"calibrated"`; existing
   projects stay `"legacy"` permanently. Test:
   `tests/test_new_project_default_mode.py` — 4 tests, all passing
   (`TestNewProjectDefaultsToCalibrated::test_brand_new_project_no_config_gets_calibrated`,
   `::test_brand_new_project_explicit_legacy_respected`,
   `::test_refit_of_new_default_project_stays_calibrated`,
   `TestExistingProjectStaysLegacy::test_project_with_null_config_refit_without_mention_stays_legacy`).

Commit per logical change; don't push. Stop after these checks and
report the numbers — don't proceed to (d)/(e) or the
programmatic-integration work without a further go-ahead.

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
