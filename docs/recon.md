# Step 0 — Recon

No code changes in this step except one pre-approved, separable fix (item 6, the
`_check_categorical_drift` debug-print replacement — committed as `bd76e48`,
unrelated to the questions below but done now per instruction). Everything else
here is read-only investigation. All numbers are either read directly from
source or produced by a script/query run in this session; none are estimated.

---

## 1. Tabular baseline storage — does KS have a real reference sample?

**Yes.** The README's "Baseline Storage" bullet (`README.md:162`, *"Tabular: IQR
fences (Q1, Q3) for continuous features, frequency tables for categorical
features"*) is **incomplete, not wrong** — it omits the thing the KS/PSI batch
test actually runs against.

- `db/crud.py`'s `insert_baseline()` stores the **full cleaned reference array**
  (not just Q1/Q3) as JSON in the `baselines.reference_data` column
  (`db/crud.py:144-216`), merging continuous + categorical columns into one
  dict. Continuous columns are coerced/cleaned first (`db/crud.py:172-178`,
  using the same `coerce_numeric_column` threshold rule the profiler uses).
- Separately, `_calculate_boundaries()` computes the IQR fences (`q1`/`q3`) and
  categorical `allowed_values` from that same cleaned data, stored in
  `baselines.iqr_fences` — this is what `/predict` (real-time IQR scoring)
  reads, via `compute_iqr_anomalies()` (`drift/detector.py:5-46`).
- At `/analyze` time (`main.py:306-328`), the handler calls
  `crud.get_baseline(project_id)`, which returns `state["reference_data"]` —
  the full raw array — and feeds it directly into
  `DistributionDetector.fit_baseline()` (`drift/detector.py:60-65`), which just
  does `self.reference_data[feature_name] = np.array(data)`. **No summary
  statistics are used for KS/PSI** — the whole point is a real two-sample
  comparison against real reference rows.

**Size and how it's drawn:** there is no explicit cap or sampling logic for the
tabular reference — it's exactly however many rows were passed to
`/fit/{project_id}`'s `reference_data`/`categorical_data`. This is different
from the embedding path, which explicitly caps stored reference embeddings at
`max_reference_samples=3000` (`db/crud.py:244-255`, `insert_embedding_baseline`).

**Measured for the actual local Citi Bike baseline** (`project_id=citi_bike_v1`,
still present in `drift.db` from earlier validation runs):

| Metric | Value |
|---|---|
| Stored `reference_data` size (SQLite column, `length()`) | 309,418 bytes (0.31 MB) |
| Continuous feature row count (`pickup_longitude` etc.) | 5,000 rows each |
| Categorical feature row count (`gender_id`, `month`) | 2,000 rows each |
| SQLite read time (5 runs, warm) | 0.38–0.96 ms |
| `json.loads` time | 2.68–3.52 ms |
| `DistributionDetector.fit_baseline()` time | 0.74–1.16 ms |
| **Total per `/analyze` call (excluding module import)** | **4.20–5.25 ms** |

(Measured via a standalone script reading directly from `drift.db` and calling
the real `fit_baseline()`; import overhead of `drift.detector` excluded since
that only happens once at process startup, not per request.)

**Implication for Step 2:** at this size, reference loading is not a
bottleneck (~5ms). Per your instruction, no cap is being proposed yet — this
is just the baseline measurement Step 2 will reference when it proposes one
with a quantified KS-power trade-off.

**README fix needed (deferred to Step 9):** the "Baseline Storage" bullet
should say the full cleaned reference sample is stored (uncapped), not just
"fences and frequency tables."

---

## 2. Decision rules, thresholds, and multiple-testing correction

| Detector | Rule | Threshold | Where |
|---|---|---|---|
| KS (continuous, tabular) | `p_value < p_value_threshold` | `0.05` (constructor default, `DistributionDetector.__init__`, `drift/detector.py:53-58`; also hardcoded at the call site `main.py:321`) | `drift/detector.py:64-72` |
| PSI (categorical, tabular) | `psi_score > 0.2` | `0.2`, hardcoded, not a constructor parameter | `drift/detector.py:123` |
| IQR (real-time, tabular) | `value < Q1 - 3·IQR` or `value > Q3 + 3·IQR` | fence multiplier `3.0`, hardcoded | `drift/detector.py:32-33` |
| DCT (text/image/joint) | `auc > auc_threshold` | `0.65` default (`EmbeddingDriftDetector.__init__`, `drift/embedding_detector.py:28`); joint uses the same detector class with a different **classifier**, not a different threshold |  `drift/embedding_detector.py:71` |

**No multiple-testing correction exists anywhere today.** `system_alert_triggered`
is computed as a plain OR:

- Tabular: `analyze_production_window()` sets `system_alert = True` the moment
  any one feature's `drift_detected` is `True` (`drift/detector.py:130-138`).
- Text/image/joint: there's only ever one feature (`"embedding_drift"`), so
  `system_alert_triggered` is just that single result's `drift_detected`
  (`main.py`, each `/analyze/.../text|image|joint` handler).

For a project with `k` monitored tabular features each tested at `alpha=0.05`
independently, the system-level false-alarm rate under a true null is
approximately `1 - (1 - 0.05)^k`, not 0.05 — e.g. ≈0.23 at k=5. This is exactly
what Step 1's A/A test is meant to measure empirically rather than just derive
algebraically.

---

## 3. PSI binning and the `month` PSI=16.27 finding

There is **no continuous-style binning** in `_check_categorical_drift()` — it
operates on already-discrete category values via `np.unique` (`drift/detector.py:83-84`),
so "binning" here just means "distinct category labels," not bucketed ranges.

The epsilon substitution (`drift/detector.py:104-118`) is what produces large
PSI values for a category that's **completely absent** from one side:

```python
epsilon = 0.0001
...
if expected_pct == 0.0:
    expected_pct = epsilon
if actual_pct == 0.0:
    actual_pct = epsilon
psi_score += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
```

`month` in the Citi Bike validation is drifted **by construction**: the
baseline is fit on Jan–Mar data and production batches are drawn from Apr–Dec
(`tests/test_drift_engine.py` — baseline/production split described in the
script's header comments). Every Apr–Dec month value is a category with
`expected_pct = 0` in the Jan–Mar reference, so each contributes
`(actual_pct - 0.0001) * ln(actual_pct / 0.0001)`. For an actual month
frequency around 0.10–0.15, a single such category contributes roughly
`0.10 * ln(1000) ≈ 0.69` to `0.15 * ln(1500) ≈ 1.10`. Summed across the ~9
Apr–Dec month categories that don't appear in Jan–Mar, this lands in the
observed 16.27 range — it's not a bug, it's the intended behavior of PSI's
epsilon-substitution for zero-frequency categories, applied to a categorical
feature whose categories are disjoint between baseline and production by
construction. (This matches the README's own "Bugs Found" / "Per-Feature
Detection Rate" framing, which already treats `month` as an intentionally
extreme case, not a normal one.)

---

## 4. Validation suite structure (`tests/test_drift_engine.py`, `split_citi_bike.py`)

- **"Trial"** = one call to `send_batch_to_analyze()` (`tests/test_drift_engine.py:89-119`),
  i.e. one sampled batch sent to `/analyze/{project_id}` and scored once.
- **Decision unit**: the confusion-matrix evaluation (`run_classification_evaluation`,
  `tests/test_drift_engine.py:276-403`) operates at **batch × feature**, not
  per-batch. Each trial's response contains one `drift_detected` per monitored
  feature (7 features: 5 continuous + `gender_id` + `month`), and every
  feature-result across every trial becomes one row in the confusion matrix.
  `n_trials=15`, so Case A and Case B each contribute `15 × 7 = 105` decision
  units (matches the previously observed `n=105` in this script's printed
  output); Case C injects synthetic drift only into continuous features, so it
  contributes `15 × 5 = 75` decision units.
- **Batch types feeding the headline metrics** (`run_classification_evaluation`):
  - Case A — real **baseline-period** rows (`baseline_df`, Jan–Mar), sampled
    fresh per trial via `send_batch_to_analyze(baseline_df, ...)`
    (`tests/test_drift_engine.py:303-311`) — labeled "should not trigger."
  - Case B — real **production-period** rows (`prod_df`, Apr–Dec), sampled
    fresh per trial (`tests/test_drift_engine.py:313-323`) — labeled per the
    independent PSI/KS ground truth from `compute_ground_truth()`.
  - Case C — a fresh `baseline_df` sample with synthetic "moderate" (1.5σ)
    drift injected per continuous feature (`tests/test_drift_engine.py:325-331`)
    — labeled as a known positive.
  - **`tests/test_drift_engine.py` never calls `/fit` at all** — it only reads
    `tests/citi_bike_baseline.csv`/`tests/citi_bike_production.csv`
    (`tests/test_drift_engine.py:80-81`) and calls `/analyze/citi_bike_v1`,
    assuming that project's baseline is already locked. **`split_citi_bike.py`,
    which the README explicitly tells users to run first
    (`README.md:549`, "Prepare the test datasets first:
    `python split_citi_bike.py`"), does not exist anywhere in this repo** —
    confirmed via a full-repo filename search. So the actual process that (a)
    split the raw Citi Bike data into those two CSVs and (b) called
    `/fit/citi_bike_v1` to lock the live baseline (measured in §1: 5,000
    continuous rows / 2,000 categorical rows) is **not reproducible from
    anything currently in this repo** — it must have happened via an ad-hoc,
    unpreserved call. This means **I cannot determine from the repo whether
    Case A's `baseline_df` sample overlaps with the rows that were actually
    fit** — the fitted reference's exact source rows are simply unknown.
    Step 1 will need to either locate/reconstruct the original split logic or
    treat this as a hard blocker for a clean A/A design and re-fit the
    baseline itself from a documented, reproducible split before running the
    A/A test.
- **Which batch size produced which table**: `PRODUCTION_BATCH_SIZE = 25000`
  (`tests/test_drift_engine.py:49`) is the fixed size used for the sensitivity
  test, the classification/confusion-matrix evaluation, and (by inspection)
  the headline Precision/Recall/F1/Accuracy numbers in the README — **all at
  n=25,000**, not at any of the sample-size-sweep sizes. The sweep
  (`run_sample_size_sweep`, sizes `(1000, 3000, 5000, 10000, 20000, 50000)`,
  `tests/test_drift_engine.py:405`) is a **separate** experiment with its own
  `n_trials=8` — its rows are not meant to reproduce the headline number
  directly, which is exactly why the prompt's "headline recall exceeds every
  sweep row" observation is a real discrepancy worth Step 1's raw-count dump,
  not something resolvable from reading the code alone.
- **Synthetic drift injection**: `inject_synthetic_drift(df, column, severity, seed)`
  (`tests/test_drift_engine.py:216-230`) — a mean shift of `severity × σ` added
  to the sampled column, where `σ` is the column's own baseline standard
  deviation and `severity ∈ {0.5, 1.5, 3.0}` (mild/moderate/severe).
- **Latency test mixing** (`measure_detection_latency`, `tests/test_drift_engine.py:471-495`):
  fixed `batch_size=100`, `step=10`; for `n_drifted` from 0 to 100 in steps of
  10, mixes `n_drifted` synthetically-drifted (severity="moderate", i.e. 1.5σ)
  rows with `n_clean = batch_size - n_drifted` baseline rows, and reports the
  smallest `n_drifted` at which the feature is flagged.

I have **not** re-run this script in Step 0 (out of scope — Step 1 owns
re-running and dumping raw counts). Everything above is read from the current
source, not re-derived from a fresh run, and should be treated as "what the
code does," not yet as "what numbers it produces" — that's explicitly Step 1's
job with saved raw TP/FP/FN/TN.

---

## 5. Domain Classifier Test mechanics; joint embedding and its tests

**DCT** (`drift/embedding_detector.py`):
- CV: `StratifiedKFold(n_splits=min(5, min_class_count), shuffle=True, random_state=42)`
  (`drift/embedding_detector.py:59-62`) — fixed seed, so results are
  deterministic given the same input embeddings.
- Classifier: `LogisticRegression(max_iter=1000)` by default, injectable via
  the `classifier` constructor param (`drift/embedding_detector.py:39`) —
  verified in an earlier session pass that no caller other than
  `adapters/joint.py` passes a non-default classifier, so text/image are
  provably unaffected by the parameter's existence.
- AUC: `cross_val_predict(..., method="predict_proba")[:, 1]` → out-of-fold
  probabilities → `roc_auc_score(y, probs)` (`drift/embedding_detector.py:66-68`).
  This is a genuine out-of-fold AUC, not a resubstitution/in-sample one.
- Reference embedding cap: `max_reference_samples=3000`, random subsample via
  `np.random.default_rng(42)` if the fitted batch exceeds it
  (`db/crud.py:244-255`).
- Minimum samples: `HARD_MIN_SAMPLES=4` (raises `ValueError` below this),
  `RECOMMENDED_MIN_SAMPLES=40` (non-fatal warning) (`drift/embedding_detector.py:13-14`).

**Joint** (`adapters/joint.py`):
- Vector construction (`JointAdapter.transform`): per record, concatenates —
  a z-scored/frequency-encoded tabular sub-vector (dimension = number of
  monitored tabular fields), the 384-dim text embedding (zeros if text
  absent), the 512-dim image embedding (zeros if image absent), a bounded
  **interaction block**, and a 3-dim presence mask (tabular/text/image
  present flags).
- **The interaction feature, precisely**: a **fixed, seeded** (not trained)
  random projection of the tabular sub-vector into the text/image embedding
  dimensionality — `_random_projection(source_dim, target_dim)` draws from
  `np.random.default_rng(1337)` scaled by `1/sqrt(source_dim)`
  (Johnson–Lindenstrauss-style) — then the interaction term is
  `tabular_vector * (text_or_image_embedding @ projection)`, i.e. an
  elementwise product that is exactly zero whenever either side (tabular
  fields or that modality) is absent for a record. This is what lets the
  joint vector carry cross-modality *pairing* information, not just each
  modality's independent presence.
- Classifier: `build_joint_classifier()` returns
  `LogisticRegression(max_iter=2000, solver="liblinear", C=2.0, l1_ratio=1.0)`
  — L1-penalized, still linear, passed into the shared `EmbeddingDriftDetector`
  via its `classifier` param, so the shared default stays untouched.
- **What `tests/test_joint_adapter.py` covers** (13 tests across 3 classes):
  - `TestJointAdapterShape` (5 tests): full-record shape/determinism,
    partial-modality records, interaction blocks are exactly zero when a
    modality is absent, a record with *no* modality raises, empty-batch
    handling.
  - `TestCategoricalFrequencyEncoding` (2 tests): known-frequency encoding is
    correct; an unseen category at analyze time encodes as zero rather than
    crashing.
  - `TestJointEmbeddingDriftDetector` (4 tests): same-distribution AUC centers
    near 0.5; a tuned-magnitude correlation inversion is detected via the
    interaction feature; that detection generalizes to a noisier (80/20) same-
    magnitude pairing; a **smaller-separation** inversion is confirmed **not**
    detected (documented gap, not silently passing).

---

## 6. Auth — corrected per your instructions

**Not `st.login()`.** `dashboard.py:531` constructs a third-party
`streamlit_google_auth.Authenticate(...)`, and the actual class run in
production is **not** the pip-installed one as-is.

**Installed version, and pin:**
- `requirements.txt:18` pins `streamlit-google-auth==1.1.8` exactly.
- The locally installed package (via this same requirements.txt) is also
  `1.1.8` (confirmed via `pip show streamlit_google_auth`) — so the pin and
  the installed version match exactly; there is no drift between what's
  declared and what's installed in this dev environment.

**Diff of `patched_init.py` against the installed 1.1.8 `__init__.py`:**

Ran `diff -u` and `diff -uw` (whitespace-insensitive) between
`patched_init.py` and the real installed
`.../site-packages/streamlit_google_auth/__init__.py`. Result: **the only
difference is two added blank lines** inside `get_authorization_url()` and
`login()` — no logic, no signatures, no imports, nothing behavioral differs.
For the currently pinned version (1.1.8), this file is functionally a
byte-for-byte copy with cosmetic reformatting.

`git log --follow` shows `patched_init.py` was added in commit `4bfe51d`
("Updated dashboard.py with environment variables for Render", 2026-06-20).
The commit message doesn't explain the original motivation, and given the
diff is purely cosmetic against the exact currently-pinned version, I can't
determine from the repo alone whether this ever fixed a real bug in some
different version of the upstream package, or whether it was introduced as a
defensive "always ship exactly this known-good file" measure regardless of
what pip resolves. **I'm not speculating further than what the diff shows.**

**What breaks if the upstream version changes:** the Dockerfiles
unconditionally overwrite `site-packages/streamlit_google_auth/__init__.py`
with this repo's frozen copy at build time (`Dockerfile:22`,
`Dockerfile.dashboard:19`, both `... || true`, so a failed copy is silently
ignored rather than failing the build). If a future `pip install` resolved a
different version of `streamlit-google-auth` with a changed `Authenticate`
class (different constructor signature, different cookie-handling module
path, different method names), this blind overwrite would replace that
version's real implementation with this frozen 1.1.8-era copy — silently,
with `|| true` swallowing any copy failure and no verification that the
result is even compatible with the installed `cookie` submodule it imports
from (`from .cookie import CookieHandler`). This is precisely the risk your
Step 7 build-time-guard proposal is meant to close; I have not implemented
that guard, only documented the risk it addresses, per your instructions
("propose, don't implement").

**Session JWT minting and verification:**
- **Minted** in `dashboard.py`'s `mint_session_token(user_info)`
  (`dashboard.py:80-96`), after `streamlit_google_auth` has already completed
  the Google OAuth flow and populated `st.session_state["user_info"]`. Payload
  is `{email, name, iat, exp}` (1-hour expiry), signed
  `jwt.encode(payload, os.getenv("COOKIE_KEY"), algorithm="HS256")`.
- **Verified** in `main.py`'s `verify_access()` (`main.py:72-89`): decodes with
  `jwt.decode(credentials.credentials, COOKIE_KEY, algorithms=["HS256"])`
  against the same `COOKIE_KEY` env var (`main.py:58-60`, raises at import time
  if unset), rejecting on any `jwt.PyJWTError` or a missing `email` claim.
- **Secret**: `COOKIE_KEY`, shared between the dashboard and backend processes
  via environment variable (`docker-compose.yml` passes it to both the `api`
  and `dashboard` services identically). No asymmetric signing — pure shared-
  secret HS256.

**Design note for Step 3 (per your instruction #2, recorded now for later):**
PAT verification will live entirely in the backend's auth dependency, as a
sibling check to the existing JWT branch in (what is currently)
`verify_access()` — not integrated into `streamlit_google_auth` at all. The
only dashboard-side surface will be a token-management page that needs the
logged-in user's identity, which is already available today as
`st.session_state["user_info"]` (populated by
`streamlit_google_auth.Authenticate.check_authentification()` — see
`patched_init.py`'s `check_authentification` method) — specifically `email`
and `name`, the same fields `mint_session_token()` already reads. No new
dependency on the auth widget's internals is needed for PATs.

---

## 7. Deployment — three topologies, precisely

**Topology A — the main `Dockerfile` (single combined container):**
`Dockerfile:1-42` installs `nginx` and `supervisor`, builds the app, copies
`supervisord.conf` to `/etc/supervisor/conf.d/supervisord.conf`, configures
nginx to reverse-proxy `/api/` → `localhost:8000` and everything else →
`localhost:8501`, exposes port 80, and its `CMD` is
`supervisord -c /etc/supervisor/conf.d/supervisord.conf`.
`supervisord.conf` itself defines **three** managed processes: `backend`
(`uvicorn main:app --port 8000`), `dashboard` (`streamlit run dashboard.py
--server.port=8501`), and `nginx`. **This single image runs the entire stack
by itself** — it is not backend-only despite the README describing it that
way.

**Topology B — `docker-compose.yml`:** defines two services, `api` and
`dashboard`, **both built from `build: .`** (the same main `Dockerfile`,
Topology A's image). The `api` service takes no command override, so it runs
the full supervisord stack (backend + dashboard + nginx, all three, on port
80 internally, though compose only publishes `8000:8000` for this service —
meaning nginx's port 80 isn't even published here, so only the raw backend on
8000 is reachable from outside, though dashboard+nginx processes still start
inside the container). The `dashboard` service uses the **same image** but
overrides `command:` to run `streamlit run dashboard.py --server.port 8501`
directly, bypassing supervisord/nginx entirely for that container. Net
effect: compose runs **two containers**, each containing a full copy of
everything, with the `api` container additionally running a second,
unpublished/unused internal dashboard+nginx that compose never routes to.

**Topology C — `Dockerfile.dashboard`:** a separate, simpler image — no
nginx, no supervisor, `CMD ["streamlit", "run", "dashboard.py",
"--server.port=8000", ...]`. **`docker-compose.yml` never references this
file at all** — it's dead weight from compose's perspective, only relevant if
something outside this repo (e.g. a Render service definition) points at it
directly.

**Topology D — what the README says** (`README.md`, "Deployment" section):
*"Create two Web Services on Render pointing to your fork — one using
`Dockerfile` (backend) and one using `Dockerfile.dashboard` (dashboard)."*
This describes Topology C being used for the dashboard, and implies
`Dockerfile` is used *backend-only* for the other service — but Topology A
shows `Dockerfile` is not backend-only; it's the full combined stack. If
Render is actually run this way (two services, one per Dockerfile, per the
README), then the "backend" service is actually running backend+dashboard+
nginx together (Topology A's image), just with only port 80 exposed/routed by
Render's own port detection — dashboard and supervisord would still be
running inside that container, redundantly with the actual dedicated
dashboard service.

**I have not determined which of these is what's actually live on Render.**
Per your instruction, no changes to any Dockerfile, `docker-compose.yml`,
`supervisord.conf`, or `runtime.txt` are made or proposed until you confirm
which topology is real. `supervisord.conf` is **not** classified as stale —
it's actively referenced by the main `Dockerfile` regardless of which
topology is live in production.

---

## 8. Python version mismatch

| Source | Version | 
|---|---|
| `README.md` (badge + Prerequisites) | "3.12+" |
| `runtime.txt` | `python-3.12.4` |
| `Dockerfile` | `FROM python:3.10-slim` |
| `Dockerfile.dashboard` | `FROM python:3.10-slim` |

Both actual deployment images run **Python 3.10**, not 3.12. Since deployment
is Docker-based (per both the README's own instructions and the actual
Dockerfiles), Render's native-buildpack `runtime.txt` mechanism does not apply
to a Docker-based Web Service — meaning `runtime.txt`'s `3.12.4` is very
likely inert regardless of which topology (Q7) is live, though I haven't
found a way to confirm "Render ignores this file for Docker services" from
inside this repo — that's a platform-behavior claim, not something I can
verify by reading source. Flagging as high-confidence but not code-verified.

**Consequence for Step 7:** CI must target **3.10** (the actually-deployed
version), not 3.12, per your instruction.

---

## 9. File parsing, `/logs`, and `project_id` vs `model_id`

**File parsing** lives entirely in `dashboard.py`, not in any shared/importable
module yet: `_read_csv_with_encoding_fallback()` (`dashboard.py:126-150`) and
`read_uploaded_file()` (`dashboard.py:152`+) handle encoding fallback,
CSV/TSV/Excel/JSON/Parquet/ARFF/libsvm `.dat`/gzip/zip parsing. This is
exactly the "move into `ingest/readers.py`" target for Step 3 — currently it's
dashboard-only, un-importable from a script or the backend.

**`/logs/{project_id}`** (`main.py:139-148`, `db/crud.py:317-336`): stores/
returns raw per-prediction rows from the `logs` table — `input_data` (JSON-
encoded feature dict), `score` (max IQR deviation), `is_ood` (0/1) — written
by `/predict/{project_id}`'s background task (`main.py:294`) on every
real-time scoring call. Read via `GET /logs/{project_id}`, capped at the most
recent 1000 rows (`db/crud.py:333`, `ORDER BY rowid DESC LIMIT 1000`). This is
**tabular-only** — text/image/joint never write to `logs` since they have no
per-record real-time scoring endpoint, only batch `/analyze`.

**`project_id` vs `model_id`**: these are the same identifier space
everywhere except one endpoint. Every fit/analyze/baseline/logs endpoint uses
`project_id` consistently as the path parameter and as the `baselines.project_id`
/ `logs.project_id` / `projects.id` column. The sole exception is
`DELETE /models/{model_id}` (`main.py:596-623`), which takes a path parameter
literally named `model_id` but immediately uses it as `project_id` against
those same three tables/columns (`main.py:610,613,614`) — it is not a
different concept, just an inconsistent name at one endpoint. This directly
supports Step 3's plan (`DELETE /projects/{project_id}` as the primary name,
`DELETE /models/{model_id}` kept as a deprecated alias) — they already share
one ID space today, so that change is a rename/alias, not a data-model change.

---

## Anything else that contradicts the README (beyond what's already listed above)

- **Debug prints in production** (fixed in this step, commit `bd76e48`): four
  `print(f"[DEBUG] ...")` calls in `_check_categorical_drift` dumped raw
  category arrays to stdout on every PSI check. Not README-related, but a
  real production-logging issue found during this pass. Replaced with
  `logging.debug(...)` — nothing above DEBUG level logs category values.
- **`patched_init.py` is functionally a no-op patch for the current pin**
  (see §6) — the README/architecture never mentions this file's existence or
  purpose at all; worth a "How It Works" or "Project Structure" mention in
  Step 9 given how much deployment behavior depends on it working correctly.
- **The three deployment topologies (§7)** are the single biggest
  README-contradicting finding from this pass — the README describes a clean
  two-service split that the actual Dockerfiles don't implement as described.
- **`split_citi_bike.py` doesn't exist.** The README tells users to run it as
  the first step of reproducing the tabular validation suite (`README.md:549`),
  but it is not present anywhere in the repo (confirmed by a full-repo
  filename search). The two CSVs it's supposed to produce
  (`tests/citi_bike_baseline.csv`, `tests/citi_bike_production.csv`) do exist
  locally (large, gitignored), so someone ran *something* that produced them
  at some point — but that script isn't checked in, and the live
  `citi_bike_v1` baseline's exact provenance (which rows were actually fit)
  is consequently unverifiable from the repo alone (see §4).

---

## Summary of what's still open (not resolved in Step 0, by design)

1. Which deployment topology (§7) is actually live on Render — **waiting on
   you**, per your instruction.
2. The exact cause of the recall/F1 discrepancies in the validation suite's
   headline numbers (§4) — requires a fresh run with raw counts saved, which
   is Step 1's job, not Step 0's.
3. Whether Case A's clean-batch samples in `run_classification_evaluation`
   overlap with the rows used to `/fit` the baseline — **cannot be determined
   from the repo** since `split_citi_bike.py` (the script that would show
   this) doesn't exist (§4). Step 1 needs a decision from you: reconstruct
   a plausible split, or re-fit `citi_bike_v1` from a fresh, documented,
   reproducible split before running the A/A test.
