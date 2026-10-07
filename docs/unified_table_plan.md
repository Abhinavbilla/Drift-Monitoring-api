# Unified Table Monitoring — Implementation Plan (A–S)

Status: **M1–M4 approved and implemented (2026-10-07).**
Written 2026-10-06 from the user's specification plus six agreed corrections:
(1) relationship detectors invariant to marginal drift, (2) batch-size-matched,
disjoint null distributions, (3) real-embedding nulls for text/image so they can
join Holm correctly, (4) a row-aligned reference store, (5) background jobs and a
defined staged-upload lifecycle, (6) milestone approvals, with experimental
features allowed to stay off by default when validation doesn't support them.

Approval is **per milestone** (M1–M4, section P). Each milestone ends with a
demo, a full test-suite run and a written check-in. Within an approved
milestone, implementation proceeds without redesign. If a requirement turns out
to be infeasible or statistically unsound, I stop, report the evidence, and
propose the smallest honest alternative. I don't quietly weaken it.

Terminology used throughout (precise, never conflated):

| Term | Meaning |
|---|---|
| **Marginal / univariate drift** | P(X_i) changed for one column. |
| **Relationship / dependency drift** | The dependence between columns changed, **beyond** what their marginal changes explain. |
| **Joint / multivariate drift** | The overall joint distribution P(X_1..X_n) changed (experimental whole-row signal). |
| **Point OOD / anomaly** | One observation is unusual relative to the reference (existing IQR scoring; secondary). |

"No individual feature drifted" never means "no data drift". The report says
which of the signals above were checked, and which were not.

---

## A. Current repository assessment

Inspected 2026-10-06 (read-only).

- **Backend:** FastAPI monolith, `main.py` (~1,756 lines, 24 routes). Synchronous style throughout (sqlite3, requests, smtplib). One daemon thread already exists: the webhook delivery sweep, started from `lifespan`.
- **Storage:** SQLite (`drift.db`) through one central connection helper in `db/crud.py`. Tables: `baselines`, `baseline_versions`, `baseline_active_version`, `projects`, `analysis_runs`, `alert_events`, `logs`, `api_tokens`, `webhooks`, `webhook_deliveries`. **No WAL mode or busy timeout is configured**, which a background worker will need.
- **Project model:** one `baselines` row per project with a single `modality` (`tabular`, `text`, `image` or `joint`). Text/image reference embeddings are stored as **JSON text** in `baselines.embedding_reference`, capped at 3,000 rows.
- **Tabular cleaning:** `insert_baseline` cleans **each column independently**: it drops nulls and non-numeric cells per column. That destroys row alignment, which is why correction (4) is needed.
- **Ingestion:** `ingest/readers.py` handles CSV/TXT (encoding and delimiter sniffing), TSV, Excel, JSON/JSONL/NDJSON, Parquet, ARFF, libsvm `.dat`, `.gz`, and `.zip` containing one table. **Finding:** `.xls` asks for the `xlrd` engine, which is not installed and not in `requirements.txt`, and no test covers `.xls`. `.xls` uploads therefore fail today. `.xlsx` works (`openpyxl`); Parquet works (`pyarrow` 24).
- **Profiler:** `utils/profiler.py` (~200 lines). `compute_signals` → `classify_column` returns `monitor ∈ {True, "Categorical", False, "Review"}` plus a reason. There is no text/image type, no confidence and no evidence object. `/profile` is JSON-only and maps any error to a 500.
- **Detection:** `drift/detector.py` (`DistributionDetector`: KS for continuous, PSI for categorical, legacy vs calibrated two-gate, Holm via `drift/calibration.py`), `compute_iqr_anomalies` (point anomalies for `/predict`), `drift/embedding_detector.py` (Domain Classifier Test, logistic regression with 5-fold CV), `drift/dct_calibration.py` (synthetic-Gaussian null grid — measured over-confident on real embeddings in Step 2(e)).
- **Operations:** history (`analysis_runs`), idempotency, `schema_report` + `schema_policy`, baseline versioning, k-of-m alert state machine, signed webhooks with SSRF guard, PAT + Google auth, owner-namespaced project IDs with two-user isolation tests.
- **Frontend:** React/Vite/Tailwind (`frontend/`), with pages for Fit/Analyze/Predict/History/Baselines/Webhooks/Logs/Projects per modality. The image picker now accepts zips.
- **Tests:** 31 test files, ~297 test functions; the full suite passed (315 collected) at the last run.
- **Deployment:** `Dockerfile` runs **uvicorn + Streamlit + nginx** under supervisord. **The React frontend is not containerized.** `docker-compose.yml` mounts `./data` and `./drift.db`. There is **no `render.yaml`** in the repo; Render is presumably configured in its dashboard (to verify). The README notes Render free tier spin-down.
- **Known open items, unrelated to this plan:** leaked OAuth secrets in public git history (rotate before multi-team use); text/image calibrated p-values are over-confident (Step 2(e)).

## B. Current architecture summary

```
client (React / Python client / curl)
   │  Bearer: Google-session JWT or PAT
   ▼
main.py ── verify_access / verify_project_access (owner-namespaced project key)
   ├─ /profile ─────────── utils/profiler.profile_columns
   ├─ /fit, /fit/upload ── ingest.readers → profiler → feature_types → crud.insert_baseline (+ versions)
   ├─ /analyze(+upload) ── DistributionDetector (KS/PSI, calibrated two-gate, Holm) → schema_report
   │                       → analysis_runs → alert state machine → alert_events → webhooks queue
   ├─ /fit|analyze/text|image ── adapters.{text,image} → EmbeddingDriftDetector (+dct grid)
   ├─ /fit|analyze/joint ──────── adapters.joint (concat + interaction, L1 LR)
   ├─ /predict ─────────── IQR fences (point anomalies) → logs
   └─ /history /baselines /webhooks /projects /logs /health /auth/google
db/crud.py (SQLite) ── drift/webhooks.py daemon sweep thread
```

## C. What already exists and will be reused

| Existing piece | How it is reused |
|---|---|
| `ingest/readers.read_uploaded_file` | The table path's only table reader. All formats kept. `.xls` gets fixed (add `xlrd`, add a test) as a small Phase 1 item, since the spec requires every format to work. |
| `utils/profiler.compute_signals`, `classify_column`, `coerce_numeric_column` | Called unchanged by the new `profile_table()`. `profile_columns()` output stays **byte-identical** (golden tests). |
| `/profile` | Kept as-is for existing clients. The table path profiles inside a staging job. |
| `feature_types` overrides, `excluded_columns` | Their semantics carry into the confirmed schema (`final_type`, `final_monitor`, exclusion reasons). |
| `DistributionDetector` (KS, PSI, calibrated two-gate) | Used **verbatim** for numeric and categorical column marginal tests, which is what guarantees equivalence. |
| `drift.calibration.holm_adjust`, `apply_two_gate`, `CalibrationConfig` | The single Holm family and the two-gate decision for every test, relationships included. |
| `compute_iqr_anomalies` | Kept for `/predict` point anomalies. Not part of batch drift decisions. |
| `adapters.text.TextAdapter`, `adapters.image.ImageAdapter` (`_decode_image`) | Text/image column embedding (same models: all-MiniLM-L6-v2, resnet18). |
| `EmbeddingDriftDetector._compute_auc` | The per-column text/image marginal statistic (the DCT). |
| `crud` versioning (`baseline_versions`, `baseline_active_version`, activate) | A table baseline is one version. New artifacts are keyed by `(project, version)`. |
| `analysis_runs`, idempotency, `schema_report`/`schema_policy`, alert state machine, webhooks | The table `/analyze` writes the same rows, with additive columns. |
| Auth, `verify_project_access`, `_require_existing_project`, isolation test patterns | All new endpoints. Each gets a two-user isolation test. |
| Webhook sweep thread pattern | The job worker and the staged-upload expiry sweep use the same idempotent daemon-thread design. |
| Step 2(e) harness (`scripts/step2e_*`) | Reused for text/image calibration checks and the validation experiments. |

## D. What must change (and why)

1. **Row-aligned reference store (new).** Relationship statistics need values from the *same row*. The existing per-column store stays untouched, so per-column results remain equivalent; the row-aligned store is added alongside it. The duplication is accepted and documented.
2. **Profiler extension.** Add text/image/ignore detection, confidence, evidence and relationship proposals in a *new* `profile_table()`. `profile_columns()` is untouched.
3. **SQLite concurrency.** Enable `PRAGMA journal_mode=WAL` and `busy_timeout` in the central connection helper. This is needed once a worker thread writes concurrently with requests. Behavior is otherwise unchanged; covered by the existing test suite.
4. **Large-artifact storage.** Embeddings and staged uploads must not live as JSON text in SQLite. Add a `BlobStore` interface with a filesystem backend under `DATA_DIR` (the `./data` mount already exists in docker-compose). It can be swapped for object storage later.
5. **`.xls` reading** (add `xlrd` and a test). This is a bug fix, not a behavior change.
6. **`analysis_runs`** gets additive columns for relationship metrics and report kind.
7. **Deployment.** Containerize the React frontend (nginx serves the static build). The Streamlit process stays until you decide to retire it. Large image tables need a persistent disk; the free tier is documented as unsuitable.

No existing endpoint, request model or response field is removed or changes meaning.

## E. New files / modules

Only where no existing module fits:

| File | Purpose |
|---|---|
| `utils/profiler.py` (extend) | `profile_table()`, text/image detectors, confidence/evidence, relationship proposals. Same file; it is ~200 lines today. |
| `ingest/images.py` | Safe image-ZIP index (path normalization, limits, bomb checks), filename↔row association, base64/data-URI decoding, per-row image status. |
| `drift/relationship_detector.py` | num↔num, cat↔cat, num↔cat statistics + null grids + decisions; later the probes. |
| `drift/table_monitor.py` | Orchestrates a table analysis: schema validation → per-column → relationships → Holm family → report. Keeps `main.py` thin. |
| `db/blob_store.py` | `BlobStore` interface + filesystem implementation (`put/get/delete/delete_prefix`, sha256, size). |
| `db/crud.py` (extend) | New tables, migrations, CRUD. |
| `jobs.py` (top level, beside `main.py`) | Job queue table access, single worker thread, restart recovery. Kept out of `drift/` because it isn't detection logic. |
| `models.py` (extend) | Table request/response models. |
| `frontend/src/pages/TableWorkflowPage.tsx` (+ components) | Upload → column review → relationship review → fit → analyze → report. |
| `tests/test_table_*.py` | Per phase (see P). |
| `scripts/validate_table_*.py` | Validation experiments (section L). |

## F. Database schema changes

All changes are additive, following the existing self-healing migration pattern (guarded `CREATE TABLE IF NOT EXISTS` / `ALTER TABLE ADD COLUMN`). JSON is stored as TEXT; there is no SQLite-only SQL beyond what exists today, which keeps a PostgreSQL migration possible.

**Altered**
- `baselines.modality` accepts a new value, `'table'`. The materialized row holds the active version's per-column KS/PSI state exactly as today (num/cat columns), so `get_baseline`, `/baselines`, activate and history keep working.
- `analysis_runs`: `+ relationship_metrics TEXT` (JSON, statistics only), `+ report_kind TEXT DEFAULT 'tabular'`, `+ job_id TEXT`.

**New**
```
table_schemas(                       -- one row per column per baseline version
  project_id TEXT, version INTEGER, column_name TEXT, ordinal INTEGER,
  proposed_type TEXT, proposed_monitor INTEGER, confidence REAL,
  evidence TEXT,            -- JSON aggregates only (no sample values persisted)
  reason TEXT, alternative_type TEXT,
  final_type TEXT, final_monitor INTEGER,
  decided_by TEXT, decided_at TEXT,
  PRIMARY KEY (project_id, version, column_name))

table_column_baselines(              -- per monitored column per version
  project_id TEXT, version INTEGER, column_name TEXT, col_type TEXT,
  state TEXT,               -- JSON: num/cat → pointer to existing fences/frequencies;
                            -- text/image → model_name, embedding_dim, n_ref, pca info
  embeddings_blob TEXT,     -- BlobStore key (float16 .npy, row-aligned, with validity mask)
  null_blob TEXT,           -- BlobStore key (null grid for the marginal test, text/image)
  PRIMARY KEY (project_id, version, column_name))

table_reference_rows(                -- the row-aligned reference store
  project_id TEXT, version INTEGER,
  rows_blob TEXT,           -- BlobStore key: Parquet of monitored num/cat columns,
                            -- nulls preserved, plus row_id
  n_rows INTEGER, sample_seed INTEGER, dedup_dropped INTEGER,
  PRIMARY KEY (project_id, version))

table_relationships(
  id INTEGER PRIMARY KEY, project_id TEXT, version INTEGER,
  col_a TEXT, col_b TEXT, kind TEXT,   -- num_num | cat_cat | num_cat | probe | text_image
  proposed INTEGER, proposal_reason TEXT, proposal_strength REAL,
  final_monitor INTEGER, decided_by TEXT,
  reference_state TEXT,     -- JSON: rho / V_bc / interaction λ / level maps / per-category
                            -- medians / probe reference score
  materiality_floor REAL,
  null_blob TEXT,           -- BlobStore key: null grid {batch_size: sorted draws}
  probe_blob TEXT,          -- BlobStore key: PCA + coefficients as arrays (never pickle)
  created_at TEXT)

staged_uploads(
  id TEXT PRIMARY KEY,      -- 128-bit random
  owner_email TEXT, project_id TEXT, purpose TEXT,   -- 'fit' | 'analyze'
  status TEXT,              -- uploaded | profiling | profiled | consumed | expired | deleted | failed
  table_blob TEXT, zip_blob TEXT, table_filename TEXT,
  table_bytes INTEGER, zip_bytes INTEGER, table_sha256 TEXT, zip_sha256 TEXT,
  n_rows INTEGER, profile TEXT,   -- JSON profile result (sample values allowed here; deleted with the stage)
  created_at TEXT, expires_at TEXT)

jobs(
  id TEXT PRIMARY KEY, owner_email TEXT, project_id TEXT,
  kind TEXT,                -- profile | fit | analyze
  status TEXT,              -- queued | running | succeeded | failed | interrupted
  progress INTEGER, progress_message TEXT,
  stage_id TEXT, idempotency_key TEXT,
  result TEXT,              -- JSON (report / fit summary); statistics only
  error TEXT,               -- sanitized: never contains cell values
  created_at TEXT, started_at TEXT, finished_at TEXT)
```

**Lifecycle rules**
- Deleting a project (`DELETE /projects/{id}`) cascades to all table rows, blobs (`delete_prefix`), stages and jobs. Tested.
- Activating an old version re-materializes `baselines` from that version, same as today. Table artifacts are already version-keyed.
- Version retention: unchanged (no cap, by the earlier decision). The fit report includes bytes stored per version, so growth is visible. Pruning remains future work.

## G. API changes

Decision: **a new additive `/tables` group**, not an extension of `/fit`/`/analyze`. Reasons: today's `/fit` is a JSON dict-of-lists contract; the table flow needs two files, a two-step stage → confirm, and asynchronous jobs. Overloading `/fit` would change its meaning for existing clients.

| # | Endpoint | Purpose |
|---|---|---|
| 1 | `POST /tables/{project_id}/stage` | multipart: `file` (required, any supported format) + `images` (optional zip). Validates sizes, stores to BlobStore, enqueues a **profile** job. → `202 {stage_id, job_id, expires_at}` |
| 2 | `GET /jobs/{job_id}` | Status, progress, and on success the result: a profile (columns + proposed relationships), a fit summary, or an analysis report. Owner-only. |
| 3 | `POST /tables/{project_id}/fit` | JSON `{stage_id, columns:[{name,type,monitor}], relationships:[{col_a,col_b,monitor}], calibration_config?, schema_policy?, alert_policy?, model_version_label?, relationship_floors?}` → `202 {job_id}`. On success: new baseline version; the stage's raw blobs are deleted. |
| 4 | `POST /tables/{project_id}/analyze` | multipart: `file` + optional `images`; query `baseline_version?`; header `Idempotency-Key?`. Internally stages with purpose=analyze, enqueues an analyze job → `202 {job_id}`. Raw production data is deleted when the job finishes. |
| 5 | `GET /tables/{project_id}/baseline?version=` | Confirmed schema (proposed vs final type, monitor, who/when), per-column baseline summary, relationships with reference statistics. Answers the four audit questions in the spec. |
| 6 | `DELETE /tables/stages/{stage_id}` | Explicit discard of staged raw data (privacy). |

Reused unchanged (with additive fields only): `GET /history/{project_id}` (+`relationship_metrics`, `report_kind`), `GET /baselines/{project_id}`, `POST /baselines/{project_id}/activate`, webhooks CRUD (payload gains `relationship_metrics`, statistics only), `/projects`, `DELETE /projects/{id}` (cascade extended), auth.

**Analyze report model (job result and `analysis_runs`)**
```
TableAnalysisReport {
  baseline_version, n_rows, overall: {status: DRIFT|STABLE|DATA_ISSUES, alert, sustained_alert,
            alert_state, transition, triggered_by: [test ids]},
  column_drift: {col: {type, test (KS|PSI|DCT), statistic, reference_summary, current_summary,
            p_value, p_value_adjusted, effect_size, effect_floor, in_family, status
            (DRIFT|STABLE|NOT_TESTED), reason}},
  relationship_drift: {"a<->b": {kind, statistic_name, reference_value, current_value,
            p_value, p_value_adjusted, effect_size, effect_floor, status, explanation,
            confounded_by: [cols]}},
  joint_signal: {experimental: true, ...} | null,
  screening: {emerged_dependencies: [...]} ,          -- informational, never alerts
  schema_report: {...same shape as today, extended issue types},
  data_quality: {missing_images, corrupt_images, empty_text, malformed_rows, dedup_dropped, ...},
  family: {method: "holm", alpha, size, members: [test ids], excluded: [{test, why}]}
}
```

**Python client:** add `stage_table / fit_table / analyze_table / wait_job`. Existing methods are untouched.

## H. Dashboard / UI changes

A new **Table** workflow, the default entry in the nav. Existing per-modality pages are kept (moved under "Legacy modalities" only after M4).

1. **Upload:** table file + optional image zip, with the existing file picker, size hints and format list.
2. **Profiling progress:** job polling, with clear errors (missing images, unreadable files) shown with counts.
3. **Review columns:** one row per column: name · proposed type (editable dropdown) · confidence · reason · monitor toggle · expandable evidence (aggregates plus up to 5 truncated sample values or image thumbnails, shown only to the owner, never persisted). Columns proposed `ignore` default to off; nothing is monitored until confirmed.
4. **Review relationships:** proposed pairs with kind, strength and reason ("Spearman 0.72 in reference"), accept/reject toggles, a searchable list of non-proposed pairs to add manually, and a live count of tests in the Holm family ("adding pairs lowers per-test sensitivity").
5. **Fit:** a job with progress (embedding N images…), then a summary: version, monitored columns, relationships, data-quality counts, bytes stored.
6. **Analyze:** upload a production table (+zip) → job → report.
7. **Report:** overall status banner; a **Column drift** table and a **Relationship drift** table (status + before → after values); a data-quality panel; an "experimental joint signal" card when enabled; an expandable "statistical details" section (p, adjusted p, floor, family size). History charts reuse the existing History page with a relationship tab.

## I. Statistical design for every detector

Notation: reference R (N rows after cleaning and dedup, capped by a seeded sample to `N_MAX = 5000`); production batch B (n rows). Every test returns `(statistic, effect_size, raw p, in_family flag)`. Decisions use the existing two-gate rule: **Holm-adjusted p < α AND effect ≥ floor**. All floors are **product choices**, labelled provisional and configurable per project. Validation reports results at several floors, as Step 2(e) did. It does not tune them on evaluation data.

### I.1 Marginal tests (one per monitored column)
- **Numeric:** existing KS through `DistributionDetector` (calibrated mode; KS p-value; floor `ks_d`). Unchanged.
- **Categorical:** existing PSI with bootstrap p-value (floor `psi`). Unchanged.
- **Text / image:** DCT AUC from `_compute_auc`, run on the column's embeddings, **PCA-reduced to k=64** (PCA fitted on the reference, stored). Reasons: per-project null calibration has to be affordable, and Step 2(e)'s small-reference results suggest high dimension hurts small samples. M3 validates PCA-64 against the raw embeddings on the same draws; if PCA loses power, raw stays. The legacy endpoints are unaffected either way.
  - Its p-value comes from the **real-embedding null** (section J). Until that null passes its calibration gate, the column is decided by the legacy rule (AUC > 0.65), and is reported as `in_family: false`.

### I.2 num ↔ num — Spearman
- **Statistic:** T = |ρ_B − ρ_R|, using pairwise-complete rows.
- **Marginal invariance:** ρ is a function of ranks, so any strictly monotone change of either column (shift, scale, log, etc.) leaves it unchanged. Non-monotone marginal changes (e.g. a new mode) can move ρ; this is documented as a limitation.
- **Effect:** |Δρ|; floor default 0.10 (provisional). The report shows ρ_R, ρ_B and the direction (including sign flips).
- **Minimum:** n_pair ≥ 30 in both R and B, else NOT_TESTED (reported, outside the family).
- **Ties:** average ranks (scipy).
- **Known limitation:** changes that keep monotone association but alter its shape, or non-monotone dependence (e.g. U-shapes), aren't captured. Listed in README.

### I.3 cat ↔ cat — homogeneity of association (log-linear 3-way interaction)
- **Level handling (fixed at fit time from R):**
  - levels with < max(5 rows, 1 %) of R merge into `__other__`;
  - at most 20 levels per variable (top-19 + `__other__`).
  - Production values unseen in R map to `__unseen__`. They are **excluded from this test** (counted and reported); unseen categories remain a schema issue.
- **Statistic:** stack R and B into a 3-way table A × C × S (S = source). Fit the log-linear model **[AC][AS][CS]** by iterative proportional fitting. It allows each source its own marginals of A and C but forces the same A–C association (the same odds ratios). T = G² deviance against the saturated model.
  - This is marginal-invariant by construction: changing category proportions alone does not change odds ratios.
  - Unlike comparing Cramér's V, it also detects **re-pairing at equal strength** (Breed X now goes with Type Y instead of Z).
  - Zero cells get a 0.5 pseudo-count (Haldane) inside IPF.
- **Effect (margin-free):** RMS change of the centred interaction parameters λ_AC between R and B (log-odds scale). Floor default log(1.25) ≈ 0.22 (provisional).
- **Descriptive output:** bias-corrected Cramér's V (Bergsma) for R and B, plus the top changed cells ("Beagle × Adopted: log-OR +1.1 → −0.4").
- **Minimum:** ≥ 50 pairwise-complete rows in each source and ≥ 2 retained levels per variable, else NOT_TESTED.
- **Calibration:** never chi-square asymptotics (sparse tables break them). Only the disjoint-split null (J).

### I.4 num ↔ cat — composition-adjusted conditional distributions
- **Goal:** detect changes in P(num | cat) that are not explained by a change in the numeric column's marginal or in category proportions.
- **Transform:** convert the numeric values to probability-integral-transform scores **within each source**, using that source's ECDF **re-weighted to the reference category composition**. Batch row i gets weight p_R(k_i)/p_B(k_i) when building the batch's weighted ECDF; the reference uses its own ECDF.
  - If P(num | cat) is unchanged and only proportions change → the weighted ECDF matches → no signal.
  - If the numeric marginal shifts monotonically for all categories → the PIT absorbs it → no signal.
  - If the categories' relative positions change (e.g. "Beagles became the expensive breed") → signal.
- **Statistic (revised in M2):** T = Σ_k w_k · |mean_R(u | k) − mean_B(u | k)|, where u is the **mid-rank** PIT and w_k = p_R(k) normalized over the tested categories. The original per-category KS distance on PIT values was dropped after the M2 validation showed it breaks under heavy ties: tie groups land on slightly different PIT values in each sample, which inflated the null to around 0.6. Limitation: it tracks each category's relative position, not spread changes within a category. Floor default 0.05.
- **Category handling:**
  - a category is tested only if it has ≥ 20 rows in both R and B;
  - rarer reference categories are merged into `__other__` (tested if large enough);
  - unseen categories are excluded and reported;
  - fewer than 2 testable categories → NOT_TESTED.
- **Effect:** T itself (a weighted mean KS on the PIT scale). Floor default 0.10 (provisional).
- **Descriptive output:** per-category median in original units, before → after ("Beagle median Fee 100 → 250").
- **Limitation:** a common *non-monotone* transformation of the numeric column can leave residual signal. Documented.

### I.5 Probes: text/image ↔ categorical/numeric (Phase 10)
- **Source representation:** the column's PCA-64 embeddings (shared with I.1).
- **Categorical target:** multinomial logistic regression (levels handled as in I.3). The score is **balanced accuracy**, which is insensitive to label-prior shift.
- **Numeric target:** ridge regression on the target's reference-ECDF scores. The score is Spearman(predicted, actual), which is invariant to monotone target shifts.
- **At fit:** a cross-fitted reference score S_R (5-fold). The final probe is trained on all of R and stored as arrays: PCA components plus coefficients. **No pickles.**
- **At analyze:** S_B = the stored probe's score on B. T = S_R − S_B (one-sided: the relationship weakened).
- **Confounding:** covariate shift in the source embeddings can lower S_B with the pairing intact. If the source column's marginal test is significant, the result carries `confounded_by: [source]`, and the report says the relationship result is not attributable on its own. Validation measures how often this happens.
- **Scope limit:** a probe trained on R cannot detect a *new* relationship that appears only in production. Documented.

### I.6 text ↔ image — cross-modal matching (Phase 10, experimental)
- **At fit:** ridge-map the image PCA-64 vectors to text PCA-64 space.
- **Score:** the matching AUC = P(cosine(mapped image_i, text_i) > cosine(mapped image_i, text_j)) for random j ≠ i, estimated within a source.
- **Statistic:** T = AUC_R(cross-fitted) − AUC_B.
- Detects broken image↔text pairing (e.g. shuffled descriptions) while each marginal stays the same. Same confounding rule as I.5.

### I.7 Optional joint whole-row signal (experimental, off by default)
- **Inputs:** reference-ECDF-transformed numerics, one-hot categoricals (merged levels), and PCA-16 of each embedding column, concatenated.
- **Test:** DCT (logistic regression, the existing CV code) with a per-project disjoint-split null.
- Reported as `joint_signal` (experimental). **Never in the Holm family, never triggers an alert** unless formal validation (L) supports promoting it, and you approve that.

### I.8 Emergence screening (informational)
- At analyze, compute Spearman for all num–num pairs and bias-corrected V for all cat–cat pairs (only if the monitored columns number ≤ 60).
- Report pairs that were weak in R (|ρ| < 0.1 / V < 0.1) and strong in B (|ρ| ≥ 0.3 / V ≥ 0.2) as "new dependency appeared", with a recommendation to re-fit or add the pair.
- **Never alerts, never in the family.** It's a heuristic and is labelled as one.

### I.9 Point OOD (unchanged, secondary)
- The existing IQR fences on numeric columns via `/predict`. Not part of table drift decisions.
- Wording everywhere: "out-of-distribution relative to the configured reference input distribution". Never "the model was not trained on this".

## J. Calibration strategy

**Principle:** each project's null distribution comes from its own reference data, using **disjoint splits at the production batch size**. That means it estimates how much the statistic fluctuates between a reference and an independent batch of size n under no drift.

**Procedure for one statistic and one batch size s**
1. Draw a random split of R into R_A (N − s rows) and R_B (s rows), disjoint.
2. Compute the statistic exactly as at analyze time, with R_A as "reference" and R_B as "batch". Same code path; for I.5/I.6 the probe is retrained on R_A.
3. Repeat K times → sorted null draws.
4. The observed test uses the full R as reference. The null's reference (R_A) is slightly smaller, so the null is slightly wider. That bias is conservative, and it's documented.

**Batch-size matching**
- **Null grid sizes:** S = {30, 50, 100, 200, 400, 800, 1600, 3200}, kept only where s ≤ N/2. Computed in the **fit job**.
- **At analyze**, with batch size n:
  - If n ∈ S → use that size.
  - Otherwise → use the largest grid size **≤ n**. A smaller size has a wider null, so this is conservative.
  - If n is below the smallest grid size → NOT_TESTED.
  - If n > N/2 → use the largest size available, flag `approximate_null: true`, and recommend a reference at least twice the batch size.

**Draw counts and p-value resolution**
- Empirical p = (1 + #{T* ≥ T}) / (K + 1).
- Holm needs p ≤ α/m for the smallest p, where m is the family size. With K draws the smallest possible p is 1/(K+1).
- **Cheap statistics (I.2–I.4):** K = max(1000, ⌈10·m_max/α⌉) at fit, where m_max is the family-size cap (default 200 → K = 40,000). These are vectorized; affordable inside a job. Stored as sorted float32 arrays.
- **Expensive statistics (I.1 text/image DCT, I.5, I.6):** K = 200 per grid size. Beyond the empirical range, p comes from a **Gaussian tail fitted to the null draws** (mean, sd), flagged `tail_extrapolated: true`. This is an approximation; the M3/M4 A/A runs must show it's calibrated, or the gate below fails.

**Gate for text/image tests joining the Holm family (option (a))**
- Run the A/A protocol (L.A) on PetFinder.
- **Pass:** P(p < 0.05) lies within the 95 % Wilson interval of 5 % in every tested cell, and FWER ≤ α within its interval.
- **On pass, and with your approval:** text/image marginal tests and probes become `in_family: true` (config `embedding_tests_in_family`).
- **On fail:** they stay legacy-decided and outside the family, and the README says so with the numbers.

**Determinism:** every split uses `np.random.default_rng(seed)`, where the seed is derived from (project, version, statistic id, size) with crc32, as in Step 2(e). Null grids are reproducible and stored with the version.

**Assumption, stated explicitly:** reference rows are exchangeable, i.e. roughly i.i.d. Time-ordered or clustered data (e.g. many listings per rescuer) violates this, and the null then comes out too narrow. The fit report warns when a monotone index or a heavily duplicated key is detected. This is documented, not solved.

## K. Multiple-testing strategy

**Family F, per analyze call**
- **Members:**
  - every monitored column's marginal test (numeric KS, categorical PSI, and text/image DCT **only if** the J gate has passed);
  - every confirmed relationship test with sufficient data (I.2, I.3, I.4, and I.5/I.6 only if gated in).
- **Excluded** (listed in `family.excluded` with the reason):
  - NOT_TESTED results;
  - schema issues (handled by `schema_policy` severities, as today);
  - point anomalies;
  - the experimental joint signal;
  - emergence screening;
  - text/image tests that haven't passed the gate.

**Workflow**
1. Compute the raw p for each member.
2. Run `holm_adjust` across all of F at α (default 0.05).
3. Two-gate decision per member: adjusted p < α AND effect ≥ floor.
4. **Project-level alert** = any member DRIFT, **or** any schema issue with severity `alert`, **or** any legacy-decided text/image column DRIFT. That last case is reported separately as "decided outside the family".
5. The k-of-m `sustained_alert` and the alert state machine run on the project-level alert, unchanged.
- **Guarantee:** Holm controls the family-wise error rate under arbitrary dependence, given valid p-values. That holds even though relationship and marginal tests are correlated. FWER is measured in L.A.
- **Family-size cap:** m_max = 200 (config). The relationship review UI shows the running family size.
- **Equivalence:** a numeric/categorical-only table with relationships disabled has exactly today's tabular family, so results are byte-identical to the calibrated tabular path (regression test).

## L. Validation strategy

**Datasets**
1. **PetFinder.my** (local, gitignored), treated as one table: Age, Fee, Quantity (numeric), Breed1, Gender, Color1, State… (categorical), Description (text), first Photo (image).
2. **A second dataset**, chosen at the M4 gate with you, because it depends on download access and licence. Candidates: Avito Demand Prediction (tabular + Russian text + images; large; Kaggle account), Inside Airbnb (CC BY licensing of the data; photos would need downloading), Yelp Open Dataset (academic licence). I can't confirm their current availability or licence from this machine; that's checked at the gate. You may need to download it.

**Protocol rules**
- Pre-register labels and acceptance criteria in each script's docstring before running.
- Seeded draws.
- Raw results in `results/`.
- Every number in docs comes from an actual run that I have read.

**Experiments**
- **A. A/A:** R and B disjoint, from the same pool. N ∈ {500, 2000}, n ∈ {50, 100, 300, 1000}, 200 draws per cell, full family enabled. Measure the per-test false-positive rate and the project-level FWER (target ≤ 0.05 within the Wilson interval). Also check the calibration of each statistic type (P(p < 0.05)).
- **B. Marginal-only drift** (the invariance check):
  - numeric: monotone transforms (x + δ·sd, x·c);
  - categorical: prevalence change by **stratified resampling of whole rows by category**, which keeps P(other | cat);
  - text/image: rows resampled by a property that also changes the column's marginal.
  - **Expected:** the right column flagged, and relationship tests **not** flagged in the monotone and stratified-resampling cases. Key metric: relationship false-positive rate under marginal-only drift.
- **C. Relationship-only drift** (core): shuffle one column within B, for a fraction f ∈ {0.1, 0.25, 0.5, 1.0} of rows.
  - **Expected:** no marginal drift on any column (exact by construction), and relationship drift on the pairs involving that column.
  - **Measure:** detection by f, and attribution (flagged pairs ⊆ the pairs involving the shuffled column).
  - For text/image pairs, this is the probe/matching test.
- **D. Joint drift:** induce a three-way change (e.g. the Breed–Fee relationship changes only for one State). Measure the experimental joint signal and which pairwise tests catch it.
- **E. Attribution:** per alerting batch, check whether the flagged set is a subset of the truly affected tests (precision), whether it contains at least one (recall), and the exact-match rate.

**Metrics:** precision / recall / F1 (at test level and batch level), FPR, detection rate by effect size, attribution accuracy, and **detection latency**, defined as batches-until-alert in a seeded stream with a change point at batch t, under the default k-of-m policy.

**Reporting labels:** *formally validated* (A–E with ≥ 100 draws per cell on ≥ 1 dataset), *smoke-tested* (fewer draws or a single scenario), *experimental* (implemented, not validated or not passing). Probes and text↔image are labelled by what is actually achieved.

**Go / no-go criteria (pre-declared)**
- A feature ships **on by default for new table projects** only if:
  - A shows FWER ≤ 0.05 within the interval with it enabled;
  - B shows its false-positive rate under marginal-only drift ≤ 0.05 within the interval;
  - C shows non-trivial detection at f = 1.0.
- Otherwise it ships **off by default / experimental**, with the measured numbers.
- Existing projects never change defaults.

## M. Security strategy

**Authorization:** every `/tables` and `/jobs` route uses `verify_project_access` / owner checks. Stages and jobs are visible only to their owner. Each gets a two-user isolation test, including guessing another user's `stage_id`/`job_id` (→ 404, not 403, so existence isn't revealed).

**Image ZIP handling (`ingest/images.py`)**
- **Never `extractall`.** Entries are read in memory or streamed from the stored blob.
- **Name normalization:** reject absolute paths, drive letters, `..` segments and NUL bytes; convert `\` to `/`. Ignore directories, symlinks (unix mode in `external_attr`) and encrypted entries (each reported).
- **Archive-bomb limits (configurable env vars):**
  - ≤ 20,000 entries;
  - ≤ 20 MB uncompressed per entry, enforced **while reading** with a counting reader, not by trusting header sizes;
  - ≤ 2 GB total uncompressed;
  - per-entry compression ratio ≤ 100;
  - nested zips are not expanded.
- **Filename association:** exact normalized relative path first, then basename if it's unique. An ambiguous basename (the same name in two folders) is an error listing the duplicates. Case-sensitive, with a hint when a case-insensitive match exists.
- **Decoding:** `PIL.Image.MAX_IMAGE_PIXELS = 50_000_000`, with `DecompressionBombWarning` escalated to an error. Format allowlist: JPEG, PNG, GIF (first frame), BMP, WEBP, TIFF. Verify-then-load; convert to RGB (existing `_decode_image`).
- **Per-row image status:** ok | missing | corrupt | too_large | unsupported. **No invalid image is dropped silently:**
  - each is counted in `data_quality` with row ids (not values);
  - the row is excluded only from the image column's tests;
  - its other columns are still used;
  - if the invalid share exceeds the `schema_policy` threshold → a schema issue.
- **Base64 / data-URI (JSON tables):** same size cap, magic-byte check, allowlist.

**Other protections**
- **Excel embedded images:** not supported in v1. If `openpyxl` detects embedded images, the profile says so and points to filename + ZIP.
- **URLs:** values that look like `http(s)://` are profiled as `ignore` with the reason "remote image URLs are not supported". Nothing is ever fetched.
- **Upload limits:** table ≤ 200 MB (existing); zip ≤ 500 MB compressed. Requests are streamed to disk in chunks rather than read fully into memory (the table path only; existing endpoints are unchanged).
- **No raw production rows** in `analysis_runs`, `jobs.result`, logs, webhook payloads or error messages:
  - job errors are sanitized through a wrapper that strips exception text containing cell values (pandas errors often echo values);
  - a test injects a sentinel string into a table and asserts it appears nowhere in the DB tables or logs after fit and analyze.
- **Raw data retention:**
  - staged raw files live ≤ 24 h (configurable), are deleted on consumption, and an expiry sweep removes the rest;
  - profile sample values exist only inside the stage record and die with it;
  - embeddings are treated as sensitive: they're deleted with the project and version, they're never in webhooks or logs, and the README says so.
- **Probe/model artifacts** are stored as numeric arrays (npz) and loaded with `allow_pickle=False`. Never pickle.
- **Webhook signing and target validation:** unchanged. The payload only gains statistics.
- **Prerequisite (separate from this build):** rotate the leaked OAuth secrets.

## N. Performance strategy

- **Measured on this machine (16 cores) in Step 2(e):** about 15,000 descriptions embedded in ~4 min and ~14,650 images in ~6 min. Render instances are much weaker; that's documented.
- **Reference cap:** N_MAX = 5,000 rows (seeded sample, recorded) for the row-aligned store and the embeddings. Per-column KS/PSI keep today's behavior.
- **Streaming:** images are decoded and embedded in chunks of 64; there's never a full in-memory list of decoded images. Tables are read once per job.
- **Jobs:** one in-process worker thread processes jobs **one at a time**, FIFO, so torch memory stays bounded. Jobs are claimed atomically (`UPDATE … WHERE status='queued'`).
  - **Restart recovery:** at startup, jobs left `running` become `interrupted` with a clear message.
  - Progress is reported per stage.
- **Null computation:** cheap statistics are vectorized with numpy (Spearman via ranks; IPF on ≤ 20×20×2 tables; weighted ECDFs). Expensive nulls run in parallel with joblib (`n_jobs` config, default min(4, cores)).
- **Rough per-column cost budget** for a DCT null with PCA-64: about 1–5 minutes on a desktop. To be measured in M3; if it's too slow, reduce the grid sizes, not the rigor.
- **Storage:** embeddings as float16 `.npy` (5,000 × 512 × 2 B ≈ 5 MB per image column per version); null grids as float32; the row store as Parquet.
- **Caps:** relationship proposals ≤ 25 by default; family ≤ 200; emergence screening only if ≤ 60 monitored columns. Profiling all pairs is skipped above 200 columns (with a warning).
- **Future-migration seams:** BlobStore (→ S3/GCS), the jobs table and worker (→ a real queue/worker process), the central DB helper (→ PostgreSQL), and analysis caching keyed by (version, batch sha256).

## O. Backward-compatibility strategy

- **No existing route, request model or response field** is removed or changes meaning. New fields are optional and default-valued.
- **`profile_columns()` is frozen:** golden-output tests over the existing profiler fixtures.
- **Equivalence test:** the same numeric/categorical table through `/fit/upload` + `/analyze/upload` versus `/tables` (relationships disabled) must give identical `feature_metrics` (statistic, p, adjusted p, decisions), identical `schema_report` issue types, and the same alert state transitions.
- **Migrations are self-healing:** an old `drift.db` opens and works (a test copies a pre-migration fixture DB).
- **The full existing suite runs after every phase.** A failure blocks the phase.
- **WAL enablement:** the full suite plus a concurrency test (worker writing while API reads).
- **Text/image/joint endpoints and their frontend pages** remain. Retiring them is a separate, later decision.
- **Defaults of existing projects never change.**

## P. Phase-by-phase implementation plan (with milestone gates)

Each phase lists its deliverable, its tests and its exit criterion. The full suite must pass at every exit.

### Milestone M1 — Unified table with per-column drift (Phases 0–5)

- **Phase 0 — Audit + plan.** This document. *Exit:* your approval of M1.
- **Phase 1 — Unified profile output + ingestion foundations.**
  - Deliverables:
    - WAL/busy_timeout in the DB helper;
    - `db/blob_store.py`; `jobs.py` (table, worker, recovery); `staged_uploads` + expiry sweep;
    - `ingest/images.py` with **all zip security from M** (moved here from Phase 12, so security ships with the attack surface);
    - `profile_table()` (text/image/ignore detection, confidence, evidence, alternatives);
    - `.xls` fix;
    - `POST /tables/{id}/stage`, `GET /jobs/{id}`, `DELETE /tables/stages/{id}`.
  - Tests: profiler classification (num/cat/text/image/ignore, ID ≠ text, date strings, URLs, templated short text), confidence/evidence shape, `profile_columns` golden outputs, zip association/missing/corrupt/ambiguous/case, path traversal, bomb ratio, entry/size caps, symlink/encrypted entries, base64/data-URI, stage TTL expiry and deletion, job lifecycle and restart recovery, two-user isolation for stage/job, `.xls` read.
  - *Exit:* a PetFinder-style table + zip profiles correctly end to end through the API.
- **Phase 2 — Human-in-the-loop schema review.**
  - Deliverables: confirmed-schema validation (types ∈ the 5, monitor flags, no image type without image data, etc.); `table_schemas` persistence; Upload + Column Review UI.
  - Tests: schema validation errors; proposed-vs-final persisted; audit questions answerable; nothing monitored before confirmation.
- **Phase 3 — Unified fit (per-column).**
  - Deliverables:
    - fit job: dedup over confirmed columns, reported;
    - num/cat via the existing `insert_baseline` path;
    - text/image embeddings (+PCA) to BlobStore;
    - row-aligned store (Parquet blob);
    - data-quality report; stage raw-blob deletion after success.
  - Tests: equivalence groundwork, embedding row alignment and validity mask, missing/corrupt images excluded only from their column, dedup counts, cleanup of stage blobs, no raw rows persisted (sentinel test).
- **Phase 4 — Unified baseline storage/versioning.**
  - Deliverables: `table_column_baselines`, `table_reference_rows`, `table_relationships` (empty for now), version linkage, activate, `GET /tables/{id}/baseline`, delete cascade.
  - Tests: version storage/retrieval, schema persistence across versions, activate re-materialization, cascade delete (DB + blobs), old-DB migration fixture.
- **Phase 5 — Unified production analyze (columns only).**
  - Deliverables:
    - analyze job; extended schema validation (missing/unexpected columns, type change, missing/corrupt images, invalid/empty text, null-rate, unseen categories, constant, malformed rows);
    - per-column tests (num/cat in-family, text/image legacy-decided);
    - the report model; `analysis_runs`/history/alerts/webhooks/idempotency integration;
    - Analyze + Report UI (column section).
  - Tests: **the equivalence regression test (O)**, schema issues, idempotent replay, alert transitions and webhook payload content (statistics only), isolation, malformed input.
- **M1 gate:** live demo (start the app, run the full flow in the UI), full suite, short written report. *You approve M2.*

### Milestone M2 — Relationship engine for numeric/categorical + calibration (Phases 6–9)

- **Phase 6 — Calibration framework + num↔num.**
  - Deliverables: the null-grid machinery (J) built **first**; Spearman detector (I.2); relationship storage of reference state and null.
  - Tests:
    - monotone-transform invariance;
    - a known correlation change is detected;
    - NOT_TESTED below the minimum;
    - null grid determinism;
    - conservative size bracketing;
    - p-value resolution.
- **Phase 7 — cat↔cat** (I.3: level merging, IPF, G², λ effect, explanations).
  - Tests: invariance to proportion changes, re-pairing at equal V detected, sparse/unseen handling, IPF convergence on edge tables.
- **Phase 8 — num↔cat** (I.4).
  - Tests: invariance to category-proportion and monotone numeric shifts, a relative-position change detected, small/rare/unseen categories.
- **Phase 9 — Holm family + relationship proposals/review.**
  - Deliverables: family assembly and exclusions (K); relationship proposals in `profile_table` (strength thresholds, cap 25, reasons); manual add/remove; emergence screening (I.8); Relationship Review UI + report section.
  - Tests:
    - Holm correctness on known p-vectors;
    - family membership/exclusion;
    - equivalence still holds with relationships off;
    - proposal ranking/cap;
    - persistence of accepted/rejected pairs.
- **Early validation slice:** run L.A, L.B and L.C on PetFinder numeric/categorical columns.
- **M2 gate:** you review the measured A/B/C numbers and decide whether relationship monitoring is on by default for new table projects (per the L criteria).

### Milestone M3 — Text/image relationships + real-embedding nulls (Phase 10)

- **Phase 10 — Real-embedding DCT nulls and probes.**
  - Deliverables:
    - per-project real-embedding null grids for text/image marginal tests;
    - the Gaussian-tail extension;
    - the PCA-64 vs raw comparison;
    - probes (I.5) and text↔image matching (I.6);
    - optional joint signal (I.7, off).
  - Tests: probe storage without pickle, deterministic probe fits, shuffle breaks matching, the confounding flag, tail-extrapolation flag, resource limits.
  - Then run the J calibration gate (A/A) and L.C for probes.
- **M3 gate:** you decide, from the measured numbers, whether text/image tests join the family (`embedding_tests_in_family`) and whether probes ship on, off or experimental.

### Milestone M4 — Product completion (Phases 11–15)

- **Phase 11 — Dashboard/report polish.** History relationship tab, expandable details, nav restructure.
- **Phase 12 — Security & robustness hardening.**
  - Fuzzing malformed tables/zips (seeded), a log-scrub audit, limit tuning, sentinel no-raw-rows test across every endpoint.
  - Also an explicit review of zip bombs, path traversal and resource exhaustion against the Phase 1 implementation.
- **Phase 13 — Formal validation.** L.A–E on PetFinder at full size plus the second dataset (chosen at this gate). Results in `results/` with a report.
- **Phase 14 — Backward-compatibility sweep.** Full suite, old-DB fixture, old frontend pages smoke-tested, Python client old/new.
- **Phase 15 — Documentation, deployment, cleanup.**
  - README (implemented / tested / smoke-tested / experimental / future), PROGRESS, API docs/examples, architecture notes, known limitations with measured numbers.
  - Docker: the React build served by nginx; DATA_DIR on a persistent volume; env-var limits.
  - Render notes: a persistent disk and a paid instance are needed for image-heavy tables; the free tier spins down and kills jobs.
  - Dead-code removal introduced by the migration.
  - Final verification: start locally, exercise the API endpoints and the UI flow, check alerts/history/webhooks and isolation live.
- **M4 gate:** final report with everything in the spec's "at the end, provide" list.

## Q. Risks and known weaknesses

1. **Text/image calibration may fail the gate** even with real-embedding nulls (small references, tail extrapolation). Mitigation: they stay legacy-decided and outside the family, and the README is honest about it.
2. **Probes are confounded by covariate shift.** Mitigated by the `confounded_by` flag; not eliminated.
3. **Marginal invariance is not perfect:** non-monotone marginal changes can leak into Spearman and num↔cat. Measured in L.B and documented.
4. **The exchangeability assumption:** clustered or time-ordered references give narrow nulls and false alarms. Warned, not solved.
5. **Power loss from Holm with many tests.** Mitigated by the proposal cap and the visible family size.
6. **Resource limits:** image-heavy fits on small instances; the in-process worker dies with the instance. Documented. A real queue is future work.
7. **Storage growth:** embeddings per version with no retention cap (by the earlier decision). Bytes are reported. Pruning is future work.
8. **SQLite + filesystem blobs** must be backed up together. Documented.
9. **Second dataset access/licence** may delay Phase 13.
10. **Floors are product choices** and may need per-domain tuning by users.
11. **Effort:** this is large (see S). Context continuity across many sessions relies on PROGRESS.md handoffs at every phase exit.
12. **Spearman and log-linear limits:** non-monotone dependence and higher-order (3-way) interactions aren't captured by the pairwise tests. Only the experimental joint signal sees them.

## R. Explicitly NOT implemented

- Remote image URLs (profiled as ignore; never fetched).
- Audio, video.
- Embedding-model selection / bake-off.
- Distributed infrastructure (Celery, Redis, separate worker service, Kubernetes).
- Nonlinear or deep joint models.
- Automatic monitoring of all pairs (only proposals + confirmation; screening is informational).
- Persisting raw data beyond the staged TTL; persisting profile sample values.
- Excel embedded-image extraction (v1).
- Expanded point-OOD features.
- Automatic baseline-version pruning.
- The PostgreSQL migration itself (seams only).
- Retiring or deprecating the existing text/image/joint endpoints or the Streamlit process.
- Changing defaults of existing projects.
- Alerts driven by emergence screening or the experimental joint signal.

## S. Estimated complexity by phase

Rough, for one engineer working focused; M = milestone.

| Phase | Size | Days (est.) |
|---|---|---|
| 0 Audit + plan | S | done |
| 1 Profile + ingestion foundations (jobs, blobs, staging, zip security) | L | 5–6 |
| 2 HITL schema review (API + UI) | M | 3 |
| 3 Unified fit | L | 4–5 |
| 4 Storage/versioning | M | 3 |
| 5 Unified analyze + report (columns) | L | 4–5 |
| **M1 total** | | **~19–22** |
| 6 Calibration framework + Spearman | M | 3–4 |
| 7 cat↔cat log-linear | M | 3–4 |
| 8 num↔cat composition-adjusted | M | 3 |
| 9 Holm family + proposals + review UI + early validation slice | M | 4–5 |
| **M2 total** | | **~13–16** |
| 10 Real-embedding nulls + probes + matching + joint (exp.) + gate runs | L | 7–9 |
| **M3 total** | | **~7–9** |
| 11 Dashboard polish | M | 3 |
| 12 Hardening | M | 3 |
| 13 Formal validation (2 datasets) | L | 5–7 (+ data acquisition) |
| 14 Compatibility sweep | S | 2 |
| 15 Docs/deploy/cleanup | M | 3 |
| **M4 total** | | **~16–18** |
| **Overall** | | **~55–65 working days** |

This is higher than the 5–7 weeks I quoted before your full specification. It now includes the job system, staging, archive security, real-embedding calibration and two-dataset validation.
