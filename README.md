# Drift Monitoring API

> Detect when production data stops looking like your training data — in individual columns *and* in how columns relate to each other — for tables that mix numbers, categories, free text and images.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-009688.svg?style=for-the-badge)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg?style=for-the-badge)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge)](https://www.docker.com/)

**Live demo:** [drift-monitoring-dashboard.onrender.com](https://drift-monitoring-dashboard.onrender.com/). It runs on Render's free tier, so the first load can take 30–60 s. It may still be running an older version until it's redeployed from this repository.

---

## What problem this solves

Models degrade quietly. A model validates well and ships, then weeks later the incoming data has shifted. Nobody notices until business metrics drop.

This project locks a **baseline** from your training data and compares every production **batch** against it. It flags statistically meaningful changes, says exactly which columns or column relationships changed, and can alert by email and signed webhooks.

It separates four kinds of change, because "no column drifted" does **not** mean "no data drift":

| Term | Meaning | Checked by |
|---|---|---|
| **Marginal (column) drift** | One column's own distribution changed | KS, PSI, embedding classifier test |
| **Relationship drift** | How two columns relate changed, beyond their own changes | Spearman, log-linear, conditional-rank, probes, image↔text matching |
| **Joint drift** | The combination of all columns changed | Optional / experimental (see [Limitations](#known-limitations)) |
| **Point anomaly** | One row is unusual vs. the reference | IQR fences (`/predict`) |

Wording note: the system says an input is *out-of-distribution relative to the configured reference*. It cannot know what a black-box model was actually trained on.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Supported data](#supported-data)
3. [Quick start](#quick-start)
4. [Usage](#usage)
5. [API reference](#api-reference)
6. [Detectors in detail](#detectors-in-detail)
7. [History, schema checks, versioning, alerting](#history-schema-checks-versioning-alerting)
8. [Validation results](#validation-results)
9. [Architecture](#architecture)
10. [Project structure](#project-structure)
11. [Security and privacy](#security-and-privacy)
12. [Deployment](#deployment)
13. [Testing](#testing)
14. [Known limitations](#known-limitations)
15. [Future plans](#future-plans)
16. [Tech stack](#tech-stack)
17. [License and about](#license-and-about)

---

## How it works

**One table = one monitoring project.** A table can mix four column types. Image columns hold filenames that match files in an uploaded ZIP, or base64 / data-URI values in JSON.

```
Upload training table (+ optional image ZIP)
        │
        ▼
1. PROFILE ─────── each column: proposed type (numeric / categorical / text / image / ignore),
        │          monitor yes/no, confidence, evidence, reason; plus proposed relationships
        ▼
2. HUMAN REVIEW ── you confirm or change every type, monitor flag and relationship
        │
        ▼
3. FIT ─────────── baseline version stored: confirmed schema, per-column baselines,
        │          embeddings (text/image), row-aligned reference rows, relationship models
        ▼
4. ANALYZE ─────── production batch: schema checks → column tests → relationship tests
        │          → one Holm-corrected decision → report
        ▼
5. HISTORY, ALERT STATE MACHINE, EMAIL, SIGNED WEBHOOKS
```

Nothing is monitored until you confirm it. Heavy steps (profiling, embedding, fitting, analysis) run as **background jobs**: the API returns a `job_id` immediately and you poll `GET /jobs/{job_id}`.

**Every decision is statistical and corrected for multiple testing.** Each test produces a p-value:
- **Column tests:** analytic for KS; a bootstrap for PSI.
- **Relationship and embedding tests:** a null distribution built from *this project's own reference data*, using disjoint splits at the batch's size.

All tests in a batch form one **Holm family**. That controls the chance of *any* false alarm in the batch, even as more columns and relationships are added. A test flags drift only if it's significant after correction **and** its effect is at least a *materiality floor*, so tiny but statistically significant changes on huge batches don't alarm.

---

## Supported data

| Item | Supported |
|---|---|
| Table formats | CSV / TXT (encoding and delimiter auto-detected), TSV, Excel (`.xlsx`, `.xls`), JSON, JSON Lines / NDJSON, Parquet, ARFF, libsvm `.dat`, gzip or zip-compressed CSV |
| Column types | numeric, categorical, text, image, ignore |
| Images | JPEG, PNG, GIF (first frame), BMP, WEBP, TIFF, via filenames plus a ZIP, or base64 / data URIs in JSON |
| Limits (defaults) | table ≤ 200 MB; image ZIP ≤ 500 MB, ≤ 20,000 entries, ≤ 2 GB uncompressed, ≤ 20 MB and ≤ 50 MP per image |
| Text model | `all-MiniLM-L6-v2` (384-dim) |
| Image model | `resnet18`, penultimate layer (512-dim) |

**Not supported:** remote image URLs (they're profiled as `ignore` and never fetched), Excel-embedded images, audio, video, streaming per-row drift, and choosing other embedding models.

The older **per-modality workflows** still work unchanged for existing projects: tabular, text-only, image-only and joint. See [API reference](#api-reference).

---

## Quick start

**Prerequisites:** Python 3.12+, Node 20+, and a Google Cloud OAuth client ID (for dashboard sign-in).

```bash
git clone https://github.com/Abhinavbilla/Drift-Monitoring-api.git
cd Drift-Monitoring-api
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
COOKIE_KEY=a-long-random-secret          # signs session tokens
# optional
GOOGLE_CLIENT_SECRET=...                 # only for the legacy Streamlit dashboard
FRONTEND_URL=http://localhost:5173       # allowed CORS origins, comma-separated
ALERT_EMAIL=... ALERT_PASSWORD=...       # SMTP for drift emails (otherwise printed to the log)
DRIFT_DATA_DIR=data/blobs                # where staged uploads and embeddings are stored
```

**Backend** (database tables are created automatically on startup):

```bash
uvicorn main:app --port 8000
```

**Frontend** (React, in a second terminal):

```bash
cd frontend
cp .env.example .env                     # set VITE_API_BASE_URL and VITE_GOOGLE_CLIENT_ID
npm install
npm run dev                              # http://localhost:5173
```

Interactive API docs: `http://localhost:8000/docs`. On Windows, if `npm` fails with `spawn C:\...\bin ENOENT`, check that the `ComSpec` environment variable points to `C:\Windows\System32\cmd.exe`.

**Access token for scripts** (no browser login needed):

```bash
python scripts/create_token.py create --email you@example.com --name ci   # prints dm_<prefix>_<secret> once
```

---

## Usage

### Dashboard

Sign in with Google, open or create a project, and go to **Table Monitoring**:
1. Upload the training table, and an image ZIP if a column names image files.
2. Review the columns: change any proposed type and toggle monitoring. Expand **Evidence** to see why each type was proposed.
3. Review the relationships: proposed pairs are pre-selected. Add or remove pairs; the page shows how many tests share the false-alarm budget.
4. **Lock baseline.**
5. Upload a production batch to get the report: overall status, column drift, relationship drift (before → after), and data issues.

The **History**, **Baselines**, **Webhooks** and **Audit Log** pages cover the rest. The per-modality pages are under *Legacy*.

### Python client (`clients/python`)

```python
from drift_monitor_client import DriftClient

client = DriftClient("http://localhost:8000", token="dm_...")
profile = client.profile_table("pets", train_df, images_zip="train_images.zip")
for col in profile["columns"]:
    print(col["name"], col["proposed_type"], col["confidence"], col["reason"])

fit = client.fit_table("pets", profile)          # accepts the proposals; pass columns=/relationships= to edit
report = client.analyze_table("pets", batch_df, images_zip="batch_images.zip")
print(report["overall"]["status"], report["overall"]["triggered_by"])
```

The client also wraps the tabular `fit` / `analyze` endpoints, with large-frame Parquet upload and `idempotency_key=`.

### HTTP

```bash
TOKEN=dm_...
# 1. stage + profile
curl -s -H "Authorization: Bearer $TOKEN" -F file=@train.csv -F images=@images.zip \
     http://localhost:8000/tables/pets/stage                 # -> {"job_id": ..., "stage_id": ...}
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/jobs/<job_id>   # poll until "succeeded"

# 2. fit with the confirmed schema
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"stage_id": "<stage_id>",
          "columns": [{"name": "Age", "type": "numeric", "monitor": true},
                      {"name": "Breed", "type": "categorical", "monitor": true},
                      {"name": "Description", "type": "text", "monitor": true},
                      {"name": "Photo", "type": "image", "monitor": true},
                      {"name": "PetID", "type": "ignore", "monitor": false}],
          "relationships": [{"col_a": "Description", "col_b": "Breed"}]}' \
     http://localhost:8000/tables/pets/fit

# 3. analyze a batch (Idempotency-Key optional)
curl -s -H "Authorization: Bearer $TOKEN" -H "Idempotency-Key: batch-42" \
     -F file=@batch.csv -F images=@batch_images.zip http://localhost:8000/tables/pets/analyze
```

**Report shape** (abridged):

```json
{
  "overall": {"status": "DRIFT", "alert": true, "triggered_by": ["Description<->Breed"]},
  "column_drift": {"Age": {"type": "numeric", "test": "KS", "status": "STABLE", "p_value_adjusted": 1.0}},
  "relationship_drift": {"Description<->Breed": {"kind": "probe", "statistic_name": "balanced accuracy",
                          "reference_value": 0.89, "current_value": 0.48, "status": "DRIFT"}},
  "schema_report": {"Photo": [{"issue": "invalid_images", "counts": {"missing": 1}, "severity": "warn"}]},
  "family": {"method": "holm", "members": ["Age", "Breed", "Description", "Photo", "Description<->Breed"]}
}
```

---

## API reference

All endpoints take `Authorization: Bearer <token>`: either a session token from the dashboard's Google login, or a personal access token (optionally scoped to project IDs; see `auth/tokens.py`). Full schemas are at `/docs`.

### Table monitoring

| Endpoint | Method | Purpose |
|---|---|---|
| `/tables/{project_id}/stage` | POST | Multipart `file` + optional `images` ZIP. Starts a profile job. Staged files are deleted after the fit, or after 24 h |
| `/jobs/{job_id}` | GET | Status, progress, and result (profile, fit summary, or analysis report). Owner only |
| `/tables/{project_id}/fit` | POST | `{stage_id, columns, relationships, calibration_config?, schema_policy?, alert_policy?, model_version_label?}`: locks a new baseline version |
| `/tables/{project_id}/analyze` | POST | Multipart `file` + optional `images`; optional `baseline_version`, `Idempotency-Key` |
| `/tables/{project_id}/baseline` | GET | Confirmed schema of a version: proposed vs chosen type, who decided, monitored columns, relationships |
| `/tables/stages/{stage_id}` | DELETE | Discard a staged upload's raw files now |

### Operations (all project types)

| Endpoint | Method | Purpose |
|---|---|---|
| `/history/{project_id}` | GET | Analyses (statistics only) with `since` / `until` / `feature` / `alert_only` filters, pagination, per-feature time series, relationship results |
| `/baselines/{project_id}` | GET | All baseline versions, newest first |
| `/baselines/{project_id}/activate` | POST | Make an older version active again |
| `/baseline/{project_id}` | GET | Active baseline's fences, feature types and modality |
| `/webhooks/{project_id}` | POST / GET | Register (secret returned once) / list webhooks |
| `/webhooks/{project_id}/{webhook_id}` | DELETE | Remove a webhook |
| `/projects` | GET | Your projects |
| `/projects/{project_id}` | DELETE | Delete a project, including its stored files and embeddings |
| `/logs/{project_id}` | GET | Recent `/predict` logs |
| `/health/{project_id}` | GET | Burst check over recent point anomalies |
| `/predict/{project_id}` | POST | Real-time single-row IQR anomaly check |
| `/profile` | POST | Profile a JSON table without fitting (tabular profiler) |
| `/auth/google` | POST | Exchange a Google ID token for a session token |

### Per-modality endpoints (legacy, unchanged)

| Endpoint | Purpose |
|---|---|
| `/fit/{project_id}`, `/fit/{project_id}/upload` | Tabular baseline (JSON / file upload) |
| `/analyze/{project_id}`, `/analyze/{project_id}/upload` | Tabular analysis (supports `Idempotency-Key`, `baseline_version`) |
| `/fit` and `/analyze/{project_id}/text`, `/image`, `/joint` | Text-only, image-only and joint (tabular + text + image per record) workflows |
| `/models/{model_id}` (DELETE) | Deprecated alias of `DELETE /projects/{id}` |

---

## Detectors in detail

### Column (marginal) tests

| Column type | Test | Effect / materiality floor |
|---|---|---|
| Numeric | Two-sample Kolmogorov–Smirnov | KS D ≥ 0.05 |
| Categorical | PSI with a parametric-bootstrap p-value | PSI ≥ 0.2 |
| Text / image | Domain Classifier Test (5-fold cross-validated logistic regression separating reference and batch embeddings, PCA-64), with a p-value from this project's own reference-embedding null (200 disjoint splits, Gaussian tail beyond the draws) | AUC ≥ 0.55 |

### Relationship tests

Each test is built so that a change in either column's *own* distribution does not register as relationship drift.

| Pair | Statistic | Why it ignores marginal changes | Floor |
|---|---|---|---|
| numeric ↔ numeric | change in Spearman correlation | rank-based: monotone shifts and rescaling leave it unchanged | Δρ ≥ 0.10 |
| categorical ↔ categorical | G² of the log-linear model with no three-way interaction: same odds ratios in reference and batch, each with its own marginals | odds ratios don't depend on category proportions; also catches re-pairing at equal strength | interaction RMS ≥ log 1.25 |
| numeric ↔ categorical | weighted shift in each category's mean mid-rank PIT, with the batch re-weighted to the reference category mix | a common monotone numeric shift and a category-mix change both cancel; mid-ranks keep it stable under ties | 0.05 |
| text/image → categorical | **probe**: balanced accuracy of a classifier fitted on the reference | balanced accuracy ignores class-prior shift | drop ≥ 0.10 |
| text ↔ image | **matching**: ridge map image → text; AUC of own-text vs other-text similarity | measures pairing only | drop ≥ 0.10 |
| text/image → numeric | probe (Spearman of ridge predictions) | **report-only:** shown but never alerts, because it failed a population-mix check | drop ≥ 0.10 |

**Null distributions** come from disjoint splits of the reference at the batch's size. **Proposals:** the profiler proposes pairs that are clearly related in the reference: up to 25 numeric/categorical pairs and 15 embedding pairs. Weaker pairs are listed so you can add them. An informational screen also lists *unwatched* pairs that became strongly related in production. It never alerts.

**Profiler** (`utils/profiler.py`): numeric/categorical/ignore proposals come from the long-standing column profiler (numeric coercion, cardinality, dominant value, monotonic or index columns, timestamps, structured ID patterns). On top of that it detects:
- **image:** filenames matching the ZIP, or embedded images;
- **text:** high uniqueness, multi-word, substantial length;
- **ignore:** URLs, identifiers and date strings.

`confidence` is a heuristic agreement score, not a calibrated probability.

---

## History, schema checks, versioning, alerting

- **History** (`GET /history/{project_id}`): every analysis is recorded, statistics only, never raw rows. It includes per-feature time series and relationship results.
- **Idempotency:** an `Idempotency-Key` replayed with the same data returns the earlier result; with different data it returns `409`. Keys expire after 7 days. Table analyses key on a hash of the uploaded files.
- **Schema report:**
  - flags missing or unexpected columns, dtype changes, null-rate increases, unseen categories and constant columns;
  - for table projects, also missing or corrupt images and empty text.
  
  Each issue's severity comes from a per-project `schema_policy`. Valid columns are always analyzed; nothing is dropped silently.
- **Baseline versioning:** every fit creates a new version, and old ones are kept. You can activate any version, or analyze against a specific one.
- **Sustained alerts:** a per-project `{k, m}` policy (default 1 of 1). `alert_state` changes (`opened`, `still_open`, `resolved`) are logged only on transitions.
- **Webhooks:**
  - HMAC-SHA256 signed (`X-Drift-Signature-256`), fired on `opened` / `resolved`; `still_open` is opt-in;
  - retried with 1 / 4 / 16 / 64 s backoff, up to 5 attempts;
  - URLs resolving to private, loopback or link-local addresses are rejected (SSRF guard);
  - the payload carries statistics only.

  ```python
  import hashlib, hmac
  def verify(secret: str, raw_body: bytes, header: str) -> bool:
      expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
      return hmac.compare_digest(expected, header)
  ```
- **Duplicate rows:** rows identical in every column are removed only when the table has an identifier-like column (non-float, complete, unique across the distinct rows, 20+ of them). Without one they're kept and reported as `identical_rows_kept`, because they may be different records that share values.
- **Fit-time warnings:**
  - the minimum detectable effect for the reference size;
  - a Dvoretzky–Kiefer–Wolfowitz check that the KS floor is resolvable at this reference size;
  - fewer than the recommended number of samples.
- **Project isolation:** project IDs are namespaced per owner. Another user's project, stage or job always returns `404`, so it can't be detected.

---

## Validation results

Every number here comes from a run whose raw output is committed under `results/`, reproducible with the script named. Ground truth for the unstructured and relationship experiments is **defined by construction**: for example, cats mixed into a dog reference, or a column shuffled to break its pairings. Labels and scenarios were fixed in each script's docstring before the run.

| What | Dataset | Headline |
|---|---|---|
| Tabular column drift | NYC Citi Bike 2016 (4.5 M rows) | precision 1.000, recall 0.939, F1 0.969 on real production batches |
| Text/image column drift (old synthetic-grid calibration) | PetFinder.my | conservative: recall ~0.68; synthetic p-values over-confident (led to the fix below) |
| Real-embedding calibration | PetFinder.my | false alarms 1–5% (target 5%); 25%-off-population batches detected 100% vs 6–10% before |
| Relationship drift (numeric/categorical) | PetFinder.my | no-change alarms 0–1%; a shuffled column detected 100%; its own column never flagged |
| Full table (all column types, one Holm family) | PetFinder.my + Inside Airbnb Edinburgh | PetFinder: 0% no-change alarms, shuffles detected 99–100%; Airbnb: 8% no-change alarms (clustered listings), shuffles 54–100% |

### Tabular: NYC Citi Bike 2016

Four-part methodology (`tests/test_drift_engine.py`, data split by `scripts/split_citi_bike.py`). Ground truth was computed independently with `scipy.stats.ks_2samp` and PSI. Chi-square was not used: at 4.5 M rows it flags negligible shifts.

| Synthetic shift | Detection |
|---|---|
| 0.5σ / 1.5σ / 3.0σ | 100% (50/50) at each level |

| Real production batches (Apr–Dec vs Jan–Mar 2016) | Score |
|---|---|
| Precision / Recall / F1 / Accuracy | 1.000 / 0.939 / 0.969 / 0.96 |

| Feature | Method | Effect | Detection |
|---|---|---|---|
| `trip_duration` | KS | 0.0924 | 100% |
| `month` | PSI | 16.27 | 100% |
| `pickup_longitude` | KS | 0.0219 | 100% |
| `dropoff_latitude` | KS | 0.0195 | 100% |
| `dropoff_longitude` | KS | 0.0206 | 87% |
| `pickup_latitude` | KS | 0.0189 | 53% |
| `gender_id` | PSI | 0.043 | 0%, correctly stable (below the 0.2 floor) |

| Batch size | 1,000 | 3,000 | 10,000 | 20,000 | 50,000 |
|---|---|---|---|---|---|
| Recall | 0.458 | 0.729 | 0.812 | 0.896 | 0.917 |
| F1 | 0.629 | 0.843 | 0.886 | 0.945 | 0.957 |

For this dataset, 15,000–20,000 rows per batch sits at the knee of the curve. Detection latency, the share of a batch that must be drifted before an alert fires: `trip_duration` 10%, `dropoff_longitude` 10%, `dropoff_latitude` 20%, `pickup_longitude` 30%, `pickup_latitude` 30%.

The validation also found two real bugs, both fixed before these numbers were recorded:
- **Category key types:** categorical keys were stored as strings but arrived as integers, so lookups silently said "no drift".
- **Ground truth:** the chi-square ground truth mislabeled `gender_id`.

### Text/image column drift with the old synthetic-grid p-value (PetFinder.my)

Scripts: `scripts/step2e_*.py`; raw draws in `results/step2e_text_image_raw.json`.
- **Setup:** dogs as the reference, cats mixed into batches as drift; reference sizes 100 / 500, batch sizes 40–1,000.
- **Legacy AUC > 0.65 rule:** false alarms 0–2%, but recall only about 0.68 (text precision 0.998 / recall 0.678; image 0.994 / 0.679).
- **Calibration:** the synthetic-Gaussian p-value grid was over-confident on real embeddings, with up to 14.5% false alarms where 5% was the target.
- **Tiny references** (20 vs 20) false-alarmed 10–14%.

These findings motivated the real-embedding null used today. That grid is still what the *legacy* text/image endpoints use when calibrated mode is enabled.

### Real-embedding calibration, probes and matching (PetFinder.my)

Script: `scripts/validate_m3_embeddings.py`; results in `results/m3_embedding_validation.json`.

- **PCA-64 vs raw embeddings:** equal or better detection, and 0% false alarms on unchanged data for both.
- **Calibration:** with no drift, P(p < 0.05) was 1–5% in every text/image cell (target 5%). Batches that were 25% cats were detected **100%** of the time, vs 6–10% for the old AUC > 0.65 rule.
- **Probes and matching** (alarm rates):

| Test | No change | Category mix shifted | Pairing shuffled |
|---|---|---|---|
| text→Type, image→Type (categorical) | 0–2% | 0% | 100% |
| text ↔ image matching | 0% | 0% | 100% |
| text→Age (numeric) | 6% | **26%** | 100% |

The numeric probe is fooled by population-mix shifts, so it's report-only.

### Relationship drift, numeric/categorical (PetFinder.my)

Script: `scripts/validate_table_relationships.py`; results in `results/m2_relationship_validation_raw.json`. Reference 2,000 rows, 100 draws per cell, batches of 300 / 1,000.

| Scenario | Relationship alarm | Column alarm on the changed column |
|---|---|---|
| No change | 0% / 1% | — |
| Monotone transform of the numeric columns in relationships | 2% / 1% | 100% / 100% |
| Category mix shifted | 3% / 1% | 100% / 100% |
| A column shuffled across all rows | **100% / 100%** | 0% / 0% |
| 25% of rows shuffled | 3% / 2% | 0% / 0% (effect ~0.03 is below the 0.05 floor, by design) |

### Full table: all column types in one Holm family

Script: `scripts/validate_table_formal.py`; results in `results/m4_formal_validation.json`.
- **Setup:** reference 1,500 rows, batch 300.
- **PetFinder:** 100 draws per scenario; numeric, categorical, text and image columns; 7 profiler-proposed relationships.
- **Inside Airbnb Edinburgh** ([CC BY 4.0](https://insideairbnb.com/get-the-data/)): 50 draws per scenario; numeric, categorical and text columns; 12 relationships. Its photos are remote URLs, which the product never fetches.

| Scenario | PetFinder: any alarm / relationship alarm | Airbnb: any alarm / relationship alarm |
|---|---|---|
| No change | 0% / 0% | 8% / 2% |
| Monotone transform of numeric columns | 100% / 0% | 100% / 8% |
| Category mix shifted | 100% / 1% | 100% / 4% |
| Numeric column shuffled | relationship 99%, column 0% | 100%, column 0% |
| Categorical column shuffled | 100%, column 0% | 100%, column 4% |
| Text column shuffled | 100%, column 0% | 54%, column 0% |
| Image column shuffled | 100%, column 0% | — |
| Numeric column shuffled within one category | 0% (see below) | 100% |

- **Attribution:** flagged relationships involved the changed column 97–100% of the time.
- **Batch-level relationship drift:** precision / recall / F1 were 0.998 / 0.80 / 0.89 on PetFinder and 0.96 / 0.89 / 0.92 on Airbnb.
- **Latency:** where detection is 99–100%, the alert comes on the first batch.

What it found:
- **PetFinder's "within one category" scenario tested nothing.** It left the only monitored relationship of that column intact, a scenario-design flaw.
- **Airbnb's no-change alarms all involve `property_type`.** Hosts with many near-identical listings make rows clustered rather than independent, which narrows the nulls.
- **Airbnb descriptions** predict property and room type only moderately.

### Earlier smoke test (PetFinder.my, per-modality endpoints)

Dogs as the reference, cats as drift, N=100:
- **Text, image and joint:** all behaved as expected. For example, text AUC was 0.607 on the same population and 0.970 on drifted data; image 0.373 / 0.990; joint 0.489 / 0.992.
- **Tabular:** one same-population false positive, `State` with PSI 0.39 at N=100. It resolved by N=300 (PSI 0.079). That's PSI's small-sample sensitivity on many-category columns.

---

## Architecture

```
Browser (React)  ──┐                           ┌── jobs.py: one background worker thread
Python client    ──┼─► FastAPI (main.py) ──────┤   (profile / fit / analyze), restart recovery,
curl / CI        ──┘    auth, isolation,       │   24 h staged-upload expiry
                        endpoints              │
                              │                └── drift/webhooks.py: delivery sweep thread
                              ▼
        drift/table_monitor.py  (profile, fit, analyze orchestration)
          ├─ utils/profiler.py           column + relationship proposals
          ├─ ingest/readers.py           every table format
          ├─ ingest/images.py            safe ZIP index, per-row image status
          ├─ adapters/text.py, image.py  embeddings
          ├─ drift/detector.py           KS / PSI, Holm family, two-gate decision
          ├─ drift/relationship_detector.py   numeric/categorical relationship tests
          └─ drift/embedding_tests.py    real-embedding null, probes, matching
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
        SQLite (db/crud.py, WAL)     db/blob_store.py (files under DRIFT_DATA_DIR)
        projects, baselines,          staged uploads, embeddings (npz, float16),
        baseline_versions, jobs,      PCA, probe models (arrays, never pickles),
        table_schemas, table_*        row-aligned reference rows (Parquet)
        relationships, analysis_runs,
        alert_events, webhooks
```

- **Baselines:** a table baseline is one baseline version. The numeric and categorical part reuses the tabular storage, so a numeric/categorical-only table gives byte-identical results to the tabular endpoints (regression-tested). Text, image and relationship artifacts are keyed by `(project, version)`.
- **Raw data:** staged uploads are the only raw data kept, at most 24 h, and deleted as soon as a fit or analysis uses them.
- **Seams for later:** `BlobStore` (→ object storage), the `jobs` table (→ a separate worker), and the central DB helper (→ PostgreSQL).

---

## Project structure

```
drift-monitoring-api/
├── main.py                     FastAPI app: auth, isolation, all endpoints, tabular analysis core
├── jobs.py                     background job queue + worker thread, stage expiry, sanitized errors
├── models.py                   Pydantic request/response models
├── dashboard.py                legacy Streamlit dashboard (kept; React is the main UI)
├── migrate.py, patched_init.py legacy helpers
├── spec.md                     original v2.0 specification
├── requirements.txt, runtime.txt
├── Dockerfile                  multi-stage: React build + backend + nginx + Streamlit (supervisord)
├── Dockerfile.dashboard, docker-compose.yml, supervisord.conf
│
├── adapters/                   base.py, tabular.py, text.py (MiniLM), image.py (resnet18), joint.py
├── auth/tokens.py              personal access tokens
├── db/
│   ├── crud.py                 SQLite schema, self-healing migrations, all queries
│   └── blob_store.py           file storage for large artifacts
├── drift/
│   ├── table_monitor.py        unified table: profile / fit / analyze jobs, report
│   ├── detector.py             KS, PSI, IQR; calibrated two-gate; shared Holm family
│   ├── calibration.py          CalibrationConfig, Holm/BH, PSI bootstrap, DKW
│   ├── relationship_detector.py  numeric/categorical relationship tests, proposals, screening
│   ├── embedding_tests.py      real-embedding null, probes, image↔text matching, PCA
│   ├── embedding_detector.py   Domain Classifier Test
│   ├── dct_calibration.py      synthetic null grid (legacy text/image endpoints)
│   ├── webhooks.py             signing, SSRF guard, delivery sweep
│   └── alerts.py               drift email
├── ingest/
│   ├── readers.py              CSV/TSV/Excel/JSON/JSONL/Parquet/ARFF/.dat/.gz/.zip
│   └── images.py               ZIP safety, filename association, image validation
├── utils/
│   ├── profiler.py             profile_columns, profile_table, duplicate-record rule
│   └── validation.py           request structure checks
│
├── frontend/                   React 19 + Vite + Tailwind
│   └── src/
│       ├── pages/              TableWorkflowPage (main), History, Baselines, Webhooks, Logs,
│       │                       Projects, Overview, Login; legacy Fit / Analyze / Predict
│       ├── components/, layout/, routes/
│       └── lib/                api.ts (client + job polling), auth, types, file helpers
├── clients/python/             DriftClient (tabular + table methods) and live-client tests
├── examples/model_serving/     end-to-end example: a served model sending batches for analysis
├── scripts/                    validation experiments (Citi Bike, step2e_*, validate_*),
│                               calibration grid builder, create_token.py
├── results/                    committed raw outputs of every validation run
├── docs/
│   ├── unified_table_plan.md   design plan (A–S) for table monitoring, milestones M1–M4
│   ├── PROGRESS.md             running engineering log / handoff notes
│   └── step2_proposal.md, recon.md, step5_recon.md
└── tests/                      31 test modules (pytest)
```

Not committed: `.env`, `google_credentials.json`, `drift.db`, `data/` (blobs), `datasets/`.

---

## Security and privacy

- **Authentication and isolation:**
  - Google sign-in issues signed session tokens; scripts use hashed, revocable personal access tokens.
  - Every table endpoint checks ownership. Other users' projects, stages and jobs return `404`.
- **No raw data in logs or history:**
  - history, job results, webhooks and alert events hold statistics only;
  - job errors are sanitized, and the server log records exception type and stack, not messages, since those can echo cell values;
  - a test plants a sentinel value and checks every table for it.
- **Uploads:**
  - streamed to disk with size caps;
  - image ZIPs are never extracted;
  - unsafe paths (`..`, absolute, drive letters), symlinks, encrypted entries and oversized or suspiciously compressed entries (archive bombs) are rejected and counted, and the per-entry cap is enforced while reading;
  - damaged entries become a per-row `corrupt` status;
  - images are decoded with a pixel cap and a format allowlist.
- **Sensitive data at rest:**
  - embeddings and staged files are treated as sensitive and deleted with the project;
  - profile sample values are shown to the owner during review and purged with the stage.
- **Outbound:**
  - no remote URL fetching;
  - webhook targets go through an SSRF guard;
  - payloads are signed.
- **Known issue:** OAuth secrets were committed to this repository's git history in the past. They must be rotated in Google Cloud; removing them from future commits isn't enough.

---

## Deployment

**Docker (single container).** `Dockerfile` builds the React app and runs FastAPI, nginx and the legacy Streamlit dashboard under supervisord:
- `/`: the React app;
- `/api/`: the API;
- `/streamlit/`: the old dashboard.

```bash
docker build --build-arg VITE_GOOGLE_CLIENT_ID=<client-id> -t drift-monitoring .
docker run -p 80:80 --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/drift.db:/app/drift.db drift-monitoring
```

nginx accepts uploads up to 710 MB. The Dockerfile has **not been build-tested** in the development environment (no Docker there). The frontend production build (`npm run build` with `VITE_API_BASE_URL=/api`) was verified.

**Render.** The current live demo predates the React frontend. To redeploy, use `Dockerfile` as a web service and set:
- `GOOGLE_CLIENT_ID` and `COOKIE_KEY`;
- the build argument `VITE_GOOGLE_CLIENT_ID`;
- optionally `FRONTEND_URL`.

For table monitoring:
- **Persistent disk** at `/app/data`, plus `drift.db`. Back them up together.
- **Background jobs** run inside the API process. A free-tier instance that sleeps kills running jobs; they're marked `interrupted` on restart and must be resubmitted.
- **CPU and memory:** on a 16-core desktop, about 15,000 descriptions took ~4 min and ~14,650 photos ~6 min to embed. Small instances will be much slower.
- **Limits (env vars):** `DRIFT_MAX_ZIP_UPLOAD_BYTES`, `DRIFT_MAX_ZIP_ENTRIES`, `DRIFT_MAX_ZIP_UNCOMPRESSED`, `DRIFT_MAX_IMAGE_BYTES`, `DRIFT_MAX_IMAGE_PIXELS`.

---

## Testing

```bash
HF_HUB_OFFLINE=1 python -m pytest tests -q        # 363 tests; the offline flag avoids model-hub update checks
```

Highlights:
- **Table monitoring:**
  - `test_table_unified.py`: end-to-end jobs, equivalence with the tabular path, isolation, idempotency, no raw values persisted, duplicates;
  - `test_table_hardening.py`: fuzzed tables and ZIPs, damaged entries, limits;
  - `test_relationship_detector.py`: invariance to marginal changes, re-pairing, ties, Holm membership;
  - `test_embedding_tests.py`: null calibration, probes, matching.
- **Existing features:** history, idempotency, schema report, versioning, alert state machine, webhooks (including signing and SSRF), cross-user isolation, ingestion robustness, calibration, the profiler.
- **Live-client tests** in `clients/python/tests/` need a running server.

**Reproducing validation runs:** the datasets are not committed. Download PetFinder.my (Kaggle) to `datasets/petfinder/` and the Inside Airbnb Edinburgh `listings.csv.gz` to `datasets/airbnb/`. Then run the scripts named in [Validation results](#validation-results). `scripts/step2e_embed_petfinder.py` caches the PetFinder embeddings first.

---

## Known limitations

- **Exchangeability:** the null distributions assume reference rows are roughly independent. Clustered or time-ordered data (many near-identical listings per host, panel data) narrows them and raises false alarms. On Airbnb this gave 8% no-change alarms, all involving one high-cardinality column.
- **Relationship coverage:**
  - Spearman misses non-monotone dependence;
  - the numeric↔categorical test tracks relative position, not spread within a category;
  - only pairs are tested, not three-way interactions;
  - probes trained on the reference can't detect a relationship that appears only in production (the emergence screen partly covers numeric/categorical pairs).
- **Probes:** probes into numeric columns are report-only, since they're fooled by population-mix shifts. A probe whose source column also drifted is flagged `confounded_by`.
- **Small references:** tiny references (about 20 rows) give unreliable embedding tests. PSI false-alarms on many-category columns at small batch sizes; prefer 300+ rows when categorical columns have many levels.
- **Fixed embedding models:** `all-MiniLM-L6-v2` and `resnet18` are general-purpose. Specialized domains (medical images, technical text) may separate worse. There's no model selection.
- **One process, SQLite and local files:** this suits moderate traffic. Background jobs die with the instance, and scaling out needs PostgreSQL, object storage and a separate worker.
- **No retention cap:** baseline versions and their embeddings are kept indefinitely.
- **Validated on two datasets:** pet listings (with images) and Airbnb listings (text, no images). Drift is defined by construction (population mixing, shuffling), so this is not a guarantee for other domains or other kinds of drift.
- **Legacy joint endpoint:** it detects modality-pairing breaks only near the effect size it was tuned on. Its earlier AUC figures were measured with a mis-specified classifier and haven't been re-measured. The unified table path does not use it; its whole-row joint signal was optional in the plan and not built.
- **Text/image suite not run to completion:** the 20-Newsgroups / CIFAR-10 four-part suite (`tests/test_embedding_validation.py`) was verified for logic but never run to completion, so it has no published numbers.

---

## Future plans

- An optional whole-row joint signal (experimental, off by default)
- Making nulls robust to clustered data (block or cluster-aware resampling)
- Baseline-version retention and pruning
- PostgreSQL, object storage and a separate job worker for scale
- Remote image URLs behind an SSRF-guarded fetcher
- A webhook delivery-history endpoint and a "send test event" action
- Wasserstein distance as an extra numeric metric; rolling-window baselines
- Audio and video via embedding-based tests

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | React 19, TypeScript, Vite, Tailwind, TanStack Query, Recharts |
| Legacy dashboard | Streamlit |
| Storage | SQLite (WAL) + filesystem blob store (npz, Parquet) |
| Statistics | scipy, scikit-learn, numpy, pandas |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), torchvision (`resnet18`) |
| Auth | Google Identity Services, PyJWT session tokens, hashed personal access tokens |
| Packaging | Docker (multi-stage), nginx, supervisord, docker-compose |

---

## License and about

MIT. A `LICENSE` file has not been added to the repository yet.

Built by **Abhinav Billa**, B.Tech Mathematics and Computing, Indian Institute of Science (IISc), Bangalore.

This started from a paper on out-of-distribution detection and statistical process control. Every stage followed the same discipline: check the current code before changing it, write the plan before the code, validate with pre-registered experiments, and correct the docs when a finding turns out narrower than hoped. In order, roughly:
1. A tabular engine (KS, PSI, IQR), validated on 4.5 M Citi Bike rows.
2. Text, image and joint workflows via the Domain Classifier Test.
3. Calibrated decisions and Holm correction.
4. Hardening and PetFinder validation.
5. History, idempotency, versioning, alert state machine and signed webhooks.
6. A React frontend replacing Streamlit.
7. **Unified table monitoring** (plan: `docs/unified_table_plan.md`): mixed column types, a human-confirmed schema, relationship drift, real-embedding calibration, and two-dataset validation.

**GitHub:** [Abhinavbilla](https://github.com/Abhinavbilla)
