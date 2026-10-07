# Drift Monitoring API

> A REST API for detecting distribution shifts in machine learning features — before they silently degrade your models in production.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge)](https://drift-monitoring-dashboard.onrender.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1-009688.svg?style=for-the-badge)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B.svg?style=for-the-badge)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge)](https://www.docker.com/)

**[→ Try the live dashboard](https://drift-monitoring-dashboard.onrender.com/)**

---

## What Problem This Solves

Most ML teams catch model degradation too late — after it's already affecting real users and business metrics. The usual pattern looks like this:

1. Train a model → strong validation performance
2. Deploy to production → everything looks fine
3. Weeks or months pass → underlying data silently shifts
4. Model performance erodes → nobody notices until the damage is done

Drift Monitoring API addresses this by continuously comparing your live production data against a locked baseline distribution. When a feature's distribution shifts beyond a statistically meaningful threshold, the system flags it — before it becomes a business problem.

---

## Table of Contents

1. [Live Demo](#live-demo)
2. [Quick Start for API Users](#quick-start-for-api-users)
3. [Scope and Supported Data](#scope-and-supported-data)
4. [Key Features](#key-features)
5. [How It Works](#how-it-works)
6. [Validation Results](#validation-results)
7. [Tech Stack](#tech-stack)
8. [Project Structure](#project-structure)
9. [Installation](#installation)
10. [Deployment](#deployment)
11. [Usage](#usage)
12. [API Reference](#api-reference)
13. [History, Schema Validation, Versioning & Alerting](#history-schema-validation-versioning--alerting)
14. [Testing and Validation Methodology](#testing-and-validation-methodology)
15. [Known Limitations](#known-limitations)
16. [Future Plans](#future-plans)
17. [License](#license)
18. [About](#about)

---

## Live Demo

The dashboard is deployed on Render and open to try:

**[https://drift-monitoring-dashboard.onrender.com/](https://drift-monitoring-dashboard.onrender.com/)**

> Note: The free Render tier spins down after inactivity. The first load may take 30–60 seconds to wake up.

---

## Quick Start for API Users

The primary way to use this project is through the [live dashboard](https://drift-monitoring-dashboard.onrender.com/) — sign in with Google and you're authenticated for every action the UI exposes (locking baselines, analyzing batches, deleting models). There's no API key to generate or manage: the dashboard mints a short-lived session token from your Google login automatically, behind the scenes.

> **Programmatic access (outside the dashboard) currently has no self-serve credential flow.** Since auth is derived directly from Google login rather than a static key, there's no `/register`-style endpoint to hand you a long-lived credential to embed in your own script. See [Known Limitations](#known-limitations).

### Locking a Baseline and Analyzing Batches

Both happen through the dashboard's UI: upload training data to lock a baseline (`/fit` under the hood), then upload a production batch to check for drift (`/analyze` under the hood). The endpoints themselves are documented at `/docs` if you want to see their request/response shapes.

**[https://drift-monitoring-dashboard.onrender.com/docs](https://drift-monitoring-dashboard.onrender.com/docs)**

---

## Scope and Supported Data

Drift Monitoring API covers four input modalities through one shared adapter/baseline/dashboard architecture: **tabular**, **text**, **image**, and **joint** (tabular+text+image combined). Tabular uses statistically-grounded methods (KS-test, PSI, IQR); text, image, and joint share a single modality-agnostic method (the **Domain Classifier Test**), since all three reduce to comparing two sets of embedding vectors.

| Modality | Supported Inputs | Detection Method |
|----------|-------------------|-------------------|
| Tabular | CSV, TSV, Excel (.xlsx, .xls), JSON, JSON Lines, Parquet, ARFF, compressed (.gz, .zip) | Two-sample KS-test (continuous), PSI (categorical), IQR (real-time) |
| Text | Batches of raw strings | Domain Classifier Test (AUC-based) on `all-MiniLM-L6-v2` embeddings |
| Image | JPEG, PNG batches | Domain Classifier Test (AUC-based) on `resnet18` embeddings |
| Joint (multimodal) | Records combining any subset of tabular fields, text, and image per record | Domain Classifier Test on a concatenated joint embedding (tabular + text + image + a bounded interaction term) |

**Not supported:** audio, video, per-token/per-pixel drift localization, and real-time/streaming detection for any modality (all four are batch-based). Joint additionally has no dashboard UI yet — it's API-only (see [Known Limitations](#known-limitations)).

**Why this scope?**

KS-test, PSI, and IQR-based methods are mathematically grounded in continuous and categorical feature distributions, and remain tabular-only. Text and image don't share that mathematical structure with each other or with tabular data, but they *do* share one with each other — both are just vectors once embedded — so a single Domain Classifier Test (train a classifier to distinguish reference vs. current embeddings; AUC well above 0.5 indicates drift) covers both without inventing two separate systems. Joint extends the same idea one step further: instead of monitoring each modality in isolation, it concatenates all of them into one vector per record, so it can catch a drift pattern none of the single-modality checks can see on their own — a case where tabular metadata is individually normal, the text is individually normal, and the image is individually normal, but the *combination* has changed (e.g. records now systematically pairing metadata with the wrong image). See [How It Works](#how-it-works) for the mechanism and [Known Limitations](#known-limitations) for exactly how far that detection currently generalizes.

> **Validation status:** the tabular path has the full four-part validation methodology behind it (see [Validation Results](#validation-results)) run against a 4.5M-row real-world dataset. The text/image path has been smoke-tested (adapters produce correct, deterministic embeddings; the detector correctly centers near AUC=0.5 on same-distribution data and flags clearly-separated data) and manually verified end-to-end, but had not been through that same four-part methodology until Step 2 (e) (2026-10-05) — see [Text/Image Validation (Step 2 e)](#textimage-validation-step-2-e-petfindermy-2026-10-05) for measured false-alarm rates, power curves, and precision/recall/F1 on PetFinder.my (ground truth defined by construction, not measured). Headline: legacy is conservative (≤ 2% false alarms with a 100+ row reference, but recall only ~0.68 on 25%–100% off-population batches), calibrated mode at the default floor makes identical decisions, and the p-value is over-confident on real embeddings. The ~40-sample borderline case turned out to be mostly a small-*reference* problem (20 vs 20: 10%–14% false alarms). A separate real-world (not synthetic) smoke test across all four modalities, including joint, was also run against the [PetFinder.my Adoption Prediction dataset](#real-world-multimodal-smoke-test-petfindermy) — see that section for numbers; it's still a smoke test, not the four-part methodology.

---

## Key Features

**Automatic Column Profiling**

Before locking any baseline, the system profiles every column in your dataset and automatically decides whether it should be monitored as a continuous feature, a categorical feature, or ignored entirely. It catches identifiers, timestamps, monotonic sequences, near-constant columns, and panel entity codes without you manually specifying anything. Uncertain columns are surfaced to the user for manual review rather than silently discarded.

**Human-in-the-Loop Schema Verification**

The profiler makes recommendations, but the final schema is always confirmed by a human before the baseline is stored. This design decision reflects a real production concern: automated heuristics catch most cases, but edge cases (imbalanced targets, domain-specific encoding schemes, repeating panel IDs) still require human judgment. You see the profiler's reasoning for every column and can override any decision before monitoring starts.

**Statistical Drift Detection**

Two detection engines run in parallel depending on the feature type:

- Continuous features use the two-sample Kolmogorov-Smirnov test, which is nonparametric and makes no assumptions about the underlying distribution
- Categorical features use Population Stability Index (PSI), which measures the magnitude of a distributional shift rather than just its statistical significance — an important distinction at production data volumes where chi-square becomes oversensitive

**Real-time Anomaly Scoring**

Every incoming prediction is individually scored against IQR fences computed from the baseline. This catches point anomalies that batch drift metrics would smooth over, giving you two complementary monitoring signals rather than one.

**Text & Image Drift via Domain Classifier Test**

Text and image batches are embedded (`all-MiniLM-L6-v2` for text, `resnet18` for images) and compared using a classifier trained to distinguish reference from current embeddings — cross-validated to avoid the overfitting-inflates-AUC failure mode. An AUC near 0.5 means the two batches are indistinguishable (no drift); an AUC well above it means they're separable (drift). Same conceptual workflow as tabular — lock a baseline, analyze a batch — just a different math under the hood.

**Calibrated mode for text/image (2026-10-04):** opt in via `/fit/{project_id}/text`'s or `/fit/{project_id}/image`'s `calibration_config: {"decision_mode": "calibrated"}` to get the same two-gate (significance + materiality) decision tabular already has, instead of a bare `AUC > 0.65` cutoff. Significance comes from a p-value looked up against a precomputed null-distribution grid (`results/dct_null_distribution_grid.json` — 200 simulated draws per `(embedding_dim, reference_size, batch_size)` cell it was built at; nearest-neighbor matched for sizes in between; a genuinely uncalibrated embedding dimension falls back to a smaller on-the-fly permutation rather than borrowing a wrong-dimension number). Materiality keeps the existing AUC floor (0.65). **Stays `legacy` by default** for every new text/image project, even with calibrated mode available — unlike tabular's calibrated-by-default switch, there's no empirical side-by-side yet justifying that for embeddings (see [Known Limitations](#known-limitations)).

**Joint Multimodal Context Drift Detection**

Beyond monitoring tabular, text, and image data independently, `/fit/{project_id}/joint` and `/analyze/{project_id}/joint` build one joint embedding per record — z-scored/frequency-encoded tabular fields, text and image embeddings, and a bounded interaction term (a fixed random projection of tabular × text/image, catching cases where a modality is absent) all concatenated together — and run the same Domain Classifier Test against it. This is a genuinely different capability from running the three single-modality checks side by side: it can catch a **correlation-break** where every modality's own marginal distribution is completely unchanged and only the *pairing* between modalities has shifted (e.g. metadata that's individually normal but now systematically attached to the wrong image). Built as a fully separate, additive capability — `adapters/joint.py` and its own detector configuration — that leaves the existing tabular/text/image pipelines byte-for-byte unmodified. Its detection currently generalizes across noise but only at roughly the effect size it was validated on — see [Known Limitations](#known-limitations) for the precise, tested boundary of what it does and doesn't catch.

**Multi-format File Ingestion**

The file reader (implemented in `dashboard.py`) handles encoding detection automatically (UTF-8, CP1252, Latin-1, ISO-8859-1), parses ARFF attribute headers, detects libsvm-format .dat files, reads all sheets from multi-sheet Excel files with a schema consistency warning, and handles gzip and zip compressed inputs without requiring pre-processing.

**Robust Dataset Ingestion**

Every `/fit` and `/analyze` endpoint validates structure (matching column lengths, non-empty payloads, minimum sample counts, at-least-one-modality-present for joint records) before any DataFrame construction or embedding compute runs, returning a specific `422` naming exactly what's wrong instead of a raw pandas/numpy traceback. Tabular ingestion coerces mixed-type columns (a stray non-numeric cell like `"N/A"` in an otherwise-numeric column) using the same threshold-based rule the profiler already uses for classification, reporting how many values were dropped per column (`cleaning_summary` in the `/fit` response) rather than silently losing them or crashing. Categorical cardinality is capped at fit time. All four modalities catch the actual exceptions their adapters can raise (`UnidentifiedImageError`, `binascii.Error`, `UnicodeDecodeError` — verified none of these are `ValueError` subclasses, so a bare `except ValueError` previously let a corrupted image reach the client as a raw 500) rather than a bare `ValueError`.

**Secure Multi-tenant Architecture**

Each user authenticates via Google OAuth and can only access monitoring data for their own projects. Backend requests are authorized with a short-lived session token minted from that login (signed with a secret shared between the dashboard and backend) — there's no separate API key to provision or manage.

**Containerized Deployment**

The entire stack — FastAPI backend, Streamlit dashboard, and supervisor process management — is containerized with Docker and orchestrated via `docker-compose.yml`, making it straightforward to deploy on any cloud provider.

---

## How It Works

All four modalities converge on one shared pipeline shape — **adapt → (embed, for text/image/joint) → detect → store/report** — implemented behind a common `BaseAdapter` interface (`adapters/base.py`) so baseline storage, `/analyze` routing, and the dashboard treat tabular, text, and image uniformly rather than as separate bolted-together systems. Joint reuses this same shape but as an entirely separate, additive adapter (`adapters/joint.py`) — it does not modify any of the other three.

### 1. Profiler (`utils/profiler.py`) — tabular only

Takes a sample of your uploaded training data and computes a set of mathematical signals for each column: cardinality ratio, dominant value ratio, monotonicity, string length consistency, structured pattern detection, and dtype analysis after attempted coercion. These signals feed a routing decision that classifies each column as continuous, categorical, or ignored — with a reasoning string attached to every decision so it's auditable in the UI. Text and image baselines skip this step entirely — there's no per-column schema to confirm for a single embedding matrix. Joint reuses this same coercion logic for its tabular sub-fields (`JointAdapter.fit_tabular_schema`), just without the dashboard's human-in-the-loop confirmation step, since joint is API-only.

### 2. Adapters (`adapters/`) — modality-specific ingestion

- `tabular.py`: passes columnar data through unchanged (KS/PSI/IQR consume it directly)
- `text.py`: embeds a batch of raw strings via `all-MiniLM-L6-v2` (384-dim)
- `image.py`: embeds a batch of JPEG/PNG images via `resnet18`'s penultimate layer (512-dim), handling mixed sizes/formats via resize + RGB conversion
- `joint.py`: builds one vector per record by z-scoring/frequency-encoding the declared tabular fields, embedding any text/image present via the adapters above, and appending a bounded interaction term — a fixed (seeded, untrained) random projection of tabular × text and tabular × image, so the joint vector can carry the *pairing* between modalities, not just their independent presence. A record missing a modality contributes zeros for that slice rather than failing, as long as at least one modality is present.

### 3. Baseline Storage (`db/crud.py`)

Once the user confirms the schema (tabular) or uploads a reference batch (text/image/joint), the system locks a baseline in SQLite:

- Tabular: IQR fences (Q1, Q3) for continuous features, frequency tables for categorical features
- Text/Image: a capped sample of raw reference embeddings (the Domain Classifier Test needs real vectors to retrain against on every `/analyze` call, not just summary statistics)
- Joint: the same capped raw-embedding sample as text/image, plus the fitted tabular sub-schema (per-field mean/std or category frequencies) needed to vectorize new records the same way at `/analyze` time — stored via a separate `insert_joint_baseline()` so the text/image storage path stays untouched

### 4. Drift Detection (`drift/detector.py`, `drift/embedding_detector.py`)

At inference time, incoming production batches are compared against the stored baseline. Tabular: continuous features go through a two-sample KS-test, categorical through PSI, individual predictions also scored against IQR fences for real-time anomaly detection. Text/Image/Joint: current embeddings are compared against the stored reference embeddings via the Domain Classifier Test (cross-validated logistic regression, AUC-based) — joint passes an L1-regularized classifier instead of the shared default (see [Key Features](#key-features) and [Known Limitations](#known-limitations) for why, and confirmation that this is opt-in and does not change the text/image default).

### The Full Flow

```
Upload reference data (tabular / text / image / joint)
        ↓
Tabular: profiler classifies columns (auto + human confirmation)
Text/Image: adapter embeds the reference batch
Joint: tabular fields z-scored/encoded + text/image embedded + interaction term computed
        ↓
Baseline locked in SQLite
        ↓
Production batch sent to /analyze/{project_id}[/text|/image|/joint]
        ↓
Tabular: KS-test + PSI + IQR scoring (real-time)
Text/Image/Joint: Domain Classifier Test (AUC-based)
        ↓
Results surfaced in Streamlit dashboard (tabular/text/image) or via the API directly (joint)
```

---

## Validation Results

The system was validated on the **NYC Citi Bike 2016 dataset** (4.5 million rows, 7 monitored features) using a four-part methodology designed to give independently verifiable numbers rather than a single invented accuracy score.

### Ground Truth Verification

Ground truth was established independently of the API using `scipy.stats.ks_2samp` for continuous features and PSI computed directly for categorical features. Notably, chi-square was intentionally *not* used as the ground-truth criterion for categorical features, because at 4.5 million rows it flags practically any proportional shift as significant, including ones so small they fall well below the PSI threshold the production engine actually uses. Using PSI for both ground truth and detection keeps the evaluation methodology consistent.

### Synthetic Drift Sensitivity

Controlled drift was injected at three severity levels (0.5σ, 1.5σ, and 3.0σ shifts) across all five continuous features. 10 trials per feature per severity level, using independent random seeds.

| Severity | Shift Magnitude | Detection Rate |
|----------|----------------|----------------|
| Mild | 0.5σ | 100% (50/50) |
| Moderate | 1.5σ | 100% (50/50) |
| Severe | 3.0σ | 100% (50/50) |

### Production-Batch Classification Metrics

Real production batches (Apr–Dec 2016) were compared against a baseline locked on Jan–Mar 2016 data. Ground-truth labels came from the independent PSI/KS verification, not from the API's own output.

| Metric | Score |
|--------|-------|
| Precision | 1.000 |
| Recall | 0.939 |
| F1 | 0.969 |
| Accuracy | 0.96 |

Zero false positives across all 15 trials. The 6.1% missed detections are concentrated in `pickup_latitude` (KS-stat 0.0189, the smallest effect size in the dataset), which is consistent with expected statistical power limitations rather than a detection threshold problem — confirmed by the sample-size sweep below.

### Per-Feature Detection Rate (Real Production Batches)

| Feature | Method | Effect Size | Detection Rate |
|---------|--------|-------------|----------------|
| `trip_duration` | KS-test | KS=0.0924 | 100% |
| `month` (categorical) | PSI | PSI=16.27 | 100% |
| `pickup_longitude` | KS-test | KS=0.0219 | 100% |
| `dropoff_latitude` | KS-test | KS=0.0195 | 100% |
| `dropoff_longitude` | KS-test | KS=0.0206 | 87% |
| `pickup_latitude` | KS-test | KS=0.0189 | 53% |
| `gender_id` (categorical) | PSI | PSI=0.043 | 0% — correctly stable |

`gender_id` shows 0% detection because its PSI is 0.043, well below the 0.2 threshold. Chi-square flags it as significant (p≈0) due to sample size, but PSI correctly identifies the shift as practically negligible. This is the intended behavior.

### Batch Size Recommendation

Detection rate scales with batch size in a smooth, monotonically increasing curve — consistent with expected KS-test power scaling. Based on this sweep, **15,000–20,000 rows per batch** sits at the knee of the curve.

| Batch Size | Recall | F1 |
|------------|--------|----|
| 1,000 | 0.458 | 0.629 |
| 3,000 | 0.729 | 0.843 |
| 10,000 | 0.812 | 0.886 |
| 20,000 | 0.896 | 0.945 |
| 50,000 | 0.917 | 0.957 |

### Detection Latency

The minimum percentage of a batch that needs to reflect the drifted distribution before an alert fires:

| Feature | Latency |
|---------|---------|
| `trip_duration` | 10% of batch |
| `dropoff_longitude` | 10% of batch |
| `dropoff_latitude` | 20% of batch |
| `pickup_longitude` | 30% of batch |
| `pickup_latitude` | 30% of batch |

### Bugs Found During Validation

The validation suite caught two real bugs, both fixed before the final numbers above were recorded:

**Bug 1 — Categorical key type mismatch.** Baseline frequency tables stored category keys as strings (`'1'`, `'2'`, `'3'`), but production payloads sent integer values (`1`, `2`, `3`). Dictionary lookups failed silently, defaulting every categorical feature to "no drift" regardless of actual shift magnitude. Fixed by normalizing all keys to strings in `_check_categorical_drift`, with an additional guard for float-valued integers (`3.0` → `'3'`) to prevent a different class of the same bug.

**Bug 2 — Ground truth methodology mismatch.** The original validation script used chi-square significance as the categorical ground truth label. At 4.5 million rows, chi-square flagged `gender_id` as significantly drifted (p≈0) despite its PSI being 0.043 — far below the 0.2 threshold used by the production engine. The ground truth was corrected to use PSI directly, which resolved the apparent false negative and confirmed the engine was behaving correctly all along.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API Backend | FastAPI |
| Dashboard | Streamlit |
| Database | SQLite |
| Authentication | Google OAuth 2.0 (session tokens signed with PyJWT) |
| Tabular Drift Detection | scipy (KS-test), custom PSI, IQR |
| Text/Image/Joint Drift Detection | Domain Classifier Test — scikit-learn (`drift/embedding_detector.py`); joint uses an L1-regularized `LogisticRegression` (`adapters/joint.py`), text/image use the shared default |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), torchvision (`resnet18`) |
| Visualizations | Plotly |
| Data Processing | pandas, numpy |
| Containerization | Docker, docker-compose |
| Deployment | Render |

---

## Project Structure

```
drift-monitoring-api/
│
├── main.py                    # FastAPI application and all API endpoints
├── dashboard.py               # Streamlit frontend and file ingestion logic
├── models.py                  # Pydantic request/response models
├── migrate.py                 # SQLite schema migrations
├── patched_init.py            # Initialization patches
├── README.md
│
├── Dockerfile                 # Backend container definition
├── Dockerfile.dashboard       # Dashboard container definition
├── docker-compose.yml         # Multi-container orchestration
├── supervisord.conf           # Process management configuration
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python version pin for Render (python-3.12.4)
├── .env.example               # Environment variable template
├── .gitignore
│
├── utils/
│   ├── profiler.py            # Automatic column classification engine
│   └── validation.py          # Structural request validation (shared across all /fit and /analyze endpoints)
│
├── drift/
│   ├── detector.py            # KS-test, PSI, and IQR detection engines (tabular)
│   ├── embedding_detector.py  # Domain Classifier Test (text/image)
│   └── alerts.py              # Alert triggering and notification logic
│
├── adapters/
│   ├── base.py                # Shared BaseAdapter interface
│   ├── tabular.py             # Tabular data adapter
│   ├── text.py                # Text → sentence-transformer embeddings
│   └── image.py               # Image → resnet18 embeddings
│
├── db/
│   └── crud.py                # Database CRUD operations layer
│
├── tests/                     # Validation and test scripts
└── .streamlit/                # Streamlit config (not committed, contains secrets.toml)
```

> Files not committed to version control: `google_credentials.json`, `drift.db`, `.env`, `.streamlit/secrets.toml`

---

## Installation

**Prerequisites:** Python 3.12+, Docker (optional but recommended), a Google Cloud project with OAuth 2.0 credentials configured.

### Option A: Local Setup (without Docker)

```bash
# Clone the repository
git clone https://github.com/Abhinavbilla/Drift-Monitoring-api.git
cd Drift-Monitoring-api

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Environment setup:**

Copy `.env.example` to `.env` and fill in your values:

```env
COOKIE_KEY=your_random_secret_key
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret
REDIRECT_URI=http://localhost:8501
BACKEND_URL=http://localhost:8000
```

> For local development, `REDIRECT_URI` must match the port where Streamlit is running.

Place your `google_credentials.json` (downloaded from Google Cloud Console) in the project root. This file is listed in `.gitignore` and should never be committed.

**Initialize the database:**

```bash
python migrate.py
```

**Start the backend:**

```bash
uvicorn main:app --reload
```

**Start the dashboard** (in a separate terminal):

```bash
streamlit run dashboard.py
```

The dashboard will be available at `http://localhost:8501` and the API at `http://localhost:8000`. Interactive API docs are at `http://localhost:8000/docs`.

### Option B: Docker Compose

```bash
# Copy and fill in environment variables
cp .env.example .env

# Build and start all services
docker-compose up --build
```

This starts both the FastAPI backend and the Streamlit dashboard as separate containers managed by the compose file.

---

## Deployment

Drift Monitoring API is containerized with Docker and deployed on **Render** using two separate container services — one for the FastAPI backend and one for the Streamlit dashboard.

**Live deployment:** [https://drift-monitoring-dashboard.onrender.com/](https://drift-monitoring-dashboard.onrender.com/)

To deploy your own instance on Render:

1. Fork the repository
2. Create two Web Services on Render pointing to your fork — one using `Dockerfile` (backend) and one using `Dockerfile.dashboard` (dashboard)
3. Set environment variables per service as follows:

**Backend service:**

| Variable | Description |
|----------|-------------|
| `GOOGLE_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret |
| `COOKIE_KEY` | Random secret key for session cookies |

**Dashboard service:**

| Variable | Description |
|----------|-------------|
| `BACKEND_URL` | Full URL of the deployed backend service |
| `REDIRECT_URI` | Full URL of the deployed dashboard service |
| `COOKIE_KEY` | Same secret key used in the backend |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret |

4. Add `google_credentials.json` as a **Secret File** (not an environment variable) for the Dashboard service. In the Render dashboard, go to your Dashboard service → Secret Files → add the file at path `google_credentials.json`
5. Render will automatically build and deploy each service using the respective Dockerfile

> Note: The free Render tier spins the service down after inactivity. The first request after a period of inactivity may take 30–60 seconds to respond.

**`Dockerfile` (single container, updated for table monitoring).** This is a multi-stage build:
- it builds the React frontend and serves it with nginx at `/`;
- the API is at `/api/`;
- the old Streamlit dashboard is kept at `/streamlit/`.

Build with `docker build --build-arg VITE_GOOGLE_CLIENT_ID=<id> .` and set the same backend variables as above. nginx accepts uploads up to 710 MB (a 200 MB table plus a 500 MB image ZIP). This Dockerfile has **not been build-tested** in this development environment, where Docker isn't available. The frontend stage's `npm run build` with `VITE_API_BASE_URL=/api` was verified locally.

Table monitoring also has these requirements:
- **Persistent storage:** staged uploads, embeddings and reference rows live under `DRIFT_DATA_DIR` (default `data/blobs`, which docker-compose mounts at `./data`). Mount a persistent disk there, and back it up together with `drift.db`.
- **Background jobs:** they run in a worker thread inside the API process. A free-tier instance that spins down kills running jobs; they're marked `interrupted` on restart and must be resubmitted.
- **CPU and memory:** image-heavy fits are expensive. On a 16-core desktop, about 15,000 descriptions took ~4 minutes and ~14,650 photos ~6 minutes to embed. A small instance will be far slower.
- **Limits (env vars):** `DRIFT_MAX_ZIP_UPLOAD_BYTES`, `DRIFT_MAX_ZIP_ENTRIES`, `DRIFT_MAX_ZIP_UNCOMPRESSED`, `DRIFT_MAX_IMAGE_BYTES`, `DRIFT_MAX_IMAGE_PIXELS`.

---

## Usage

### Table monitoring (recommended)

One table, with any mix of numeric, categorical, text and image columns, is one project. In the dashboard, open a project's **Table Monitoring** tab:
1. Upload the training table, plus a ZIP of images if a column holds image filenames.
2. Review each column's proposed type and monitor flag.
3. Review the proposed relationships.
4. Lock the baseline.
5. Upload production batches.

From Python (`clients/python`):

```python
from drift_monitor_client import DriftClient

client = DriftClient("http://localhost:8000", token="dm_...")      # personal access token
profile = client.profile_table("pets", train_df, images_zip="train_images.zip")
for col in profile["columns"]:
    print(col["name"], col["proposed_type"], col["confidence"], col["reason"])
fit = client.fit_table("pets", profile)          # accepts the proposals; pass columns=/relationships= to edit
report = client.analyze_table("pets", batch_df, images_zip="batch_images.zip")
print(report["overall"]["status"], report["overall"]["triggered_by"])
```

The same flow over HTTP: `POST /tables/pets/stage` (multipart `file`, optional `images`), poll `GET /jobs/{job_id}`, then `POST /tables/pets/fit` with the confirmed `columns` and `relationships`, then `POST /tables/pets/analyze`. See the [API Reference](#api-reference).

### First-Time Setup (per-modality workflow)

1. Open the dashboard and sign in with Google OAuth
2. Upload your training data (CSV, Excel, JSON, Parquet, or ARFF)
3. Review the auto-generated schema — the profiler will classify each column and explain its reasoning
4. Adjust any misclassified columns using the dropdowns, then click **Start Monitoring**

### Sending Production Data

Production batches are sent to the `/analyze` endpoint through the dashboard's upload flow. Direct programmatic calls to `/analyze` need a Bearer token — either a session token (from a Google login) or a self-serve personal access token minted with `scripts/create_token.py` (no dashboard login needed; see [API Reference](#api-reference)):

```python
import requests

MODEL_ID = "your_model_id"
SESSION_TOKEN = "dm_..."  # a PAT from scripts/create_token.py, or a session token (see dashboard.py's mint_session_token)

payload = {
    "production_data": {
        "age": [34, 45, 28, 52, 41],
        "transaction_amount": [120.5, 89.0, 340.2, 55.8, 210.0],
        "merchant_category": ["retail", "food", "retail", "travel", "food"]
    }
}

response = requests.post(
    f"http://localhost:8000/analyze/{MODEL_ID}",
    json=payload,
    headers={"Authorization": f"Bearer {SESSION_TOKEN}"}
)

print(response.json())
```

**Example response:**

```json
{
  "system_alert_triggered": true,
  "feature_metrics": {
    "age": {
      "statistic": 0.312,
      "p_value": 0.0003,
      "drift_detected": true
    },
    "transaction_amount": {
      "statistic": 0.089,
      "p_value": 0.412,
      "drift_detected": false
    },
    "merchant_category": {
      "statistic": 0.341,
      "p_value": null,
      "drift_detected": true
    }
  }
}
```

**Recommended batch size:** 15,000–20,000 rows per request for the best balance of detection sensitivity and latency (see [Validation Results](#validation-results)).

---

## API Reference

### Unified table monitoring (milestones M1–M3)

One table = one project. A single table can mix **numeric, categorical, text and image** columns; image columns hold filenames resolved against an uploaded ZIP (or base64/data-URI values in JSON). Every heavy step runs as a background job, so clients poll `GET /jobs/{job_id}`. Design and roadmap: [`docs/unified_table_plan.md`](docs/unified_table_plan.md).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tables/{project_id}/stage` | POST | Multipart `file` (any supported table format) + optional `images` ZIP. Starts a profile job that proposes, per column: type (`numeric`/`categorical`/`text`/`image`/`ignore`), monitor yes/no, confidence (a heuristic score, not a probability), evidence and reason. Staged files are deleted after the fit or after 24 h |
| `/jobs/{job_id}` | GET | Job status, progress, and on success its result (profile, fit summary, or analysis report). Owner only |
| `/tables/{project_id}/fit` | POST | `{stage_id, columns: [{name, type, monitor}], calibration_config?, schema_policy?, alert_policy?, model_version_label?}`: the human-confirmed schema. Locks a new baseline version |
| `/tables/{project_id}/analyze` | POST | Multipart production `file` + optional `images` ZIP; optional `baseline_version` and `Idempotency-Key`. Result: per-column drift, schema issues (including missing/corrupt images and empty text), data quality, and which tests formed the Holm family |
| `/tables/{project_id}/baseline` | GET | The confirmed schema of a version: what the profiler proposed, what the user chose, who decided, and whether each column is monitored |
| `/tables/stages/{stage_id}` | DELETE | Discard a staged upload's raw files immediately |

What M1 does and does not do, plainly:
- **Numeric/categorical columns** use exactly the existing tabular detectors (KS, PSI, calibrated two-gate, Holm). A numeric/categorical-only table produces the same results as `/fit/upload` + `/analyze/upload` (a regression test asserts identical metrics).
- **Text/image columns** use the existing Domain Classifier Test with the legacy `AUC > 0.65` rule, reported as **outside the Holm family**, because their p-values were measured over-confident in Step 2 (e). Calibrating them is milestone M3.
- **Relationship drift (milestone M2)** is checked for numeric↔numeric, categorical↔categorical and numeric↔categorical pairs. The profiler proposes pairs that are clearly related in the training data; the user accepts, rejects or adds pairs. Each test is built to ignore a change in either column's own distribution:
  - numeric↔numeric: change in Spearman correlation;
  - categorical↔categorical: a log-linear test that the pairing pattern (odds ratios) is unchanged, which also catches re-pairing at equal strength;
  - numeric↔categorical: change in where each category sits in the numeric ordering, adjusted for the category mix.

  p-values come from disjoint splits of the reference at the batch's size, and every relationship test joins the same Holm family as the numeric/categorical column tests. A separate informational screen lists unwatched pairs that became strongly related; it never alerts. Relationships involving text or image columns are not implemented yet (M3).
- **Measured** (`scripts/validate_table_relationships.py`, PetFinder numeric/categorical columns, reference 2,000 rows, 100 seeded draws per cell, batches of 300 / 1,000; raw results in `results/m2_relationship_validation_raw.json`):

  | Scenario | Any relationship alarm | Column alarm on the changed column |
  |---|---|---|
  | No change | 0% / 1% | — (any alarm at all: 0% / 1%) |
  | Monotone transform of the numeric columns in relationships | 2% / 1% | 100% / 100% |
  | Category mix shifted (cats weighted 3×) | 3% / 1% | 100% / 100% |
  | One column shuffled across all rows (its own distribution unchanged) | **100% / 100%** | 0% / 0% |
  | Same, 25% of rows shuffled | 3% / 2% | 0% / 0% |

  Flagged relationships involved the shuffled column 98–100% of the time. The 25% shuffle is not caught: its effect (about 0.03) is below the 0.05 materiality floor, by design. The relationship set came from the real profiler, which typed the integer breed code `Breed1` as numeric; one dataset only. Limitations: Spearman misses non-monotone dependence, and the numeric↔categorical test tracks relative position, not within-category spread.
- **Text/image columns (M3, decision approved in M4).** Each text/image column test uses a p-value from this project's own reference embeddings: disjoint splits at the batch size, PCA-64, 200 draws, and a Gaussian tail beyond the draws. Step 2(e)'s synthetic grid was measured over-confident. These tests are **in the Holm family**, with an AUC materiality floor of 0.55.
- **Relationships involving text/image (M3):**
  - probes: a text/image column predicts another column;
  - text↔image matching: does each row's image still go with its own text.

  They're proposed at profile time; text/image columns are embedded once there and the vectors reused by the fit. Probes into **categorical** columns and matching are in the Holm family and alert. Probes into **numeric** columns are report-only (shown, never alerting), because they failed the population-mix check below. A probe whose source column also drifted is marked `confounded_by`.
- **Measured for M3** (`scripts/validate_m3_embeddings.py`, cached PetFinder embeddings, seeded; `results/m3_embedding_validation.json`):
  - **PCA-64 vs raw embeddings:** the same or better detection (image, 25% cats: 10% vs 4%; 50% cats: 100% vs 98%), and 0% false alarms on unchanged data for both.
  - **Calibration (reference 1,000, 100 draws per cell):** with no drift, P(p < 0.05) was 1–5% in all four text/image cells, inside the 95% interval around 5%. A batch with 25% cats was detected 100% of the time at p < 0.05; the AUC > 0.65 rule caught it 6–10% of the time.
  - **Probes and matching (reference 1,000, batch 200, 50 draws):**

    | Test | No change | Category mix shifted | Pairing shuffled |
    |---|---|---|---|
    | text→Type, image→Type (categorical) | 0–2% | 0% | 100% |
    | text↔image matching | 0% | 0% | 100% |
    | text→Age (numeric) | 6% | **26%** | 100% |

    The numeric probe is not robust to a population-mix shift (cats and dogs differ in age), so it stays report-only for good.

  Live check (restarted server, 400 real pets with photos): shuffling descriptions between pets flagged exactly the three Description relationships and nothing else.
- **Formal validation (M4)** (`scripts/validate_table_formal.py`; raw results in `results/m4_formal_validation.json`):
  - **Setup:** reference 1,500 rows, batch 300, seeded. One Holm family, as in the API; probes into numeric columns left out.
  - **PetFinder** (100 draws per scenario; numeric, categorical, text and image columns; 7 relationships proposed by the profiler).
  - **Inside Airbnb Edinburgh** (CC BY 4.0, 50 draws per scenario; numeric, categorical and text columns; 12 relationships). Its photos are remote URLs the product never fetches.

  | Scenario | PetFinder: any alarm / relationship alarm | Airbnb: any alarm / relationship alarm |
  |---|---|---|
  | No change | 0% / 0% | 8% / 2% |
  | Monotone transform of the numeric columns in relationships | 100% / 0% | 100% / 8% |
  | Category mix shifted | 100% / 1% | 100% / 4% |
  | Numeric column shuffled (its own distribution unchanged) | relationship 99%, column 0% | 100%, column 0% |
  | Categorical column shuffled | 100%, column 0% | 100%, column 4% |
  | Text column shuffled | 100%, column 0% | 54%, column 0% |
  | Image column shuffled | 100%, column 0% | — |
  | Numeric column shuffled within one category's rows | 0% (see note) | 100% |

  - **Attribution:** flagged relationship tests involved the changed column 97–100% of the time.
  - **Batch-level relationship drift (shuffles and subgroup as positives):** precision 0.998, recall 0.80, F1 0.89 on PetFinder; 0.96 / 0.89 / 0.92 on Airbnb.
  - **Detection latency:** wherever a shuffle is detected 99–100% of the time, it's caught on the first batch under the default alert policy (k=1 of m=1).

  What this found:
  - **PetFinder's subgroup scenario tested nothing.** Shuffling `Breed1` only among dogs leaves `Breed1↔Type`, its only monitored relationship, exactly intact. That's a flaw in the scenario design, not a detection failure.
  - **Airbnb's no-change alarms (4 of 50) and its monotone-scenario relationship alarms all involve `property_type`.** It's a high-cardinality categorical, and hosts with many near-identical listings make rows clustered rather than independent. That violates the exchangeability assumption the nulls rely on and makes them too narrow. This is known and not fixed.
  - **Airbnb text shuffles are caught 54% of the time.** Descriptions there predict room and property type only moderately.
- **Not implemented (optional in the plan):** the experimental whole-row joint signal; the existing `/joint` endpoint is unchanged.
- **Duplicate rows:** the fit removes only rows identical in every column, the same rule as the tabular `/fit`. A table without an identifier column can still lose legitimate repeated rows this way, which affects the tabular endpoints too; see `docs/PROGRESS.md`.
- History, alert state machine, webhooks and idempotency all work for table projects; the original tabular/text/image/joint endpoints are unchanged.
- Image ZIPs are read in memory, never extracted; unsafe paths, symlinks, encrypted entries, oversized and suspiciously compressed entries are rejected and counted.
- Measured live on this machine (120 PetFinder rows with photos): profile 1 s, fit 10 s, analysis of 100 rows 4 s. Larger tables and slower hosts will take proportionally longer.

### Per-modality endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fit/{project_id}` | POST | Lock a tabular baseline from training data (JSON body) |
| `/fit/{project_id}/upload` | POST | Same as above, via multipart CSV/Parquet upload (large reference sets) |
| `/analyze/{project_id}` | POST | Compare a tabular production batch against the stored baseline (JSON body) |
| `/analyze/{project_id}/upload` | POST | Same as above, via multipart CSV/Parquet upload. Both tabular analyze endpoints accept an optional `Idempotency-Key` header and an optional `baseline_version` query param |
| `/fit/{project_id}/text` | POST | Lock a text baseline (embeds `reference_texts`) |
| `/analyze/{project_id}/text` | POST | Compare a text production batch via the Domain Classifier Test |
| `/fit/{project_id}/image` | POST | Lock an image baseline (embeds base64-encoded `reference_images`) |
| `/analyze/{project_id}/image` | POST | Compare an image production batch via the Domain Classifier Test |
| `/fit/{project_id}/joint` | POST | Lock a joint baseline from records combining tabular/text/image (`reference_records`) |
| `/analyze/{project_id}/joint` | POST | Compare a joint production batch via the Domain Classifier Test |
| `/predict/{project_id}` | POST | Real-time single-point anomaly check against locked IQR boundaries |
| `/profile` | POST | Profile a tabular dataset's columns without locking a baseline |
| `/projects` | GET | List all projects for the authenticated user |
| `/projects/{project_id}` | DELETE | Permanently delete a project and all associated baseline/log data |
| `/models/{model_id}` | DELETE | Deprecated alias for `DELETE /projects/{project_id}` |
| `/baseline/{project_id}` | GET | Fetch IQR fences, feature types, and modality for a project's *active* baseline |
| `/baselines/{project_id}` | GET | List every baseline version ever fit for a project, newest first, flagging the active one |
| `/baselines/{project_id}/activate` | POST | Make an existing (old or current) baseline version active again |
| `/history/{project_id}` | GET | Paginated history of `/analyze` calls, with `since`/`until`/`feature`/`alert_only` filters and a per-feature time series |
| `/webhooks/{project_id}` | POST | Register a webhook for drift alerts (returns its signing secret once) |
| `/webhooks/{project_id}` | GET | List a project's webhooks (never includes the secret) |
| `/webhooks/{project_id}/{webhook_id}` | DELETE | Remove a webhook |
| `/logs/{project_id}` | GET | Retrieve recent logs for a project |
| `/health/{project_id}` | GET | Burst-alert health check (wave of recent real-time anomalies) |
| `/docs` | GET | Interactive Swagger UI |

Full request/response schemas are available at `/docs` when the server is running.

**Auth for programmatic access:** every endpoint above accepts either a session token (minted via the dashboard's Google login) or a personal access token (PAT, `dm_<prefix>_<secret>`) as a Bearer token. Mint a PAT with `scripts/create_token.py` — no need to go through the dashboard's login flow for scripted/CI use. A PAT can be scoped to specific project IDs or left unscoped (`["*"]`); see `auth/tokens.py`.

**Python client:** `clients/python/drift_monitor_client` wraps the tabular `/fit` and `/analyze` endpoints (including large-frame uploads, `calibration_config`, `feature_types`, and `idempotency_key=`) behind a small `DriftClient` class, so scripted/CI use doesn't need to hand-build requests:

```python
from drift_monitor_client import DriftClient

client = DriftClient("http://localhost:8000", token="dm_...")
client.fit("my_project", reference_df)
result = client.analyze("my_project", production_df, idempotency_key="batch-2026-10-01")
```

---

## History, Schema Validation, Versioning & Alerting

Beyond a single fit-and-compare cycle, the tabular `/analyze` endpoints (`/analyze/{project_id}` and its upload counterpart) also give each project a queryable history, a safety net against malformed batches, multiple baselines to roll between, and an alert state machine that distinguishes "drifted once" from "still drifting."

**History (`GET /history/{project_id}`):** every tabular `/analyze` call is recorded — statistics only, never the raw production rows — with `since`/`until`/`feature`/`alert_only` filters, pagination, and a per-feature time series (`statistic`, `effect_size`, `p_value_adjusted`, `significant`, `material`) so you can track how a feature's drift metric has moved over time without re-running anything.

**Idempotency:** pass an `Idempotency-Key` header on either tabular analyze endpoint. Replaying the same key with the same batch returns the stored result verbatim (no new history row, no re-sent alert email); replaying it with a *different* batch is a `409`. Keys are scoped per project and expire after 7 days.

**`schema_report`:** every tabular `/analyze` response includes a `schema_report` flagging what's wrong with the batch *before* trusting its drift numbers — missing columns, unexpected columns, a column whose dtype no longer matches what was fit (e.g. a numeric column suddenly receiving strings), a null rate well above the reference's, categories never seen at fit time, or a column that's gone constant. Valid columns are still analyzed regardless of what's flagged elsewhere in the batch — a malformed column never produces a 500 or a silent drop. Nulls and non-numeric cells are cleaned out before statistics are computed (matching how `/fit` already cleans reference data), so a batch with missing values gets a real computed statistic instead of a meaningless `NaN`. Each issue carries a severity (`alert`/`warn`/`ignore`) resolved from a per-project `schema_policy` (set via `/fit`'s optional `schema_policy` field; default: alert on missing columns, warn on everything else).

**Baseline versioning:** every `/fit` call creates a new version instead of overwriting the last one — old versions are kept indefinitely. `GET /baselines/{project_id}` lists them (optionally labeled via `/fit`'s `model_version_label`), `POST /baselines/{project_id}/activate` switches which one is live, and `/analyze` accepts an optional `baseline_version` query param to compare against a specific past version instead of whichever is currently active.

**Sustained-alert state machine:** a single alerting batch doesn't necessarily mean "something is wrong" — it could be noise. Each project has a `{k, m}` policy (default `1, 1` — today's single-batch behavior); `sustained_alert` is true once `k` of the last `m` analyses *on the same baseline version* have alerted. The resulting `alert_state` (`ok`/`open`) only logs a transition event (`opened`, `still_open`, `resolved`) when something actually changes — a long stable streak in either direction doesn't grow the event log. Set via `/fit`'s optional `alert_policy` field.

**Webhooks:** register a URL (`POST /webhooks/{project_id}`) to get an HMAC-signed `POST` whenever a project's alert state actually *transitions* — `opened` and `resolved` by default; `still_open` is opt-in per webhook (via `event_filter`), since a long sustained incident would otherwise fire one delivery per `/analyze` call for its whole duration. Every delivery carries an `X-Drift-Signature-256: sha256=<hmac>` header computed over the raw body with the webhook's own secret (returned once, at registration — verify it the same way Stripe/GitHub webhooks are verified). Failed deliveries retry with exponential backoff (1s/4s/16s/64s) across up to 5 attempts via a background sweep, independent of the request that triggered them — a slow or down endpoint never blocks `/analyze`. Registration rejects any URL that resolves to a loopback, private, link-local, reserved, or multicast address (this also blocks cloud metadata endpoints) — an SSRF guard, not optional.

```python
import hashlib, hmac

def verify_signature(secret: str, raw_body: bytes, header_value: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header_value)
```

**DKW floor warning:** `/fit` checks whether the configured KS materiality floor for a continuous feature is finer than this reference size can actually resolve, using the Dvoretzky–Kiefer–Wolfowitz distribution-free bound on the reference's own empirical-CDF estimation error — a stricter, non-asymptotic check than the existing minimum-detectable-effect warning. When it fires, the response reports the minimum reference size that would make the configured floor trustworthy.

Every project ID is namespaced per owner internally, so two different users can each have a project called `"demo"` without colliding or being able to probe each other's project names via status codes — existing (pre-namespacing) projects keep working without migration.

---

## Testing and Validation Methodology

The validation suite (`tests/test_drift_engine.py`) implements a four-part evaluation framework:

**Part 1 — Ground Truth Verification:**
Establishes which features actually drifted using `scipy.stats.ks_2samp` (continuous) and PSI (categorical), completely independent of the API. This is the reference against which all detection results are evaluated.

**Part 2 — Synthetic Drift Sensitivity:**
Injects controlled distributional shifts at three severity levels (mild: 0.5σ, moderate: 1.5σ, severe: 3.0σ) and measures detection rate across 10 independent trials per level. Answers the question: "at what magnitude does the engine start reliably catching shifts?"

**Part 3 — Confusion Matrix Evaluation:**
Builds a full confusion matrix from three types of test cases: real baseline batches (should not trigger), real production batches (labeled using Part 1 ground truth), and synthetically drifted batches (known positive). Reports Precision, Recall, and F1 via scikit-learn rather than an invented weighted formula.

**Part 4 — Detection Latency:**
Gradually increases the proportion of drifted samples in a fixed-size batch until an alert fires. Reports the minimum percentage of drifted data required to trigger detection for each feature.

To run the validation suite:

```bash
# Prepare the test datasets first
python split_citi_bike.py

# Run the full validation
python tests/test_drift_engine.py
```

**Text/image smoke tests** (`tests/test_embedding_adapters.py`) — not the four-part methodology above, but unit-level checks that the acceptance criteria in the design hold: adapters produce correctly-shaped, deterministic embeddings and tolerate edge cases (empty/long strings, mixed image sizes/formats); the Domain Classifier Test centers near AUC=0.5 on same-distribution data across repeated trials and correctly flags clearly-separated data. Run with:

```bash
python -m pytest tests/test_embedding_adapters.py -v
```

**Text/image four-part validation suite** (`tests/test_embedding_validation.py`) — the real four-part methodology adapted for embedding-based detection, using labeled public datasets as a ground truth proxy: 20 Newsgroups topic categories for text, CIFAR-10 classes for images. Ground truth is verified independently of our embedding pipeline (TF-IDF / raw pixels, not sentence-transformers/resnet18); synthetic severity is the proportion of a different category mixed into a batch (10%/30%/60%) rather than a σ-shift; confusion matrix and detection-latency parts mirror the tabular suite's structure. Run with:

```bash
python tests/test_embedding_validation.py
```

> This script's logic has been verified end-to-end (correct API calls, zero false positives on same-category batches, monotonically increasing detection with severity), but as of this writing it has not yet been run to completion against the real datasets to produce publishable numbers — CIFAR-10's ~170MB download was too unreliable on the network available at the time. **No precision/recall/F1/latency numbers for text/image should be treated as final until this script has actually been run to completion and its output reviewed.** Also worth remembering: this validates the *mechanism* using a topic/class-shift proxy, not genuine real-world drift observed over time the way the tabular Citi Bike split was.

**Ingestion robustness tests** (`tests/test_ingestion_robustness.py`) — reproduces each ingestion failure mode against the pre-fix behavior first, then proves the fix: the mixed-type tabular column crash and its order-dependent miscategorization variant, the profiler's cleaned values being discarded before storage, missing/mis-scoped/mis-typed exception handling across all four modalities' fit/analyze endpoints, absent structural validation at the API boundary, and unbounded categorical cardinality. 19 tests, run with:

```bash
python -m pytest tests/test_ingestion_robustness.py -v
```

### Text/Image Validation (Step 2 e, PetFinder.my, 2026-10-05)

The four-part methodology (false alarms, power curve, batch-size sweep, precision/recall/F1) run on real PetFinder.my descriptions (`all-MiniLM-L6-v2`) and first photos (`resnet18`), through the same `EmbeddingDriftDetector._compute_auc` that `/analyze` uses (20/20 sampled draws re-checked identical against the real `analyze()`). **Ground truth is defined by construction, not measured**: dogs are the reference population, cats the "different" one; a batch's drift label comes from how it was sampled (0% cats = no drift; 25%/50%/100% cats = drift; 5%/10% cats and within-dog splits are reported as power/exploratory only, never scored — pre-registered in `scripts/step2e_validate.py`). Reference sizes 100 and 500, batch sizes 40/100/300/1000, 200 A/A draws and 100 draws per mixture per cell. Full tables: `results/step2e_text_image_report.md`; raw draws: `results/step2e_text_image_raw.json`; reproduce with `scripts/step2e_embed_petfinder.py`, `step2e_validate.py`, `step2e_report.py` (the dataset is gitignored, so this needs a local copy).

| Rule (same draws) | Modality | False alarms, A/A | Precision | Recall | F1 |
|---|---|---|---|---|---|
| legacy (AUC > 0.65) | text | 0.0%–1.5% per cell | 0.998 | 0.678 | 0.808 |
| legacy (AUC > 0.65) | image | 0.0%–2.0% per cell | 0.994 | 0.679 | 0.807 |
| calibrated, AUC floor 0.65 (current default floor) | text / image | identical to legacy in every cell | same | same | same |
| calibrated, AUC floor 0.60 | text | 0.0%–10.0% per cell | 0.980 | 0.830 | 0.898 |
| calibrated, AUC floor 0.60 | image | 0.0%–9.0% per cell | 0.976 | 0.803 | 0.881 |
| calibrated, AUC floor 0.55 | text | 1.0%–11.0% per cell | 0.944 | 0.922 | 0.933 |
| calibrated, AUC floor 0.55 | image | 0.5%–14.5% per cell | 0.943 | 0.911 | 0.927 |

(Precision/recall/F1 pooled over all 8 size combinations per modality; positives are the 25%/50%/100% cat mixtures.)

What this found, plainly:
- **At the current 0.65 floor, calibrated mode never changed a single decision vs legacy** in this run — AUC ≥ 0.65 almost never occurs under no drift, so the floor, not the p-value, decides. Calibrated mode as shipped adds a p-value to the response but no extra detection at default settings.
- **The detector is conservative.** Under no drift it rarely alarms (legacy ≤ 2% for reference ≥ 100), but it also misses real contamination: a batch that is 25% off-population was caught only 0%–20% of the time at the legacy cutoff (per-cell range, both modalities), 50% off-population 82%–100%, fully off-population 100%.
- **The p-value is over-confident on real embeddings.** Under no drift, P(p < 0.05) was above the nominal 5% in 15 of 16 modality/size cells (up to 14.5%; P(p < 0.10) up to 26.0%). The null grid is built from synthetic Gaussian vectors at a handful of sizes (nearest-neighbor lookup), which real embeddings don't match. Lowering the AUC floor below 0.65 therefore trades recall for a false-alarm rate above 5% at small batch sizes (floor 0.60: up to 10.0% at a 40-row batch, ≤ 3.5% at 100+ rows).
- **The ~40-sample borderline case was real, and is mostly about a small *reference*.** With a 100+ row reference, a 40-row batch false-alarmed ≤ 2.0% under legacy. With both sides tiny (supplementary run, `scripts/step2e_small_batch.py`, 200 A/A draws): 20 vs 20 false-alarmed 10.0% (text) / 14.0% (image); 40 vs 40, 6.0% (text) / 4.0% (image). The original case's exact texts weren't recorded, so this measures that regime on real data rather than reproducing it. At these sizes the p-value gate also added nothing (the precomputed grid's smallest cell is 50); the only 3 of 1,200 draws where calibrated@0.65 and legacy disagreed had an AUC of exactly 0.65, which the calibrated rule's `>=` floor counts and legacy's `>` cutoff doesn't — a boundary quirk, not the p-value.

Not decided by this run: whether calibrated should become the default for text/image (it shouldn't on this evidence, until the null is built from real embeddings), and whether the AUC floor should move. Scope limits: one dataset (pet descriptions in English, pet photos), fixed embedding models, drift defined by population mixing — this says nothing about medical imaging, technical text, or drift types other than population shift. Joint remains unvalidated.

### Real-World Multimodal Smoke Test (PetFinder.my)

To exercise all four modalities — including joint, which has no dedicated public benchmark dataset — against real, non-synthetic data rather than only synthetic injections or topic-shift proxies, an ad-hoc smoke test was run locally against the [PetFinder.my Adoption Prediction dataset](https://www.kaggle.com/competitions/petfinder-adoption-prediction) (14,993 pets; tabular attributes, free-text descriptions, and photos for each). This script isn't committed (the dataset itself is gitignored, matching this project's practice of keeping large data local-only), so treat this as a documented manual run, not a reproducible CI artifact.

**Method:** for each modality, fit a baseline on a sample of dogs (N=100), then `/analyze` two batches — a disjoint dog sample ("same", drift *not* expected) and a cat sample ("drift", expected, since dog/cat tabular attributes, description vocabulary, and photo appearance are all genuinely different distributions).

| Modality | Scenario | Result | AUC / Alert |
|----------|----------|--------|--------------|
| Tabular | same (dog vs dog) | ❌ false positive | `Color1`, `State` flagged (PSI 0.06 / **0.39**) |
| Tabular | drift (dog vs cat) | ✅ correct | drift correctly flagged |
| Text | same | ✅ correct | AUC 0.607, no drift |
| Text | drift | ✅ correct | AUC 0.970, drift detected |
| Image | same | ✅ correct | AUC 0.373, no drift |
| Image | drift | ✅ correct | AUC 0.990, drift detected |
| Joint | same | ✅ correct | AUC 0.489, no drift |
| Joint | drift | ✅ correct | AUC 0.992, drift detected |

Text, image, and joint all behaved exactly as expected on real (not synthetic) multimodal data. The one failure was root-caused, not dismissed: it's PSI's known sensitivity to low-frequency categories at small N, not a bug. A follow-up sweep on the same dog-vs-dog "same" scenario at increasing sample size confirmed it:

| N | Alert | `Color1` PSI | `State` PSI |
|---|-------|--------------|-------------|
| 100 | **True (false positive)** | 0.059 | **0.391** |
| 300 | False (correct) | 0.042 | 0.079 |
| 500 | False (correct) | 0.016 | 0.019 |
| 1,000 | False (correct) | 0.029 | 0.011 |

`State` (14 categories, several with very few samples) is where PSI's log-ratio term is most sensitive to sampling noise at small N — it resolves cleanly by N=300. See [Known Limitations](#known-limitations) for what this means in practice.

---

## Known Limitations

**Categorical drift sensitivity depends on PSI threshold — and on sample size for moderate-cardinality fields:** The current threshold (PSI > 0.2) is the industry-standard cutoff for "significant population shift." Features with genuine but small proportional shifts (like `gender_id` in the Citi Bike validation, PSI=0.043) will correctly not trigger alerts, even if chi-square would flag them as statistically significant at large sample sizes. Whether this is a limitation or correct behavior depends on your use case. Separately, verified during the [PetFinder smoke test](#real-world-multimodal-smoke-test-petfindermy): at N=100 with a 14-category field, two samples drawn from the *same* population produced a false positive (PSI=0.39) purely from low-frequency-category sampling noise; the same comparison at N=300 settled to PSI=0.08. This project's own recommended minimum (40 rows) does not fully protect against false positives on higher-cardinality categorical fields — prefer 300+ rows per batch when a categorical field has more than a handful of categories.

**Batch-based detection only:** The system compares distributions over a batch of incoming data. It does not currently support online/streaming drift detection where each individual data point updates a running estimate. Point anomalies are caught via IQR scoring, but distributional drift requires a batch.

**Baseline versioning has no retention cap:** every `/fit` call keeps its version indefinitely (see [History, Schema Validation, Versioning & Alerting](#history-schema-validation-versioning--alerting)) — there's no automatic pruning yet, so a project re-fit very frequently will accumulate versions without bound. Low risk for a typical re-fit cadence (occasional, not per-request), but worth knowing before scripting frequent automated re-fits.

**SQLite at scale:** SQLite is appropriate for moderate traffic and single-server deployments. High-concurrency production environments would benefit from migrating the storage layer to PostgreSQL.

**Text/image validation is one dataset deep; joint is still only smoke-tested:** text and image now have measured false-alarm rates, power curves, and precision/recall/F1 (see [Text/Image Validation](#textimage-validation-step-2-e-petfindermy-2026-10-05)), but on a single dataset (English pet descriptions and photos), with fixed embedding models, and with drift defined by mixing two populations — not a general guarantee. Two concrete weaknesses it found: the calibrated p-value is over-confident on real embeddings (its null grid is synthetic Gaussian), so calibrated mode only adds detection if the AUC floor is lowered below 0.65, at the cost of >5% false alarms with small batches; and results are unreliable when the *reference* is tiny (20 vs 20 false-alarmed 10%–14%). On the per-modality text/image endpoints calibrated mode stays opt-in, because it still uses the synthetic grid. The unified table path uses each project's real-embedding null instead, validated in M3/M4 (see API Reference). Joint has no precision/recall/F1 numbers at all (only unit tests, manual checks, and the dog-vs-cat smoke test) and stays legacy-only; calibration for it isn't built.

**Embedding model choice is fixed, not tunable:** `all-MiniLM-L6-v2` (text) and `resnet18` (image) were chosen for their small footprint on a free-tier deployment. There's no per-use-case model selection yet — a domain with very different characteristics (e.g. highly technical text, medical imaging) may see worse separability than these general-purpose embeddings provide.

**Heavier container images for text/image support:** `torch`, `torchvision`, and `sentence-transformers` meaningfully increase image size and cold-start time versus the previous tabular-only stack, on top of Render's existing free-tier spin-down behavior.

> **UNVERIFIED (2026-10-01): every AUC number in the next two paragraphs was measured on a mis-specified classifier.** `build_joint_classifier()` was found to never actually apply `penalty="l1"` (it silently ran as L2 the whole time the numbers below were measured), fixed in the 2026-09-30 hardening pass (`docs/PROGRESS.md`). These numbers have NOT been re-measured against the corrected L1 classifier — treat every AUC figure below as provisional until Step 4 re-evaluates.

**Joint multimodal drift detects correlation inversions only at roughly the effect size it was tuned on — this is narrower than "closes the gap":** `/fit/{project_id}/joint` and `/analyze/{project_id}/joint` detect drift in the *combination* of tabular/text/image fields (e.g. metadata that's individually normal but paired with the wrong image), by concatenating per-modality embeddings plus a bounded interaction feature into one vector and running the Domain Classifier Test with an L1-regularized classifier (`adapters/joint.py`'s `build_joint_classifier()` — still a *linear* model, just L1-penalized rather than L2; not a nonlinear classifier, and used only for joint analysis — the shared `drift/embedding_detector.py` default used by text/image is untouched). This closes the originally-documented blind spot for tabular↔text and tabular↔image correlation inversions — cases where each modality's own marginal is completely unchanged and only the *pairing* flips — **for inversions of roughly the tuned magnitude**. Verified across 8 independent data seeds, but all 8 shared the same cluster separation and image classes, varying only random noise — that's validation of one fixed effect size, not general robustness. Tested separately against a smaller, different correlation-inversion scenario: **not detected** (AUC=0.39, below chance-adjacent) at the same fixed `C=2.0`. A noisier-but-same-magnitude pairing (80/20, not perfectly deterministic) *is* still detected (AUC=0.78) — so the fix generalizes along the noise axis but not the effect-size axis. Both the passing and failing cases are encoded as actual tests in `tests/test_joint_adapter.py` (`test_correlation_inversion_generalizes_to_noisy_pairing`, `test_correlation_inversion_not_detected_at_smaller_separation`), not just described here. Separately, and unaffected by this: a **pure text↔image correlation inversion on a project with zero tabular fields declared** remains undetected regardless of effect size — there's no tabular vector for the interaction term to anchor against. Dashboard UI for this modality doesn't exist yet either — it's currently API-only.

**Why the joint path uses a different classifier than text/image, and why C isn't made adaptive:** the interaction signal that closes the gap above is usually a small minority among the much larger raw text(384)/image(512) embedding dimensions, and gets diluted by the default L2 classifier's smooth weight-shrinkage (verified: isolating just the interaction dimension gave AUC=0.75 in isolation but AUC=0.40 combined under L2). Applying the same L1 fix globally was considered and rejected: verified on real `TextAdapter` output, it measurably worsens an already-documented text/image weakness (the small-batch borderline case rises from AUC=0.691, already borderline, to 0.746 under L1). Making the classifier configurable per-call (default unchanged) let the joint path get what it needs without that regression. Scaling `C` adaptively from the baseline's observed tabular spread was also considered and rejected, not left untried: after z-scoring, the tabular subvector's variance is always ~1 by construction regardless of the real underlying cluster separation, so there's no fit-time-observable signal to adapt on — and even ignoring that, sweeping `C` for the smaller-separation case found a *different* working window (~5-20) than the tuned case's, with performance degrading to *worse than chance* (AUC=0.14) at high `C`, not merely plateauing — too unstable for a lightweight heuristic to safely target.

---

## Future Plans

- Wasserstein distance as an additional continuous drift metric
- A delivery-history endpoint for webhooks (`GET /webhooks/{project_id}/{webhook_id}/deliveries`) and a "send test event" action — the underlying `webhook_deliveries` table already records every attempt, just not exposed via the API yet
- Time-windowed drift detection (rolling window rather than fixed baseline)
- PostgreSQL support for production-scale deployments
- A retention cap/pruning script for old baseline versions (see [Known Limitations](#known-limitations))
- Full four-part validation methodology for the text/image Domain Classifier Test, matching the tabular benchmark's rigor
- Extend drift detection to audio and video via embedding-based drift metrics
- Make joint correlation-inversion detection robust across effect sizes, not just the tuned one (see Known Limitations) — adaptive `C` was tried and rejected as not viable (no fit-time-observable signal to adapt on, and the `C`-vs-AUC relationship isn't safely targetable); a genuinely different classifier family (e.g. a small tree-based or nonlinear model, actually evaluated this time rather than assumed) is the more promising unexplored direction
- Close the remaining pure text↔image correlation-inversion gap for joint projects with zero tabular fields (see Known Limitations) — without regressing text/image's existing behavior the way a global classifier change would

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## About

Built by **Abhinav Billa**, B.Tech Mathematics and Computing, Indian Institute of Science (IISc), Bangalore.

This project started from a paper on out-of-distribution detection and statistical process control, and evolved into a general-purpose, four-modality drift monitoring platform over several rounds of iterative development, each one following the same discipline: verify the current code before changing it, write the plan before writing code, and correct the documentation the moment a finding turns out narrower than first assumed.

**How it got here, roughly in order:**
1. A tabular-only engine (KS-test, PSI, IQR) validated against a real 4.5M-row dataset, with the four-part methodology in [Validation Results](#validation-results).
2. Text and image support added behind the same `BaseAdapter` interface, both reducing to the Domain Classifier Test rather than inventing separate detection logic per modality.
3. Authentication migrated from a static, self-issued API key to session tokens derived directly from Google login — removing a whole class of "forgotten API key in a script" risk, at the cost of the self-serve programmatic-access flow noted in [Known Limitations](#known-limitations).
4. Joint multimodal detection added as a fourth, fully separate capability — closing a real blind spot (a correlation break between modalities that no single-modality check can see) without touching the existing tabular/text/image pipelines. When the first version of the fix turned out to only generalize along one axis (noise) and not another (effect size), that got documented as a real, tested limitation rather than smoothed over.
5. Dataset ingestion hardened end-to-end after deliberately trying to break it: a mixed-type tabular column that could crash or silently miscategorize depending on cell order, an exception type that a bare `except ValueError` didn't actually catch, and no structural validation at the API boundary — all reproduced first, then fixed, then covered by tests that check the fix, not just the absence of a crash.
6. Validated further against real (not synthetic) multimodal data — the PetFinder.my Adoption Prediction dataset — which surfaced a genuine, previously-undocumented sample-size sensitivity in the categorical PSI test, root-caused rather than dismissed (see [Real-World Multimodal Smoke Test](#real-world-multimodal-smoke-test-petfindermy)).

The validation methodology throughout — particularly the decision to use PSI rather than chi-square for categorical ground truth, the per-case confusion matrix breakdown that isolated a silent key-type bug, and treating every "does this generalize?" question as something to actually test rather than assume — came from treating validation as a first-class engineering artifact, not an afterthought.

**GitHub:** [Abhinavbilla](https://github.com/Abhinavbilla)
