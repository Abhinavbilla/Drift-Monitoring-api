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
13. [Testing and Validation Methodology](#testing-and-validation-methodology)
14. [Known Limitations](#known-limitations)
15. [Future Plans](#future-plans)
16. [License](#license)
17. [About](#about)

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

Drift Monitoring API covers three input modalities through one shared adapter/baseline/dashboard architecture: **tabular**, **text**, and **image** data. Tabular uses statistically-grounded methods (KS-test, PSI, IQR); text and image share a single modality-agnostic method (the **Domain Classifier Test**), since both reduce to comparing two sets of embedding vectors.

| Modality | Supported Inputs | Detection Method |
|----------|-------------------|-------------------|
| Tabular | CSV, TSV, Excel (.xlsx, .xls), JSON, JSON Lines, Parquet, ARFF, compressed (.gz, .zip) | Two-sample KS-test (continuous), PSI (categorical), IQR (real-time) |
| Text | Batches of raw strings | Domain Classifier Test (AUC-based) on `all-MiniLM-L6-v2` embeddings |
| Image | JPEG, PNG batches | Domain Classifier Test (AUC-based) on `resnet18` embeddings |

**Not supported:** audio, video, per-token/per-pixel drift localization, and real-time/streaming detection for any modality (all three are batch-based).

**Why this scope?**

KS-test, PSI, and IQR-based methods are mathematically grounded in continuous and categorical feature distributions, and remain tabular-only. Text and image don't share that mathematical structure with each other or with tabular data, but they *do* share one with each other — both are just vectors once embedded — so a single Domain Classifier Test (train a classifier to distinguish reference vs. current embeddings; AUC well above 0.5 indicates drift) covers both without inventing two separate systems.

> **Validation status:** the tabular path has the full four-part validation methodology behind it (see [Validation Results](#validation-results)) run against a 4.5M-row real-world dataset. The text/image path has been smoke-tested (adapters produce correct, deterministic embeddings; the detector correctly centers near AUC=0.5 on same-distribution data and flags clearly-separated data) and manually verified end-to-end, but has **not** yet been through that same four-part methodology — no precision/recall/F1 numbers are published for it, deliberately, until that's done. One finding from manual testing worth noting: at small batch sizes (~40 samples), two genuinely different-but-same-domain text batches produced a borderline AUC (0.69, just above the 0.65 threshold) — the AUC threshold and/or minimum recommended batch size for text may need tuning once the full validation is run.

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

**Multi-format File Ingestion**

The file reader (implemented in `dashboard.py`) handles encoding detection automatically (UTF-8, CP1252, Latin-1, ISO-8859-1), parses ARFF attribute headers, detects libsvm-format .dat files, reads all sheets from multi-sheet Excel files with a schema consistency warning, and handles gzip and zip compressed inputs without requiring pre-processing.

**Secure Multi-tenant Architecture**

Each user authenticates via Google OAuth and can only access monitoring data for their own projects. Backend requests are authorized with a short-lived session token minted from that login (signed with a secret shared between the dashboard and backend) — there's no separate API key to provision or manage.

**Containerized Deployment**

The entire stack — FastAPI backend, Streamlit dashboard, and supervisor process management — is containerized with Docker and orchestrated via `docker-compose.yml`, making it straightforward to deploy on any cloud provider.

---

## How It Works

All three modalities converge on one shared pipeline shape — **adapt → (embed, for text/image) → detect → store/report** — implemented behind a common `BaseAdapter` interface (`adapters/base.py`) so baseline storage, `/analyze` routing, and the dashboard treat tabular, text, and image uniformly rather than as separate bolted-together systems.

### 1. Profiler (`utils/profiler.py`) — tabular only

Takes a sample of your uploaded training data and computes a set of mathematical signals for each column: cardinality ratio, dominant value ratio, monotonicity, string length consistency, structured pattern detection, and dtype analysis after attempted coercion. These signals feed a routing decision that classifies each column as continuous, categorical, or ignored — with a reasoning string attached to every decision so it's auditable in the UI. Text and image baselines skip this step entirely — there's no per-column schema to confirm for a single embedding matrix.

### 2. Adapters (`adapters/`) — modality-specific ingestion

- `tabular.py`: passes columnar data through unchanged (KS/PSI/IQR consume it directly)
- `text.py`: embeds a batch of raw strings via `all-MiniLM-L6-v2` (384-dim)
- `image.py`: embeds a batch of JPEG/PNG images via `resnet18`'s penultimate layer (512-dim), handling mixed sizes/formats via resize + RGB conversion

### 3. Baseline Storage (`db/crud.py`)

Once the user confirms the schema (tabular) or uploads a reference batch (text/image), the system locks a baseline in SQLite:

- Tabular: IQR fences (Q1, Q3) for continuous features, frequency tables for categorical features
- Text/Image: a capped sample of raw reference embeddings (the Domain Classifier Test needs real vectors to retrain against on every `/analyze` call, not just summary statistics)

### 4. Drift Detection (`drift/detector.py`, `drift/embedding_detector.py`)

At inference time, incoming production batches are compared against the stored baseline. Tabular: continuous features go through a two-sample KS-test, categorical through PSI, individual predictions also scored against IQR fences for real-time anomaly detection. Text/Image: current embeddings are compared against the stored reference embeddings via the Domain Classifier Test (cross-validated logistic regression, AUC-based). CUSUM-based sequential detection is also available via `drift/cusum.py` for tabular gradual shifts over time.

### The Full Flow

```
Upload reference data (tabular / text / image)
        ↓
Tabular: profiler classifies columns (auto + human confirmation)
Text/Image: adapter embeds the reference batch
        ↓
Baseline locked in SQLite
        ↓
Production batch sent to /analyze/{project_id}[/text|/image]
        ↓
Tabular: KS-test + PSI + IQR scoring (real-time)
Text/Image: Domain Classifier Test (AUC-based)
        ↓
Results surfaced in Streamlit dashboard
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
| Tabular Drift Detection | scipy (KS-test), custom PSI, custom CUSUM (`drift/cusum.py`) |
| Text/Image Drift Detection | Domain Classifier Test — scikit-learn (`drift/embedding_detector.py`) |
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
│   └── helpers.py             # Shared utility functions
│
├── drift/
│   ├── detector.py            # KS-test, PSI, and IQR detection engines (tabular)
│   ├── embedding_detector.py  # Domain Classifier Test (text/image)
│   ├── alerts.py              # Alert triggering and notification logic
│   ├── baseline.py            # Baseline computation utilities
│   └── cusum.py               # Custom CUSUM sequential drift detection (tabular)
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
├── data/                      # Sample datasets for testing and validation
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

---

## Usage

### First-Time Setup

1. Open the dashboard and sign in with Google OAuth
2. Upload your training data (CSV, Excel, JSON, Parquet, or ARFF)
3. Review the auto-generated schema — the profiler will classify each column and explain its reasoning
4. Adjust any misclassified columns using the dropdowns, then click **Start Monitoring**

### Sending Production Data

Production batches are sent to the `/analyze` endpoint through the dashboard's upload flow. Direct programmatic calls to `/analyze` require a valid session token in the `Authorization` header (see [Known Limitations](#known-limitations) — there's currently no self-serve way to obtain one outside the dashboard's own login flow):

```python
import requests

MODEL_ID = "your_model_id"
SESSION_TOKEN = "..."  # minted from a Google login, see dashboard.py's mint_session_token

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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fit/{project_id}` | POST | Lock a tabular baseline from training data |
| `/analyze/{project_id}` | POST | Compare a tabular production batch against the stored baseline |
| `/fit/{project_id}/text` | POST | Lock a text baseline (embeds `reference_texts`) |
| `/analyze/{project_id}/text` | POST | Compare a text production batch via the Domain Classifier Test |
| `/fit/{project_id}/image` | POST | Lock an image baseline (embeds base64-encoded `reference_images`) |
| `/analyze/{project_id}/image` | POST | Compare an image production batch via the Domain Classifier Test |
| `/profile` | POST | Profile a tabular dataset's columns without locking a baseline |
| `/projects` | GET | List all projects for the authenticated user |
| `/baseline/{project_id}` | GET | Fetch IQR fences, feature types, and modality for a project |
| `/logs/{project_id}` | GET | Retrieve recent logs for a project |
| `/models/{model_id}` | DELETE | Permanently delete a model and associated data |
| `/docs` | GET | Interactive Swagger UI |

Full request/response schemas are available at `/docs` when the server is running.

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

---

## Known Limitations

**Categorical drift sensitivity depends on PSI threshold:** The current threshold (PSI > 0.2) is the industry-standard cutoff for "significant population shift." Features with genuine but small proportional shifts (like `gender_id` in the Citi Bike validation, PSI=0.043) will correctly not trigger alerts, even if chi-square would flag them as statistically significant at large sample sizes. Whether this is a limitation or correct behavior depends on your use case.

**Batch-based detection only:** The system compares distributions over a batch of incoming data. It does not currently support online/streaming drift detection where each individual data point updates a running estimate. Point anomalies are caught via IQR scoring, but distributional drift requires a batch.

**Single baseline per project:** Each project has one active baseline. If your model is retrained and the new model operates on a shifted feature distribution (intentionally), you need to re-fit the baseline manually. There is no automatic baseline versioning yet.

**SQLite at scale:** SQLite is appropriate for moderate traffic and single-server deployments. High-concurrency production environments would benefit from migrating the storage layer to PostgreSQL.

**No self-serve credential flow for programmatic API access:** Backend authentication is derived directly from Google login (the dashboard mints a short-lived session token after you sign in) rather than a static, separately-provisioned API key. This removes a class of "forgotten API key sitting in a script" risk, but it also means there's currently no way to obtain a valid credential for calling `/fit` or `/analyze` from your own external script without going through the dashboard's own login flow. A proper service-account/personal-access-token feature would be needed to support that use case again.

**Text/image validation is smoke-tested, not yet fully validated:** Unlike the tabular path's four-part methodology against a 4.5M-row real dataset, the text/image Domain Classifier Test has only been verified with unit-level smoke tests and manual end-to-end checks — no formal precision/recall/F1 numbers exist for it yet (see [Scope and Supported Data](#scope-and-supported-data)). Treat text/image drift verdicts as directionally useful, not benchmarked.

**Embedding model choice is fixed, not tunable:** `all-MiniLM-L6-v2` (text) and `resnet18` (image) were chosen for their small footprint on a free-tier deployment. There's no per-use-case model selection yet — a domain with very different characteristics (e.g. highly technical text, medical imaging) may see worse separability than these general-purpose embeddings provide.

**Heavier container images for text/image support:** `torch`, `torchvision`, and `sentence-transformers` meaningfully increase image size and cold-start time versus the previous tabular-only stack, on top of Render's existing free-tier spin-down behavior.

---

## Future Plans

- Wasserstein distance as an additional continuous drift metric
- Webhook support for drift alerts (Slack, email, PagerDuty)
- Baseline versioning with drift history across model versions
- Time-windowed drift detection (rolling window rather than fixed baseline)
- PostgreSQL support for production-scale deployments
- REST API client SDK (Python package)
- Full four-part validation methodology for the text/image Domain Classifier Test, matching the tabular benchmark's rigor
- Service-account/personal-access-token support for programmatic API access outside the dashboard
- Extend drift detection to audio and video via embedding-based drift metrics

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## About

Built by **Abhinav Billa**, B.Tech Mathematics and Computing, Indian Institute of Science (IISc), Bangalore.

This project started from a paper on out-of-distribution detection and statistical process control, and evolved into a general-purpose drift monitoring platform over several days of iterative development and debugging.

The validation methodology, particularly the decision to use PSI rather than chi-square for categorical ground truth, and the per-case confusion matrix breakdown that isolated a silent key-type bug, came from treating the validation suite as a first-class engineering artifact rather than an afterthought.

**GitHub:** [Abhinavbilla](https://github.com/Abhinavbilla)
