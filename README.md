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

You don't need to clone this repository or install anything to start using the API. The live deployment is open and ready.

**Base URL:** `https://drift-monitoring-dashboard.onrender.com`

### Step 1: Get an API Key

Visit the [live dashboard](https://drift-monitoring-dashboard.onrender.com/), sign in with Google, and click **Generate API Key** in the Developer Portal tab. Your key will be displayed once and should be saved securely.

### Step 2: Lock a Baseline

Send your training data to the `/fit` endpoint. This locks the reference distribution that all future production batches will be compared against.

```python
import requests

API_KEY = "your_api_key"
MODEL_ID = "my_model_v1"
BASE_URL = "https://drift-monitoring-dashboard.onrender.com"

payload = {
    "reference_data": {
        "age": [25, 34, 45, 29, 52, 38, 41],
        "transaction_amount": [120.5, 89.0, 340.2, 55.8, 210.0, 175.3, 98.6],
        "merchant_category": ["retail", "food", "retail", "travel", "food", "retail", "food"]
    }
}

response = requests.post(
    f"{BASE_URL}/fit/{MODEL_ID}",
    json=payload,
    headers={"X-API-Key": API_KEY}
)

print(response.json())
```

### Step 3: Analyze a Production Batch

Once a baseline is locked, send production batches to `/analyze`. The API returns a drift verdict per feature and a top-level system alert flag.

```python
payload = {
    "production_data": {
        "age": [55, 61, 70, 48, 63, 57, 72],
        "transaction_amount": [890.0, 1200.5, 750.0, 980.3, 1100.0, 860.0, 920.0],
        "merchant_category": ["luxury", "luxury", "travel", "luxury", "luxury", "travel", "luxury"]
    }
}

response = requests.post(
    f"{BASE_URL}/analyze/{MODEL_ID}",
    json=payload,
    headers={"X-API-Key": API_KEY}
)

print(response.json())
# {"system_alert_triggered": true, "feature_metrics": {...}}
```

### Step 4: Explore the Full API

Interactive docs with all endpoints, request schemas, and response examples are available at:

**[https://drift-monitoring-dashboard.onrender.com/docs](https://drift-monitoring-dashboard.onrender.com/docs)**

---

## Scope and Supported Data

This API is built specifically for **structured, tabular data**. It is not designed for images, audio, video, or free-text columns.

| Supported | Not Supported |
|-----------|---------------|
| CSV, TSV | Images |
| Excel (.xlsx, .xls) | Audio / Video |
| JSON, JSON Lines | Free-text (NLP) fields |
| Parquet | Unstructured data |
| ARFF (Weka format) | |
| Compressed (.gz, .zip) | |

**Why this scope?**

KS-test, PSI, and IQR-based methods are mathematically grounded in continuous and categorical feature distributions. Unstructured data requires fundamentally different approaches (embedding drift, perceptual hashing, NLP-specific metrics) that are out of scope here. Drift Monitoring API focuses on doing structured tabular monitoring well rather than doing everything poorly.

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

**Multi-format File Ingestion**

The file reader (implemented in `dashboard.py`) handles encoding detection automatically (UTF-8, CP1252, Latin-1, ISO-8859-1), parses ARFF attribute headers, detects libsvm-format .dat files, reads all sheets from multi-sheet Excel files with a schema consistency warning, and handles gzip and zip compressed inputs without requiring pre-processing.

**Secure Multi-tenant Architecture**

Each user authenticates via Google OAuth, gets a provisioned API key, and can only access monitoring data for their own projects. The developer portal exposes key generation directly from the dashboard UI.

**Containerized Deployment**

The entire stack — FastAPI backend, Streamlit dashboard, and supervisor process management — is containerized with Docker and orchestrated via `docker-compose.yml`, making it straightforward to deploy on any cloud provider.

---

## How It Works

The system is split into three layers that interact in a defined sequence:

### 1. Profiler (`utils/profiler.py`)

Takes a sample of your uploaded training data and computes a set of mathematical signals for each column: cardinality ratio, dominant value ratio, monotonicity, string length consistency, structured pattern detection, and dtype analysis after attempted coercion. These signals feed a routing decision that classifies each column as continuous, categorical, or ignored — with a reasoning string attached to every decision so it's auditable in the UI.

### 2. Baseline Storage (`db/crud.py`)

Once the user confirms the schema, the system computes and stores two kinds of baseline statistics in SQLite:

- IQR fences (Q1, Q3) for every continuous feature
- Frequency distribution tables for every categorical feature

These are the reference distributions everything in production gets compared against.

### 3. Drift Detection (`drift/detector.py`)

At inference time, incoming production batches are compared against the stored baseline. Continuous features go through a two-sample KS-test. Categorical features go through PSI computed against the stored frequency table. Individual predictions are also scored against IQR fences for real-time anomaly detection. CUSUM-based sequential detection is also available via `drift/cusum.py` for tracking gradual shifts over time.

### The Full Flow

```
Upload training data
        ↓
Profiler classifies columns (auto + human confirmation)
        ↓
Baseline locked in SQLite (IQR fences + frequency tables)
        ↓
Production data sent to /analyze
        ↓
KS-test (continuous) + PSI (categorical) + IQR scoring (real-time)
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
| Authentication | Google OAuth 2.0 |
| Drift Detection | scipy (KS-test), custom PSI, custom CUSUM (`drift/cusum.py`) |
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
│   ├── detector.py            # KS-test, PSI, and IQR detection engines
│   ├── alerts.py              # Alert triggering and notification logic
│   ├── baseline.py            # Baseline computation utilities
│   └── cusum.py               # Custom CUSUM sequential drift detection
│
├── adapters/
│   ├── base.py                # Abstract adapter interface
│   └── tabular.py             # Tabular data adapter
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
2. Click **Generate API Key** in the Developer Portal tab to provision your credentials
3. Upload your training data (CSV, Excel, JSON, Parquet, or ARFF)
4. Review the auto-generated schema — the profiler will classify each column and explain its reasoning
5. Adjust any misclassified columns using the dropdowns, then click **Start Monitoring**

### Sending Production Data

Once a baseline is locked, send production batches to the `/analyze` endpoint:

```python
import requests

API_KEY = "your_api_key"
MODEL_ID = "your_model_id"

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
    headers={"X-API-Key": API_KEY}
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
| `/register` | POST | Provision an API key for a new user |
| `/fit/{project_id}` | POST | Lock a baseline from training data |
| `/analyze/{project_id}` | POST | Compare a production batch against the stored baseline |
| `/profile` | POST | Profile a dataset's columns without locking a baseline |
| `/projects` | GET | List all projects for the authenticated user |
| `/baseline/{project_id}` | GET | Fetch IQR fences and feature types for a project |
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

---

## Known Limitations

**Categorical drift sensitivity depends on PSI threshold:** The current threshold (PSI > 0.2) is the industry-standard cutoff for "significant population shift." Features with genuine but small proportional shifts (like `gender_id` in the Citi Bike validation, PSI=0.043) will correctly not trigger alerts, even if chi-square would flag them as statistically significant at large sample sizes. Whether this is a limitation or correct behavior depends on your use case.

**Batch-based detection only:** The system compares distributions over a batch of incoming data. It does not currently support online/streaming drift detection where each individual data point updates a running estimate. Point anomalies are caught via IQR scoring, but distributional drift requires a batch.

**Single baseline per project:** Each project has one active baseline. If your model is retrained and the new model operates on a shifted feature distribution (intentionally), you need to re-fit the baseline manually. There is no automatic baseline versioning yet.

**SQLite at scale:** SQLite is appropriate for moderate traffic and single-server deployments. High-concurrency production environments would benefit from migrating the storage layer to PostgreSQL.

---

## Future Plans

- Wasserstein distance as an additional continuous drift metric
- Webhook support for drift alerts (Slack, email, PagerDuty)
- Baseline versioning with drift history across model versions
- Time-windowed drift detection (rolling window rather than fixed baseline)
- PostgreSQL support for production-scale deployments
- REST API client SDK (Python package)
- Extend drift detection support to unstructured data (text, images, audio, video) using embedding-based drift metrics and perceptual hashing

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## About

Built by **Abhinav Billa**, B.Tech Mathematics and Computing, Indian Institute of Science (IISc), Bangalore.

This project started from a paper on out-of-distribution detection and statistical process control, and evolved into a general-purpose drift monitoring platform over several days of iterative development and debugging.

The validation methodology, particularly the decision to use PSI rather than chi-square for categorical ground truth, and the per-case confusion matrix breakdown that isolated a silent key-type bug, came from treating the validation suite as a first-class engineering artifact rather than an afterthought.

**GitHub:** [Abhinavbilla](https://github.com/Abhinavbilla)
