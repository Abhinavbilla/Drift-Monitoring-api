```markdown
# Drift-Monitoring-api

**Production-grade drift monitoring for machine learning models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

**Drift-Monitoring-api** is a production-ready MLOps platform that automatically detects distribution shifts in your ML features – before they impact your business.

It combines:
- **KS-test** for continuous feature drift
- **PSI (Population Stability Index)** for categorical feature drift
- **IQR (3× interquartile range)** for real-time anomaly detection

All wrapped in an interactive Streamlit dashboard and a high-performance FastAPI backend with Google OAuth authentication.

---

## Benchmarks

Validated on the **NYC Citi Bike dataset** (4.5M rows, 7 monitored features) with independently verified ground truth.

| Metric | Score |
|--------|-------|
| **Precision** | **1.000** |
| **Recall** | **0.945** |
| **F1 Score** | **0.972** |

### Per-feature detection rate (real production data)

| Feature | Effect Size | Detection Rate |
|---------|-------------|----------------|
| `trip_duration` | KS=0.0924 | 100% |
| `month` (categorical) | PSI=16.27 | 100% |
| `pickup_longitude` | KS=0.0219 | 100% |
| `dropoff_latitude` | KS=0.0195 | 100% |
| `dropoff_longitude` | KS=0.0206 | 87% |
| `pickup_latitude` | KS=0.0189 | 53% |
| `gender_id` (categorical) | PSI=0.043 | 0% (no practical drift) |

> Detection rate correlates with effect size – this is expected statistical behaviour, not a system flaw.

### Recommended production batch size: **20,000 rows**

| Batch Size | Recall | F1 |
|------------|--------|-----|
| 1,000 | 0.458 | 0.629 |
| 3,000 | 0.729 | 0.843 |
| 10,000 | 0.812 | 0.886 |
| **20,000** | **0.896** | **0.945** |
| 50,000 | 0.917 | 0.957 |

> 20,000 rows sits at the knee of the recall curve – capturing most sensitivity without diminishing returns.

---

## How It Works

| Feature Type | Detection Method |
|--------------|------------------|
| Continuous | KS-test (p < 0.05) |
| Categorical | PSI (threshold > 0.2) |
| Real-time anomalies | IQR (3×) |

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Google OAuth credentials](https://console.cloud.google.com/apis/credentials)

### Installation

```bash
# Clone the repository
git clone https://github.com/Abhinavbilla/Drift-Monitoring-api.git
cd Drift-Monitoring-api

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Google OAuth client ID, secret, and a cookie key
```