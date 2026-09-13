```sh
# Drift Monitoring API
## Specification Document (v2.0 — Structured + Unstructured Data)

## 0. Problem Statement

Most ML teams catch model degradation too late — after a model's underlying data has silently shifted and performance has already eroded in production. Drift Monitoring API addresses this by continuously comparing live production data against a locked baseline distribution and flagging statistically meaningful shifts before they become business problems.

The system originally covered only **structured, tabular data** (CSV, Excel, JSON, Parquet, ARFF) using KS-test, PSI, and IQR-based methods — techniques mathematically grounded in continuous/categorical feature distributions. This left a real gap: production ML systems increasingly consume **text** (reviews, support tickets, chat logs) and **images** (product photos, scans, satellite imagery) as model inputs, and these drift too — a sentiment classifier trained on formal reviews silently degrading against slang-heavy text, or a vision model failing when fed a batch of visually different images. Teams needed either a second, unrelated tool for these inputs, or nothing at all.

**What this spec covers:** a unified system where structured and unstructured data share the same baseline-lock/compare-batch/alert workflow, the same auth, storage, and dashboard — while using the detection math appropriate to each modality under the hood. The codebase already anticipated this: an abstract adapter interface (`adapters/base.py`) and a stubbed `adapters/image.py` existed alongside the working `adapters/tabular.py` from the start.

**What "solved" looks like:** a user locks a baseline from tabular data, text, or images — through the same conceptual workflow — and gets back a drift verdict with the same rigor (validated detection rates, confusion-matrix-backed precision/recall, documented limitations) regardless of modality.

---

## 1. Purpose

Provide one drift-monitoring system that detects distribution shift across three input modalities — tabular, text, and image — through a shared adapter-based architecture, shared baseline storage, shared auth, and a shared dashboard, using modality-appropriate detection methods underneath a common interface.

---

## 2. Scope

**In scope:**
- Tabular data: CSV, TSV, Excel, JSON/JSONL, Parquet, ARFF, compressed inputs (existing, unchanged)
- Text data: batches of raw strings (reviews, tickets, short documents)
- Image data: batches of images (JPEG, PNG)
- Detection methods:
  - Tabular: two-sample KS-test (continuous), PSI (categorical), IQR-based real-time anomaly scoring (existing, unchanged)
  - Text & Image: embedding-based **Domain Classifier Test** (train a classifier to distinguish reference vs. current embeddings; AUC significantly above 0.5 indicates drift) — modality-agnostic, same logic for both
- Shared baseline storage (SQLite), shared Google OAuth/API-key auth, shared Streamlit dashboard
- Human-in-the-loop schema confirmation for tabular (existing); equivalent lightweight confirmation step for text/image baselines (e.g. confirming embedding model choice, sample count)

**Out of scope (this version):**
- Video, audio
- Fine-grained per-token or per-pixel drift localization — batch-level verdicts only
- Real-time/streaming embedding computation — batch-based only, consistent with existing tabular limitation
- Migrating storage off SQLite (existing known limitation, unchanged)
- CUSUM-style sequential detection for text/image (tabular-only for now, via `drift/cusum.py`)

---

## 3. Architecture

Three input modalities converge on one shared pipeline shape: **adapt → (embed, for unstructured) → detect → store/report.**
```

```sh

**Why the Domain Classifier Test
 for unstructured data:** it needs no distributional assumptions (unlike KS-test, which requires continuous scalar values), works identically regardless of modality since both text and image embeddings are just vectors, and produces one interpretable number (AUC) that maps cleanly onto the existing dashboard's drift-score display — a genuine unifying method rather than a bolted-on second system.

---

## 4. Components

| Component | File | Status | Purpose |
|---|---|---|---|
| Column profiler | `utils/profiler.py` | existing | Auto-classify tabular columns |
| Tabular adapter | `adapters/tabular.py` | existing | Tabular ingestion |
| Text adapter | `adapters/text.py` | **new** | Text batch → embedding matrix |
| Image adapter | `adapters/image.py` | **complete stub** | Image batch → embedding matrix |
| Tabular detector | `drift/detector.py` | existing | KS-test, PSI, IQR |
| Embedding detector | `drift/embedding_detector.py` | **new** | Domain Classifier Test (AUC-based) |
| CUSUM detector | `drift/cusum.py` | existing (tabular only) | Sequential drift tracking |
| Baseline storage | `crud.py`, `db/crud.py` | existing, **extend** | Add embedding/summary storage for text/image |
| API endpoints | `main.py` | existing, **extend** | Route `/fit`, `/analyze` (or new endpoints) by modality |
| Dashboard | `dashboard.py` | existing, **extend** | Display drift results across all three modalities |

All new/completed adapters **must implement `adapters/base.py`'s abstract interface** — non-negotiable, since that's what lets baseline storage, `/analyze` routing, and the dashboard treat all three modalities uniformly rather than as three separate systems bolted together.

---

## 5. Contracts

### 5.1 Tabular Adapter — *(existing, unchanged)*
As currently implemented in `adapters/tabular.py`.

### 5.2 Text Adapter (`adapters/text.py`) — new
- **Input:** list of raw strings
- **Output:** `(n_samples, embedding_dim)` numpy array
- **Acceptance criterion:** deterministic given the same model version; handles empty strings and very long strings without crashing (truncate per model's max token length)

### 5.3 Image Adapter (`adapters/image.py`) — complete the stub
- **Input:** list of image file paths or byte streams
- **Output:** `(n_samples, embedding_dim)` numpy array
- **Acceptance criterion:** handles common formats (JPEG, PNG) and mismatched image sizes without crashing (resize/preprocess per model requirements)

### 5.4 Tabular Detector — *(existing, unchanged)*
As currently implemented in `drift/detector.py`.

### 5.5 Embedding Detector (`drift/embedding_detector.py`) — new
- **Input:** reference embedding matrix, current embedding matrix
- **Output:** AUC score, drift boolean (thresholded), optional per-sample classifier confidence (future localization hook)
- **Acceptance criterion:** on two batches drawn from the *same* distribution, AUC should center near 0.5 across repeated trials (no systematic false-positive bias)

### 5.6 Baseline Storage — extend
- **New capability:** store either raw reference embeddings or embedding summary statistics per project, keyed by modality
- **Acceptance criterion:** tabular baselines are unaffected by this change; existing tests (`tests/test_drift_engine.py`) still pass unmodified

---

## 6. Validation Plan (all modalities, same rigor)

Follow the same four-part methodology used for the existing tabular validation on the Citi Bike dataset — applied per modality:

1. **Ground Truth Verification** — for tabular: `scipy.stats.ks_2samp` / PSI (existing). For text/image: a known, labeled dataset with a clear distribution split (e.g. two distinct text corpora, two distinct image categories) as ground truth.
2. **Synthetic Drift Sensitivity** — inject controlled shifts (tabular: σ-based shifts, existing; text/image: increasing proportion of out-of-distribution samples) across multiple severity levels, multiple trials per level.
3. **Confusion Matrix Evaluation** — same-distribution batches (should not trigger) vs. known-drifted batches (should trigger); report Precision, Recall, F1 via scikit-learn for each modality.
4. **Detection Latency** — minimum proportion of a batch that must be drifted before reliable detection, per modality.

**No numbers from the text/image path go into a README or resume until this validation is actually run** — same discipline already established for the tabular benchmark.

---

## 7. Success Criteria (v2.0 demo)

- [ ] `adapters/text.py` and `adapters/image.py` both implement `adapters/base.py`'s interface and produce fixed-dim embeddings
- [ ] Domain Classifier Test correctly distinguishes at least one clearly-drifted text scenario and one clearly-drifted image scenario
- [ ] Same-distribution batches (text and image) do not falsely trigger drift across repeated trials
- [ ] Existing tabular pipeline is fully unaffected — `tests/test_drift_engine.py` still passes unmodified
- [ ] Dashboard displays drift results for all three modalities (even a minimal per-modality metric card is enough for v2.0)
- [ ] Full four-part validation plan (Section 6) run at least once per modality, with real numbers, before any resume or README claim
- [ ] README's "Scope and Supported Data" table is updated to reflect the new modalities once this is actually working (not before)

---

## 8. Known Limitations (carried forward + new)

- **SQLite at scale** — unchanged existing limitation, now also applies to embedding storage.
- **Batch-based detection only** — unstructured detection inherits this; no streaming embedding support in v2.0.
- **Single baseline per project** — unchanged; applies across all modalities.
- **New:** embedding model choice trades off speed vs. detection quality — no per-use-case auto-selection yet.
- **New:** `torch` + `sentence-transformers` + CLIP/ResNet meaningfully increase dependency and container size versus the existing lightweight tabular-only stack — may affect Render deployment (existing cold-start delay could worsen).

---

## 9. Open Questions
- Embedding model choice: `all-MiniLM-L6-v2` (small, fast, 384-dim) vs. a larger sentence-transformer?
- Store raw reference embeddings (larger, more flexible) vs. summary statistics (smaller, less flexible) in SQLite?
- Does Render's container memory/size limit tolerate the new dependencies, or does this push toward a different deployment target for the unstructured path specifically?
- Equivalent "significant drift" AUC threshold to PSI's existing 0.2 cutoff — fixed, or tuned per modality?
```