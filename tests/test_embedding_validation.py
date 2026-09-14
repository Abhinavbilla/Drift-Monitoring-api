"""
Text / Image Drift Detection — Validation Suite (Domain Classifier Test)
==========================================================================

Validates the v2.0 text/image drift monitoring path using the same
four-part methodology as tests/test_drift_engine.py's tabular validation,
adapted for embedding-based detection:

1. GROUND TRUTH VERIFICATION
   Ground truth here comes from labeled, well-established public datasets
   (20 Newsgroups topic categories; CIFAR-10 image classes) rather than
   being computed independently the way the tabular suite used scipy.
   We still verify the chosen categories are genuinely separable using a
   classifier INDEPENDENT of our embedding pipeline (TF-IDF for text, raw
   pixels for images — not sentence-transformers/resnet18), so "ground
   truth" isn't just an unverified assumption.

2. SYNTHETIC DRIFT SENSITIVITY
   Unlike tabular's sigma-based shifts, severity here is the PROPORTION
   of a different category mixed into an otherwise same-category batch
   (embeddings don't have a natural magnitude-of-shift the way numeric
   features do). Mild/moderate/severe = 10%/30%/60% foreign samples.

3. CLASSIFICATION METRICS (Precision/Recall/F1)
   Confusion matrix from three case types: same-category batches (no
   drift), moderately-mixed batches (drift), and fully-different-category
   batches (drift) — mirrors the tabular suite's Case A/B/C structure.

4. DETECTION LATENCY
   Minimum proportion of foreign-category samples needed in a fixed-size
   batch before the Domain Classifier Test fires.

IMPORTANT CAVEAT: this validates the *mechanism* — correct math, no false
positives on same-distribution data, a real detection curve — but the
"drift" here is a topic/class-shift proxy, not genuine real-world drift
observed over time the way the tabular suite's Citi Bike months-apart
split was. Report these numbers with that caveat attached, not with the
same claim of real-world realism as the tabular results.

Run:
    python tests/test_embedding_validation.py
"""

import io
import os
import base64
import time

import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

from _session_auth import mint_session_token

load_dotenv()
VALIDATION_OWNER_EMAIL = os.getenv("VALIDATION_OWNER_EMAIL", "validation-suite@drift-sentinel.local")
API_BASE = os.getenv("API_BASE", "http://api:8000")
HEADERS = {"Authorization": f"Bearer {mint_session_token(VALIDATION_OWNER_EMAIL)}"}

TEXT_MODEL_ID = "validation_text_v1"
IMAGE_MODEL_ID = "validation_image_v1"

TEXT_BASELINE_CATEGORY = "sci.space"
TEXT_DRIFT_CATEGORY = "rec.sport.hockey"

CIFAR_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cifar10_cache")
CIFAR_BASELINE_CLASS = 0  # airplane
CIFAR_DRIFT_CLASS = 3     # cat

BASELINE_SAMPLE_SIZE = 150
BATCH_SIZE = 80
N_TRIALS = 8
SEVERITIES = {"mild": 0.10, "moderate": 0.30, "severe": 0.60}


# ============================================
# DATA LOADING
# ============================================
def load_text_categories():
    print("Loading 20 Newsgroups categories...")
    data = fetch_20newsgroups(
        subset="all",
        categories=[TEXT_BASELINE_CATEGORY, TEXT_DRIFT_CATEGORY],
        remove=("headers", "footers", "quotes"),
    )
    baseline_docs = [d for d, t in zip(data.data, data.target) if data.target_names[t] == TEXT_BASELINE_CATEGORY and d.strip()]
    drift_docs = [d for d, t in zip(data.data, data.target) if data.target_names[t] == TEXT_DRIFT_CATEGORY and d.strip()]
    print(f"  {TEXT_BASELINE_CATEGORY}: {len(baseline_docs)} docs, {TEXT_DRIFT_CATEGORY}: {len(drift_docs)} docs\n")
    return baseline_docs, drift_docs


def load_image_classes():
    import torchvision

    print("Loading CIFAR-10 classes (downloads ~170MB on first run)...")
    ds = torchvision.datasets.CIFAR10(root=CIFAR_CACHE_DIR, train=True, download=True)
    baseline_imgs = [img for img, label in ds if label == CIFAR_BASELINE_CLASS]
    drift_imgs = [img for img, label in ds if label == CIFAR_DRIFT_CLASS]
    print(f"  class {CIFAR_BASELINE_CLASS} (airplane): {len(baseline_imgs)} images, "
          f"class {CIFAR_DRIFT_CLASS} (cat): {len(drift_imgs)} images\n")
    return baseline_imgs, drift_imgs


def _pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sample_mixed_batch(baseline_pool, drift_pool, n, drift_fraction, seed):
    """Samples n items, drift_fraction of them from drift_pool and the rest from baseline_pool."""
    rng = np.random.default_rng(seed)
    n_drift = int(round(n * drift_fraction))
    n_baseline = n - n_drift

    items = []
    if n_baseline > 0:
        idx = rng.choice(len(baseline_pool), size=n_baseline, replace=False)
        items += [baseline_pool[i] for i in idx]
    if n_drift > 0:
        idx = rng.choice(len(drift_pool), size=n_drift, replace=False)
        items += [drift_pool[i] for i in idx]

    rng.shuffle(items)
    return items


# ============================================
# API HELPERS
# ============================================
def fit_baseline(model_id, modality, samples):
    key = "reference_texts" if modality == "text" else "reference_images"
    payload_samples = samples if modality == "text" else [_pil_to_b64(s) for s in samples]
    resp = requests.post(
        f"{API_BASE}/fit/{model_id}/{modality}",
        json={key: payload_samples},
        headers=HEADERS,
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Fit failed for {model_id}: {resp.status_code} {resp.text}")
    return resp.json()


def analyze_batch(model_id, modality, samples):
    key = "production_texts" if modality == "text" else "production_images"
    payload_samples = samples if modality == "text" else [_pil_to_b64(s) for s in samples]
    try:
        resp = requests.post(
            f"{API_BASE}/analyze/{model_id}/{modality}",
            json={key: payload_samples},
            headers=HEADERS,
            timeout=180,
        )
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return None
    if resp.status_code != 200:
        print(f"  ❌ API Error: {resp.status_code} - {resp.text}")
        return None
    return resp.json()["feature_metrics"]["embedding_drift"]


def cleanup(model_id):
    try:
        requests.delete(f"{API_BASE}/models/{model_id}", headers=HEADERS, timeout=30)
    except Exception:
        pass


# ============================================
# PART 1 — GROUND TRUTH VERIFICATION
# ============================================
def verify_text_ground_truth(baseline_docs, drift_docs):
    print("=" * 70)
    print("PART 1: GROUND TRUTH VERIFICATION — TEXT (independent of our embedding pipeline)")
    print("=" * 70)
    X = baseline_docs + drift_docs
    y = [0] * len(baseline_docs) + [1] * len(drift_docs)
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X_vec = vectorizer.fit_transform(X)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X_vec, y, cv=5)
    print(f"  TF-IDF + LogisticRegression cross-val accuracy distinguishing "
          f"'{TEXT_BASELINE_CATEGORY}' vs '{TEXT_DRIFT_CATEGORY}': {scores.mean():.3f} (+/- {scores.std():.3f})")
    print("  (TF-IDF features, NOT the sentence-transformer embeddings our detector uses —")
    print("   genuinely independent confirmation these are separable domains.)\n")
    return scores.mean()


def verify_image_ground_truth(baseline_imgs, drift_imgs, n=300):
    print("=" * 70)
    print("PART 1: GROUND TRUTH VERIFICATION — IMAGE (independent of our embedding pipeline)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    b_idx = rng.choice(len(baseline_imgs), size=min(n, len(baseline_imgs)), replace=False)
    d_idx = rng.choice(len(drift_imgs), size=min(n, len(drift_imgs)), replace=False)
    X = np.array(
        [np.asarray(baseline_imgs[i]).flatten() for i in b_idx]
        + [np.asarray(drift_imgs[i]).flatten() for i in d_idx]
    )
    y = [0] * len(b_idx) + [1] * len(d_idx)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
    print(f"  Raw-pixel + LogisticRegression cross-val accuracy distinguishing "
          f"class {CIFAR_BASELINE_CLASS} vs class {CIFAR_DRIFT_CLASS}: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print("  (Raw flattened pixels, NOT the resnet18 embeddings our detector uses.)\n")
    return scores.mean()


# ============================================
# PART 2 — SYNTHETIC DRIFT SENSITIVITY
# ============================================
def run_sensitivity_test(modality, model_id, baseline_pool, drift_pool):
    label = modality.upper()
    print("=" * 70)
    print(f"PART 2: SYNTHETIC DRIFT SENSITIVITY — {label}")
    print("=" * 70)

    results = {}
    for severity, frac in SEVERITIES.items():
        detections = 0
        for trial in range(N_TRIALS):
            batch = sample_mixed_batch(baseline_pool, drift_pool, BATCH_SIZE, frac, seed=trial)
            metric = analyze_batch(model_id, modality, batch)
            if metric and metric["drift_detected"]:
                detections += 1
            time.sleep(0.2)
        rate = detections / N_TRIALS * 100
        results[severity] = rate
        print(f"  {severity:9s} ({frac:.0%} foreign) drift: {detections}/{N_TRIALS} detected ({rate:.0f}%)")
    print()
    return results


# ============================================
# PART 3 — CLASSIFICATION METRICS
# ============================================
def run_classification_evaluation(modality, model_id, baseline_pool, drift_pool):
    label = modality.upper()
    print("=" * 70)
    print(f"PART 3: CLASSIFICATION METRICS — {label} (Precision / Recall / F1)")
    print("=" * 70)

    y_true, y_pred = [], []
    case_results = {
        "A (same-category, expect NO drift)": [],
        "B (moderate mix, expect DRIFT)": [],
        "C (fully different category, expect DRIFT)": [],
    }

    for trial in range(N_TRIALS):
        cases = [
            ("A (same-category, expect NO drift)", 0.0, 0, 1000 + trial),
            ("B (moderate mix, expect DRIFT)", SEVERITIES["moderate"], 1, 2000 + trial),
            ("C (fully different category, expect DRIFT)", 1.0, 1, 3000 + trial),
        ]
        for case_label, frac, truth, seed in cases:
            batch = sample_mixed_batch(baseline_pool, drift_pool, BATCH_SIZE, frac, seed=seed)
            metric = analyze_batch(model_id, modality, batch)
            if metric:
                detected = metric["drift_detected"]
                y_true.append(truth)
                y_pred.append(1 if detected else 0)
                case_results[case_label].append(detected)
            time.sleep(0.2)

    print("\n  Per-case breakdown:")
    for case_label, detections in case_results.items():
        if not detections:
            print(f"    {case_label}: no data collected")
            continue
        rate = sum(detections) / len(detections) * 100
        print(f"    {case_label}: {sum(detections)}/{len(detections)} detected ({rate:.0f}%)")

    if not y_true:
        print("  ❌ No results collected — check API connectivity.")
        return None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n  Confusion Matrix:")
    print("                  Predicted No Drift   Predicted Drift")
    print(f"  Actual No Drift   {cm[0][0]:>15d}   {cm[0][1]:>15d}")
    print(f"  Actual Drift      {cm[1][0]:>15d}   {cm[1][1]:>15d}")
    print(f"\n  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}\n")
    print(classification_report(y_true, y_pred, target_names=["No Drift", "Drift"], zero_division=0))

    return {"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm.tolist()}


# ============================================
# PART 4 — DETECTION LATENCY
# ============================================
def measure_detection_latency(modality, model_id, baseline_pool, drift_pool, batch_size=60, step_fraction=0.1):
    fraction = 0.0
    while fraction <= 1.0001:
        batch = sample_mixed_batch(baseline_pool, drift_pool, batch_size, fraction, seed=99)
        metric = analyze_batch(model_id, modality, batch)
        if metric and metric["drift_detected"]:
            return fraction
        time.sleep(0.2)
        fraction += step_fraction
    return None


def run_latency_test(modality, model_id, baseline_pool, drift_pool):
    label = modality.upper()
    print("=" * 70)
    print(f"PART 4: DETECTION LATENCY — {label} (minimum % foreign-category to trigger alert)")
    print("=" * 70)
    latency = measure_detection_latency(modality, model_id, baseline_pool, drift_pool)
    if latency is not None:
        print(f"  Alert fires once {latency:.0%} of batch is from the foreign category\n")
    else:
        print("  Never triggered within tested range (needs investigation)\n")
    return latency


# ============================================
# ORCHESTRATION
# ============================================
def run_validation_for_modality(modality, model_id, baseline_pool, drift_pool):
    label = modality.upper()
    print("\n" + "#" * 70)
    print(f"# {label} VALIDATION")
    print("#" * 70 + "\n")

    print(f"Locking {modality} baseline ({BASELINE_SAMPLE_SIZE} samples from the baseline category)...")
    baseline_batch = sample_mixed_batch(baseline_pool, drift_pool, BASELINE_SAMPLE_SIZE, 0.0, seed=42)
    fit_baseline(model_id, modality, baseline_batch)
    print("Baseline locked.\n")

    sensitivity_results = run_sensitivity_test(modality, model_id, baseline_pool, drift_pool)
    classification_results = run_classification_evaluation(modality, model_id, baseline_pool, drift_pool)
    latency = run_latency_test(modality, model_id, baseline_pool, drift_pool)

    return {
        "sensitivity": sensitivity_results,
        "classification": classification_results,
        "latency": latency,
    }


def print_summary(text_results, image_results):
    print("=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    for label, results in [("TEXT", text_results), ("IMAGE", image_results)]:
        print(f"\n{label}:")
        print(f"  Sensitivity: {results['sensitivity']}")
        if results["classification"]:
            c = results["classification"]
            print(f"  Precision: {c['precision']:.3f}  Recall: {c['recall']:.3f}  F1: {c['f1']:.3f}")
        latency = results["latency"]
        if latency is not None:
            print(f"  Detection latency: {latency:.0%} foreign samples")
        else:
            print("  Detection latency: not triggered")
    print()
    print("Caveat: these numbers validate the MECHANISM using labeled public")
    print("datasets as a topic/class-shift proxy for drift, not genuine real-world")
    print("drift observed over time the way the tabular Citi Bike validation was.")
    print("Don't present them with the same claim of real-world realism.")
    print("=" * 70)


if __name__ == "__main__":
    text_baseline_docs, text_drift_docs = load_text_categories()
    verify_text_ground_truth(text_baseline_docs, text_drift_docs)
    try:
        text_results = run_validation_for_modality("text", TEXT_MODEL_ID, text_baseline_docs, text_drift_docs)
    finally:
        cleanup(TEXT_MODEL_ID)

    image_baseline_imgs, image_drift_imgs = load_image_classes()
    verify_image_ground_truth(image_baseline_imgs, image_drift_imgs)
    try:
        image_results = run_validation_for_modality("image", IMAGE_MODEL_ID, image_baseline_imgs, image_drift_imgs)
    finally:
        cleanup(IMAGE_MODEL_ID)

    print_summary(text_results, image_results)
