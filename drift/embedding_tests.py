"""
Embedding-based tests for the unified table path (milestone M3,
docs/unified_table_plan.md sections I.1, I.5, I.6, J).

- Column drift for text/image: the existing Domain Classifier Test AUC, with a
  p-value from disjoint splits of THIS project's reference embeddings at the
  batch size (real embeddings, unlike drift/dct_calibration.py's synthetic
  Gaussian grid, which Step 2(e) measured as over-confident).
- Probes: how well a text/image column predicts a categorical or numeric
  column. Reference score is cross-fitted; production score uses the probe
  fitted on the whole reference. A drop means the pairing weakened.
- Matching: whether a row's image still goes with its own text (ridge map
  image -> text space, AUC of own-text vs other-text cosine).

Everything runs on PCA-reduced embeddings (fitted on the reference): it keeps
the repeated classifier fits affordable and the small-sample noise down.
Probe/matching models are stored as plain arrays, never pickles.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

from drift.relationship_detector import spearman

PCA_DIM = 64
NULL_DRAWS = 200
MIN_ROWS = 30
FLOORS = {"probe": 0.10, "text_image": 0.10}
PROPOSE_MIN = {"probe_categorical": 0.10, "probe_numeric": 0.3, "text_image": 0.10}
_MATCH_NEGATIVES = 5


# ---------------------------------------------------------
# Representation
# ---------------------------------------------------------
def fit_pca(emb: np.ndarray, dim: int = PCA_DIM) -> Dict[str, np.ndarray]:
    mean = emb.mean(axis=0)
    k = min(dim, emb.shape[0] - 1, emb.shape[1])
    _, _, vt = np.linalg.svd(emb - mean, full_matrices=False)
    return {"mean": mean.astype(np.float32), "components": vt[:k].astype(np.float32)}


def project(emb: np.ndarray, pca: Dict[str, np.ndarray]) -> np.ndarray:
    return (emb.astype(np.float32) - pca["mean"]) @ pca["components"].T


# ---------------------------------------------------------
# p-values from a null sample
# ---------------------------------------------------------
def tail_pvalue(observed: float, null: np.ndarray) -> Tuple[float, bool]:
    """Empirical p with the +1 correction. When the observation is beyond
    every null draw, the empirical floor 1/(K+1) can't reach Holm's
    thresholds for larger families, so a Gaussian tail fitted to the draws is
    used instead -- an approximation, flagged as such."""
    exceed = int(np.sum(null >= observed))
    p = (1 + exceed) / (len(null) + 1)
    if exceed == 0 and null.std() > 0:
        return float(min(p, norm.sf((observed - null.mean()) / null.std()))), True
    return float(p), False


def split_null(statistic: Callable[[np.ndarray, np.ndarray], float], n_ref: int, n_batch: int,
               draws: int, seed: int) -> Tuple[np.ndarray, bool]:
    """Null sample of statistic(ref_index, batch_index) over disjoint splits of
    the reference at the batch size (capped at half the reference, flagged)."""
    size = min(n_batch, n_ref // 2)
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for i in range(draws):
        perm = rng.permutation(n_ref)
        out[i] = statistic(perm[size:], perm[:size])
    return out, size < n_batch


# ---------------------------------------------------------
# Column drift: Domain Classifier Test with a real-embedding null
# ---------------------------------------------------------
def dct_test(auc_fn: Callable[[np.ndarray, np.ndarray], float], ref: np.ndarray, bat: np.ndarray,
             draws: int, seed: int) -> Dict[str, Any]:
    observed = auc_fn(ref, bat)
    null, approximate = split_null(lambda a, b: auc_fn(ref[a], ref[b]), len(ref), len(bat), draws, seed)
    p, tail = tail_pvalue(observed, null)
    return {"auc": observed, "p_value": p, "tail_extrapolated": tail, "approximate_null": approximate,
            "null_draws": draws}


# ---------------------------------------------------------
# Probes: embedding column -> categorical / numeric column
# ---------------------------------------------------------
def _fit(x: np.ndarray, y: np.ndarray, target: str):
    if target == "categorical":
        return LogisticRegression(max_iter=1000).fit(x, y)
    return Ridge(alpha=1.0).fit(x, rankdata(y) / len(y))  # ranks: monotone target shifts don't matter


def _score(model, x: np.ndarray, y: np.ndarray, target: str) -> float:
    pred = model.predict(x)
    if target == "categorical":
        return float(balanced_accuracy_score(y, pred))  # insensitive to class-prior shift
    return spearman(pred, y.astype(float))


def cross_fitted_score(x: np.ndarray, y: np.ndarray, target: str, seed: int = 0) -> Optional[float]:
    if len(y) < MIN_ROWS:
        return None
    if target == "categorical":
        counts = np.unique(y, return_counts=True)[1]
        if len(counts) < 2 or counts.min() < 2:
            return None
        folds = StratifiedKFold(n_splits=min(5, int(counts.min())), shuffle=True, random_state=seed).split(x, y)
    else:
        folds = KFold(n_splits=5, shuffle=True, random_state=seed).split(x)
    pred = np.empty(len(y), dtype=object if target == "categorical" else float)
    for tr, te in folds:
        pred[te] = _fit(x[tr], y[tr], target).predict(x[te])
    if target == "categorical":
        return float(balanced_accuracy_score(y, pred.astype(y.dtype)))
    return spearman(pred, y.astype(float))


def probe_params(model, target: str) -> Dict[str, np.ndarray]:
    params = {"coef": np.atleast_2d(model.coef_).astype(np.float32),
              "intercept": np.atleast_1d(model.intercept_).astype(np.float32)}
    if target == "categorical":
        params["classes"] = model.classes_.astype(str)
    return params


def fit_probe(x: np.ndarray, y: np.ndarray, target: str) -> Dict[str, np.ndarray]:
    return probe_params(_fit(x, y, target), target)


def probe_predict(params: Dict[str, np.ndarray], x: np.ndarray, target: str) -> np.ndarray:
    scores = x @ params["coef"].T + params["intercept"]
    if target != "categorical":
        return scores.ravel()
    classes = params["classes"]
    if len(classes) == 2:
        return classes[(scores.ravel() > 0).astype(int)]
    return classes[np.argmax(scores, axis=1)]


def probe_strength(x: np.ndarray, y: np.ndarray, target: str) -> Optional[float]:
    """Chance-adjusted predictability on the reference, for proposals."""
    score = cross_fitted_score(x, y, target)
    if score is None:
        return None
    return score - 1 / len(np.unique(y)) if target == "categorical" else score


def probe_test(x_ref: np.ndarray, y_ref: np.ndarray, x_bat: np.ndarray, y_bat: np.ndarray, target: str,
               params: Dict[str, np.ndarray], ref_score: float, draws: int, seed: int) -> Dict[str, Any]:
    if len(y_bat) < MIN_ROWS or (target == "categorical" and len(np.unique(y_bat)) < 2):
        return {"testable": False, "reason": f"Fewer than {MIN_ROWS} usable rows (or a single class) in this batch."}
    pred = probe_predict(params, x_bat, target)
    bat_score = float(balanced_accuracy_score(y_bat, pred)) if target == "categorical" else spearman(pred, y_bat.astype(float))
    observed = ref_score - bat_score

    def null_stat(a, b):
        return ref_score - _score(_fit(x_ref[a], y_ref[a], target), x_ref[b], y_ref[b], target)

    null, approximate = split_null(null_stat, len(y_ref), len(y_bat), draws, seed)
    p, tail = tail_pvalue(observed, null)
    return {"testable": True, "observed": observed, "effect": observed, "p_value": p, "tail_extrapolated": tail,
            "approximate_null": approximate, "null_draws": draws, "reference_value": round(ref_score, 4),
            "current_value": round(bat_score, 4),
            "statistic_name": "balanced accuracy" if target == "categorical" else "Spearman(prediction, actual)"}


# ---------------------------------------------------------
# text <-> image matching (experimental)
# ---------------------------------------------------------
def fit_matching(img: np.ndarray, txt: np.ndarray) -> Dict[str, np.ndarray]:
    model = Ridge(alpha=1.0).fit(img, txt)
    return {"coef": model.coef_.T.astype(np.float32), "intercept": model.intercept_.astype(np.float32)}


def matching_auc(params: Dict[str, np.ndarray], img: np.ndarray, txt: np.ndarray, seed: int = 0) -> float:
    """P(cos(mapped image_i, own text_i) > cos(mapped image_i, another text_j))."""
    mapped = img @ params["coef"] + params["intercept"]
    mapped /= np.linalg.norm(mapped, axis=1, keepdims=True) + 1e-12
    t = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-12)
    own = np.sum(mapped * t, axis=1)
    rng = np.random.default_rng(seed)
    n, wins = len(own), 0.0
    for _ in range(_MATCH_NEGATIVES):
        other = np.sum(mapped * t[(np.arange(n) + rng.integers(1, n, size=n)) % n], axis=1)
        wins += np.mean(own > other) + 0.5 * np.mean(own == other)
    return float(wins / _MATCH_NEGATIVES)


def cross_fitted_matching(img: np.ndarray, txt: np.ndarray, seed: int = 0) -> Optional[float]:
    if len(img) < MIN_ROWS:
        return None
    scores = [matching_auc(fit_matching(img[tr], txt[tr]), img[te], txt[te], seed)
              for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(img)]
    return float(np.mean(scores))


def matching_test(img_ref: np.ndarray, txt_ref: np.ndarray, img_bat: np.ndarray, txt_bat: np.ndarray,
                  params: Dict[str, np.ndarray], ref_score: float, draws: int, seed: int) -> Dict[str, Any]:
    if len(img_bat) < MIN_ROWS:
        return {"testable": False, "reason": f"Fewer than {MIN_ROWS} rows with both a valid image and text."}
    bat_score = matching_auc(params, img_bat, txt_bat, seed)
    observed = ref_score - bat_score

    def null_stat(a, b):
        return ref_score - matching_auc(fit_matching(img_ref[a], txt_ref[a]), img_ref[b], txt_ref[b], seed)

    null, approximate = split_null(null_stat, len(img_ref), len(img_bat), draws, seed)
    p, tail = tail_pvalue(observed, null)
    return {"testable": True, "observed": observed, "effect": observed, "p_value": p, "tail_extrapolated": tail,
            "approximate_null": approximate, "null_draws": draws, "reference_value": round(ref_score, 4),
            "current_value": round(bat_score, 4), "statistic_name": "image-text matching AUC"}

