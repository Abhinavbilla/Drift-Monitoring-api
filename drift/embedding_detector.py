from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score


class EmbeddingDriftDetector:
    """
    Domain Classifier Test for text/image drift: trains a classifier to
    distinguish reference embeddings (label 0) from current embeddings
    (label 1). AUC near 0.5 means the classifier can't tell them apart
    (no drift); AUC well above 0.5 means the two batches are separable
    (drift). Modality-agnostic — both text and image adapters produce
    plain embedding matrices.
    """

    def __init__(self, auc_threshold: float = 0.65, n_splits: int = 5):
        self.auc_threshold = auc_threshold
        self.n_splits = n_splits

    def analyze(self, ref_embeddings: np.ndarray, cur_embeddings: np.ndarray) -> Dict[str, Any]:
        ref_embeddings = np.asarray(ref_embeddings)
        cur_embeddings = np.asarray(cur_embeddings)

        if len(ref_embeddings) == 0 or len(cur_embeddings) == 0:
            raise ValueError("Both reference and current embeddings must be non-empty.")

        X = np.vstack([ref_embeddings, cur_embeddings])
        y = np.concatenate([
            np.zeros(len(ref_embeddings), dtype=int),
            np.ones(len(cur_embeddings), dtype=int),
        ])

        n_splits = min(self.n_splits, np.bincount(y).min())
        if n_splits < 2:
            raise ValueError(
                "Need at least 2 samples in each of the reference and current "
                "batches to run the Domain Classifier Test."
            )

        classifier = LogisticRegression(max_iter=1000)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        probs = cross_val_predict(classifier, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, probs)

        return {
            "statistic": float(auc),
            "p_value": None,
            "drift_detected": bool(auc > self.auc_threshold),
        }
