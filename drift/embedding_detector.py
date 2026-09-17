from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

# Empirically, AUC estimates get noisy/unreliable well below this per-batch
# sample count (see the manual smoke test in the multimodal review: two
# genuinely same-domain-but-different text batches at n=40 total, 20 unique
# each, produced a borderline AUC=0.69 against the 0.65 threshold). Below
# HARD_MIN_SAMPLES the detector's own cross-validation can't run at all.
RECOMMENDED_MIN_SAMPLES = 40
HARD_MIN_SAMPLES = 4


class EmbeddingDriftDetector:
    """
    Domain Classifier Test for text/image drift: trains a classifier to
    distinguish reference embeddings (label 0) from current embeddings
    (label 1). AUC near 0.5 means the classifier can't tell them apart
    (no drift); AUC well above 0.5 means the two batches are separable
    (drift). Modality-agnostic — both text and image adapters produce
    plain embedding matrices.
    """

    def __init__(self, auc_threshold: float = 0.65, n_splits: int = 5, classifier: Optional[ClassifierMixin] = None):
        """
        classifier: defaults to exactly today's LogisticRegression(max_iter=1000)
        when omitted, so every existing caller (text/image endpoints) is
        provably unaffected by this parameter's existence. A caller can
        opt into a different classifier — e.g. adapters/joint.py's
        build_joint_classifier() — without changing anyone else's behavior.
        See build_joint_classifier()'s docstring for why joint embeddings
        specifically need a different one (L1 sparsity vs. this default's
        L2), and why that classifier is NOT used here as the new default.
        """
        self.auc_threshold = auc_threshold
        self.n_splits = n_splits
        self.classifier = classifier if classifier is not None else LogisticRegression(max_iter=1000)

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

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        probs = cross_val_predict(self.classifier, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, probs)

        return {
            "statistic": float(auc),
            "p_value": None,
            "drift_detected": bool(auc > self.auc_threshold),
        }
