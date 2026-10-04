from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from drift.calibration import CalibrationConfig, adjust_p_values, apply_two_gate
from drift.dct_calibration import dct_pvalue

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

    Step 2 (d), 2026-10-04: analyze() accepts an optional calibration_config
    (same CalibrationConfig used by the tabular DistributionDetector).
    Omitted, or decision_mode="legacy": behavior is byte-identical to
    before this existed -- a bare AUC > auc_threshold cutoff, no p-value.
    decision_mode="calibrated": the AUC gets a p-value from a precomputed
    null-distribution grid (drift/dct_calibration.py), keyed by
    (embedding_dim, n_ref, n_batch) -- falls back to an on-the-fly
    permutation null if this exact embedding dimension was never
    calibrated (e.g. a future model swap), rather than silently using a
    wrong-dimension grid entry or fabricating a p-value.
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

    def _compute_auc(self, ref_embeddings: np.ndarray, cur_embeddings: np.ndarray) -> float:
        """The statistic itself, factored out so the calibration grid's
        own simulation (scripts/build_dct_calibration_grid.py) can call
        the EXACT SAME code path on synthetic null draws -- no risk of
        the calibration drifting out of sync with what analyze() actually
        computes at runtime."""
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
        return float(roc_auc_score(y, probs))

    def analyze(
        self, ref_embeddings: np.ndarray, cur_embeddings: np.ndarray,
        calibration_config: Optional[CalibrationConfig] = None,
    ) -> Dict[str, Any]:
        auc = self._compute_auc(ref_embeddings, cur_embeddings)

        decision_mode = calibration_config.decision_mode if calibration_config else "legacy"
        if decision_mode != "calibrated":
            # Byte-identical to every version of this method before Step 2 (d).
            return {
                "statistic": auc,
                "p_value": None,
                "drift_detected": bool(auc > self.auc_threshold),
            }

        n_ref = len(np.asarray(ref_embeddings))
        n_batch = len(np.asarray(cur_embeddings))
        embedding_dim = np.asarray(ref_embeddings).shape[1]
        p_value = dct_pvalue(auc, embedding_dim, n_ref, n_batch, auc_fn=self._compute_auc)
        p_value_adjusted = adjust_p_values([p_value], calibration_config.multiple_testing)[0]
        floor = calibration_config.effect_floor_for("embedding_drift", "dct_auc")

        gate = apply_two_gate(
            effect_size=auc, effect_floor=floor, p_value=p_value, p_value_adjusted=p_value_adjusted,
            alpha=calibration_config.alpha, decision_mode="calibrated",
            threshold_used=f"p_adj<{calibration_config.alpha} [DCT-null (grid)], AUC>={floor}",
        )
        return {
            "statistic": auc,
            "p_value": gate.p_value,
            "drift_detected": gate.drift_detected,
            "effect_size": gate.effect_size,
            "effect_floor": gate.effect_floor,
            "p_value_adjusted": gate.p_value_adjusted,
            "significant": gate.significant,
            "material": gate.material,
            "decision_mode": gate.decision_mode,
            "threshold_used": gate.threshold_used,
        }
