"""
Step 2 core: two-gate calibrated decisions (significance + materiality),
multiple-testing correction, and the minimum-detectable-D helper.

This module is purely additive -- nothing in drift/detector.py or
drift/embedding_detector.py is modified to use it yet. Legacy behavior
(today's single-threshold decisions) is unaffected by this module's mere
existence; it's wired in as an opt-in path.

Decisions recorded (user, 2026-09-29), NOT re-derived here:
  - alpha = 0.05, corrected family-wise per batch via Holm across features
    by default; BH available as an option.
  - Effect floors: KS D = 0.05 (dataset-independent, deliberately not
    tuned to the Citi Bike knife-edge cluster), PSI = 0.2, DCT AUC = 0.65
    (provisional/unvalidated until Step 8).
  - decision_mode default = "legacy" for all projects (new and existing);
    switching new-project default to "calibrated" needs a separate
    approval after the side-by-side comparison.
  - DCT calibration default = "precomputed", clamped (never extrapolated)
    at analyze time for batch sizes outside the calibrated grid.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Defaults (see docstring above for the decision record)
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 0.05
DEFAULT_MULTIPLE_TESTING = "holm"
DEFAULT_EFFECT_FLOORS = {"ks_d": 0.05, "psi": 0.2, "dct_auc": 0.65}
DEFAULT_DECISION_MODE = "legacy"
DEFAULT_PSI_NULL_DRAWS = 1000
DEFAULT_DCT_CALIBRATION = "precomputed"
DEFAULT_DCT_PERMUTATIONS = 100

# Standard asymptotic two-sided KS critical constant at common alpha levels.
# Used only for the minimum-detectable-D display/warning -- a continuous-case
# approximation, not exact for near-discrete features (see
# results/tabular_validation_notes.md's "Minimum-detectable-D formula caveat").
_KS_C_ALPHA = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}


def ks_c_alpha(alpha: float) -> float:
    """Returns c(alpha) for the asymptotic KS critical value formula. Falls
    back to interpolating/approximating for alpha values not in the standard
    table, via the closed-form c(alpha) = sqrt(-0.5 * ln(alpha/2))."""
    if alpha in _KS_C_ALPHA:
        return _KS_C_ALPHA[alpha]
    return math.sqrt(-0.5 * math.log(alpha / 2))


def minimum_detectable_d(n: int, m: int, alpha: float = DEFAULT_ALPHA) -> float:
    """The asymptotic KS critical-value floor for a batch of size n compared
    against a reference of size m, at significance alpha:
        c(alpha) * sqrt((n+m) / (n*m))
    This is the smallest population D that COULD be detected at this (n, m)
    -- not what will be detected (that depends on the actual sample), but
    the floor below which detection is asymptotically impossible regardless
    of batch size. Continuous-case approximation; see the caveat above."""
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive.")
    c = ks_c_alpha(alpha)
    return c * math.sqrt((n + m) / (n * m))


# ---------------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------------

def holm_adjust(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values. Family-wise error rate
    control: rejecting all H_i with holm_adjust(p)[i] <= alpha bounds the
    probability of ANY false rejection in the family at alpha, which is
    what bounds the per-batch system-level false-alarm rate when the
    system alert is "any feature flagged" (the decision recorded 2026-09-29).

    Standard algorithm: sort ascending, p_adj_(i) = max_{j<=i} min(1, (m-j+1)*p_(j)),
    enforced monotone non-decreasing, mapped back to original order.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(indexed):  # rank is 0-indexed; i = rank+1 in 1-indexed formula
        raw = (m - rank) * p_values[idx]
        running_max = max(running_max, raw)
        adjusted_sorted[rank] = min(1.0, running_max)
    out = [0.0] * m
    for rank, idx in enumerate(indexed):
        out[idx] = adjusted_sorted[rank]
    return out


def bh_adjust(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg step-up adjusted p-values (controls false
    discovery rate, not family-wise error rate -- offered as an option per
    instruction, Holm remains the default).

    Standard algorithm: sort ascending, p_adj_(i) = min_{j>=i} min(1, (m/j)*p_(j)),
    enforced monotone non-increasing from the largest rank down, mapped
    back to original order.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):  # iterate from largest rank down to smallest
        idx = indexed[rank]
        raw = (m / (rank + 1)) * p_values[idx]
        running_min = min(running_min, raw)
        adjusted_sorted[rank] = min(1.0, running_min)
    out = [0.0] * m
    for rank, idx in enumerate(indexed):
        out[idx] = adjusted_sorted[rank]
    return out


def adjust_p_values(p_values: List[float], method: str) -> List[float]:
    if method == "holm":
        return holm_adjust(p_values)
    if method == "bh":
        return bh_adjust(p_values)
    if method == "none":
        return list(p_values)
    raise ValueError(f"Unknown multiple_testing method: {method!r} (expected 'holm', 'bh', or 'none').")


# ---------------------------------------------------------------------------
# Two-gate decision
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    effect_size: float
    effect_floor: float
    p_value: Optional[float]
    p_value_adjusted: Optional[float]
    significant: bool
    material: bool
    decision_mode: str
    threshold_used: str

    @property
    def drift_detected(self) -> bool:
        """The single boolean the rest of the system consumes: in
        calibrated mode, flagged iff BOTH gates pass (significant AND
        material). In legacy mode this mirrors today's single-threshold
        behavior exactly (see apply_two_gate's legacy branch)."""
        if self.decision_mode == "legacy":
            return self.significant
        return self.significant and self.material


def apply_two_gate(
    effect_size: float,
    effect_floor: float,
    p_value: Optional[float],
    p_value_adjusted: Optional[float],
    alpha: float,
    decision_mode: str,
    threshold_used: str,
) -> GateResult:
    """Combines Gate 1 (significance: p_value_adjusted < alpha, or for
    detectors with no p-value concept, effect_size > threshold_used alone)
    and Gate 2 (materiality: effect_size >= effect_floor).

    decision_mode="legacy": Gate 2 is not applied -- `material` is always
    True and `drift_detected` (via GateResult.drift_detected) equals
    Gate 1 alone, exactly reproducing today's single-threshold behavior.
    """
    if p_value_adjusted is not None:
        significant = p_value_adjusted < alpha
    else:
        # Detectors without a p-value (legacy DCT/PSI paths) fall back to
        # the existing single-threshold comparison as "significance".
        significant = effect_size > effect_floor if decision_mode == "legacy" else effect_size > 0

    material = True if decision_mode == "legacy" else (effect_size >= effect_floor)

    return GateResult(
        effect_size=effect_size,
        effect_floor=effect_floor,
        p_value=p_value,
        p_value_adjusted=p_value_adjusted,
        significant=significant,
        material=material,
        decision_mode=decision_mode,
        threshold_used=threshold_used,
    )


# ---------------------------------------------------------------------------
# PSI parametric bootstrap significance test
# ---------------------------------------------------------------------------

def compute_psi(ref_freq: Dict[str, float], cur_freq: Dict[str, float], epsilon: float = 0.0001) -> float:
    """Same PSI formula as drift/detector.py's _check_categorical_drift
    (epsilon substitution for zero-frequency categories), operating on
    pre-computed frequency dicts so it can be reused for both the observed
    statistic and each bootstrap null draw without recomputing from raw
    arrays each time."""
    all_categories = set(ref_freq) | set(cur_freq)
    psi = 0.0
    for cat in all_categories:
        expected = ref_freq.get(cat, 0.0) or epsilon
        actual = cur_freq.get(cat, 0.0) or epsilon
        psi += (actual - expected) * math.log(actual / expected)
    return float(psi)


def psi_bootstrap_pvalue(
    reference_frequencies: Dict[str, float],
    observed_psi: float,
    batch_size: int,
    null_draws: int = DEFAULT_PSI_NULL_DRAWS,
    seed: Optional[int] = None,
) -> float:
    """Parametric bootstrap p-value for PSI at the ACTUAL batch size,
    replacing the flat PSI>0.2 cutoff's implicit assumption that the same
    threshold is equally meaningful at any n. Draws `null_draws` multinomial
    samples of size `batch_size` from the reference frequencies (i.e.
    simulates batches that are genuinely drawn from the reference
    distribution, the null hypothesis), computes PSI for each, and returns
    (1 + #{null_PSI >= observed_PSI}) / (null_draws + 1) -- the "+1"
    correction avoids a p-value of exactly 0, which would otherwise be
    possible whenever the observed PSI exceeds every simulated null draw.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    rng = np.random.default_rng(seed)
    categories = list(reference_frequencies.keys())
    probs = np.array([reference_frequencies[c] for c in categories], dtype=float)
    probs = probs / probs.sum()  # guard against tiny floating-point drift

    exceed_count = 0
    for _ in range(null_draws):
        draw_counts = rng.multinomial(batch_size, probs)
        draw_freq = {c: draw_counts[i] / batch_size for i, c in enumerate(categories)}
        null_psi = compute_psi(reference_frequencies, draw_freq)
        if null_psi >= observed_psi:
            exceed_count += 1
    return (1 + exceed_count) / (null_draws + 1)


# ---------------------------------------------------------------------------
# Per-project calibration config
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    decision_mode: str = DEFAULT_DECISION_MODE
    alpha: float = DEFAULT_ALPHA
    multiple_testing: str = DEFAULT_MULTIPLE_TESTING
    effect_floors: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EFFECT_FLOORS))
    per_feature_effect_floors: Dict[str, float] = field(default_factory=dict)
    psi_null_draws: int = DEFAULT_PSI_NULL_DRAWS
    dct_calibration: str = DEFAULT_DCT_CALIBRATION
    dct_permutations: int = DEFAULT_DCT_PERMUTATIONS

    def effect_floor_for(self, feature: str, floor_type: str) -> float:
        """Per-feature override takes precedence over the project-level
        floor for that detector type (per instruction: 'overridable per
        feature')."""
        if feature in self.per_feature_effect_floors:
            return self.per_feature_effect_floors[feature]
        return self.effect_floors[floor_type]

    def to_dict(self) -> dict:
        return {
            "decision_mode": self.decision_mode,
            "alpha": self.alpha,
            "multiple_testing": self.multiple_testing,
            "effect_floors": dict(self.effect_floors),
            "per_feature_effect_floors": dict(self.per_feature_effect_floors),
            "psi_null_draws": self.psi_null_draws,
            "dct_calibration": self.dct_calibration,
            "dct_permutations": self.dct_permutations,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "CalibrationConfig":
        """Absent/None config (e.g. an existing project fit before Step 2
        shipped) resolves to the legacy default -- existing projects stay
        legacy permanently, per instruction, without needing a migration
        to write an explicit config row."""
        if not d:
            return cls()
        return cls(
            decision_mode=d.get("decision_mode", DEFAULT_DECISION_MODE),
            alpha=d.get("alpha", DEFAULT_ALPHA),
            multiple_testing=d.get("multiple_testing", DEFAULT_MULTIPLE_TESTING),
            effect_floors={**DEFAULT_EFFECT_FLOORS, **d.get("effect_floors", {})},
            per_feature_effect_floors=d.get("per_feature_effect_floors", {}),
            psi_null_draws=d.get("psi_null_draws", DEFAULT_PSI_NULL_DRAWS),
            dct_calibration=d.get("dct_calibration", DEFAULT_DCT_CALIBRATION),
            dct_permutations=d.get("dct_permutations", DEFAULT_DCT_PERMUTATIONS),
        )
