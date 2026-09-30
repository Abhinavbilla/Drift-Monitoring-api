"""
Tests for drift/detector.py's Step 2 integration: legacy-mode identity
(must be byte-for-byte unaffected by calibration_config's mere existence)
and calibrated-mode two-gate behavior.

Run:
    python -m pytest tests/test_detector_calibration.py -v
"""

import numpy as np
import pytest

from drift.calibration import CalibrationConfig
from drift.detector import DistributionDetector


def _fit(config=None, ref_continuous=None, ref_categorical=None):
    d = DistributionDetector(calibration_config=config)
    ref = {}
    types = {}
    if ref_continuous is not None:
        ref["x"] = ref_continuous
        types["x"] = "continuous"
    if ref_categorical is not None:
        ref["y"] = ref_categorical
        types["y"] = "categorical"
    d.fit_baseline(ref, types)
    return d


class TestLegacyModeUnaffected:
    """The whole point of calibration_config being opt-in: every existing
    caller (main.py's DistributionDetector(p_value_threshold=0.05), with no
    calibration_config argument at all) must see zero behavior change."""

    def test_no_config_matches_config_none_explicit(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 1000).tolist()
        prod = rng.normal(0.5, 1, 1000).tolist()

        d_no_arg = DistributionDetector()
        d_no_arg.fit_baseline({"x": ref}, {"x": "continuous"})
        r1 = d_no_arg.analyze_production_window({"x": prod})

        d_explicit_none = DistributionDetector(calibration_config=None)
        d_explicit_none.fit_baseline({"x": ref}, {"x": "continuous"})
        r2 = d_explicit_none.analyze_production_window({"x": prod})

        assert r1 == r2

    def test_legacy_decision_mode_matches_no_config(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 1000).tolist()
        prod = rng.normal(0.3, 1, 1000).tolist()

        d_none = _fit(config=None, ref_continuous=ref)
        r_none = d_none.analyze_production_window({"x": prod})

        d_legacy = _fit(config=CalibrationConfig(decision_mode="legacy"), ref_continuous=ref)
        r_legacy = d_legacy.analyze_production_window({"x": prod})

        assert r_none == r_legacy

    def test_legacy_response_shape_unchanged(self):
        """Legacy mode's feature_metrics must contain EXACTLY the original
        three keys -- no new fields leaking in when decision_mode=legacy."""
        d = _fit(config=None, ref_continuous=[1, 2, 3, 4, 5] * 20)
        r = d.analyze_production_window({"x": [1, 2, 3, 4, 5] * 20})
        assert set(r["feature_metrics"]["x"].keys()) == {"statistic", "p_value", "drift_detected"}

    def test_legacy_categorical_response_shape_unchanged(self):
        d = _fit(config=None, ref_categorical=["a", "b"] * 50)
        r = d.analyze_production_window({"y": ["a", "b"] * 50})
        assert set(r["feature_metrics"]["y"].keys()) == {"statistic", "p_value", "drift_detected"}
        assert r["feature_metrics"]["y"]["p_value"] is None  # legacy PSI never gets a p-value


class TestCalibratedModeTwoGate:
    def test_response_has_all_new_fields(self):
        cfg = CalibrationConfig(decision_mode="calibrated")
        d = _fit(config=cfg, ref_continuous=list(range(100)))
        r = d.analyze_production_window({"x": list(range(100))})
        expected_new_fields = {"effect_size", "effect_floor", "p_value_adjusted", "significant",
                                "material", "decision_mode", "threshold_used"}
        assert expected_new_fields <= set(r["feature_metrics"]["x"].keys())
        # Original fields must still be present too (additive, not replaced)
        assert {"statistic", "p_value", "drift_detected"} <= set(r["feature_metrics"]["x"].keys())

    def test_significant_but_not_material_does_not_flag(self):
        """The exact documented Citi Bike scenario: a tiny population effect
        that's highly significant at large batch size but below the
        materiality floor must not be flagged as drift."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 5000).tolist()
        prod = rng.normal(0.03, 1, 50000).tolist()  # tiny shift, huge batch -> significant, small effect

        cfg = CalibrationConfig(decision_mode="calibrated")
        d = _fit(config=cfg, ref_continuous=ref)
        r = d.analyze_production_window({"x": prod})
        m = r["feature_metrics"]["x"]
        assert m["significant"] is True
        assert m["material"] is False
        assert m["drift_detected"] is False
        assert r["system_alert_triggered"] is False

    def test_large_effect_flags_both_gates(self):
        rng = np.random.default_rng(43)
        ref = rng.normal(0, 1, 5000).tolist()
        prod = rng.normal(0.5, 1, 5000).tolist()

        cfg = CalibrationConfig(decision_mode="calibrated")
        d = _fit(config=cfg, ref_continuous=ref)
        r = d.analyze_production_window({"x": prod})
        m = r["feature_metrics"]["x"]
        assert m["significant"] is True
        assert m["material"] is True
        assert m["drift_detected"] is True
        assert r["system_alert_triggered"] is True

    def test_no_drift_neither_gate_fires(self):
        rng = np.random.default_rng(44)
        ref = rng.normal(0, 1, 2000).tolist()
        prod = rng.normal(0, 1, 2000).tolist()

        cfg = CalibrationConfig(decision_mode="calibrated")
        d = _fit(config=cfg, ref_continuous=ref)
        r = d.analyze_production_window({"x": prod})
        assert r["feature_metrics"]["x"]["drift_detected"] is False

    def test_categorical_gets_bootstrap_pvalue_not_none(self):
        """Legacy PSI never has a p-value; calibrated mode must produce a
        real one via the bootstrap so it can join the corrected family."""
        cfg = CalibrationConfig(decision_mode="calibrated")
        d = _fit(config=cfg, ref_categorical=["a"] * 70 + ["b"] * 30)
        r = d.analyze_production_window({"y": ["a"] * 50 + ["b"] * 50})
        m = r["feature_metrics"]["y"]
        assert m["p_value"] is not None
        assert 0.0 <= m["p_value"] <= 1.0
        assert m["p_value_adjusted"] is not None

    def test_holm_correction_reduces_false_positives_across_features(self):
        """Multiple genuinely-null features tested together should see
        fewer false positives under Holm correction than the raw per-test
        alpha would predict -- exercised with several null features at once."""
        rng = np.random.default_rng(45)
        n_features = 10
        ref = {f"f{i}": rng.normal(0, 1, 500).tolist() for i in range(n_features)}
        types = {f"f{i}": "continuous" for i in range(n_features)}

        cfg = CalibrationConfig(decision_mode="calibrated")
        d = DistributionDetector(calibration_config=cfg)
        d.fit_baseline(ref, types)

        # All features drawn from the identical null distribution -- no
        # feature should be flagged as material even if one or two happen
        # to cross raw significance by chance (correction should catch it).
        prod = {f"f{i}": rng.normal(0, 1, 500).tolist() for i in range(n_features)}
        r = d.analyze_production_window(prod)
        n_flagged = sum(1 for m in r["feature_metrics"].values() if m["drift_detected"])
        assert n_flagged <= 1  # allow at most one spurious flag out of 10 nulls

    def test_per_feature_floor_override_respected(self):
        cfg = CalibrationConfig(decision_mode="calibrated", per_feature_effect_floors={"x": 0.01})
        rng = np.random.default_rng(46)
        ref = rng.normal(0, 1, 5000).tolist()
        prod = rng.normal(0.02, 1, 50000).tolist()  # small effect, but below the OVERRIDDEN floor 0.01? no -- above it

        d = _fit(config=cfg, ref_continuous=ref)
        r = d.analyze_production_window({"x": prod})
        assert r["feature_metrics"]["x"]["effect_floor"] == 0.01
