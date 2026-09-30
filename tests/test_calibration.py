"""
Tests for drift/calibration.py -- Step 2's two-gate calibrated decision
engine. Covers: Holm/BH correction against hand-computed examples (checked
against known standard-software output, e.g. R's p.adjust), the
minimum-detectable-D formula, and the two-gate combination logic
(legacy mode must reduce to today's single-threshold behavior exactly).

Run:
    python -m pytest tests/test_calibration.py -v
"""

import math

import numpy as np
import pytest

from drift.calibration import (
    CalibrationConfig,
    apply_two_gate,
    bh_adjust,
    compute_psi,
    holm_adjust,
    ks_c_alpha,
    minimum_detectable_d,
    psi_bootstrap_pvalue,
    recommended_batch_size,
)


class TestHolmAdjust:
    def test_evenly_spaced_pvalues_matches_hand_computation(self):
        """p = [0.01, 0.02, 0.03, 0.04, 0.05] -- verified by hand against
        R's p.adjust(p, method="holm"): [0.05, 0.08, 0.09, 0.09, 0.09]."""
        p = [0.01, 0.02, 0.03, 0.04, 0.05]
        adjusted = holm_adjust(p)
        expected = [0.05, 0.08, 0.09, 0.09, 0.09]
        for a, e in zip(adjusted, expected):
            assert a == pytest.approx(e, abs=1e-9)

    def test_out_of_order_input_matches_hand_computation(self):
        """p = [0.01, 0.04, 0.03, 0.005] (unsorted) -- hand-computed:
        sorted view [0.005,0.01,0.03,0.04] -> adjusted [0.02,0.03,0.06,0.06],
        mapped back to original order -> [0.03, 0.06, 0.06, 0.02]."""
        p = [0.01, 0.04, 0.03, 0.005]
        adjusted = holm_adjust(p)
        expected = [0.03, 0.06, 0.06, 0.02]
        for a, e in zip(adjusted, expected):
            assert a == pytest.approx(e, abs=1e-9)

    def test_single_pvalue_unchanged(self):
        assert holm_adjust([0.03])[0] == pytest.approx(0.03)

    def test_empty_list(self):
        assert holm_adjust([]) == []

    def test_monotone_nondecreasing_in_sorted_order(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, size=20).tolist()
        adjusted = holm_adjust(p)
        order = sorted(range(len(p)), key=lambda i: p[i])
        sorted_adjusted = [adjusted[i] for i in order]
        assert all(sorted_adjusted[i] <= sorted_adjusted[i + 1] + 1e-12 for i in range(len(sorted_adjusted) - 1))

    def test_bounded_and_at_least_raw_pvalue(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, size=15).tolist()
        adjusted = holm_adjust(p)
        for raw, adj in zip(p, adjusted):
            assert 0.0 <= adj <= 1.0
            assert adj >= raw - 1e-12


class TestBHAdjust:
    def test_evenly_spaced_pvalues_all_equal(self):
        """Classic BH property: for p = [0.01,0.02,0.03,0.04,0.05], every
        adjusted value comes out equal (0.05) -- hand-verified and a known
        textbook case for evenly-spaced p-values."""
        p = [0.01, 0.02, 0.03, 0.04, 0.05]
        adjusted = bh_adjust(p)
        for a in adjusted:
            assert a == pytest.approx(0.05, abs=1e-9)

    def test_bh_less_or_equal_conservative_than_holm(self):
        """BH controls FDR (less conservative) vs Holm's FWER control --
        BH-adjusted p-values should never exceed Holm-adjusted for the same
        input."""
        rng = np.random.default_rng(2)
        p = rng.uniform(0, 0.2, size=10).tolist()
        holm = holm_adjust(p)
        bh = bh_adjust(p)
        for h, b in zip(holm, bh):
            assert b <= h + 1e-9

    def test_empty_list(self):
        assert bh_adjust([]) == []

    def test_bounded(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 1, size=12).tolist()
        adjusted = bh_adjust(p)
        assert all(0.0 <= a <= 1.0 for a in adjusted)


class TestMinimumDetectableD:
    def test_matches_known_values_from_step1(self):
        """Cross-checked against results/tabular_validation_notes.md's
        'Reference size vs. batch size' section: m=5000 floor ~0.0192,
        m=50000 floor ~0.0061, n=20000/m=5000 ~0.0215 (as n->infinity for
        the first two)."""
        assert minimum_detectable_d(n=10**9, m=5000) == pytest.approx(0.0192, abs=1e-3)
        assert minimum_detectable_d(n=10**9, m=50000) == pytest.approx(0.0061, abs=1e-3)
        assert minimum_detectable_d(n=20000, m=5000) == pytest.approx(0.0215, abs=1e-3)

    def test_symmetric_in_n_and_m(self):
        assert minimum_detectable_d(100, 200) == pytest.approx(minimum_detectable_d(200, 100))

    def test_decreases_as_either_size_grows(self):
        small = minimum_detectable_d(1000, 1000)
        large = minimum_detectable_d(10000, 1000)
        assert large < small

    def test_rejects_nonpositive_sizes(self):
        with pytest.raises(ValueError):
            minimum_detectable_d(0, 100)
        with pytest.raises(ValueError):
            minimum_detectable_d(100, -5)

    def test_c_alpha_standard_table_values(self):
        assert ks_c_alpha(0.05) == 1.36
        assert ks_c_alpha(0.01) == 1.63
        assert ks_c_alpha(0.10) == 1.22


class TestRecommendedBatchSize:
    def test_boundary_crosses_exactly_at_recommended_n(self):
        """The whole point of the formula: min_detectable_d at the
        recommended n must be <= floor, and at n-1 it must be > floor."""
        for m, floor in [(5000, 0.05), (50000, 0.05), (5000, 0.02), (1000, 0.1)]:
            n_rec = recommended_batch_size(m, floor)
            assert n_rec is not None
            assert minimum_detectable_d(n_rec, m) <= floor
            assert minimum_detectable_d(n_rec - 1, m) > floor

    def test_matches_hand_computed_example(self):
        """m=5000, floor=0.05, alpha=0.05 -- hand-solved n >= c^2*m/(floor^2*m-c^2)
        = 1.8496*5000/(0.0025*5000-1.8496) = 9248/10.6504 ~= 868.4 -> 869."""
        assert recommended_batch_size(5000, 0.05) == 869

    def test_infeasible_returns_none(self):
        """A reference too small to ever reach the floor (floor below the
        fit-time n->infinity floor) has no finite recommended batch size."""
        assert recommended_batch_size(100, 0.01) is None

    def test_larger_reference_needs_smaller_batch(self):
        n_small_ref = recommended_batch_size(5000, 0.05)
        n_large_ref = recommended_batch_size(50000, 0.05)
        assert n_large_ref < n_small_ref

    def test_tighter_floor_needs_larger_batch(self):
        n_loose = recommended_batch_size(5000, 0.05)
        n_tight = recommended_batch_size(5000, 0.02)
        assert n_tight > n_loose

    def test_rejects_nonpositive_inputs(self):
        with pytest.raises(ValueError):
            recommended_batch_size(0, 0.05)
        with pytest.raises(ValueError):
            recommended_batch_size(5000, 0)


class TestTwoGateLegacyReducesToToday:
    def test_legacy_significant_alone_drives_detection(self):
        """Legacy mode: drift_detected must equal Gate 1 (significance)
        alone, since Gate 2 is bypassed -- material is always True."""
        result = apply_two_gate(
            effect_size=0.02, effect_floor=0.05, p_value=0.01, p_value_adjusted=0.01,
            alpha=0.05, decision_mode="legacy", threshold_used="p<0.05",
        )
        assert result.material is True
        assert result.significant is True
        assert result.drift_detected is True  # even though effect_size (0.02) < effect_floor (0.05)

    def test_legacy_not_significant_not_detected(self):
        result = apply_two_gate(
            effect_size=0.10, effect_floor=0.05, p_value=0.20, p_value_adjusted=0.20,
            alpha=0.05, decision_mode="legacy", threshold_used="p<0.05",
        )
        assert result.drift_detected is False

    def test_calibrated_requires_both_gates(self):
        """Calibrated mode: significant but not material (Citi Bike's
        documented case at D_gt=0.05 default -- population D~0.02 is
        significant at large n but below the 0.05 floor) must NOT flag."""
        result = apply_two_gate(
            effect_size=0.02, effect_floor=0.05, p_value=1e-10, p_value_adjusted=1e-10,
            alpha=0.05, decision_mode="calibrated", threshold_used="p<0.05, D>=0.05",
        )
        assert result.significant is True
        assert result.material is False
        assert result.drift_detected is False

    def test_calibrated_both_gates_pass(self):
        result = apply_two_gate(
            effect_size=0.10, effect_floor=0.05, p_value=1e-10, p_value_adjusted=1e-10,
            alpha=0.05, decision_mode="calibrated", threshold_used="p<0.05, D>=0.05",
        )
        assert result.drift_detected is True

    def test_calibrated_material_but_not_significant(self):
        """Large effect but not statistically significant (e.g. tiny batch)
        -- Gate 1 fails, must not flag even though Gate 2 would pass."""
        result = apply_two_gate(
            effect_size=0.10, effect_floor=0.05, p_value=0.5, p_value_adjusted=0.5,
            alpha=0.05, decision_mode="calibrated", threshold_used="p<0.05, D>=0.05",
        )
        assert result.material is True
        assert result.significant is False
        assert result.drift_detected is False


class TestPSIBootstrap:
    def test_compute_psi_matches_known_example(self):
        """Two categories, one doubled in frequency -- hand-computable PSI:
        PSI = sum((actual-expected)*ln(actual/expected))."""
        ref = {"a": 0.5, "b": 0.5}
        cur = {"a": 0.6, "b": 0.4}
        psi = compute_psi(ref, cur)
        expected = (0.6 - 0.5) * math.log(0.6 / 0.5) + (0.4 - 0.5) * math.log(0.4 / 0.5)
        assert psi == pytest.approx(expected, abs=1e-9)

    def test_identical_distributions_zero_psi(self):
        ref = {"a": 0.3, "b": 0.3, "c": 0.4}
        assert compute_psi(ref, ref) == pytest.approx(0.0, abs=1e-9)

    def test_false_rejection_rate_near_alpha_under_null(self):
        """THE key calibration check: if the 'observed' batch is ACTUALLY
        drawn from the reference distribution (the null is true), the
        bootstrap p-value should be < alpha at approximately the alpha rate
        across many repeated trials -- this is what 'calibrated' means."""
        rng = np.random.default_rng(123)
        ref_freq = {"a": 0.5, "b": 0.3, "c": 0.2}
        categories = list(ref_freq.keys())
        probs = np.array([ref_freq[c] for c in categories])
        batch_size = 500
        alpha = 0.05
        n_trials = 200

        false_rejections = 0
        for t in range(n_trials):
            draw_counts = rng.multinomial(batch_size, probs)
            observed_freq = {c: draw_counts[i] / batch_size for i, c in enumerate(categories)}
            observed_psi = compute_psi(ref_freq, observed_freq)
            p = psi_bootstrap_pvalue(ref_freq, observed_psi, batch_size, null_draws=200, seed=1000 + t)
            if p < alpha:
                false_rejections += 1

        rate = false_rejections / n_trials
        # Binomial 95% CI half-width at n=200, p=0.05 is roughly +/-0.03 --
        # allow a generous margin since this is a stochastic test.
        assert rate < 0.15, f"false rejection rate {rate:.3f} is far above alpha={alpha}"

    def test_never_zero_pvalue(self):
        """The +1 correction must prevent p=0 even when observed PSI
        exceeds every null draw."""
        ref_freq = {"a": 0.99, "b": 0.01}
        p = psi_bootstrap_pvalue(ref_freq, observed_psi=100.0, batch_size=100, null_draws=50, seed=1)
        assert p > 0.0

    def test_larger_batch_more_powerful_for_same_frequency_gap(self):
        """The same frequency deviation should be MORE significant (smaller
        p-value) at a larger batch size -- this is the whole point of the
        bootstrap replacing the flat threshold."""
        ref_freq = {"a": 0.5, "b": 0.5}
        cur_freq = {"a": 0.55, "b": 0.45}
        psi = compute_psi(ref_freq, cur_freq)
        p_small = psi_bootstrap_pvalue(ref_freq, psi, batch_size=50, null_draws=500, seed=1)
        p_large = psi_bootstrap_pvalue(ref_freq, psi, batch_size=5000, null_draws=500, seed=1)
        assert p_large < p_small


class TestCalibrationConfig:
    def test_none_resolves_to_legacy_default(self):
        """Existing projects (no stored config) must resolve to legacy --
        no migration needed to keep them behaving exactly as before."""
        cfg = CalibrationConfig.from_dict(None)
        assert cfg.decision_mode == "legacy"
        assert cfg.alpha == 0.05
        assert cfg.effect_floors == {"ks_d": 0.05, "psi": 0.2, "dct_auc": 0.65}

    def test_roundtrip(self):
        cfg = CalibrationConfig(decision_mode="calibrated", alpha=0.05, multiple_testing="bh")
        d = cfg.to_dict()
        cfg2 = CalibrationConfig.from_dict(d)
        assert cfg2.decision_mode == "calibrated"
        assert cfg2.multiple_testing == "bh"

    def test_per_feature_override(self):
        cfg = CalibrationConfig(per_feature_effect_floors={"trip_duration": 0.1})
        assert cfg.effect_floor_for("trip_duration", "ks_d") == 0.1
        assert cfg.effect_floor_for("pickup_longitude", "ks_d") == 0.05

    def test_partial_dict_fills_defaults(self):
        cfg = CalibrationConfig.from_dict({"decision_mode": "calibrated"})
        assert cfg.decision_mode == "calibrated"
        assert cfg.alpha == 0.05  # default filled in
        assert cfg.dct_calibration == "precomputed"
