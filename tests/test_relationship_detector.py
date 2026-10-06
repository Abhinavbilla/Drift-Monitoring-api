"""
Relationship drift statistics (milestone M2): each test must ignore marginal
drift and catch dependency changes; relationship tests join the column
tests' Holm family.

Run:
    python -m pytest tests/test_relationship_detector.py -v
"""

import numpy as np
import pytest

from drift import relationship_detector as rel
from drift.calibration import CalibrationConfig, holm_adjust
from drift.detector import DistributionDetector

DRAWS = 400


def _p(kind, state, ref, bat, seed=1):
    prep = rel.prepare_test(kind, state, ref, bat)
    assert prep["testable"], prep
    return rel.finish_test(prep, DRAWS, seed)


def _correlated(rng, n, rho):
    x = rng.normal(size=n)
    return x, rho * x + np.sqrt(1 - rho ** 2) * rng.normal(size=n)


# ---------------------------------------------------------
# numeric <-> numeric
# ---------------------------------------------------------
def test_spearman_ignores_monotone_marginal_changes():
    rng = np.random.default_rng(0)
    x, y = _correlated(rng, 500, 0.7)
    assert rel.num_num_statistic((x, y), (np.exp(x) * 3 + 10, y ** 3)) == pytest.approx(0, abs=1e-12)


def test_spearman_detects_broken_correlation_and_passes_same_distribution():
    rng = np.random.default_rng(1)
    rx, ry = _correlated(rng, 2000, 0.8)
    state = rel.reference_state("num_num", rx, ry)
    bx, by = _correlated(rng, 300, 0.8)
    assert _p("num_num", state, (rx, ry), (bx, by))["p_value"] > 0.05
    broken = _p("num_num", state, (rx, ry), (bx, rng.permutation(by)))
    assert broken["p_value"] < 0.01 and broken["current_value"] < 0.2


def test_num_num_needs_minimum_rows():
    rng = np.random.default_rng(2)
    rx, ry = _correlated(rng, 500, 0.5)
    prep = rel.prepare_test("num_num", {}, (rx, ry), (rx[:10], ry[:10]))
    assert not prep["testable"] and "complete rows" in prep["reason"]


# ---------------------------------------------------------
# categorical <-> categorical
# ---------------------------------------------------------
def _cat_pairs(rng, n, pairing):
    a = rng.choice(["x", "y", "z"], size=n, p=[0.5, 0.3, 0.2])
    b = np.array([pairing[v] if rng.random() < 0.8 else rng.choice(["p", "q", "r"]) for v in a], dtype=object)
    return a.astype(object), b


def test_cat_cat_ignores_category_proportion_changes():
    rng = np.random.default_rng(3)
    ra, rb = _cat_pairs(rng, 3000, {"x": "p", "y": "q", "z": "r"})
    state = rel.reference_state("cat_cat", ra, rb)
    # Resample whole rows stratified by A with new proportions: P(B|A) unchanged, P(A) and P(B) shift.
    idx = np.concatenate([rng.choice(np.flatnonzero(ra == v), size=k) for v, k in (("x", 60), ("y", 120), ("z", 220))])
    assert _p("cat_cat", state, (ra, rb), (ra[idx], rb[idx]))["p_value"] > 0.05


def test_cat_cat_detects_repairing_at_equal_strength():
    rng = np.random.default_rng(4)
    ra, rb = _cat_pairs(rng, 3000, {"x": "p", "y": "q", "z": "r"})
    state = rel.reference_state("cat_cat", ra, rb)
    ba, bb = _cat_pairs(rng, 400, {"x": "q", "y": "p", "z": "r"})  # same strength, swapped pairing
    result = _p("cat_cat", state, (ra, rb), (ba, bb))
    assert abs(result["current_value"] - result["reference_value"]) < 0.1  # V barely moves...
    assert result["p_value"] < 0.01                                         # ...but the pairing did


def test_cat_cat_unseen_values_excluded_and_counted():
    rng = np.random.default_rng(5)
    ra, rb = _cat_pairs(rng, 1000, {"x": "p", "y": "q", "z": "r"})
    state = rel.reference_state("cat_cat", ra, rb)
    ba, bb = _cat_pairs(rng, 200, {"x": "p", "y": "q", "z": "r"})
    ba[:7] = "brand_new"
    assert rel.prepare_test("cat_cat", state, (ra, rb), (ba, bb))["excluded_unseen"] == 7


# ---------------------------------------------------------
# numeric <-> categorical
# ---------------------------------------------------------
def _fee_by_breed(rng, n, means, probs):
    k = rng.choice(list(means), size=n, p=probs).astype(object)
    return np.array([rng.normal(means[c], 10) for c in k]), k


def test_num_cat_ignores_common_shift_and_mix_change():
    rng = np.random.default_rng(6)
    means = {"lab": 100, "beagle": 150, "poodle": 200}
    rx, rk = _fee_by_breed(rng, 3000, means, [0.5, 0.3, 0.2])
    state = rel.reference_state("num_cat", rx, rk)
    bx, bk = _fee_by_breed(rng, 400, means, [0.2, 0.3, 0.5])  # category mix changed
    assert _p("num_cat", state, (rx, rk), (bx * 1.5 + 40, bk))["p_value"] > 0.05  # plus a common monotone shift


def test_num_cat_detects_change_in_relative_position():
    rng = np.random.default_rng(7)
    rx, rk = _fee_by_breed(rng, 3000, {"lab": 100, "beagle": 150, "poodle": 200}, [0.4, 0.3, 0.3])
    state = rel.reference_state("num_cat", rx, rk)
    bx, bk = _fee_by_breed(rng, 400, {"lab": 200, "beagle": 150, "poodle": 100}, [0.4, 0.3, 0.3])
    result = _p("num_cat", state, (rx, rk), (bx, bk))
    assert result["p_value"] < 0.01 and "median" in result["explanation"]


def test_num_cat_handles_heavily_tied_codes():
    """Regression: integer codes with big tie groups (like PetFinder's Breed1)
    made a KS-on-PIT statistic large even with no drift, hiding real breaks."""
    rng = np.random.default_rng(11)
    k = rng.choice(["dog", "cat"], size=3000).astype(object)
    x = np.where(k == "dog", rng.choice([307, 179, 20], size=3000, p=[0.7, 0.2, 0.1]),
                 rng.choice([266, 265, 299], size=3000, p=[0.6, 0.3, 0.1])).astype(float)
    state = rel.reference_state("num_cat", x, k)
    same = _p("num_cat", state, (x[:2000], k[:2000]), (x[2000:2300], k[2000:2300]))
    assert same["effect"] < 0.05  # the old KS-on-PIT statistic was ~0.6 here
    shuffled = _p("num_cat", state, (x[:2000], k[:2000]), (rng.permutation(x[2000:2300]), k[2000:2300]))
    assert shuffled["p_value"] < 0.01 and shuffled["effect"] > 0.1  # each category moves ~0.12 toward 0.5


# ---------------------------------------------------------
# Family / Holm integration
# ---------------------------------------------------------
def test_relationship_tests_join_the_holm_family():
    rng = np.random.default_rng(8)
    ref = {"a": list(rng.normal(size=500)), "b": list(rng.normal(size=500))}
    batch = {"a": list(rng.normal(size=200)), "b": list(rng.normal(size=200))}
    det = DistributionDetector(calibration_config=CalibrationConfig(decision_mode="calibrated"))
    det.fit_baseline(ref, {"a": "continuous", "b": "continuous"})
    without = det.analyze_production_window(batch)
    extra = [{"name": "a<->b", "p_value": 0.001, "effect_size": 0.5, "effect_floor": 0.1, "label": "test"}]
    with_rel = det.analyze_production_window(batch, extra)

    raw = [without["feature_metrics"][f]["p_value"] for f in ("a", "b")] + [0.001]
    expected = holm_adjust(raw)
    assert [with_rel["feature_metrics"][f]["p_value_adjusted"] for f in ("a", "b")] == expected[:2]
    assert with_rel["extra_metrics"]["a<->b"]["p_value_adjusted"] == expected[2]
    assert with_rel["extra_metrics"]["a<->b"]["drift_detected"] and with_rel["system_alert_triggered"]
    assert "extra_metrics" not in without  # no extras: output shape unchanged


def test_null_draws_resolve_holm_first_threshold():
    for m in (1, 10, 80):
        k = rel.null_draws_for_family(m, 0.05)
        assert 1 / (k + 1) < 0.05 / m


# ---------------------------------------------------------
# Proposals and screening
# ---------------------------------------------------------
def test_proposals_rank_strong_pairs_and_list_weak_ones():
    rng = np.random.default_rng(9)
    x, y = _correlated(rng, 400, 0.8)
    noise = rng.normal(size=400)
    out = rel.propose_relationships({"x": ("numeric", x), "y": ("numeric", y), "noise": ("numeric", noise)})
    by_pair = {(c["col_a"], c["col_b"]): c for c in out}
    assert by_pair[("x", "y")]["proposed"] and out[0]["col_a"] == "x"
    assert not by_pair[("noise", "x")]["proposed"] and "(weak)" in by_pair[("noise", "x")]["reason"]


def test_screening_flags_dependency_that_emerged():
    rng = np.random.default_rng(10)
    ref = {"x": rng.normal(size=500), "y": rng.normal(size=500)}
    bx, by = _correlated(rng, 300, 0.8)
    found = rel.screen_emerged(ref, {"x": bx, "y": by}, {"x": "numeric", "y": "numeric"}, watched=set())
    assert found and found[0]["current"] >= 0.3
    assert rel.screen_emerged(ref, {"x": bx, "y": by}, {"x": "numeric", "y": "numeric"},
                              watched={frozenset(("x", "y"))}) == []
