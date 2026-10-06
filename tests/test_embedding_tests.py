"""
Embedding-based tests (milestone M3): real-embedding null for the column
DCT, probes, text<->image matching. Synthetic embeddings, seeded.

Run:
    python -m pytest tests/test_embedding_tests.py -v
"""

import io

import numpy as np
from sklearn.linear_model import LogisticRegression

from drift import embedding_tests as et
from drift.embedding_detector import EmbeddingDriftDetector

AUC = EmbeddingDriftDetector()._compute_auc
DRAWS = 60


def test_tail_pvalue_uses_gaussian_tail_only_beyond_all_draws():
    null = np.random.default_rng(0).normal(0.5, 0.02, size=200)
    p, tail = et.tail_pvalue(0.9, null)
    assert tail and p < 1 / 201
    p, tail = et.tail_pvalue(0.5, null)
    assert not tail and 0.2 < p < 0.8


def test_pca_caps_dimension_and_projects():
    emb = np.random.default_rng(1).normal(size=(40, 100))
    pca = et.fit_pca(emb, dim=64)
    assert pca["components"].shape == (39, 100) and et.project(emb, pca).shape == (40, 39)


def test_dct_real_null_separates_shift_from_no_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(400, 16))
    same = et.dct_test(AUC, ref, rng.normal(size=(100, 16)), DRAWS, seed=1)
    shifted = et.dct_test(AUC, ref, rng.normal(0.6, 1, size=(100, 16)), DRAWS, seed=1)
    assert same["auc"] < 0.65 and shifted["p_value"] < 0.02


def test_probe_params_reproduce_sklearn_predictions_and_survive_npz():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 8))
    for classes in (2, 3):
        y = np.array([f"c{i}" for i in np.argmax(x[:, :classes], axis=1)])
        params = et.fit_probe(x, y, "categorical")
        buf = io.BytesIO()
        np.savez(buf, **params)
        loaded = dict(np.load(io.BytesIO(buf.getvalue()), allow_pickle=False))
        expected = LogisticRegression(max_iter=1000).fit(x, y).predict(x)
        assert (et.probe_predict(loaded, x, "categorical") == expected).all()


def _linked(rng, n):
    x = rng.normal(size=(n, 8))
    return x, np.where(x[:, 0] > 0, "a", "b")


def test_probe_detects_broken_pairing_not_a_kept_one():
    rng = np.random.default_rng(4)
    xr, yr = _linked(rng, 600)
    score = et.cross_fitted_score(xr, yr, "categorical")
    params = et.fit_probe(xr, yr, "categorical")
    xb, yb = _linked(rng, 150)
    kept = et.probe_test(xr, yr, xb, yb, "categorical", params, score, DRAWS, seed=1)
    broken = et.probe_test(xr, yr, xb, rng.permutation(yb), "categorical", params, score, DRAWS, seed=1)
    assert kept["effect"] < 0.1
    assert broken["p_value"] < 0.02 and broken["effect"] > 0.3


def test_numeric_probe_and_too_few_rows():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(500, 8))
    y = x[:, 0] * 3 + rng.normal(scale=0.5, size=500)
    score = et.cross_fitted_score(x, y, "numeric")
    assert score > 0.8
    params = et.fit_probe(x, y, "numeric")
    assert not et.probe_test(x, y, x[:10], y[:10], "numeric", params, score, DRAWS, seed=1)["testable"]


def test_matching_detects_shuffled_pairs():
    rng = np.random.default_rng(6)
    img = rng.normal(size=(500, 12))
    txt = img @ rng.normal(size=(12, 10)) + rng.normal(scale=0.3, size=(500, 10))
    score = et.cross_fitted_matching(img[:400], txt[:400])
    params = et.fit_matching(img[:400], txt[:400])
    kept = et.matching_test(img[:400], txt[:400], img[400:], txt[400:], params, score, DRAWS, seed=1)
    broken = et.matching_test(img[:400], txt[:400], img[400:], txt[400:][rng.permutation(100)], params, score,
                              DRAWS, seed=1)
    assert score > 0.9 and kept["effect"] < 0.1 and broken["p_value"] < 0.02
