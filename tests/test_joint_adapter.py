"""
Smoke tests for the joint multimodal context drift path (JointAdapter +
the existing, unmodified EmbeddingDriftDetector) — NOT a full validation
suite, matching the discipline already established for text/image in
tests/test_embedding_adapters.py.

These tests check:
  - shape/determinism and correct handling of partial (missing-modality)
    records
  - the categorical frequency-encoding function's correctness in isolation
  - same-distribution batches center near AUC=0.5 (no false positives)
  - a KNOWN, DOCUMENTED LIMITATION: a pure correlation inversion between a
    continuous tabular field and an image class, with zero marginal
    movement in either modality (verified via an independent KS-test and
    the image-only Domain Classifier Test), is NOT detected by the joint
    embedding. This is a structural consequence of using a linear
    classifier (LogisticRegression, shared with text/image via
    drift/embedding_detector.py, deliberately left unmodified) over
    concatenated features — see the test's docstring for the full
    reasoning. Correlation breaks that co-occur with any marginal movement,
    or with a shift in which modalities are typically present, are not
    subject to this limitation.

Run:
    python -m pytest tests/test_joint_adapter.py -v
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image
from scipy import stats

from adapters.joint import JointAdapter, TEXT_DIM, IMAGE_DIM, PRESENCE_DIM
from adapters.image import ImageAdapter
from drift.embedding_detector import EmbeddingDriftDetector


def _b64_png(color, size=(32, 32)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _noisy_b64_png(channel_bias, seed, size=(32, 32)):
    """Same synthetic-image-class generator used earlier in this project's
    manual multimodal validation: a color-biased noise image standing in
    for a distinct 'visual class'."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 80, size=(*size, 3), dtype=np.uint8)
    arr[:, :, channel_bias] = rng.integers(180, 256, size=size, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestJointAdapterShape:
    def test_full_records_shape_and_determinism(self):
        adapter = JointAdapter()
        records = [
            {"tabular": {"price": 10.0, "category": "a"}, "text": "hello world", "image": _b64_png((255, 0, 0))},
            {"tabular": {"price": 20.0, "category": "b"}, "text": "another sentence", "image": _b64_png((0, 255, 0))},
        ]
        stats_ = adapter.fit_tabular_schema(records)
        emb1 = adapter.transform(records, stats_)
        emb2 = adapter.transform(records, stats_)

        expected_dim = len(stats_) + TEXT_DIM + IMAGE_DIM + PRESENCE_DIM
        assert emb1.shape == (2, expected_dim)
        assert np.allclose(emb1, emb2)

    def test_partial_records_missing_modalities(self):
        adapter = JointAdapter()
        records = [
            {"tabular": {"price": 10.0}, "text": "some text", "image": None},
            {"tabular": None, "text": None, "image": _b64_png((0, 0, 255))},
            {"tabular": {"price": 30.0}, "text": None, "image": None},
        ]
        stats_ = adapter.fit_tabular_schema(records)
        emb = adapter.transform(records, stats_)

        f_dim = len(stats_)
        expected_dim = f_dim + TEXT_DIM + IMAGE_DIM + PRESENCE_DIM
        assert emb.shape == (3, expected_dim)

        presence_offset = f_dim + TEXT_DIM + IMAGE_DIM
        # record 0: tabular + text, no image
        assert list(emb[0, presence_offset:presence_offset + 3]) == [1.0, 1.0, 0.0]
        assert not np.allclose(emb[0, f_dim:f_dim + TEXT_DIM], 0.0)  # text embedding present, non-zero
        assert np.allclose(emb[0, f_dim + TEXT_DIM:presence_offset], 0.0)  # image segment zero-filled
        # record 1: image only
        assert list(emb[1, presence_offset:presence_offset + 3]) == [0.0, 0.0, 1.0]
        # record 2: tabular only
        assert list(emb[2, presence_offset:presence_offset + 3]) == [1.0, 0.0, 0.0]

    def test_record_with_no_modalities_raises(self):
        adapter = JointAdapter()
        records = [{"tabular": None, "text": None, "image": None}]
        with pytest.raises(ValueError):
            adapter.transform(records, {})

    def test_empty_batch(self):
        adapter = JointAdapter()
        emb = adapter.transform([], {})
        assert emb.shape[0] == 0


class TestCategoricalFrequencyEncoding:
    def test_known_frequencies_encoded_correctly(self):
        adapter = JointAdapter()
        records = [{"tabular": {"tier": v}} for v in ["gold", "gold", "gold", "silver"]]
        stats_ = adapter.fit_tabular_schema(records)
        assert stats_["tier"]["type"] == "categorical"
        assert stats_["tier"]["frequencies"]["gold"] == pytest.approx(0.75)
        assert stats_["tier"]["frequencies"]["silver"] == pytest.approx(0.25)

        vec = adapter._tabular_subvector({"tier": "gold"}, ["tier"], stats_)
        assert vec[0] == pytest.approx(0.75)

    def test_unseen_category_encodes_as_zero(self):
        adapter = JointAdapter()
        records = [{"tabular": {"tier": v}} for v in ["gold", "gold", "silver"]]
        stats_ = adapter.fit_tabular_schema(records)
        vec = adapter._tabular_subvector({"tier": "platinum"}, ["tier"], stats_)
        assert vec[0] == 0.0


class TestJointEmbeddingDriftDetector:
    def test_same_distribution_centers_near_point_five(self):
        adapter = JointAdapter()

        def make_records(seed):
            rng = np.random.default_rng(seed)
            return [
                {"tabular": {"score": float(rng.normal(10, 1))}, "image": _noisy_b64_png(0, seed * 1000 + i)}
                for i in range(30)
            ]

        aucs = []
        for trial in range(6):
            ref_records = make_records(trial)
            cur_records = make_records(trial + 100)
            stats_ = adapter.fit_tabular_schema(ref_records)
            ref_emb = adapter.transform(ref_records, stats_)
            cur_emb = adapter.transform(cur_records, stats_)
            result = EmbeddingDriftDetector(auc_threshold=0.65).analyze(ref_emb, cur_emb)
            aucs.append(result["statistic"])

        mean_auc = np.mean(aucs)
        assert 0.35 <= mean_auc <= 0.65, f"Expected AUC to center near 0.5, got mean={mean_auc:.3f}"

    def test_pure_correlation_inversion_is_a_known_blind_spot(self):
        """
        DOCUMENTS A KNOWN LIMITATION — this test is expected to show the
        joint detector NOT catching the drift, and that's the point.

        Setup: baseline pairs a 'low' tabular-score pool with image-class-A
        and a 'high' tabular-score pool with image-class-B. Production
        re-pairs the SAME two fixed pools the other way around. Because
        it's a permutation of fixed pools (not resampling), both marginals
        are preserved exactly, by construction — only the correlation
        between the two modalities inverts.

        Why the joint detector misses this: EmbeddingDriftDetector uses
        LogisticRegression — a purely additive linear model — over the
        concatenated [tabular | text | image | presence] vector. Catching
        this swap requires "image looks like class A" to mean *reference*
        when paired with a low score but mean *current* when paired with a
        high score — an XOR-style interaction no additive linear model can
        represent, regardless of how much signal is in the data. This was
        found empirically while building the joint feature (see spec/plan
        history) and is a structural property of concatenation + linear
        classification, not a bug to fix in this pass. The trade-off
        deliberately accepted: no changes to the shared
        drift/embedding_detector.py (which text/image also depend on) and
        no new trainable components in adapters/joint.py, at the cost of
        this specific "zero marginal movement anywhere" edge case going
        undetected. Correlation breaks that co-occur with *any* marginal
        movement in a modality, or with a shift in which modalities are
        typically present (via the presence-mask dimensions), are not
        subject to this limitation.
        """
        n = 40
        rng = np.random.default_rng(0)
        low_scores = rng.normal(10, 1, size=n).tolist()
        high_scores = rng.normal(50, 1, size=n).tolist()

        class_a_images = [_noisy_b64_png(0, seed=1000 + i) for i in range(n)]  # red-biased
        class_b_images = [_noisy_b64_png(2, seed=2000 + i) for i in range(n)]  # blue-biased

        def build_records(score_pool_a, image_pool_a, score_pool_b, image_pool_b):
            records = [{"tabular": {"score": s}, "image": img} for s, img in zip(score_pool_a, image_pool_a)]
            records += [{"tabular": {"score": s}, "image": img} for s, img in zip(score_pool_b, image_pool_b)]
            return records

        baseline_records = build_records(low_scores, class_a_images, high_scores, class_b_images)
        scrambled_records = build_records(low_scores, class_b_images, high_scores, class_a_images)

        # --- (a) marginals are genuinely unchanged ---
        baseline_scores = np.array([r["tabular"]["score"] for r in baseline_records])
        scrambled_scores = np.array([r["tabular"]["score"] for r in scrambled_records])
        ks_stat, ks_p = stats.ks_2samp(baseline_scores, scrambled_scores)
        assert ks_p > 0.99, "Tabular field marginal should be exactly unchanged (same multiset of values)"

        image_adapter = ImageAdapter()
        baseline_image_emb = image_adapter.transform([r["image"] for r in baseline_records])
        scrambled_image_emb = image_adapter.transform([r["image"] for r in scrambled_records])
        image_only_result = EmbeddingDriftDetector(auc_threshold=0.65).analyze(baseline_image_emb, scrambled_image_emb)
        assert not image_only_result["drift_detected"], (
            f"Image-only marginal should be unchanged (same multiset of images), "
            f"got AUC={image_only_result['statistic']:.3f}"
        )

        # --- (b) the joint embedding does NOT catch a pure correlation
        # inversion with zero marginal movement anywhere — see the
        # docstring above for why this is expected, not a bug.
        joint_adapter = JointAdapter()
        tabular_stats = joint_adapter.fit_tabular_schema(baseline_records)
        baseline_joint_emb = joint_adapter.transform(baseline_records, tabular_stats)
        scrambled_joint_emb = joint_adapter.transform(scrambled_records, tabular_stats)

        joint_result = EmbeddingDriftDetector(auc_threshold=0.65).analyze(baseline_joint_emb, scrambled_joint_emb)
        assert not joint_result["drift_detected"], (
            f"Expected this known blind spot to persist (see docstring) — if this now "
            f"fails, the detector's separating power has changed and the limitation "
            f"documented here (and in adapters/joint.py / README) needs re-checking, "
            f"not just this assertion. Got AUC={joint_result['statistic']:.3f}"
        )
