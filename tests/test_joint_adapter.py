"""
Smoke tests for the joint multimodal context drift path (JointAdapter +
the shared EmbeddingDriftDetector, used here with the joint-only classifier
from build_joint_classifier()) — NOT a full validation suite, matching the
discipline already established for text/image in
tests/test_embedding_adapters.py.

These tests check:
  - shape/determinism and correct handling of partial (missing-modality)
    records, including the new interaction_text/interaction_image blocks
  - the categorical frequency-encoding function's correctness in isolation
  - same-distribution batches center near AUC=0.5 (no false positives),
    using the actual joint-only classifier the production endpoint uses
  - a pure correlation inversion between a continuous tabular field and an
    image class, with zero marginal movement in either modality (verified
    via an independent KS-test and the image-only Domain Classifier Test),
    IS detected at the originally-tuned effect size — via the bounded
    interaction feature in adapters/joint.py plus build_joint_classifier()'s
    L1 regularization (still a LINEAR classifier, just L1-penalized — not
    a nonlinear model).
  - that fix's ACTUAL, verified scope — not an idealized one: it
    generalizes to a noisy/imperfect pairing at the same effect size, but
    NOT to a smaller correlation-inversion magnitude at a fixed C=2.0 (a
    real, reproducible gap, encoded as a passing test rather than left as
    a comment). See JointAdapter's class docstring and
    build_joint_classifier()'s docstring for the full mechanism, what was
    tried and rejected (adaptive C), and the remaining gaps.

Run:
    python -m pytest tests/test_joint_adapter.py -v
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image
from scipy import stats

from adapters.joint import JointAdapter, build_joint_classifier, TEXT_DIM, IMAGE_DIM, PRESENCE_DIM
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

        f_dim = len(stats_)
        expected_dim = f_dim + TEXT_DIM + IMAGE_DIM + 2 * f_dim + PRESENCE_DIM
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
        expected_dim = f_dim + TEXT_DIM + IMAGE_DIM + 2 * f_dim + PRESENCE_DIM
        assert emb.shape == (3, expected_dim)

        interaction_text_offset = f_dim + TEXT_DIM + IMAGE_DIM
        interaction_image_offset = interaction_text_offset + f_dim
        presence_offset = interaction_image_offset + f_dim

        # record 0: tabular + text, no image
        assert list(emb[0, presence_offset:presence_offset + 3]) == [1.0, 1.0, 0.0]
        assert not np.allclose(emb[0, f_dim:f_dim + TEXT_DIM], 0.0)  # text embedding present, non-zero
        assert np.allclose(emb[0, f_dim + TEXT_DIM:interaction_text_offset], 0.0)  # image segment zero-filled
        # record 1: image only
        assert list(emb[1, presence_offset:presence_offset + 3]) == [0.0, 0.0, 1.0]
        # record 2: tabular only
        assert list(emb[2, presence_offset:presence_offset + 3]) == [1.0, 0.0, 0.0]

    def test_interaction_blocks_are_exactly_zero_when_either_side_is_absent(self):
        """
        Verified, not assumed: a linear projection of a zero-filled
        modality is exactly zero, so both interaction blocks must be
        exactly zero whenever tabular is absent (regardless of text/image)
        or whenever text/image is absent (regardless of tabular).
        """
        adapter = JointAdapter()
        records = [
            {"tabular": {"price": 10.0}, "text": None, "image": None},           # tabular only
            {"tabular": None, "text": "some text", "image": _b64_png((1, 2, 3))},  # no tabular at all
            {"tabular": {"price": 30.0}, "text": "another sentence", "image": _b64_png((4, 5, 6))},  # all three
        ]
        stats_ = adapter.fit_tabular_schema(records)
        emb = adapter.transform(records, stats_)

        f_dim = len(stats_)
        assert f_dim > 0
        interaction_text_offset = f_dim + TEXT_DIM + IMAGE_DIM
        interaction_image_offset = interaction_text_offset + f_dim
        presence_offset = interaction_image_offset + f_dim

        # record 0: text/image absent -> both interaction blocks zero
        assert np.allclose(emb[0, interaction_text_offset:presence_offset], 0.0)
        # record 1: tabular absent -> both interaction blocks zero, even though text+image are present
        assert np.allclose(emb[1, interaction_text_offset:presence_offset], 0.0)
        # record 2: all three present -> interaction blocks should generically be non-zero
        assert not np.allclose(emb[2, interaction_text_offset:presence_offset], 0.0)

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
            result = EmbeddingDriftDetector(auc_threshold=0.65, classifier=build_joint_classifier()).analyze(ref_emb, cur_emb)
            aucs.append(result["statistic"])

        mean_auc = np.mean(aucs)
        assert 0.35 <= mean_auc <= 0.65, f"Expected AUC to center near 0.5, got mean={mean_auc:.3f}"

    def test_correlation_inversion_is_detected_via_interaction_feature(self):
        """
        Setup: baseline pairs a 'low' tabular-score pool with image-class-A
        and a 'high' tabular-score pool with image-class-B. Production
        re-pairs the SAME two fixed pools the other way around. Because
        it's a permutation of fixed pools (not resampling), both marginals
        are preserved exactly, by construction — only the correlation
        between the two modalities inverts.

        This was originally a documented blind spot: plain concatenation
        fed to a linear classifier structurally cannot represent the
        XOR-style interaction such a swap requires. It's closed by two
        additions working together (see JointAdapter's class docstring for
        the full mechanism): the interaction_text/interaction_image blocks
        in adapters/joint.py, and build_joint_classifier()'s L1
        regularization (needed because that interaction signal is diluted
        among the much larger raw embedding dimensions under the default
        L2 classifier — verified empirically, not assumed: isolating just
        the interaction_image column gives AUC=0.75 in isolation but
        AUC=0.40 when combined under L2). Verified across 8 independent
        data seeds at this exact effect size (AUC 0.97-1.00 at C=2.0) —
        but all 8 seeds share this test's cluster separation and image
        classes, varying only the random noise. That is NOT the same as
        general robustness: see
        test_correlation_inversion_not_detected_at_smaller_separation
        below for a different, smaller-separation scenario where this
        same fix does not work, and
        test_correlation_inversion_generalizes_to_noisy_pairing for a
        scenario (at this test's separation, but a noisy/imperfect
        pairing) where it does. Read all three together for the honest
        picture, not this one in isolation.

        Things this does NOT close: a pure text<->image inversion on a
        project with zero tabular fields (no anchor for the interaction
        terms), and correlation inversions with a smaller effect size than
        this test's — see the two tests below.
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

        # --- (b) the joint embedding + interaction feature + L1 classifier
        # DOES catch this pure correlation inversion — see docstring above.
        joint_adapter = JointAdapter()
        tabular_stats = joint_adapter.fit_tabular_schema(baseline_records)
        baseline_joint_emb = joint_adapter.transform(baseline_records, tabular_stats)
        scrambled_joint_emb = joint_adapter.transform(scrambled_records, tabular_stats)

        joint_result = EmbeddingDriftDetector(auc_threshold=0.65, classifier=build_joint_classifier()).analyze(
            baseline_joint_emb, scrambled_joint_emb
        )
        assert joint_result["drift_detected"], (
            f"Expected the interaction feature + L1 classifier to catch this correlation "
            f"inversion (see docstring) — if this now fails, something in that mechanism "
            f"regressed. Got AUC={joint_result['statistic']:.3f}"
        )

    def test_correlation_inversion_generalizes_to_noisy_pairing(self):
        """
        Same correlation-inversion setup and effect size as
        test_correlation_inversion_is_detected_via_interaction_feature,
        but the pairing is 80/20 noisy (not a perfectly deterministic
        swap) in both the baseline and the flipped/scrambled batch. This
        checks the fix isn't limited to an unrealistically clean toy case
        — it generalizes along the noise axis (unlike the effect-size
        axis; see test_correlation_inversion_not_detected_at_smaller_separation).
        """
        n = 40
        rng = np.random.default_rng(1234)
        low_scores = rng.normal(10, 1, size=n).tolist()
        high_scores = rng.normal(50, 1, size=n).tolist()
        class_a_pool = [_noisy_b64_png(0, seed=70000 + i) for i in range(20)]
        class_b_pool = [_noisy_b64_png(2, seed=80000 + i) for i in range(20)]

        def noisy_pairing(seed, flip, noise=0.2):
            local_rng = np.random.default_rng(seed)
            records = []
            for s in low_scores:
                use_b = (local_rng.random() < noise) if not flip else (local_rng.random() >= noise)
                pool = class_b_pool if use_b else class_a_pool
                records.append({"tabular": {"score": s}, "image": pool[local_rng.integers(len(pool))]})
            for s in high_scores:
                use_a = (local_rng.random() < noise) if not flip else (local_rng.random() >= noise)
                pool = class_a_pool if use_a else class_b_pool
                records.append({"tabular": {"score": s}, "image": pool[local_rng.integers(len(pool))]})
            return records

        baseline_records = noisy_pairing(1, flip=False)
        scrambled_records = noisy_pairing(2, flip=True)

        adapter = JointAdapter()
        tabular_stats = adapter.fit_tabular_schema(baseline_records)
        baseline_emb = adapter.transform(baseline_records, tabular_stats)
        scrambled_emb = adapter.transform(scrambled_records, tabular_stats)

        result = EmbeddingDriftDetector(auc_threshold=0.65, classifier=build_joint_classifier()).analyze(
            baseline_emb, scrambled_emb
        )
        assert result["drift_detected"], (
            f"Expected the fix to tolerate a noisy (80/20) pairing at the tuned effect size. "
            f"Got AUC={result['statistic']:.3f}"
        )

    def test_correlation_inversion_not_detected_at_smaller_separation(self):
        """
        DOCUMENTS A REAL, VERIFIED GAP — this test is expected to show the
        fix NOT catching the drift, and that's the point (same spirit as
        the original pre-fix blind-spot test this replaced).

        Same permutation-of-fixed-pools construction as
        test_correlation_inversion_is_detected_via_interaction_feature,
        but with a smaller tabular cluster separation (values ~30 vs ~45,
        z-score gap ~1.4, instead of ~10 vs ~50, z-score gap ~2.0) and
        different image classes (green-biased vs dim-red-biased, instead
        of red vs blue) — i.e. a scenario NOT used while tuning C=2.0.

        At the current fixed C=2.0, this is NOT detected (AUC well below
        the 0.65 threshold, and below 0.5). A different C window (~5-20)
        does rescue this specific case empirically, but it is a different
        window than the one that works for the larger-separation test
        above, and no single fixed C was found that covers both. Adaptive
        C was considered and rejected — see build_joint_classifier()'s
        docstring for why. This is the accurate, current scope of the fix:
        validated for correlation inversions of roughly the originally-
        tuned effect size, not a general solution.
        """
        n = 40
        rng = np.random.default_rng(999)
        val_lo = rng.normal(30, 2, size=n).tolist()
        val_hi = rng.normal(45, 2, size=n).tolist()
        imgs_x = [_noisy_b64_png(1, seed=50000 + i) for i in range(n)]  # green-biased
        imgs_y = [_noisy_b64_png(0, seed=60000 + i) for i in range(n)]  # dim red-biased

        def build_records(score_pool_a, image_pool_a, score_pool_b, image_pool_b):
            records = [{"tabular": {"score": s}, "image": img} for s, img in zip(score_pool_a, image_pool_a)]
            records += [{"tabular": {"score": s}, "image": img} for s, img in zip(score_pool_b, image_pool_b)]
            return records

        baseline_records = build_records(val_lo, imgs_x, val_hi, imgs_y)
        scrambled_records = build_records(val_lo, imgs_y, val_hi, imgs_x)

        adapter = JointAdapter()
        tabular_stats = adapter.fit_tabular_schema(baseline_records)
        baseline_emb = adapter.transform(baseline_records, tabular_stats)
        scrambled_emb = adapter.transform(scrambled_records, tabular_stats)

        result = EmbeddingDriftDetector(auc_threshold=0.65, classifier=build_joint_classifier()).analyze(
            baseline_emb, scrambled_emb
        )
        assert not result["drift_detected"], (
            f"Expected this smaller-separation case to remain undetected at the current "
            f"fixed C=2.0 (see docstring) — if this now passes, either the fix generalized "
            f"further than documented (update the docs!) or something else changed. "
            f"Got AUC={result['statistic']:.3f}"
        )
