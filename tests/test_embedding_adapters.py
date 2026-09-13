"""
Smoke tests for the v2.0 text/image adapters and the embedding-based
Domain Classifier Test — NOT the full four-part validation suite from
spec.md §6 (that requires real labeled text/image drift datasets and is
a deliberate follow-up, per spec §7).

These tests check the acceptance criteria spelled out in spec.md §5.2,
§5.3, and §5.5 directly:
  - adapters produce fixed-dim, deterministic embeddings and tolerate
    edge-case inputs without crashing
  - the Domain Classifier Test centers near AUC=0.5 on same-distribution
    batches and flags drift on clearly separated ones

Run:
    python -m pytest tests/test_embedding_adapters.py -v
"""

import io

import numpy as np
import pytest
from PIL import Image

from adapters.text import TextAdapter
from adapters.image import ImageAdapter
from drift.embedding_detector import EmbeddingDriftDetector


def _solid_color_image_b64(color, size=(64, 64), fmt="PNG"):
    import base64
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestTextAdapter:
    def test_shape_and_determinism(self):
        texts = ["hello world", "drift monitoring is useful", "a third sentence"]
        adapter = TextAdapter()
        emb1 = adapter.transform(texts)
        emb2 = adapter.transform(texts)

        assert emb1.shape == (3, 384)
        assert np.allclose(emb1, emb2)

    def test_empty_batch(self):
        adapter = TextAdapter()
        emb = adapter.transform([])
        assert emb.shape == (0, 384)

    def test_empty_and_long_strings_do_not_crash(self):
        adapter = TextAdapter()
        long_string = "word " * 2000
        emb = adapter.transform(["", long_string, "normal sentence"])
        assert emb.shape == (3, 384)


class TestImageAdapter:
    def test_shape_and_mixed_sizes_formats(self):
        images = [
            _solid_color_image_b64((255, 0, 0), size=(32, 32), fmt="PNG"),
            _solid_color_image_b64((0, 255, 0), size=(200, 100), fmt="JPEG"),
        ]
        adapter = ImageAdapter()
        emb = adapter.transform(images)
        assert emb.shape == (2, 512)

    def test_empty_batch(self):
        adapter = ImageAdapter()
        emb = adapter.transform([])
        assert emb.shape == (0, 512)

    def test_grayscale_and_rgba_do_not_crash(self):
        import base64
        buf_l = io.BytesIO()
        Image.new("L", (48, 48), color=128).save(buf_l, format="PNG")
        buf_rgba = io.BytesIO()
        Image.new("RGBA", (48, 48), color=(10, 20, 30, 128)).save(buf_rgba, format="PNG")

        images = [
            base64.b64encode(buf_l.getvalue()).decode("ascii"),
            base64.b64encode(buf_rgba.getvalue()).decode("ascii"),
        ]
        adapter = ImageAdapter()
        emb = adapter.transform(images)
        assert emb.shape == (2, 512)


class TestEmbeddingDriftDetector:
    def test_same_distribution_centers_near_point_five(self):
        rng = np.random.default_rng(0)
        detector = EmbeddingDriftDetector(auc_threshold=0.65)

        aucs = []
        for trial in range(10):
            rng_trial = np.random.default_rng(trial)
            ref = rng_trial.normal(size=(200, 16))
            cur = rng_trial.normal(size=(200, 16))
            result = detector.analyze(ref, cur)
            aucs.append(result["statistic"])

        mean_auc = np.mean(aucs)
        assert 0.4 <= mean_auc <= 0.6, f"Expected AUC to center near 0.5, got mean={mean_auc:.3f}"

    def test_separated_clusters_trigger_drift(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(loc=0.0, scale=1.0, size=(200, 16))
        cur = rng.normal(loc=5.0, scale=1.0, size=(200, 16))

        detector = EmbeddingDriftDetector(auc_threshold=0.65)
        result = detector.analyze(ref, cur)

        assert result["statistic"] > 0.9
        assert result["drift_detected"] is True

    def test_too_few_samples_raises_clear_error(self):
        detector = EmbeddingDriftDetector()
        with pytest.raises(ValueError):
            detector.analyze(np.random.normal(size=(1, 4)), np.random.normal(size=(1, 4)))
