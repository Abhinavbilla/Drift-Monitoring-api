"""
Step 2 (d), 2026-10-04: calibrated-mode DCT p-values for text/image.

Grid-lookup tests use a small, fast, monkeypatched fake grid (not the
real large precomputed one) for deterministic, quick unit assertions --
end-to-end tests against the real grid/API are separate.

Run:
    python -m pytest tests/test_dct_calibration.py -v
"""

import random
import sys

import numpy as np
import pytest

sys.path.insert(0, "tests")
from _session_auth import mint_session_token  # noqa: E402

from fastapi.testclient import TestClient

import db.crud as crud
import drift.dct_calibration as dct_calibration
from drift.calibration import CalibrationConfig
from drift.embedding_detector import EmbeddingDriftDetector
from main import app, _internal_project_key

client = TestClient(app)
EMAIL = "dct-cal-api-test@example.com"
EMAIL_B = "dct-cal-api-test-b@example.com"
TOKEN = mint_session_token(EMAIL)
TOKEN_B = mint_session_token(EMAIL_B)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
HEADERS_B = {"Authorization": f"Bearer {TOKEN_B}"}


def _internal(public_id, email=EMAIL):
    return _internal_project_key(public_id, email)


@pytest.fixture(autouse=True)
def cleanup_api_projects():
    yield
    conn = crud.get_connection()
    cur = conn.cursor()
    for table, col in [("baselines", "project_id"), ("projects", "id"),
                        ("baseline_versions", "project_id"), ("baseline_active_version", "project_id")]:
        cur.execute(f"DELETE FROM {table} WHERE {col} LIKE '%test_dctcal_api_%'")
    conn.commit()
    conn.close()


def _fake_auc_fn(ref, cur):
    """Deterministic stand-in for the real classifier -- just returns a
    fixed value, fast, no actual model fit. Used only where the test
    cares about plumbing (lookup/fallback selection), not the real
    statistic's numeric value."""
    return 0.5


class TestGridLookupAndPValue:
    def test_exact_grid_cell_match(self, monkeypatch):
        fake_grid = {"grid": {"384": [{"n_ref": 100, "n_batch": 100, "null_auc_draws": [0.5] * 9 + [0.9]}]}}
        monkeypatch.setattr(dct_calibration, "_load_grid", lambda: fake_grid)
        p = dct_calibration.dct_pvalue(0.9, 384, 100, 100, auc_fn=_fake_auc_fn)
        assert p == pytest.approx(2 / 11)  # 1 draw >= 0.9, +1 correction, 10 draws

    def test_nearest_neighbor_match_for_unseen_size(self, monkeypatch):
        fake_grid = {"grid": {"384": [
            {"n_ref": 50, "n_batch": 50, "null_auc_draws": [0.5] * 10},
            {"n_ref": 5000, "n_batch": 5000, "null_auc_draws": [0.9] * 10},
        ]}}
        monkeypatch.setattr(dct_calibration, "_load_grid", lambda: fake_grid)
        # n_ref=60 is much closer (in log-space) to 50 than to 5000.
        p_near_small = dct_calibration.dct_pvalue(0.5, 384, 60, 60, auc_fn=_fake_auc_fn)
        assert p_near_small == pytest.approx(11 / 11)  # every null draw (0.5) >= observed (0.5)

    def test_missing_dimension_falls_back_to_live_permutation(self, monkeypatch):
        fake_grid = {"grid": {"384": [{"n_ref": 100, "n_batch": 100, "null_auc_draws": [0.5] * 10}]}}
        monkeypatch.setattr(dct_calibration, "_load_grid", lambda: fake_grid)
        called = {"count": 0}

        def counting_auc_fn(ref, cur):
            called["count"] += 1
            return 0.5

        # 999 is not a calibrated dimension in fake_grid -- must fall back.
        dct_calibration.dct_pvalue(0.6, 999, 50, 50, auc_fn=counting_auc_fn)
        assert called["count"] == dct_calibration.LIVE_FALLBACK_DRAWS

    def test_no_grid_file_falls_back_to_live_permutation(self, monkeypatch):
        monkeypatch.setattr(dct_calibration, "_load_grid", lambda: None)
        called = {"count": 0}

        def counting_auc_fn(ref, cur):
            called["count"] += 1
            return 0.5

        dct_calibration.dct_pvalue(0.6, 384, 50, 50, auc_fn=counting_auc_fn)
        assert called["count"] == dct_calibration.LIVE_FALLBACK_DRAWS

    def test_pvalue_never_exactly_zero(self, monkeypatch):
        """The +1 correction -- an observed AUC exceeding every null draw
        still gets a nonzero p-value."""
        fake_grid = {"grid": {"384": [{"n_ref": 100, "n_batch": 100, "null_auc_draws": [0.5] * 50}]}}
        monkeypatch.setattr(dct_calibration, "_load_grid", lambda: fake_grid)
        p = dct_calibration.dct_pvalue(0.99, 384, 100, 100, auc_fn=_fake_auc_fn)
        assert p > 0
        assert p == pytest.approx(1 / 51)


class TestEmbeddingDriftDetectorLegacyUnaffected:
    def test_no_calibration_config_is_byte_identical_shape(self):
        rng = np.random.default_rng(1)
        ref = rng.standard_normal((50, 16))
        cur = rng.standard_normal((50, 16))
        result = EmbeddingDriftDetector().analyze(ref, cur)
        assert result["p_value"] is None
        assert set(result.keys()) == {"statistic", "p_value", "drift_detected"}

    def test_explicit_legacy_decision_mode_same_shape(self):
        rng = np.random.default_rng(1)
        ref = rng.standard_normal((50, 16))
        cur = rng.standard_normal((50, 16))
        config = CalibrationConfig(decision_mode="legacy")
        result = EmbeddingDriftDetector().analyze(ref, cur, calibration_config=config)
        assert result["p_value"] is None
        assert set(result.keys()) == {"statistic", "p_value", "drift_detected"}


class TestEmbeddingDriftDetectorCalibrated:
    def test_calibrated_mode_adds_two_gate_fields(self, monkeypatch):
        fake_grid = {"grid": {"16": [{"n_ref": 50, "n_batch": 50, "null_auc_draws": [0.5] * 20}]}}
        monkeypatch.setattr(dct_calibration, "_load_grid", lambda: fake_grid)

        rng = np.random.default_rng(1)
        ref = rng.standard_normal((50, 16))
        cur = rng.standard_normal((50, 16)) + 5  # shifted -- should separate easily
        config = CalibrationConfig(decision_mode="calibrated")
        result = EmbeddingDriftDetector().analyze(ref, cur, calibration_config=config)

        for key in ("effect_size", "effect_floor", "p_value_adjusted", "significant", "material",
                    "decision_mode", "threshold_used"):
            assert key in result
        assert result["decision_mode"] == "calibrated"
        assert result["effect_floor"] == 0.65  # DEFAULT_EFFECT_FLOORS["dct_auc"]
        assert result["drift_detected"] is True  # obviously separable batches


class TestFitAcceptsCalibrationConfig:
    """DB-layer check: insert_embedding_baseline persists and preserves
    calibration_config the same way insert_baseline does for tabular."""

    def test_calibration_config_persists_and_survives_refit_without_mention(self):
        pid = "test_dctcal_crud_persist"
        crud.insert_embedding_baseline(
            pid, "text", [[0.1, 0.2]] * 10, "fake-model",
            calibration_config={"decision_mode": "calibrated"},
        )
        assert crud.get_baseline(pid)["calibration_config"]["decision_mode"] == "calibrated"

        crud.insert_embedding_baseline(pid, "text", [[0.3, 0.4]] * 10, "fake-model")  # no mention
        assert crud.get_baseline(pid)["calibration_config"]["decision_mode"] == "calibrated"

        conn = crud.get_connection()
        conn.execute("DELETE FROM baselines WHERE project_id = ?", (pid,))
        conn.commit()
        conn.close()

    def test_new_embedding_project_defaults_to_legacy(self):
        """No calibration_config given at all -- resolves to legacy,
        same as tabular's pre-2026-09-30 projects. Unlike tabular, there
        is NO new-project override to 'calibrated' for embeddings (user
        decision, 2026-10-04: legacy-by-default until Step 2 (e) validates)."""
        pid = "test_dctcal_crud_default"
        crud.insert_embedding_baseline(pid, "text", [[0.1, 0.2]] * 10, "fake-model")
        assert crud.get_baseline(pid)["calibration_config"] is None

        conn = crud.get_connection()
        conn.execute("DELETE FROM baselines WHERE project_id = ?", (pid,))
        conn.commit()
        conn.close()


class TestTextAPIEndToEnd:
    """Real sentence-transformers embeddings via the actual /fit and
    /analyze/text endpoints -- proves the wiring (request ->
    insert_embedding_baseline -> analyze_text_batch -> EmbeddingDriftDetector)
    end to end, not just each layer in isolation."""

    def _fit(self, pid, headers=HEADERS, calibration_config=None):
        payload = {"reference_texts": [f"sentence number {i} about cats" for i in range(45)]}
        if calibration_config is not None:
            payload["calibration_config"] = calibration_config
        return client.post(f"/fit/{pid}/text", json=payload, headers=headers)

    def test_new_text_project_defaults_to_legacy_response_shape(self):
        pid = "test_dctcal_api_default"
        assert self._fit(pid).status_code == 200
        resp = client.post(f"/analyze/{pid}/text",
                            json={"production_texts": [f"sentence {i} about dogs" for i in range(45)]},
                            headers=HEADERS)
        metrics = resp.json()["feature_metrics"]["embedding_drift"]
        assert metrics["p_value"] is None  # legacy shape, unaffected

    def test_explicit_calibrated_mode_returns_two_gate_fields(self):
        pid = "test_dctcal_api_calibrated"
        assert self._fit(pid, calibration_config={"decision_mode": "calibrated"}).status_code == 200
        resp = client.post(f"/analyze/{pid}/text",
                            json={"production_texts": [f"completely different topic {i} about finance" for i in range(45)]},
                            headers=HEADERS)
        assert resp.status_code == 200
        metrics = resp.json()["feature_metrics"]["embedding_drift"]
        assert metrics["decision_mode"] == "calibrated"
        assert metrics["p_value_adjusted"] is not None

    def test_user_b_calibration_config_independent_of_user_a(self):
        pid = "test_dctcal_api_iso"
        self._fit(pid, headers=HEADERS, calibration_config={"decision_mode": "calibrated"})
        self._fit(pid, headers=HEADERS_B)  # B's own project, no calibration_config -> legacy

        resp_b = client.post(f"/analyze/{pid}/text",
                              json={"production_texts": [f"sentence {i}" for i in range(45)]},
                              headers=HEADERS_B)
        assert resp_b.json()["feature_metrics"]["embedding_drift"]["p_value"] is None


class TestImageAPIEndToEnd:
    """Same wiring as text, but exercises the 512-dim (resnet18) grid
    lookup specifically -- a different cell than text's 384-dim one."""

    def _solid_color_b64(self, color, size=(32, 32)):
        import base64
        import io
        from PIL import Image
        img = Image.new("RGB", size, color=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def test_calibrated_image_analyze_uses_512dim_grid(self):
        pid = "test_dctcal_api_image"
        ref_images = [self._solid_color_b64((10, 10, 10)) for _ in range(45)]
        resp = client.post(
            f"/fit/{pid}/image",
            json={"reference_images": ref_images, "calibration_config": {"decision_mode": "calibrated"}},
            headers=HEADERS,
        )
        assert resp.status_code == 200

        batch_images = [self._solid_color_b64((240, 240, 240)) for _ in range(45)]
        resp2 = client.post(f"/analyze/{pid}/image", json={"production_images": batch_images}, headers=HEADERS)
        assert resp2.status_code == 200
        metrics = resp2.json()["feature_metrics"]["embedding_drift"]
        assert metrics["decision_mode"] == "calibrated"
        assert metrics["drift_detected"] is True  # black vs near-white, trivially separable
