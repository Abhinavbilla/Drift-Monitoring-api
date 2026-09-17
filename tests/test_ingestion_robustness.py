"""
Ingestion robustness — reproduces each ingestion gap found while making
dataset ingestion "clean/validate at the earliest point" (mixed-type
tabular crash, discarded cleaning, missing/mis-scoped/mis-typed exception
handling, no structural validation, unbounded categorical cardinality),
then proves the fix. Each test starts from the concrete failure mode
confirmed against the pre-fix code before this pass, not a hypothetical.

Run:
    python -m pytest tests/test_ingestion_robustness.py -v
"""

import base64
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.crud import _calculate_boundaries, insert_baseline, get_baseline, MAX_CATEGORICAL_CARDINALITY
from utils.validation import (
    ValidationError,
    validate_tabular_columns,
    validate_min_samples,
    validate_joint_records,
)
from tests._session_auth import mint_session_token

import main
from main import app

client = TestClient(app)
TOKEN = mint_session_token("ingestion-robustness-tests@example.com")
AUTH = {"Authorization": f"Bearer {TOKEN}"}


# ---------------------------------------------------------
# Gap 1: a single stray non-numeric cell crashed/miscategorized a column
# depending on cell ORDER, instead of being coerced-and-reported.
# ---------------------------------------------------------
class TestMixedTypeColumnCoercion:
    def test_stray_string_after_numerics_no_longer_crashes(self):
        """
        Reproduced pre-fix: TypeError: unsupported operand type(s) for -:
        'numpy.str_' and 'numpy.str_' -- np.percentile on a mixed list
        because classification only checked the first value's type.
        """
        fences, cleaning_summary = _calculate_boundaries(
            {"mixed_col": [5.2, 6.1, 7.3, 8.0, 9.1, 10.4, 11.0, 12.5, 13.2, "N/A"]}
        )
        assert fences[0]["type"] == "continuous"
        assert cleaning_summary["mixed_col"]["dropped_non_numeric"] == 1

    def test_stray_string_first_no_longer_forces_categorical(self):
        """
        Pre-fix, a bad value landing FIRST silently sent an otherwise
        numeric column down the categorical path instead (no crash, but
        wrong) -- order shouldn't change the classification.
        """
        fences, cleaning_summary = _calculate_boundaries(
            {"mixed_col": ["N/A", 5.2, 6.1, 7.3, 8.0, 9.1, 10.4, 11.0, 12.5, 13.2]}
        )
        assert fences[0]["type"] == "continuous"
        assert cleaning_summary["mixed_col"]["dropped_non_numeric"] == 1

    def test_mostly_string_column_stays_categorical(self):
        """A column that's genuinely categorical (below the coercion
        threshold) must not be force-coerced just because a few cells
        happen to parse as numbers."""
        fences, cleaning_summary = _calculate_boundaries(
            {"cat_col": ["red", "blue", "green", "red", "5"]}
        )
        assert fences[0]["type"] == "categorical"
        assert "cat_col" not in cleaning_summary

    def test_cleaned_values_actually_propagate_to_storage(self):
        """
        Pre-fix: the profiler's own coercion was computed for
        classification only and then discarded -- the RAW, uncoerced
        reference_data reached crud.insert_baseline() and storage,
        so DistributionDetector's separate batch KS-test silently
        compared numpy-upcasted-to-string values (a wrong answer, not a
        crash -- the more dangerous half of this gap).
        """
        cleaning_summary = insert_baseline(
            project_id="test_ingestion_robustness_cleaning_propagation",
            feature_types={"mixed_col": "continuous"},
            reference_data={"mixed_col": [5.2, 6.1, 7.3, 8.0, 9.1, 10.4, 11.0, 12.5, 13.2, "N/A"]},
            categorical_data={},
        )
        assert cleaning_summary["mixed_col"]["dropped_non_numeric"] == 1

        state = get_baseline("test_ingestion_robustness_cleaning_propagation")
        stored = state["reference_data"]["mixed_col"]
        assert len(stored) == 9
        assert all(isinstance(v, (int, float)) for v in stored)


# ---------------------------------------------------------
# Gap 7: unbounded categorical cardinality
# ---------------------------------------------------------
class TestCategoricalCardinalityCap:
    def test_high_cardinality_categorical_column_rejected(self):
        many_values = [f"id_{i}" for i in range(MAX_CATEGORICAL_CARDINALITY + 1)]
        with pytest.raises(ValueError, match="exceeding"):
            _calculate_boundaries({"free_text_id": many_values})

    def test_categorical_column_at_the_cap_is_accepted(self):
        values = [f"id_{i}" for i in range(MAX_CATEGORICAL_CARDINALITY)]
        fences, _ = _calculate_boundaries({"cat_col": values})
        assert fences[0]["type"] == "categorical"


# ---------------------------------------------------------
# Gap 5: structural validation helpers (unit-level)
# ---------------------------------------------------------
class TestStructuralValidationHelpers:
    def test_mismatched_column_lengths_rejected(self):
        with pytest.raises(ValidationError, match="same number of values"):
            validate_tabular_columns({"a": [1, 2, 3], "b": [1, 2]})

    def test_empty_columns_rejected(self):
        with pytest.raises(ValidationError):
            validate_tabular_columns({})

    def test_no_values_rejected(self):
        with pytest.raises(ValidationError, match="no values"):
            validate_tabular_columns({"a": [], "b": []})

    def test_below_hard_min_samples_rejected(self):
        with pytest.raises(ValidationError, match="at least"):
            validate_min_samples(2, hard_min=4, recommended_min=40, label="reference text")

    def test_between_hard_and_recommended_returns_warning_not_error(self):
        warning = validate_min_samples(10, hard_min=4, recommended_min=40, label="reference text")
        assert warning is not None
        assert "40" in warning

    def test_joint_record_with_no_modality_rejected(self):
        with pytest.raises(ValidationError, match="index 1"):
            validate_joint_records([
                {"tabular": {"a": 1}, "text": None, "image": None},
                {"tabular": None, "text": None, "image": None},
            ])

    def test_joint_records_empty_list_rejected(self):
        with pytest.raises(ValidationError):
            validate_joint_records([])


# ---------------------------------------------------------
# Gap 3/4/5 at the HTTP boundary: malformed requests must return a clean
# 4xx with a specific message, never a raw 500 traceback.
# ---------------------------------------------------------
class TestEndpointBoundaryErrors:
    def test_tabular_fit_rejects_mismatched_column_lengths(self):
        resp = client.post(
            "/fit/test_ingestion_robustness_tabular_422",
            json={"reference_data": {"a": [1, 2, 3], "b": [1, 2]}},
            headers=AUTH,
        )
        assert resp.status_code == 422
        assert "same number of values" in resp.json()["detail"]

    def test_tabular_fit_rejects_completely_empty_payload(self):
        resp = client.post(
            "/fit/test_ingestion_robustness_tabular_empty",
            json={"reference_data": {}, "categorical_data": {}},
            headers=AUTH,
        )
        assert resp.status_code == 422

    def test_image_fit_rejects_corrupted_base64_cleanly(self):
        """
        Reproduced pre-fix: binascii.Error / UnidentifiedImageError are
        NOT ValueError subclasses, so the old bare `except ValueError`
        (and the missing try/except entirely on /fit) let this reach the
        client as a raw 500.
        """
        resp = client.post(
            "/fit/test_ingestion_robustness_image_bad_b64/image",
            json={"reference_images": ["not-valid-base64!!!"] * 4},
            headers=AUTH,
        )
        assert resp.status_code == 400
        assert resp.status_code != 500

    def test_image_fit_rejects_non_image_bytes_cleanly(self):
        garbage = base64.b64encode(b"this is not an image, just text bytes").decode("ascii")
        resp = client.post(
            "/fit/test_ingestion_robustness_image_not_an_image/image",
            json={"reference_images": [garbage, garbage, garbage, garbage]},
            headers=AUTH,
        )
        assert resp.status_code == 400
        assert resp.status_code != 500

    def test_joint_fit_rejects_record_with_no_modality(self):
        resp = client.post(
            "/fit/test_ingestion_robustness_joint_empty_record/joint",
            json={"reference_records": [{"tabular": None, "text": None, "image": None}]},
            headers=AUTH,
        )
        assert resp.status_code == 422
        assert "index 0" in resp.json()["detail"]

    def test_tabular_fit_reports_cleaning_summary_in_response(self):
        resp = client.post(
            "/fit/test_ingestion_robustness_tabular_cleaning_summary",
            json={
                "reference_data": {
                    "price": [5.2, 6.1, 7.3, 8.0, 9.4, 10.1, 11.7, 12.3, 13.9, "N/A"] * 5,
                },
                "categorical_data": {},
            },
            headers=AUTH,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "price" in body["cleaning_summary"]
        assert body["cleaning_summary"]["price"]["dropped_non_numeric"] == 5
