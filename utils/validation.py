"""
Structural validation for ingestion endpoints, applied at the earliest
possible point in each request -- before any DataFrame construction,
profiling, or embedding compute runs. Failures raise ValidationError
(caught in main.py and converted to a clean HTTPException(422, ...))
naming exactly what's wrong, instead of letting a raw pandas/numpy/PIL
exception reach the client (see tests/test_ingestion_robustness.py for
the reproduced failures this closes).
"""

from typing import Any, Dict, List, Optional


class ValidationError(ValueError):
    """Raised for a structural problem with an ingestion request. Caught
    in main.py and converted to a 422 response with this message."""


TABULAR_RECOMMENDED_MIN_SAMPLES = 40


def validate_tabular_columns(*column_dicts: Dict[str, List[Any]]) -> int:
    """
    Validates that every column across the given dict(s) has the same
    length, and that there's at least one non-empty column. Call BEFORE
    pd.DataFrame(...) so a length mismatch produces a specific message
    ("columns 'a' (3) and 'b' (2) disagree") instead of pandas' generic
    "All arrays must be of the same length". Returns the row count.
    """
    lengths: Dict[str, int] = {}
    for column_dict in column_dicts:
        for name, values in column_dict.items():
            lengths[name] = len(values)

    if not lengths:
        raise ValidationError(
            "reference_data (and categorical_data, if provided) must contain at least one column."
        )

    distinct = set(lengths.values())
    if len(distinct) > 1:
        detail = ", ".join(f"'{name}' ({n} values)" for name, n in sorted(lengths.items()))
        raise ValidationError(f"All columns must have the same number of values, but they don't: {detail}.")

    n_rows = next(iter(lengths.values()))
    if n_rows == 0:
        raise ValidationError("Columns are present but contain no values.")

    return n_rows


def warn_if_below_recommended_samples(n_rows: int, minimum: int = TABULAR_RECOMMENDED_MIN_SAMPLES, label: str = "rows") -> Optional[str]:
    """
    Returns a warning string if n_rows is below the recommended minimum,
    else None. Non-fatal by design -- matches the dashboard's existing
    text/image sample-size guidance, which warns rather than blocks.
    """
    if n_rows < minimum:
        return f"Only {n_rows} {label} provided; {minimum}+ is recommended for a statistically reliable baseline."
    return None


def validate_min_samples(n_samples: int, hard_min: int, recommended_min: int, label: str) -> Optional[str]:
    """
    Raises ValidationError if n_samples < hard_min -- matches
    drift/embedding_detector.py's own hard floor (HARD_MIN_SAMPLES),
    checked here too so a request fails before any embedding compute is
    wasted, not just inside the detector afterward. Returns a non-fatal
    warning string if between hard_min and recommended_min, else None.
    """
    if n_samples < hard_min:
        raise ValidationError(
            f"{label} needs at least {hard_min} samples to run the Domain Classifier Test, got {n_samples}."
        )
    if n_samples < recommended_min:
        return f"Only {n_samples} {label} samples provided; {recommended_min}+ is recommended for reliable results."
    return None


def validate_joint_records(records: List[Dict[str, Any]]) -> None:
    """
    Validates every joint record has at least one modality present,
    BEFORE any embedding work starts. JointAdapter.transform() already
    checks this, but deep inside, after any earlier records in the batch
    have already been embedded -- checking here fails fast for the whole
    batch up front instead of wasting that compute on a request that was
    always going to be rejected.
    """
    if not records:
        raise ValidationError("At least one record is required.")
    for i, record in enumerate(records):
        if not (record.get("tabular") or record.get("text") or record.get("image")):
            raise ValidationError(
                f"Record at index {i} has no tabular, text, or image content — "
                "at least one modality is required per record."
            )
