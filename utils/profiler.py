import pandas as pd
import re
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Shared with compute_signals' own text->numeric promotion rule below, so a
# column's classification (continuous vs categorical) and its actual cleaned
# values always agree -- two independent implementations of "80%" drifting
# apart was exactly the kind of silent inconsistency that let a stray
# non-numeric cell reach storage uncoerced (see db/crud.py's
# _calculate_boundaries and tests/test_ingestion_robustness.py).
NUMERIC_COERCION_THRESHOLD = 0.80


def coerce_numeric_column(values: List[Any], threshold: float = NUMERIC_COERCION_THRESHOLD) -> Tuple[Optional[List[float]], int]:
    """
    Attempts to coerce a column's raw values to numeric. If the success
    rate is above `threshold`, returns (cleaned_values, dropped_count) --
    the successfully-coerced values only, with unconvertible cells dropped
    rather than crashing downstream on a mixed-type list. If coercion
    success is below threshold, returns (None, 0): the column should be
    treated as categorical instead, values unchanged by the caller.
    """
    non_null = [x for x in values if x is not None]
    if not non_null:
        return [], 0

    series = pd.Series(non_null)
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.notna().mean() <= threshold:
        return None, 0

    clean_values = coerced.dropna().tolist()
    dropped = len(non_null) - len(clean_values)
    return clean_values, dropped


@dataclass
class ColumnSignals:
    n_rows: int
    n_unique: int
    cardinality_ratio: float
    dominant_ratio: float
    is_float: bool
    is_integer: bool
    is_text: bool
    is_datetime: bool
    is_monotonic: bool
    has_structured_pattern: bool
    value_range: float
    mean_str_length: float
    str_length_std: float


def compute_signals(series: pd.Series, n_rows: int) -> ColumnSignals:
    is_float = pd.api.types.is_float_dtype(series)
    is_integer = pd.api.types.is_integer_dtype(series)
    
    # FIX 2: Bulletproof text detection
    is_text = pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)

    # --- Feature Coercion ---
    if is_text and not is_datetime:
        coerced = pd.to_numeric(series, errors='coerce')
        if coerced.notna().mean() > NUMERIC_COERCION_THRESHOLD:
            series = coerced
            is_float = pd.api.types.is_float_dtype(series)
            is_integer = pd.api.types.is_integer_dtype(series)
            is_text = False

    n_unique = series.nunique()
    cardinality_ratio = n_unique / n_rows if n_rows > 0 else 0
    dominant_ratio = series.value_counts(normalize=True).iloc[0] if n_rows > 0 else 0
    value_range = 0.0
    if (is_float or is_integer) and len(series) > 1:
        val_min, val_max = series.min(), series.max()
        if pd.notna(val_min) and pd.notna(val_max):
            value_range = float(val_max - val_min)

    # --- Monotonicity ---
    is_monotonic = (
        (series.is_monotonic_increasing or series.is_monotonic_decreasing)
        and cardinality_ratio > 0.85
    )

    # --- String statistics ---
    mean_str_length = 0.0
    str_length_std = 0.0
    if is_text:
        str_lengths = series.dropna().astype(str).apply(len)
        if len(str_lengths) > 0:
            mean_str_length = float(str_lengths.mean())
            str_length_std = float(str_lengths.std()) if len(str_lengths) > 1 else 0.0

    # --- Structured ID Pattern Detection ---
    has_structured_pattern = False
    if is_text:
        sample = series.dropna().sample(min(100, len(series)), random_state=42)
        pattern = re.compile(r'^[A-Z0-9]{3,}[._\-\/][A-Z0-9]{2,}', re.IGNORECASE)
        match_ratio = sample.apply(lambda x: bool(pattern.match(str(x)))).mean()
        has_structured_pattern = match_ratio > 0.75 and mean_str_length > 6.0

    return ColumnSignals(
        n_rows=n_rows, n_unique=n_unique,
        cardinality_ratio=cardinality_ratio, dominant_ratio=dominant_ratio,
        is_float=is_float, is_integer=is_integer,
        is_text=is_text, is_datetime=is_datetime,
        is_monotonic=is_monotonic, has_structured_pattern=has_structured_pattern,
        value_range=value_range, mean_str_length=mean_str_length,
        str_length_std=str_length_std
    )


def classify_column(sig: ColumnSignals) -> Tuple[Any, str]:
    # 1. Constant / near‑constant
    if sig.dominant_ratio > 0.99 and sig.n_unique <= 5:
        return False, "Near-constant column (No drift possible)"
    if sig.is_datetime:
        return False, "Datetime column (Always drifts trivially)"
    if sig.is_monotonic:
        return False, "Monotonic sequence (Likely Time or Row Index)"

    # 2. Categorical
    if sig.is_text and sig.n_unique <= 50:
        return "Categorical", "Low-cardinality text (Suitable for PSI)"
    if sig.is_integer and sig.n_unique <= 20:
        return "Categorical", "Low-cardinality integer (Suitable for PSI)"

    # 3. Continuous
    if sig.is_float or (sig.is_integer and sig.n_unique > 20):
        return True, "Suitable for continuous drift monitoring"

    # 4. Ignore
    if sig.is_text and sig.has_structured_pattern and sig.str_length_std < 4.0:
        return False, "Structured code/ID pattern (Panel Entity ID)"
    if sig.cardinality_ratio > 0.95 and not sig.is_float:
        return False, "High cardinality integer/text (Likely an identifier)"
    if sig.is_text and sig.cardinality_ratio > 0.50 and not sig.has_structured_pattern:
        return False, "High-cardinality free text (Not monitorable)"

    # 5. Fallback
    return "Review", "Could not classify — recommend manual review"


def profile_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    n_rows = len(df)
    profiles = []

    for col in df.columns:
        series = df[col].dropna()

        # FIX 1: Hard-code defaults for empty columns so it doesn't crash
        if len(series) == 0:
            profiles.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "cardinality_ratio": 0.0,
                "dominant_ratio": 1.0,
                "is_datetime": False,
                "is_numeric": False,
                "monitor": False,
                "reason": "Empty column",
                "best_guess": "ignore"
            })
            continue

        sig = compute_signals(series, n_rows)
        monitor_status, reason = classify_column(sig)
        
        if col.lower() == "time" and sig.is_integer and sig.n_unique == sig.n_rows:
            monitor_status = False
            reason = "Monotonic time index (Should be ignored – will cause false drift)"
            best_guess = "ignore"
        
        best_guess = None
        if monitor_status == "Review":
            if sig.is_float or sig.is_integer:
                best_guess = "continuous"
            elif sig.is_text and sig.n_unique <= 50:
                best_guess = "categorical"
            else:
                best_guess = "ignore"

        profiles.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "cardinality_ratio": round(sig.cardinality_ratio, 4),
            "dominant_ratio": round(sig.dominant_ratio, 4),
            "is_datetime": sig.is_datetime,
            "is_numeric": sig.is_float or sig.is_integer,
            "monitor": monitor_status,
            "reason": reason,
            "best_guess": best_guess
        })

    return profiles

# ---------------------------------------------------------
# Unified table profiling (docs/unified_table_plan.md, Phase 1)
# ---------------------------------------------------------
# profile_table() proposes one of five types per column, with a monitor
# recommendation, confidence, evidence and reason. Numeric/categorical/ignore
# proposals come straight from profile_columns() above (unchanged), so the
# table path classifies those columns exactly like the existing tabular
# path; string columns additionally get image / URL / identifier / date /
# free-text detection. `confidence` is a heuristic agreement score in [0, 1],
# not a calibrated probability. The human confirms or edits every proposal.
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_HEX_ID_RE = re.compile(r"^(?:[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}|[0-9a-f]{16,})$",
                        re.IGNORECASE)
_SAMPLE_SIZE = 2000


def _proposal(ptype: str, monitor: bool, confidence: float, reason: str, alternative: Optional[str] = None):
    return {"proposed_type": ptype, "proposed_monitor": monitor, "confidence": round(float(confidence), 2),
            "reason": reason, "alternative_type": alternative}


def _string_evidence(strs: pd.Series) -> Dict[str, float]:
    lengths = strs.str.len()
    tokens = strs.str.split().str.len()
    non_space = strs.str.replace(r"\s", "", regex=True)
    letters = non_space.str.count(r"[^\W\d_]")
    total_chars = non_space.str.len().sum()
    return {
        "mean_length": round(float(lengths.mean()), 1),
        "mean_tokens": round(float(tokens.mean()), 2),
        "alpha_ratio": round(float(letters.sum() / total_chars), 3) if total_chars else 0.0,
    }


def _string_column_proposal(strs: pd.Series, ev: Dict[str, Any], archive) -> Optional[Dict[str, Any]]:
    """Proposal for a non-numeric string column, or None to fall back to profile_columns()."""
    from ingest.images import IMAGE_EXTENSIONS, decode_inline  # local: keeps profiler importable on its own

    head = strs.head(50)
    inline_ratio = float(head.map(lambda v: decode_inline(v) is not None).mean())
    if inline_ratio >= 0.8:
        return _proposal("image", True, inline_ratio, "Values are embedded base64/data-URI images.")

    ext_ratio = float(strs.str.lower().str.strip().str.endswith(IMAGE_EXTENSIONS).mean())
    if archive is not None:
        match_ratio = float(strs.map(lambda v: archive.resolve(v)[0] == "ok").mean())
        ev["image_match_ratio"] = round(match_ratio, 3)
        if match_ratio >= 0.5:
            return _proposal("image", True, match_ratio,
                             f"{match_ratio:.0%} of values match files in the image ZIP.")
        if ext_ratio >= 0.8:
            return _proposal("image", False, 0.5,
                             f"Values look like image filenames but only {match_ratio:.0%} match files in the "
                             f"image ZIP -- check the ZIP before monitoring this column.")
    elif ext_ratio >= 0.8:
        return _proposal("image", False, 0.6,
                         "Values look like image filenames, but no image ZIP was uploaded -- upload one to "
                         "monitor this column.")

    if float(strs.str.match(_URL_RE).mean()) >= 0.8:
        return _proposal("ignore", False, 0.95, "Remote URLs -- fetching them is not supported.")

    if (ev["mean_tokens"] <= 1.2 and ev["unique_ratio"] >= 0.95) or float(strs.str.match(_HEX_ID_RE).mean()) >= 0.8:
        return _proposal("ignore", False, 0.95, "Identifier-like: unique single-token values.")

    has_digits = float(strs.str.contains(r"\d").mean())
    if has_digits >= 0.9 and ev["mean_tokens"] <= 3:
        sample = strs.head(200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if float(parsed.notna().mean()) >= 0.9:
            return _proposal("ignore", False, 0.9, "Date/time values (always drift trivially over time).")

    criteria = [ev["mean_tokens"] >= 3, ev["mean_length"] >= 15, ev["alpha_ratio"] >= 0.5,
                ev["unique_ratio"] >= 0.5 or ev["n_unique"] > 50]
    if ev["mean_tokens"] >= 3 and sum(criteria) >= 3:
        if ev["n_unique"] <= 50:
            if ev["mean_tokens"] >= 5 and ev["mean_length"] >= 40:
                return _proposal("text", True, 0.6,
                                 "Repeated multi-word sentences -- could also be categorical templates; review.",
                                 alternative="categorical")
            return None
        return _proposal("text", True, 0.6 + 0.1 * sum(criteria),
                         "High uniqueness + multi-word strings + substantial average length.")
    return None


def _proposal_from_profile(p: Dict[str, Any], ev: Dict[str, Any]) -> Dict[str, Any]:
    status, reason = p["monitor"], p["reason"]
    if status is True:
        return _proposal("numeric", True, ev.get("numeric_fraction", 1.0),
                         f"Successful numeric coercion ({ev.get('numeric_fraction', 1.0):.0%} of values). {reason}")
    if status == "Categorical":
        return _proposal("categorical", True, 0.95 if ev["n_unique"] <= 20 else 0.8,
                         f"Repeated limited-cardinality values. {reason}")
    if status == "Review":
        guess = {"continuous": "numeric", "categorical": "categorical"}.get(p["best_guess"], "ignore")
        return _proposal(guess, guess != "ignore", 0.5, f"{reason} Best guess: {guess}.")
    return _proposal("ignore", False, 0.95, reason)


def profile_table(df: pd.DataFrame, archive=None) -> List[Dict[str, Any]]:
    """One proposal per column: name, proposed_type, proposed_monitor,
    confidence, reason, alternative_type, evidence (aggregates only, safe to
    persist) and sample_values (up to 5 truncated values -- shown to the
    owner during review, never persisted). `archive` is an optional
    ingest.images.ImageArchive used to recognize filename columns."""
    n_rows = len(df)
    base = {p["name"]: p for p in profile_columns(df)}
    proposals = []
    for col in df.columns:
        p = base[col]
        non_null = df[col].dropna()
        n_unique = int(non_null.astype(str).nunique()) if len(non_null) else 0
        ev: Dict[str, Any] = {
            "n_rows": n_rows,
            "null_rate": round(1 - len(non_null) / n_rows, 4) if n_rows else 1.0,
            "n_unique": n_unique,
            "unique_ratio": round(n_unique / len(non_null), 4) if len(non_null) else 0.0,
            "dominant_ratio": p["dominant_ratio"],
        }
        proposal = None
        if len(non_null):
            sample = non_null.sample(min(_SAMPLE_SIZE, len(non_null)), random_state=42)
            ev["numeric_fraction"] = round(float(pd.to_numeric(sample, errors="coerce").notna().mean()), 3)
            if not p["is_numeric"] and not p["is_datetime"] and ev["numeric_fraction"] <= NUMERIC_COERCION_THRESHOLD:
                strs = sample.astype(str)
                ev.update(_string_evidence(strs))
                proposal = _string_column_proposal(strs, ev, archive)
        if proposal is None:
            proposal = _proposal_from_profile(p, ev)
        samples = [str(v)[:80] for v in pd.unique(non_null.astype(str))[:5]] if len(non_null) else []
        proposals.append({"name": col, **proposal, "evidence": ev, "sample_values": samples})
    return proposals
