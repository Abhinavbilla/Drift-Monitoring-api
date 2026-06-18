import pandas as pd
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


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
        if coerced.notna().mean() > 0.80:
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