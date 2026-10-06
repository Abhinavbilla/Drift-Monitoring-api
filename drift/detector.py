import logging
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional

from drift.calibration import CalibrationConfig, adjust_p_values, apply_two_gate, psi_bootstrap_pvalue

logger = logging.getLogger(__name__)

def compute_iqr_anomalies(input_data: dict, baselines: list) -> tuple:
    """
    Evaluates a single incoming data point for OOD anomalies.
    Uses IQR for continuous numbers, and Allowed Sets for categorical strings.
    """
    results = {}
    is_ood = 0
    
    for row in baselines:
        feature = row["feature_name"]
        
        if feature in input_data:
            value = input_data[feature]
            deviation = 0
            
            # --- THE NEW ROUTER LOGIC ---
            # Check if this feature was saved as categorical
            if row.get("type") == "categorical":
                allowed_values = row.get("allowed_values", [])
                
                # If the incoming string is NOT in the allowed list, it's an anomaly!
                if value not in allowed_values:
                    deviation = 1.0  # Represents a 100% categorical failure
            
            # Otherwise, process it using the standard IQR math
            else:
                q1 = row.get("q1", 0) 
                q3 = row.get("q3", 0) 
                
                iqr = q3 - q1
                lower_bound = q1 - (3.0 * iqr)
                upper_bound = q3 + (3.0 * iqr)
                
                if value < lower_bound:
                    deviation = (lower_bound - value) / iqr if iqr != 0 else 1.0
                elif value > upper_bound:
                    deviation = (value - upper_bound) / iqr if iqr != 0 else 1.0
                    
            results[feature] = deviation
            
            if deviation > 0:
                is_ood = 1
                
    score = max(results.values()) if results else 0
    return score, is_ood, results


class DistributionDetector:
    def __init__(self, p_value_threshold: float = 0.05, calibration_config: Optional[CalibrationConfig] = None):
        # Setting a standard threshold.
        # If the p-value dips below this, we sound the alarm.
        self.p_value_threshold = p_value_threshold
        self.reference_data: Dict[str, np.ndarray] = {}
        self.feature_types: Dict[str, str] = {}
        # None (every existing caller) -> legacy behavior, byte-identical to
        # before this parameter existed. Only main.py's calibrated-mode path
        # passes an actual CalibrationConfig; see analyze_production_window.
        self.calibration_config = calibration_config

    def fit_baseline(self, reference_features: Dict[str, List[Any]], feature_types: Dict[str, str]) -> None:
        """
        Feed in the clean, idealized dataset here. This locks in our 'ground truth'.
        """
        self.feature_types = feature_types
        for feature_name, data in reference_features.items():
            self.reference_data[feature_name] = np.array(data)

    def _check_continuous_drift(self, ref_data: np.ndarray, prod_data: np.ndarray) -> Dict[str, Any]:
        """
        Run a Kolmogorov-Smirnov test. It's solid for figuring out if the 
        overall shape of our continuous data has shifted over time.
        """
        statistic, p_value = stats.ks_2samp(ref_data, prod_data)
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < self.p_value_threshold)
        }
    def _check_categorical_drift(self, ref_data: np.ndarray, prod_data: np.ndarray) -> Dict[str, Any]:
        """
        Checks if the frequency of categories has shifted using Population Stability Index (PSI).
        """
        ref_elements, ref_counts = np.unique(ref_data, return_counts=True)
        prod_elements, prod_counts = np.unique(prod_data, return_counts=True)
        
        # FIX: Convert keys to strings to match stored baseline format.
        # Also normalize floats like 3.0 -> "3" instead of "3.0" so integer-valued
        # categories stored as strings always match regardless of dtype drift
        # between how reference_data and live production payloads arrive.
        logger.debug("ref_elements: %s", ref_elements)
        logger.debug("prod_elements: %s", prod_elements)
        logger.debug("ref_counts: %s", ref_counts)
        logger.debug("prod_counts: %s", prod_counts)

        def _normalize_key(k):
            if isinstance(k, float) and k.is_integer():
                return str(int(k))
            return str(k)

        ref_freq = {_normalize_key(k): v for k, v in zip(ref_elements, ref_counts / len(ref_data))}
        prod_freq = {_normalize_key(k): v for k, v in zip(prod_elements, prod_counts / len(prod_data))}
        
        psi_score = 0.0
        epsilon = 0.0001
        
        all_categories = set(ref_freq.keys()).union(set(prod_freq.keys()))
        
        for category in all_categories:
            expected_pct = ref_freq.get(category, 0.0)
            if expected_pct == 0.0:
                expected_pct = epsilon
                
            actual_pct = prod_freq.get(category, 0.0)
            if actual_pct == 0.0:
                actual_pct = epsilon
                
            psi_score += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        
        return {
            "statistic": float(psi_score),
            "p_value": None,
            "drift_detected": bool(psi_score > 0.2)
        }

    @staticmethod
    def _reference_frequencies(ref_data: np.ndarray) -> Dict[str, float]:
        """Same category-key normalization as _check_categorical_drift (kept
        as a separate helper rather than refactoring that method, so the
        legacy PSI code path is provably untouched). Used only by the
        calibrated-mode PSI bootstrap below."""
        elements, counts = np.unique(ref_data, return_counts=True)

        def _normalize_key(k):
            if isinstance(k, float) and k.is_integer():
                return str(int(k))
            return str(k)

        return {_normalize_key(k): float(v) for k, v in zip(elements, counts / len(ref_data))}

    def analyze_production_window(self, production_features: Dict[str, List[Any]],
                                  extra_tests: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Scan a fresh batch of production data to see if the model is going off the rails.

        Legacy mode (self.calibration_config is None, or decision_mode ==
        "legacy"): byte-identical to the pre-Step-2 behavior below -- single
        per-feature threshold, no correction, no materiality gate.

        Calibrated mode: raw statistics are computed with the EXACT SAME
        _check_continuous_drift/_check_categorical_drift calls (so D/PSI
        values never differ between modes for the same data), then
        Holm/BH-corrected across every feature in this batch and passed
        through the two-gate (significance + materiality) decision.

        extra_tests (table projects: relationship tests) are dicts with name,
        p_value, effect_size, effect_floor, label. In calibrated mode they
        join the SAME correction family as the features; in legacy mode
        there is no family, so each is decided on its raw p. Results come
        back under "extra_metrics". Without extra_tests nothing changes.
        """
        extra_tests = extra_tests or []
        cfg = self.calibration_config
        is_calibrated = cfg is not None and cfg.decision_mode == "calibrated"

        raw_results: Dict[str, Dict[str, Any]] = {}
        feature_kind: Dict[str, str] = {}

        for feature_name, prod_data_list in production_features.items():
            if feature_name not in self.reference_data:
                continue

            ref_data = self.reference_data[feature_name]
            prod_data = np.array(prod_data_list)
            f_type = self.feature_types.get(feature_name, 'continuous')
            feature_kind[feature_name] = f_type

            if f_type == 'continuous':
                raw_results[feature_name] = self._check_continuous_drift(ref_data, prod_data)
            else:
                raw_results[feature_name] = self._check_categorical_drift(ref_data, prod_data)

        if not is_calibrated:
            # --- LEGACY: exactly the pre-Step-2 aggregation, untouched ---
            drift_report = {}
            system_alert = False
            for feature_name, result in raw_results.items():
                if result["drift_detected"]:
                    system_alert = True
                drift_report[feature_name] = result
            extra = {t["name"]: self._gate_extra(t, t["p_value"], self.p_value_threshold, "none") for t in extra_tests}
            report = {
                "system_alert_triggered": system_alert or any(m["drift_detected"] for m in extra.values()),
                "feature_metrics": drift_report,
            }
            if extra_tests:
                report["extra_metrics"] = extra
            return report

        # --- CALIBRATED: two-gate decision with family-wise correction ---
        # Categorical features need an actual p-value (legacy PSI has none)
        # so they can join the same corrected family as the continuous ones.
        p_values: List[float] = []
        feature_order: List[str] = []
        for feature_name, result in raw_results.items():
            if feature_kind[feature_name] == 'continuous':
                p = result["p_value"]
            else:
                ref_freq = self._reference_frequencies(self.reference_data[feature_name])
                batch_size = len(production_features[feature_name])
                p = psi_bootstrap_pvalue(
                    ref_freq, result["statistic"], batch_size,
                    null_draws=cfg.psi_null_draws, seed=42,
                )
            p_values.append(p)
            feature_order.append(feature_name)

        adjusted_all = adjust_p_values(p_values + [t["p_value"] for t in extra_tests], cfg.multiple_testing)
        adjusted, extra_adjusted = adjusted_all[:len(p_values)], adjusted_all[len(p_values):]
        extra = {t["name"]: self._gate_extra(t, p_adj, cfg.alpha, cfg.multiple_testing)
                 for t, p_adj in zip(extra_tests, extra_adjusted)}

        drift_report = {}
        system_alert = False
        for feature_name, p_raw, p_adj in zip(feature_order, p_values, adjusted):
            result = raw_results[feature_name]
            f_type = feature_kind[feature_name]
            floor_type = "ks_d" if f_type == "continuous" else "psi"
            floor = cfg.effect_floor_for(feature_name, floor_type)
            threshold_label = ("KS" if f_type == "continuous" else "PSI-bootstrap") + f" ({cfg.multiple_testing})"

            gate = apply_two_gate(
                effect_size=result["statistic"],
                effect_floor=floor,
                p_value=p_raw,
                p_value_adjusted=p_adj,
                alpha=cfg.alpha,
                decision_mode="calibrated",
                threshold_used=f"p_adj<{cfg.alpha} [{threshold_label}], effect>={floor}",
            )

            entry = dict(result)  # keep "statistic" as-is; "p_value"/"drift_detected" overwritten below
            entry.update({
                "p_value": gate.p_value,  # categorical: legacy None replaced with the real bootstrap p-value
                "effect_size": gate.effect_size,
                "effect_floor": gate.effect_floor,
                "p_value_adjusted": gate.p_value_adjusted,
                "significant": gate.significant,
                "material": gate.material,
                "decision_mode": gate.decision_mode,
                "threshold_used": gate.threshold_used,
                "drift_detected": gate.drift_detected,  # calibrated semantics: significant AND material
            })
            drift_report[feature_name] = entry
            if gate.drift_detected:
                system_alert = True

        report = {
            "system_alert_triggered": system_alert or any(m["drift_detected"] for m in extra.values()),
            "feature_metrics": drift_report,
        }
        if extra_tests:
            report["extra_metrics"] = extra
        return report

    @staticmethod
    def _gate_extra(test: Dict[str, Any], p_adj: float, alpha: float, method: str) -> Dict[str, Any]:
        gate = apply_two_gate(
            effect_size=test["effect_size"], effect_floor=test["effect_floor"], p_value=test["p_value"],
            p_value_adjusted=p_adj, alpha=alpha, decision_mode="calibrated",
            threshold_used=f"p_adj<{alpha} [{test['label']} ({method})], effect>={test['effect_floor']}",
        )
        return {"statistic": test["effect_size"], "p_value": gate.p_value, "drift_detected": gate.drift_detected,
                "effect_size": gate.effect_size, "effect_floor": gate.effect_floor,
                "p_value_adjusted": gate.p_value_adjusted, "significant": gate.significant,
                "material": gate.material, "decision_mode": gate.decision_mode,
                "threshold_used": gate.threshold_used}