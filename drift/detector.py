import numpy as np
from scipy import stats
from typing import Dict, Any, List

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
    def __init__(self, p_value_threshold: float = 0.05):
        # Setting a standard threshold. 
        # If the p-value dips below this, we sound the alarm.
        self.p_value_threshold = p_value_threshold
        self.reference_data: Dict[str, np.ndarray] = {}
        self.feature_types: Dict[str, str] = {} 

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
         # ========== DEBUG: Add this ==========
        print(f"[DEBUG] ref_elements: {ref_elements}")
        print(f"[DEBUG] prod_elements: {prod_elements}")
        print(f"[DEBUG] ref_counts: {ref_counts}")
        print(f"[DEBUG] prod_counts: {prod_counts}")
    # ======================================
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
    def analyze_production_window(self, production_features: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Scan a fresh batch of production data to see if the model is going off the rails.
        """
        drift_report = {}
        system_alert = False

        for feature_name, prod_data_list in production_features.items():
            if feature_name not in self.reference_data:
                continue

            ref_data = self.reference_data[feature_name]
            prod_data = np.array(prod_data_list)
            f_type = self.feature_types.get(feature_name, 'continuous')

            if f_type == 'continuous':
                result = self._check_continuous_drift(ref_data, prod_data)
            else:
                result = self._check_categorical_drift(ref_data, prod_data)

            if result["drift_detected"]:
                system_alert = True

            drift_report[feature_name] = result

        return {
            "system_alert_triggered": system_alert,
            "feature_metrics": drift_report
        }