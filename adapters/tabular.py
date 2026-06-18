from typing import Dict, Any

class TabularAdapter:
    def __init__(self, feature_configs: Dict[str, Dict[str, str]] = None):
        """
        Holds configurations for incoming production data.
        Since the detector engine is now fully non-parametric (IQR + Categorical Sets), 
        we no longer need heavy mathematical transformations here.
        """
        self.feature_configs = feature_configs or {}

    # Notice the return type is now Dict[str, Any] instead of Dict[str, float]
    def clean_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a raw incoming data point, handles missing values, 
        and safely passes strings while casting numericals to floats.
        """
        cleaned_data = {}
        
        for feature, value in raw_data.items():
            # Drop missing values entirely so they don't break the detector's math
            if value is None:
                continue
                
            # If the value is a string (categorical), pass it through safely
            if isinstance(value, str):
                cleaned_data[feature] = value
            # Otherwise, it's a number, so strictly cast it to float
            else:
                cleaned_data[feature] = float(value)
                
        return cleaned_data