from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class FitBaselineRequest(BaseModel):
    reference_data: Dict[str, List[Any]]                        # continuous columns
    categorical_data: Optional[Dict[str, List[Any]]] = {}       # categorical columns

class FitBaselineResponse(BaseModel):
    status: str
    message: str
    inferred_feature_types: Dict[str, str] = Field(
        description="Shows whether the engine classified each feature as 'continuous' or 'categorical'."
    )






class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ..., 
        description="A single incoming production data point. Keys are feature names, values are the raw data."
    )

class PredictResponse(BaseModel):
    is_anomaly: bool = Field(description="True if the data point breached the IQR fences.")
    anomaly_score: float = Field(description="The maximum deviation score across all features.")
    feature_deviations: Dict[str, float] = Field(
        description="Detailed breakdown of how far each feature deviated from its normal bounds."
    )






class AnalyzeBatchRequest(BaseModel):
    production_data: Dict[str, List[Any]] = Field(
        ..., 
        description="A recent batch of production data to compare against the baseline."
    )

class FeatureDriftMetric(BaseModel):
    statistic: float = Field(description="The KS statistic or TVD distance.")
    p_value: Optional[float] = Field(description="P-value for continuous tests. Null for categorical TVD.")
    drift_detected: bool = Field(description="True if statistical drift was confirmed.")

class AnalyzeBatchResponse(BaseModel):
    system_alert_triggered: bool = Field(description="True if ANY feature in the batch is drifting.")
    feature_metrics: Dict[str, FeatureDriftMetric] = Field(
        description="Detailed drift metrics for every feature evaluated."
    )
    
    
    
    
    
    

class HealthCheckResponse(BaseModel):
    system_status: str = Field(description="Overall health status of the monitored project ('Healthy' or 'Degraded').")
    is_burst_alert: bool = Field(description="True if the recent anomaly rate exceeds the safety threshold.")
    drift_ratio: float = Field(description="The exact percentage of recent requests flagged as anomalous (0.0 to 1.0).")