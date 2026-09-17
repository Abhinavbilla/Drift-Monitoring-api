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




# ---------------------------------------------------------
# TEXT / IMAGE (EMBEDDING-BASED) DRIFT MONITORING — v2.0
# ---------------------------------------------------------

class FitTextBaselineRequest(BaseModel):
    reference_texts: List[str] = Field(
        ..., description="Baseline batch of raw strings to lock as the reference distribution."
    )

class AnalyzeTextBatchRequest(BaseModel):
    production_texts: List[str] = Field(
        ..., description="Recent batch of raw strings to compare against the text baseline."
    )

class FitImageBaselineRequest(BaseModel):
    reference_images: List[str] = Field(
        ..., description="Baseline batch of base64-encoded images to lock as the reference distribution."
    )

class AnalyzeImageBatchRequest(BaseModel):
    production_images: List[str] = Field(
        ..., description="Recent batch of base64-encoded images to compare against the image baseline."
    )

class EmbeddingFitResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------
# JOINT MULTIMODAL CONTEXT DRIFT MONITORING
# ---------------------------------------------------------

class JointRecord(BaseModel):
    tabular: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured fields for this record, if present."
    )
    text: Optional[str] = Field(default=None, description="Associated text, if present.")
    image: Optional[str] = Field(default=None, description="Associated base64-encoded image, if present.")

class FitJointBaselineRequest(BaseModel):
    reference_records: List[JointRecord] = Field(
        ..., description="Baseline batch of joint records (each with any subset of tabular/text/image) to lock as the reference distribution."
    )

class AnalyzeJointBatchRequest(BaseModel):
    production_records: List[JointRecord] = Field(
        ..., description="Recent batch of joint records to compare against the joint baseline."
    )