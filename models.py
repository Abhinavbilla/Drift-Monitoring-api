from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class FitBaselineRequest(BaseModel):
    reference_data: Dict[str, List[Any]]                        # continuous columns
    categorical_data: Optional[Dict[str, List[Any]]] = {}       # categorical columns
    calibration_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Step 2 two-gate decision config (decision_mode, alpha, multiple_testing, "
                    "effect_floors, per_feature_effect_floors, psi_null_draws). Omit to leave the "
                    "project's existing config untouched (or default to legacy for a brand-new "
                    "project) -- this is never required. See drift.calibration.CalibrationConfig."
    )
    feature_types: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional explicit classification ({column: 'continuous'|'categorical'}), "
                    "overriding the server-side profiler for the columns named -- e.g. a "
                    "low-cardinality integer column you specifically want monitored as continuous. "
                    "Columns not named here are still profiled normally. Every named column must "
                    "appear in reference_data or categorical_data (either works, regardless of "
                    "which one matches the type given here); an unrecognized value or an unknown "
                    "column name is a 422."
    )
    schema_policy: Optional[Dict[str, str]] = Field(
        default=None,
        description="Step 5 item 4: per-project severity policy for /analyze's schema_report. Keys: "
                    "'missing_columns' and 'default' (every other issue type); values: 'alert', "
                    "'warn', or 'ignore'. Omit to leave the project's existing policy untouched (or "
                    "default to {missing_columns: alert, default: warn} for a brand-new project) -- "
                    "never required."
    )
    model_version_label: Optional[str] = Field(
        default=None,
        description="Step 5 item 5: an optional human-readable label (e.g. 'v2-retrained-on-march-data') "
                    "stored on the baseline version this /fit call creates. Purely descriptive -- never "
                    "required, never interpreted by the server."
    )

class ActivateBaselineVersionRequest(BaseModel):
    version: int = Field(description="The existing baseline version number to make active.")


class FeatureCalibrationInfo(BaseModel):
    minimum_detectable_d: Optional[float] = Field(
        default=None,
        description="The best-case asymptotic KS critical-value floor this reference size could ever "
                    "support (as the future batch size -> infinity), at the project's alpha. Continuous "
                    "features only (None for categorical, which uses PSI, not KS)."
    )
    effect_floor: Optional[float] = Field(
        default=None,
        description="The configured materiality floor for this feature (ks_d or psi, whichever applies)."
    )
    reference_too_small_for_floor: Optional[bool] = Field(
        default=None,
        description="True if minimum_detectable_d > effect_floor for a continuous feature -- i.e. this "
                    "reference size cannot reliably resolve an effect as small as the configured floor, "
                    "no matter how large future production batches are. A prompt to enlarge the "
                    "reference or raise the floor, not an error. False/None otherwise."
    )
    recommended_batch_size: Optional[int] = Field(
        default=None,
        description="REDEFINED 2026-09-30: the smallest /analyze batch size (n) at which this "
                    "reference's asymptotic KS critical value reaches HALF the configured floor: the "
                    "smallest n with c(alpha)*sqrt((n+m)/(n*m)) <= floor/2. Simulation-verified effect "
                    "(results/step2_hardening_batch_size_simulation.md): this does NOT meaningfully "
                    "improve power at true D=floor (already ~99% at the smaller min_batch_size_at_floor "
                    "n, for both reference sizes tested) -- its real benefit is a much lower "
                    "false-material rate for a batch at or below the floor (simulated null, m=5,000: "
                    "5.45%->0%; true D=0.02, m=5,000: 26.15%->0.60%), at the cost of substantially more "
                    "data. None if reference_too_small_for_floor is True. Continuous features only. "
                    "Asymptotic approximation for continuous data -- not exact for small n or "
                    "discrete/near-discrete features."
    )
    min_batch_size_at_floor: Optional[int] = Field(
        default=None,
        description="The OLD recommended_batch_size definition (kept, unchanged, under this name): "
                    "the smallest n with c(alpha)*sqrt((n+m)/(n*m)) <= floor (not floor/2). At exactly "
                    "this n, the critical value equals the floor -- since that critical value is an "
                    "alpha-level null-rejection threshold, a batch with NO true drift still has "
                    "~alpha (5%) probability of exceeding the floor from sampling noise alone "
                    "(simulation-confirmed: 5.45%/4.65% for m=5,000/50,000). See recommended_batch_size "
                    "for the more conservative figure. None if reference_too_small_for_floor is True. "
                    "Continuous features only. Also an asymptotic approximation for continuous data."
    )

class FitBaselineResponse(BaseModel):
    status: str
    message: str
    inferred_feature_types: Dict[str, str] = Field(
        description="Shows whether the engine classified each feature as 'continuous' or 'categorical'."
    )
    cleaning_summary: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Per-column counts of values dropped during ingestion (e.g. non-numeric cells in a continuous column), so silent data loss is visible rather than hidden."
    )
    calibration_info: Dict[str, FeatureCalibrationInfo] = Field(
        default_factory=dict,
        description="Per-feature minimum-detectable-D and configured effect floor, shown regardless "
                    "of decision_mode so a caller can see what this reference size can and cannot "
                    "detect before choosing floors. API field only -- no dashboard UI for this."
    )
    version: int = Field(
        description="Step 5 item 5: the baseline version this /fit call just created. Old versions "
                    "are kept, not overwritten -- see GET /baselines/{project_id}."
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
    # Step 2 additions -- all Optional/default None, so a legacy-mode
    # response (the only kind that existed before Step 2) is unaffected in
    # substance: these simply serialize as null. Populated only when the
    # project's decision_mode is "calibrated" (drift/detector.py).
    effect_size: Optional[float] = Field(default=None, description="Same value as statistic; named for clarity in calibrated mode.")
    effect_floor: Optional[float] = Field(default=None, description="The configured materiality floor for this feature.")
    p_value_adjusted: Optional[float] = Field(default=None, description="p_value after multiple-testing correction across this batch's features.")
    significant: Optional[bool] = Field(default=None, description="Gate 1: p_value_adjusted < alpha.")
    material: Optional[bool] = Field(default=None, description="Gate 2: effect_size >= effect_floor.")
    decision_mode: Optional[str] = Field(default=None, description="'legacy' or 'calibrated' for this analysis.")
    threshold_used: Optional[str] = Field(default=None, description="Human-readable description of the exact decision rule applied.")

class AnalyzeBatchResponse(BaseModel):
    system_alert_triggered: bool = Field(description="True if ANY feature in the batch is drifting.")
    feature_metrics: Dict[str, FeatureDriftMetric] = Field(
        description="Detailed drift metrics for every feature evaluated."
    )
    schema_report: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Step 5 item 4 (tabular /analyze and /analyze/upload only): per-column schema "
                    "issues found in this batch -- missing_column, unexpected_column, dtype_change, "
                    "null_rate, unseen_categories, constant_column -- each with a severity "
                    "(alert/warn/ignore) resolved from the project's schema_policy. Empty for the "
                    "text/image/joint analyze endpoints, which have no column-level schema concept."
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