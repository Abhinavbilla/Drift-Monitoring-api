export type Modality = "tabular" | "text" | "image" | "joint";

export interface FeatureMetric {
  statistic: number;
  p_value: number | null;
  drift_detected: boolean;
  effect_size?: number | null;
  effect_floor?: number | null;
  p_value_adjusted?: number | null;
  significant?: boolean | null;
  material?: boolean | null;
  decision_mode?: string | null;
  threshold_used?: string | null;
}

export interface SchemaIssue {
  issue: string;
  severity: "alert" | "warn" | "ignore";
  [key: string]: unknown;
}

export interface AnalyzeResponse {
  system_alert_triggered: boolean;
  feature_metrics: Record<string, FeatureMetric>;
  schema_report: Record<string, SchemaIssue[]>;
  alert: boolean | null;
  sustained_alert: boolean | null;
  windows_considered: number | null;
  alert_state: "ok" | "open" | null;
  transition: "opened" | "resolved" | "still_open" | null;
}

export interface FeatureCalibrationInfo {
  minimum_detectable_d?: number | null;
  effect_floor?: number | null;
  reference_too_small_for_floor?: boolean | null;
  recommended_batch_size?: number | null;
  min_batch_size_at_floor?: number | null;
  floor_below_dkw_bound?: boolean | null;
  minimum_reference_size_for_dkw_safe_floor?: number | null;
}

export interface FitResponse {
  status: string;
  message: string;
  inferred_feature_types?: Record<string, string>;
  cleaning_summary?: Record<string, { dropped_non_numeric: number }>;
  calibration_info?: Record<string, FeatureCalibrationInfo>;
  version?: number;
}

export interface BaselineVersion {
  version: number;
  created_at: string;
  model_version_label: string | null;
  modality: string;
  active: boolean;
}

export interface HistoryRun {
  id: number;
  ts: string;
  baseline_version: number | null;
  batch_size: number | null;
  decision_mode: string | null;
  system_alert: boolean;
  sustained_alert: boolean | null;
  feature_metrics: Record<string, FeatureMetric>;
  schema_report: Record<string, SchemaIssue[]> | null;
}

export interface HistoryResponse {
  project_id: string;
  total: number;
  limit: number;
  offset: number;
  runs: HistoryRun[];
  feature_time_series: Record<
    string,
    { ts: string; statistic: number; effect_size: number | null; p_value_adjusted: number | null;
      significant: boolean | null; material: boolean | null }[]
  >;
}

export interface Webhook {
  id: string;
  project_id: string;
  url: string;
  event_filter: string[];
  enabled: boolean;
  created_at: string;
  secret?: string | null;
}

export interface PredictResponse {
  is_anomaly: boolean;
  anomaly_score: number;
  feature_deviations: Record<string, unknown>;
}

export interface HealthResponse {
  system_status: string;
  is_burst_alert: boolean;
  drift_ratio: number;
}

export interface LogEntry {
  input_data: Record<string, unknown>;
  score: number;
  is_ood: boolean;
}

export interface BaselineInfo {
  fences: { feature_name: string; type: string; q1?: number; q3?: number; allowed_values?: string[] }[];
  feature_types: Record<string, string>;
  modality: Modality;
}
