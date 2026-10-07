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
  relationship_metrics?: Record<string, FeatureMetric>;
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

// -- unified table path --
export type TableColumnType = "numeric" | "categorical" | "text" | "image" | "ignore";

export interface JobAccepted {
  job_id: string;
  stage_id: string | null;
  expires_at: string | null;
  replay: boolean;
}

export interface Job<R> {
  job_id: string;
  kind: "profile" | "fit" | "analyze";
  status: "queued" | "running" | "succeeded" | "failed" | "interrupted";
  progress: number;
  progress_message: string | null;
  result: R | null;
  error: string | null;
}

export interface ColumnProposal {
  name: string;
  proposed_type: TableColumnType;
  proposed_monitor: boolean;
  confidence: number;
  reason: string;
  alternative_type: TableColumnType | null;
  evidence: Record<string, number>;
  sample_values: string[];
}

export interface RelationshipProposal {
  col_a: string;
  col_b: string;
  kind: "num_num" | "cat_cat" | "num_cat" | "probe" | "text_image";
  strength: number;
  proposed: boolean;
  reason: string;
}

export interface TableRelationshipChoice {
  col_a: string;
  col_b: string;
  monitor: boolean;
}

export interface TableProfile {
  stage_id: string;
  n_rows: number;
  columns: ColumnProposal[];
  relationships: RelationshipProposal[];
  image_zip: { image_entries: number; rejected: Record<string, number> } | null;
}

export interface TableColumnChoice {
  name: string;
  type: TableColumnType;
  monitor: boolean;
}

export interface TableFitResult {
  version: number;
  message: string;
  monitored: Record<string, string[]>;
  not_monitored: string[];
  duplicate_rows_dropped: number;
  data_quality: Record<string, { valid: number; invalid: Record<string, number> }>;
}

export interface ColumnDrift {
  type: TableColumnType;
  test: string;
  status: "DRIFT" | "STABLE" | "NOT_TESTED";
  statistic?: number;
  p_value?: number | null;
  p_value_adjusted?: number | null;
  effect_floor?: number | null;
  in_family: boolean;
  threshold_used?: string | null;
  reason?: string;
}

export interface RelationshipDrift {
  kind: string;
  status: "DRIFT" | "STABLE" | "NOT_TESTED";
  statistic_name: string;
  reference_value: number | null;
  current_value: number | null;
  explanation?: string;
  p_value?: number | null;
  p_value_adjusted?: number | null;
  in_family: boolean;
  report_only?: boolean;
  confounded_by?: string[];
  reason?: string;
}

export interface TableReport {
  baseline_version: number;
  n_rows: number;
  overall: { status: "DRIFT" | "STABLE" | "DATA_ISSUES"; alert: boolean; alert_state: string | null; triggered_by: string[] };
  column_drift: Record<string, ColumnDrift>;
  relationship_drift: Record<string, RelationshipDrift>;
  screening?: { emerged_dependencies: { col_a: string; col_b: string; reference: number; current: number }[] };
  schema_report: Record<string, { issue: string; severity: string }[]>;
  data_quality: Record<string, { valid: number; invalid: Record<string, number> }>;
  family: { method: string; members: string[]; excluded: { test: string; why: string }[] };
}
