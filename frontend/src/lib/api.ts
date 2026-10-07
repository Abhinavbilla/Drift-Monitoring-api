import type {
  AnalyzeResponse,
  BaselineInfo,
  BaselineVersion,
  FitResponse,
  HealthResponse,
  HistoryResponse,
  Job,
  JobAccepted,
  LogEntry,
  TableBaselineInfo,
  TableColumnChoice,
  TableRelationshipChoice,
  PredictResponse,
  Webhook,
} from "./types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(`[${status}] ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

let getToken: () => string | null = () => null;
export function registerTokenGetter(fn: () => string | null) {
  getToken = fn;
}

async function request<T>(
  path: string,
  options: { method?: string; body?: unknown; params?: Record<string, string | number | boolean | undefined>;
             headers?: Record<string, string>; isForm?: boolean } = {},
): Promise<T> {
  const { method = "GET", body, params, headers = {}, isForm = false } = options;
  // Concatenate rather than resolve: new URL("/x", "https://host/api") would drop "/api".
  // window.location.origin makes a relative base like "/api" (Docker/nginx) work too.
  const url = new URL(BASE_URL.replace(/\/$/, "") + path, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) url.searchParams.set(k, String(v));
    }
  }

  const token = getToken();
  const finalHeaders: Record<string, string> = { ...headers };
  if (token) finalHeaders["Authorization"] = `Bearer ${token}`;

  let payload: BodyInit | undefined;
  if (body !== undefined) {
    if (isForm) {
      payload = body as FormData;
    } else {
      finalHeaders["Content-Type"] = "application/json";
      payload = JSON.stringify(body);
    }
  }

  const resp = await fetch(url.toString(), { method, headers: finalHeaders, body: payload });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const data = await resp.json();
      if (data?.detail) detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* non-JSON error body, keep statusText */
    }
    throw new ApiError(resp.status, detail);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json() as Promise<T>;
}

export const api = {
  // -- auth --
  loginWithGoogle: (idToken: string) =>
    request<{ session_token: string; email: string; name: string }>("/auth/google", {
      method: "POST",
      body: { id_token: idToken },
    }),

  // -- projects --
  listProjects: () => request<{ projects: string[] }>("/projects"),
  deleteProject: (projectId: string) => request<{ status: string }>(`/projects/${projectId}`, { method: "DELETE" }),
  getBaseline: (projectId: string) => request<BaselineInfo>(`/baseline/${projectId}`),

  // -- tabular fit/analyze --
  fitTabular: (projectId: string, body: Record<string, unknown>) =>
    request<FitResponse>(`/fit/${projectId}`, { method: "POST", body }),
  analyzeTabular: (projectId: string, productionData: Record<string, unknown[]>, idempotencyKey?: string, baselineVersion?: number) =>
    request<AnalyzeResponse>(`/analyze/${projectId}`, {
      method: "POST",
      body: { production_data: productionData },
      params: { baseline_version: baselineVersion },
      headers: idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {},
    }),
  predict: (projectId: string, features: Record<string, unknown>) =>
    request<PredictResponse>(`/predict/${projectId}`, { method: "POST", body: { features } }),

  // -- text/image/joint fit/analyze --
  fitText: (projectId: string, referenceTexts: string[], calibrationConfig?: Record<string, unknown>) =>
    request<FitResponse>(`/fit/${projectId}/text`, {
      method: "POST", body: { reference_texts: referenceTexts, calibration_config: calibrationConfig },
    }),
  analyzeText: (projectId: string, productionTexts: string[]) =>
    request<AnalyzeResponse>(`/analyze/${projectId}/text`, { method: "POST", body: { production_texts: productionTexts } }),
  fitImage: (projectId: string, referenceImages: string[], calibrationConfig?: Record<string, unknown>) =>
    request<FitResponse>(`/fit/${projectId}/image`, {
      method: "POST", body: { reference_images: referenceImages, calibration_config: calibrationConfig },
    }),
  analyzeImage: (projectId: string, productionImages: string[]) =>
    request<AnalyzeResponse>(`/analyze/${projectId}/image`, { method: "POST", body: { production_images: productionImages } }),
  fitJoint: (projectId: string, referenceRecords: unknown[]) =>
    request<FitResponse>(`/fit/${projectId}/joint`, { method: "POST", body: { reference_records: referenceRecords } }),
  analyzeJoint: (projectId: string, productionRecords: unknown[]) =>
    request<AnalyzeResponse>(`/analyze/${projectId}/joint`, { method: "POST", body: { production_records: productionRecords } }),

  // -- history --
  getHistory: (projectId: string, filters: { since?: string; until?: string; feature?: string; alertOnly?: boolean; limit?: number; offset?: number } = {}) =>
    request<HistoryResponse>(`/history/${projectId}`, {
      params: { since: filters.since, until: filters.until, feature: filters.feature, alert_only: filters.alertOnly, limit: filters.limit, offset: filters.offset },
    }),

  // -- baseline versioning --
  listBaselineVersions: (projectId: string) => request<{ project_id: string; versions: BaselineVersion[] }>(`/baselines/${projectId}`),
  activateBaselineVersion: (projectId: string, version: number) =>
    request<{ status: string; active_version: number }>(`/baselines/${projectId}/activate`, { method: "POST", body: { version } }),

  // -- webhooks --
  listWebhooks: (projectId: string) => request<{ project_id: string; webhooks: Webhook[] }>(`/webhooks/${projectId}`),
  registerWebhook: (projectId: string, url: string, eventFilter?: string[]) =>
    request<Webhook>(`/webhooks/${projectId}`, { method: "POST", body: { url, event_filter: eventFilter } }),
  deleteWebhook: (projectId: string, webhookId: string) =>
    request<{ status: string }>(`/webhooks/${projectId}/${webhookId}`, { method: "DELETE" }),

  // -- logs / health --
  getLogs: (projectId: string) => request<LogEntry[]>(`/logs/${projectId}`),
  getHealth: (projectId: string) => request<HealthResponse>(`/health/${projectId}`),

  // -- upload variants --
  fitUpload: (projectId: string, file: File, extra?: Record<string, string>) => {
    const form = new FormData();
    form.append("file", file);
    if (extra) for (const [k, v] of Object.entries(extra)) form.append(k, v);
    return request<FitResponse>(`/fit/${projectId}/upload`, { method: "POST", body: form, isForm: true });
  },
  analyzeUpload: (projectId: string, file: File, idempotencyKey?: string) => {
    const form = new FormData();
    form.append("file", file);
    return request<AnalyzeResponse>(`/analyze/${projectId}/upload`, {
      method: "POST", body: form, isForm: true,
      headers: idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {},
    });
  },

  // -- unified table path (one table = one project; heavy steps run as jobs) --
  stageTable: (projectId: string, file: File, images?: File) =>
    request<JobAccepted>(`/tables/${projectId}/stage`, { method: "POST", body: tableForm(file, images), isForm: true }),
  fitTable: (projectId: string, stageId: string, columns: TableColumnChoice[], relationships: TableRelationshipChoice[]) =>
    request<JobAccepted>(`/tables/${projectId}/fit`, { method: "POST", body: { stage_id: stageId, columns, relationships } }),
  analyzeTable: (projectId: string, file: File, images?: File) =>
    request<JobAccepted>(`/tables/${projectId}/analyze`, { method: "POST", body: tableForm(file, images), isForm: true }),
  getJob: <R,>(jobId: string) => request<Job<R>>(`/jobs/${jobId}`),
  getTableBaseline: (projectId: string) => request<TableBaselineInfo>(`/tables/${projectId}/baseline`),
};

function tableForm(file: File, images?: File): FormData {
  const form = new FormData();
  form.append("file", file);
  if (images) form.append("images", images);
  return form;
}

/** Polls a job until it finishes; resolves with its result or throws its error. */
export async function waitForJob<R>(jobId: string, onProgress?: (job: Job<R>) => void): Promise<R> {
  for (;;) {
    const job = await api.getJob<R>(jobId);
    onProgress?.(job);
    if (job.status === "succeeded") return job.result as R;
    if (job.status === "failed" || job.status === "interrupted") throw new ApiError(422, job.error || "Job failed.");
    await new Promise((r) => setTimeout(r, 1000));
  }
}
