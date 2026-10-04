import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router-dom";
import { Badge, Button, Card, EmptyState, ErrorBanner, Field, PageHeader, Spinner, TextInput } from "../components/ui";
import { api, ApiError } from "../lib/api";
import type { PredictResponse } from "../lib/types";

export function PredictPage() {
  const { projectId = "" } = useParams();
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictResponse | null>(null);

  const { data: baseline, isLoading } = useQuery({
    queryKey: ["baseline", projectId],
    queryFn: () => api.getBaseline(projectId),
    retry: false,
  });

  const mutation = useMutation({
    mutationFn: () => {
      const features: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(values)) {
        const n = Number(v);
        features[k] = Number.isNaN(n) || v.trim() === "" ? v : n;
      }
      return api.predict(projectId, features);
    },
    onSuccess: setResult,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16 text-slate-400">
        <Spinner className="h-6 w-6" />
      </div>
    );
  }

  if (!baseline || baseline.modality !== "tabular" || Object.keys(baseline.feature_types).length === 0) {
    return (
      <div>
        <PageHeader title="Predict" subtitle="Real-time single-point anomaly check." />
        <EmptyState title="Not available" description="Predict requires a tabular baseline with monitored features." />
      </div>
    );
  }

  return (
    <div>
      <PageHeader title="Predict" subtitle="Score one incoming data point against the locked IQR fences." />

      <Card className="max-w-xl p-6 space-y-4">
        {Object.entries(baseline.feature_types).map(([name, type]) => (
          <Field key={name} label={name} hint={type}>
            <TextInput
              value={values[name] ?? ""}
              onChange={(e) => setValues((prev) => ({ ...prev, [name]: e.target.value }))}
            />
          </Field>
        ))}
        <Button onClick={() => mutation.mutate()} disabled={mutation.isPending}>
          {mutation.isPending ? "Scoring..." : "Score Point"}
        </Button>
        {mutation.error && (
          <ErrorBanner message={mutation.error instanceof ApiError ? mutation.error.detail : "Prediction failed."} />
        )}
      </Card>

      {result && (
        <Card className="mt-6 max-w-xl p-6">
          <div className="mb-4 flex items-center justify-between">
            <p className="text-sm font-semibold text-slate-700">Result</p>
            {result.is_anomaly ? <Badge tone="alert">Anomaly</Badge> : <Badge tone="ok">Normal</Badge>}
          </div>
          <p className="text-sm text-slate-500">
            Anomaly score: <span className="font-mono text-slate-700">{result.anomaly_score.toFixed(4)}</span>
          </p>
          {Object.keys(result.feature_deviations).length > 0 && (
            <div className="mt-4 space-y-1">
              {Object.entries(result.feature_deviations).map(([k, v]) => (
                <div key={k} className="flex justify-between text-sm">
                  <span className="text-slate-500">{k}</span>
                  <span className="font-mono text-slate-700">{typeof v === "number" ? v.toFixed(4) : String(v)}</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
