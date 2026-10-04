import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router-dom";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Badge, Card, EmptyState, ErrorBanner, PageHeader, Spinner } from "../components/ui";
import { api, ApiError } from "../lib/api";

export function HistoryPage() {
  const { projectId = "" } = useParams();
  const [alertOnly, setAlertOnly] = useState(false);
  const [feature, setFeature] = useState<string>("");

  const { data, isLoading, error } = useQuery({
    queryKey: ["history", projectId, alertOnly, feature],
    queryFn: () => api.getHistory(projectId, { alertOnly, feature: feature || undefined, limit: 50 }),
    retry: false,
  });

  const featureNames = data ? Object.keys(data.feature_time_series) : [];

  return (
    <div>
      <PageHeader
        title="History"
        subtitle="Every /analyze call for this project -- statistics only, never raw rows."
      />

      <div className="mb-6 flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-slate-600">
          <input type="checkbox" checked={alertOnly} onChange={(e) => setAlertOnly(e.target.checked)} />
          Alerting runs only
        </label>
        {featureNames.length > 0 && (
          <select
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm"
            value={feature}
            onChange={(e) => setFeature(e.target.value)}
          >
            <option value="">All features</option>
            {featureNames.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        )}
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-16 text-slate-400">
          <Spinner className="h-6 w-6" />
        </div>
      )}

      {error && <ErrorBanner message={error instanceof ApiError ? error.detail : "Failed to load history."} />}

      {data && data.total === 0 && (
        <EmptyState title="No analysis history yet" description="Run an /analyze call to start building history." />
      )}

      {data && Object.entries(data.feature_time_series).map(([name, series]) => (
        <Card key={name} className="mb-6 p-5">
          <p className="mb-3 text-sm font-semibold text-slate-700">{name} &mdash; statistic over time</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={series.map((p) => ({ ...p, tsLabel: new Date(p.ts).toLocaleString() }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="tsLabel" tick={{ fontSize: 11 }} minTickGap={40} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="statistic" stroke="#4f46e5" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      ))}

      {data && data.runs.length > 0 && (
        <Card className="overflow-hidden">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-5 py-3">Time</th>
                <th className="px-5 py-3">Baseline</th>
                <th className="px-5 py-3">Batch size</th>
                <th className="px-5 py-3">Decision mode</th>
                <th className="px-5 py-3">Alert</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.runs.map((run) => (
                <tr key={run.id}>
                  <td className="px-5 py-2.5 text-slate-600">{new Date(run.ts).toLocaleString()}</td>
                  <td className="px-5 py-2.5 text-slate-600">v{run.baseline_version}</td>
                  <td className="px-5 py-2.5 text-slate-600">{run.batch_size}</td>
                  <td className="px-5 py-2.5 text-slate-600">{run.decision_mode ?? "legacy"}</td>
                  <td className="px-5 py-2.5">
                    {run.system_alert ? <Badge tone="alert">Alert</Badge> : <Badge tone="ok">Clear</Badge>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}
