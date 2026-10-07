import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Button, Card, EmptyState, ErrorBanner, PageHeader, Spinner, StatusPill } from "../components/ui";
import { IconClock } from "../components/icons";
import { pairLabel } from "../components/vocab";
import { api, ApiError } from "../lib/api";
import { timeAgo } from "../lib/format";
import type { HistoryRun } from "../lib/types";

function drifted(run: HistoryRun): string[] {
  const names = (metrics: Record<string, { drift_detected: boolean }> = {}) =>
    Object.entries(metrics).filter(([, m]) => m.drift_detected).map(([name]) => pairLabel(name));
  return [...names(run.feature_metrics), ...names(run.relationship_metrics)];
}

const shortDate = (ts: string) => new Date(ts).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

export function HistoryPage() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const [alertOnly, setAlertOnly] = useState(false);
  const [feature, setFeature] = useState<string>("");

  const { data, isLoading, error } = useQuery({
    queryKey: ["history", projectId, alertOnly, feature],
    queryFn: () => api.getHistory(projectId, { alertOnly, feature: feature || undefined, limit: 50 }),
    retry: false,
  });
  const featureNames = data ? Object.keys(data.feature_time_series) : [];

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Past checks"
        title="Every check you've run"
        subtitle="Each row is one batch you checked. We keep the results and statistics only — never your raw data."
      />

      <div className="flex flex-wrap items-center gap-3">
        <label className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm text-slate-700">
          <input type="checkbox" className="accent-brand-600" checked={alertOnly} onChange={(e) => setAlertOnly(e.target.checked)} />
          Only checks that found changes
        </label>
        {featureNames.length > 0 && (
          <select className="rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-sm text-slate-700"
                  value={feature} onChange={(e) => setFeature(e.target.value)}>
            <option value="">All columns</option>
            {featureNames.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        )}
      </div>

      {isLoading && <div className="flex justify-center py-16 text-slate-400"><Spinner className="h-6 w-6" /></div>}
      {error && <ErrorBanner message={error instanceof ApiError ? error.detail : "We couldn't load past checks."} />}

      {data && data.total === 0 && (
        <EmptyState icon={<IconClock />} title="No checks yet"
                    description="Once you check a batch of new data, it shows up here so you can see how things change over time."
                    action={<Button onClick={() => navigate(`/projects/${encodeURIComponent(projectId)}/table`)}>Check new data</Button>} />
      )}

      {data && data.runs.length > 0 && (
        <Card className="overflow-hidden">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-200 bg-slate-50/80 text-left text-xs font-semibold text-slate-500">
              <tr>
                <th className="px-5 py-3">When</th>
                <th className="px-5 py-3">Result</th>
                <th className="px-5 py-3">What changed</th>
                <th className="px-5 py-3 text-right">Rows</th>
                <th className="px-5 py-3 text-right">Baseline</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.runs.map((run) => {
                const changed = drifted(run);
                return (
                  <tr key={run.id} className="hover:bg-slate-50/60">
                    <td className="px-5 py-3">
                      <p className="font-medium text-slate-800">{timeAgo(run.ts)}</p>
                      <p className="text-xs text-slate-400">{new Date(run.ts).toLocaleString()}</p>
                    </td>
                    <td className="px-5 py-3">
                      {run.system_alert ? <StatusPill tone="alert">Changes found</StatusPill> : <StatusPill tone="ok">Normal</StatusPill>}
                    </td>
                    <td className="max-w-md px-5 py-3 text-slate-600">{changed.join(", ") || <span className="text-slate-400">Nothing</span>}</td>
                    <td className="px-5 py-3 text-right tabular-nums text-slate-600">{run.batch_size?.toLocaleString()}</td>
                    <td className="px-5 py-3 text-right text-slate-500">v{run.baseline_version}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Card>
      )}

      {data && featureNames.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold text-slate-900">Trends by column</h2>
          <p className="mt-1 text-sm text-slate-500">How different each column was from normal in each check. Higher means more different.</p>
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            {Object.entries(data.feature_time_series).map(([name, series]) => (
              <Card key={name} className="p-5">
                <p className="mb-3 text-sm font-semibold text-slate-800">{name}</p>
                <ResponsiveContainer width="100%" height={150}>
                  <LineChart data={series.map((p) => ({ ...p, when: shortDate(p.ts) }))} margin={{ left: 0, right: 8, top: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" vertical={false} />
                    <XAxis dataKey="when" tick={{ fontSize: 11, fill: "#94a3b8" }} minTickGap={40} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} width={52} />
                    <Tooltip formatter={(v) => [Number(v).toFixed(3), "difference"]} />
                    <Line type="monotone" dataKey="statistic" stroke="#4f46e5" strokeWidth={2.5} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
