import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { Badge, Card, EmptyState, ErrorBanner, PageHeader, Spinner } from "../components/ui";
import { api, ApiError } from "../lib/api";

export function LogsPage() {
  const { projectId = "" } = useParams();

  const { data, isLoading, error } = useQuery({
    queryKey: ["logs", projectId],
    queryFn: () => api.getLogs(projectId),
    retry: false,
  });

  return (
    <div>
      <PageHeader eyebrow="Activity log" title="Single-row checks" subtitle="The most recent rows checked one at a time with “Check a single row” (up to the last 1,000)." />

      {isLoading && (
        <div className="flex items-center justify-center py-16 text-slate-400">
          <Spinner className="h-6 w-6" />
        </div>
      )}

      {error && <ErrorBanner message={error instanceof ApiError ? error.detail : "Failed to load logs."} />}

      {data && data.length === 0 && (
        <EmptyState title="No logs yet" description="Logs appear here after you call /predict for this project." />
      )}

      {data && data.length > 0 && (
        <Card className="overflow-hidden">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-5 py-3">Input</th>
                <th className="px-5 py-3">Score</th>
                <th className="px-5 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.slice(0, 200).map((log, i) => (
                <tr key={i}>
                  <td className="max-w-md truncate px-5 py-2.5 font-mono text-xs text-slate-500">
                    {JSON.stringify(log.input_data)}
                  </td>
                  <td className="px-5 py-2.5 text-slate-600">{log.score.toFixed(4)}</td>
                  <td className="px-5 py-2.5">
                    {log.is_ood ? <Badge tone="alert">Anomaly</Badge> : <Badge tone="ok">Normal</Badge>}
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
