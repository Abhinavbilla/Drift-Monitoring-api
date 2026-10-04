import { Badge, Card } from "./ui";
import type { AnalyzeResponse, FeatureMetric, SchemaIssue } from "../lib/types";

function severityTone(severity: string): "alert" | "warn" | "slate" {
  if (severity === "alert") return "alert";
  if (severity === "warn") return "warn";
  return "slate";
}

function FeatureMetricsTable({ metrics }: { metrics: Record<string, FeatureMetric> }) {
  const rows = Object.entries(metrics);
  if (rows.length === 0) return <p className="text-sm text-slate-400">No features were analyzed.</p>;
  return (
    <table className="w-full text-sm">
      <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
        <tr>
          <th className="px-5 py-3">Feature</th>
          <th className="px-5 py-3">Statistic</th>
          <th className="px-5 py-3">p-value</th>
          <th className="px-5 py-3">Drift detected</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-slate-100">
        {rows.map(([name, m]) => (
          <tr key={name}>
            <td className="px-5 py-2.5 font-medium text-slate-700">{name}</td>
            <td className="px-5 py-2.5 text-slate-600">{m.statistic.toFixed(4)}</td>
            <td className="px-5 py-2.5 text-slate-600">{m.p_value != null ? m.p_value.toExponential(3) : "--"}</td>
            <td className="px-5 py-2.5">
              {m.drift_detected ? <Badge tone="alert">Drifted</Badge> : <Badge tone="ok">Stable</Badge>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SchemaReportList({ report }: { report: Record<string, SchemaIssue[]> }) {
  const entries = Object.entries(report);
  if (entries.length === 0) {
    return <p className="text-sm text-ok-600">No schema issues found in this batch.</p>;
  }
  return (
    <ul className="space-y-2">
      {entries.map(([column, issues]) =>
        issues.map((issue, i) => (
          <li key={`${column}-${i}`} className="flex items-start gap-3 rounded-lg border border-slate-100 px-4 py-2.5">
            <Badge tone={severityTone(issue.severity)}>{issue.severity}</Badge>
            <div className="text-sm">
              <span className="font-semibold text-slate-700">{column}</span>
              <span className="text-slate-500"> &mdash; {issue.issue.replace(/_/g, " ")}</span>
            </div>
          </li>
        )),
      )}
    </ul>
  );
}

export function AnalyzeResultPanel({ result }: { result: AnalyzeResponse }) {
  const hasAlertState = result.alert_state != null;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card className="p-5">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">System alert</p>
          <div className="mt-2">
            {result.system_alert_triggered ? <Badge tone="alert">Triggered</Badge> : <Badge tone="ok">Clear</Badge>}
          </div>
        </Card>
        {hasAlertState && (
          <>
            <Card className="p-5">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Alert state</p>
              <div className="mt-2">
                {result.alert_state === "open" ? <Badge tone="alert">Open</Badge> : <Badge tone="ok">Ok</Badge>}
              </div>
              {result.windows_considered != null && (
                <p className="mt-1 text-xs text-slate-400">{result.windows_considered} window(s) considered</p>
              )}
            </Card>
            <Card className="p-5">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Transition</p>
              <div className="mt-2">
                {result.transition ? <Badge tone="warn">{result.transition}</Badge> : <Badge tone="slate">none</Badge>}
              </div>
            </Card>
          </>
        )}
      </div>

      <Card className="overflow-hidden">
        <div className="border-b border-slate-200 px-5 py-3">
          <p className="text-sm font-semibold text-slate-700">Feature metrics</p>
        </div>
        <FeatureMetricsTable metrics={result.feature_metrics} />
      </Card>

      {Object.keys(result.schema_report).length >= 0 && (
        <Card className="p-5">
          <p className="mb-3 text-sm font-semibold text-slate-700">Schema report</p>
          <SchemaReportList report={result.schema_report} />
        </Card>
      )}
    </div>
  );
}
