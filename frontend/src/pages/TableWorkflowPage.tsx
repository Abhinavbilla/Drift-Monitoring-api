import { useState } from "react";
import { useParams } from "react-router-dom";
import { Badge, Button, Card, ErrorBanner, Field, FileInput, PageHeader, SuccessBanner } from "../components/ui";
import { ApiError, api, waitForJob } from "../lib/api";
import { TABULAR_ACCEPT } from "../lib/constants";
import type {
  ColumnProposal, Job, TableColumnChoice, TableColumnType, TableFitResult, TableProfile, TableReport,
} from "../lib/types";

const TYPES: TableColumnType[] = ["numeric", "categorical", "text", "image", "ignore"];
const STATUS_TONE = { DRIFT: "alert", STABLE: "ok", NOT_TESTED: "slate", DATA_ISSUES: "warn" } as const;

function errorText(e: unknown): string {
  return e instanceof ApiError ? e.detail : e instanceof Error ? e.message : "Something went wrong.";
}

function JobProgress({ job }: { job: Job<unknown> | null }) {
  if (!job || job.status === "succeeded") return null;
  return <p className="text-sm text-slate-500">{job.progress_message || job.status}… {job.progress}%</p>;
}

function ColumnReviewRow({ proposal, choice, onChange }: {
  proposal: ColumnProposal; choice: TableColumnChoice; onChange: (c: TableColumnChoice) => void;
}) {
  return (
    <tr className="border-t border-slate-100 align-top">
      <td className="py-2 pr-3 font-medium text-slate-800">{proposal.name}</td>
      <td className="py-2 pr-3">
        <select
          className="rounded-md border border-slate-300 px-2 py-1 text-sm"
          value={choice.type}
          onChange={(e) => {
            const type = e.target.value as TableColumnType;
            onChange({ ...choice, type, monitor: type !== "ignore" && choice.monitor });
          }}
        >
          {TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
      </td>
      <td className="py-2 pr-3">
        <input
          type="checkbox"
          checked={choice.monitor}
          disabled={choice.type === "ignore"}
          onChange={(e) => onChange({ ...choice, monitor: e.target.checked })}
          aria-label={`Monitor ${proposal.name}`}
        />
      </td>
      <td className="py-2 pr-3 text-sm text-slate-600">{Math.round(proposal.confidence * 100)}%</td>
      <td className="py-2 text-sm text-slate-600">
        {proposal.reason}
        <details className="mt-1">
          <summary className="cursor-pointer text-xs text-brand-600">Evidence</summary>
          <dl className="mt-1 grid grid-cols-2 gap-x-3 text-xs text-slate-500">
            {Object.entries(proposal.evidence).map(([k, v]) => (
              <div key={k} className="contents"><dt>{k}</dt><dd>{String(v)}</dd></div>
            ))}
          </dl>
          {proposal.sample_values.length > 0 && (
            <p className="mt-1 text-xs text-slate-500">Samples: {proposal.sample_values.join(" · ")}</p>
          )}
        </details>
      </td>
    </tr>
  );
}

function Report({ report }: { report: TableReport }) {
  const issues = Object.entries(report.schema_report).flatMap(([col, list]) => list.map((i) => ({ col, ...i })));
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Badge tone={STATUS_TONE[report.overall.status]}>{report.overall.status}</Badge>
        <span className="text-sm text-slate-600">
          {report.n_rows} rows vs baseline v{report.baseline_version}
          {report.overall.triggered_by.length > 0 && ` · drift in ${report.overall.triggered_by.join(", ")}`}
        </span>
      </div>
      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-700">Column drift</h3>
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase text-slate-400">
            <tr><th className="py-1">Column</th><th>Type</th><th>Test</th><th>Statistic</th><th>Status</th><th>Details</th></tr>
          </thead>
          <tbody>
            {Object.entries(report.column_drift).map(([col, r]) => (
              <tr key={col} className="border-t border-slate-100 align-top">
                <td className="py-2 font-medium text-slate-800">{col}</td>
                <td>{r.type}</td>
                <td>{r.test}</td>
                <td>{r.statistic !== undefined ? r.statistic.toFixed(3) : "—"}</td>
                <td><Badge tone={STATUS_TONE[r.status]}>{r.status}</Badge></td>
                <td className="text-xs text-slate-500">
                  <details>
                    <summary className="cursor-pointer text-brand-600">More</summary>
                    {r.reason && <p>{r.reason}</p>}
                    {r.threshold_used && <p>Rule: {r.threshold_used}</p>}
                    {r.p_value_adjusted != null && <p>Adjusted p: {r.p_value_adjusted.toPrecision(3)}</p>}
                    <p>{r.in_family ? "Part of the Holm family" : "Decided outside the Holm family"}</p>
                  </details>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 text-xs text-slate-400">Relationship drift between columns arrives in the next milestone.</p>
      </div>
      {issues.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-slate-700">Data issues</h3>
          <ul className="space-y-1 text-sm text-slate-600">
            {issues.map((i, n) => (
              <li key={n}><Badge tone={i.severity === "alert" ? "alert" : "warn"}>{i.severity}</Badge> {i.col}: {i.issue.replace(/_/g, " ")}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function TableWorkflowPage() {
  const { projectId = "" } = useParams();
  const [trainFiles, setTrainFiles] = useState<File[]>([]);
  const [trainZip, setTrainZip] = useState<File[]>([]);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchZip, setBatchZip] = useState<File[]>([]);
  const [profile, setProfile] = useState<TableProfile | null>(null);
  const [choices, setChoices] = useState<Record<string, TableColumnChoice>>({});
  const [fitResult, setFitResult] = useState<TableFitResult | null>(null);
  const [report, setReport] = useState<TableReport | null>(null);
  const [job, setJob] = useState<Job<unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
      setJob(null);
    }
  };

  const startProfile = () => run(async () => {
    setFitResult(null);
    const accepted = await api.stageTable(projectId, trainFiles[0], trainZip[0]);
    const result = await waitForJob<TableProfile>(accepted.job_id, setJob);
    setProfile(result);
    setChoices(Object.fromEntries(result.columns.map((c) => [
      c.name, { name: c.name, type: c.proposed_type, monitor: c.proposed_monitor },
    ])));
  });

  const lockBaseline = () => run(async () => {
    if (!profile) return;
    const accepted = await api.fitTable(projectId, profile.stage_id, Object.values(choices));
    setFitResult(await waitForJob<TableFitResult>(accepted.job_id, setJob));
    setProfile(null);
  });

  const analyze = () => run(async () => {
    const accepted = await api.analyzeTable(projectId, batchFiles[0], batchZip[0]);
    setReport(await waitForJob<TableReport>(accepted.job_id, setJob));
  });

  return (
    <div className="space-y-6">
      <PageHeader
        title="Table monitoring"
        subtitle="One table can mix numeric, categorical, text and image columns. Upload it, confirm each column, lock a baseline, then compare production batches."
      />
      {error && <ErrorBanner message={error} />}

      <Card className="space-y-4 p-6">
        <h2 className="text-base font-semibold text-slate-800">1. Upload training data</h2>
        <Field label="Training table" hint="CSV, TSV, Excel, JSON/JSONL, Parquet, ARFF, or a compressed CSV.">
          <FileInput accept={TABULAR_ACCEPT} files={trainFiles} onFiles={setTrainFiles} />
        </Field>
        <Field label="Image ZIP (optional)" hint="Only if a column holds image filenames. Names must match the files in the ZIP.">
          <FileInput accept=".zip" files={trainZip} onFiles={setTrainZip} />
        </Field>
        <Button onClick={startProfile} disabled={busy || trainFiles.length === 0}>Profile table</Button>
        <JobProgress job={job} />
      </Card>

      {profile && (
        <Card className="space-y-4 p-6">
          <h2 className="text-base font-semibold text-slate-800">2. Review columns ({profile.n_rows} rows)</h2>
          <p className="text-sm text-slate-500">
            These are suggestions. Change any type, and choose which columns to monitor. Nothing is monitored until you lock the baseline.
          </p>
          {profile.image_zip && Object.keys(profile.image_zip.rejected).length > 0 && (
            <p className="text-sm text-warn-600">
              Skipped ZIP entries: {Object.entries(profile.image_zip.rejected).map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`).join(", ")}
            </p>
          )}
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-400">
              <tr><th className="py-1">Column</th><th>Type</th><th>Monitor</th><th>Confidence</th><th>Why</th></tr>
            </thead>
            <tbody>
              {profile.columns.map((p) => (
                <ColumnReviewRow
                  key={p.name}
                  proposal={p}
                  choice={choices[p.name]}
                  onChange={(c) => setChoices((prev) => ({ ...prev, [p.name]: c }))}
                />
              ))}
            </tbody>
          </table>
          <Button onClick={lockBaseline} disabled={busy}>Lock baseline</Button>
          <JobProgress job={job} />
        </Card>
      )}

      {fitResult && (
        <SuccessBanner
          message={`Baseline v${fitResult.version} locked. Monitoring ${Object.entries(fitResult.monitored)
            .filter(([, cols]) => cols.length).map(([t, cols]) => `${t}: ${cols.join(", ")}`).join(" · ")}.`
            + (fitResult.duplicate_rows_dropped ? ` Removed ${fitResult.duplicate_rows_dropped} duplicate row(s).` : "")}
        />
      )}

      <Card className="space-y-4 p-6">
        <h2 className="text-base font-semibold text-slate-800">3. Analyze a production batch</h2>
        <Field label="Production table" hint="Same columns as the training table.">
          <FileInput accept={TABULAR_ACCEPT} files={batchFiles} onFiles={setBatchFiles} />
        </Field>
        <Field label="Image ZIP (optional)">
          <FileInput accept=".zip" files={batchZip} onFiles={setBatchZip} />
        </Field>
        <Button onClick={analyze} disabled={busy || batchFiles.length === 0}>Run analysis</Button>
        <JobProgress job={job} />
        {report && <Report report={report} />}
      </Card>
    </div>
  );
}
