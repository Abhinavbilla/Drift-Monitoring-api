import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import {
  Badge, Button, Card, Details, ErrorBanner, FileInput, PageHeader, StatusPill, Stepper, Tabs,
} from "../components/ui";
import { IconAlert, IconArrowRight, IconCheckCircle, IconInfo, IconLink, IconPlus, IconSparkles } from "../components/icons";
import { COLUMN_TYPES, TypeIcon, connectionSentence, pairLabel } from "../components/vocab";
import { ApiError, api, waitForJob } from "../lib/api";
import { TABULAR_ACCEPT } from "../lib/constants";
import { plural } from "../lib/format";
import type {
  ColumnDrift, ColumnProposal, Job, RelationshipDrift, RelationshipProposal, TableColumnChoice, TableColumnType,
  TableFitResult, TableProfile, TableRelationshipChoice, TableReport,
} from "../lib/types";

const TYPES: TableColumnType[] = ["numeric", "categorical", "text", "image", "ignore"];
const SETUP_STEPS = ["Upload your data", "Check the columns", "Choose connections", "Done"];
const pairKey = (a: string, b: string) => [a, b].sort().join("<->");

function errorText(e: unknown): string {
  return e instanceof ApiError ? e.detail : e instanceof Error ? e.message : "Something went wrong. Please try again.";
}

function certainty(confidence: number): { label: string; tone: "ok" | "brand" | "warn" } {
  if (confidence >= 0.85) return { label: "Confident", tone: "ok" };
  if (confidence >= 0.6) return { label: "Fairly sure", tone: "brand" };
  return { label: "Please check", tone: "warn" };
}

// ---------------------------------------------------------------- shared pieces

function Progress({ job, fallback }: { job: Job<unknown> | null; fallback: string }) {
  const pct = job?.progress ?? 5;
  return (
    <Card className="animate-fade-up p-6">
      <div className="flex items-center gap-3">
        <span className="relative flex h-3 w-3">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand-400 opacity-60" />
          <span className="relative inline-flex h-3 w-3 rounded-full bg-brand-600" />
        </span>
        <p className="font-medium text-slate-800">{job?.progress_message || fallback}…</p>
      </div>
      <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full bg-gradient-to-r from-brand-400 to-brand-600 transition-all duration-500"
             style={{ width: `${Math.max(5, pct)}%` }} />
      </div>
      <p className="mt-3 text-xs text-slate-500">Photos and long text take the longest. You can leave this page open while it works.</p>
    </Card>
  );
}

function Toggle({ checked, onChange, disabled, label }: {
  checked: boolean; onChange: (v: boolean) => void; disabled?: boolean; label: string;
}) {
  return (
    <button type="button" role="switch" aria-checked={checked} aria-label={label} disabled={disabled}
            onClick={() => onChange(!checked)}
            className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors disabled:opacity-40 ${
              checked ? "bg-brand-600" : "bg-slate-300"}`}>
      <span className={`inline-block h-5 w-5 rounded-full bg-white shadow transition-transform ${checked ? "translate-x-5" : "translate-x-0.5"}`} />
    </button>
  );
}

// ---------------------------------------------------------------- set-up flow

function UploadStep({ busy, onStart }: { busy: boolean; onStart: (table: File, zip?: File) => void }) {
  const [table, setTable] = useState<File[]>([]);
  const [zip, setZip] = useState<File[]>([]);
  return (
    <Card className="p-6 sm:p-8">
      <h2 className="text-lg font-semibold text-slate-900">Upload the data your model learned from</h2>
      <p className="mt-1 text-sm leading-relaxed text-slate-500">
        This becomes your <span className="font-medium text-slate-700">baseline</span> — what “normal” looks like.
        Later batches are compared against it.
      </p>
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <p className="mb-2 text-sm font-medium text-slate-700">Your data file</p>
          <FileInput accept={TABULAR_ACCEPT} files={table} onFiles={setTable} prompt="Choose a spreadsheet" />
          <p className="mt-2 text-xs text-slate-500">CSV or Excel work best. JSON, Parquet and a few others are fine too.</p>
        </div>
        <div>
          <p className="mb-2 text-sm font-medium text-slate-700">Photos <span className="font-normal text-slate-400">(optional)</span></p>
          <FileInput accept=".zip" files={zip} onFiles={setZip} prompt="Choose a ZIP of photos" />
          <p className="mt-2 text-xs text-slate-500">Only if a column lists photo file names, like “cat-01.jpg”. Put those photos in one ZIP.</p>
        </div>
      </div>
      <div className="mt-8 flex items-center justify-between gap-4 border-t border-slate-100 pt-6">
        <p className="flex items-center gap-2 text-xs text-slate-500">
          <IconInfo className="h-4 w-4 shrink-0" /> Your file is deleted after setup. We keep summaries, not your raw data.
        </p>
        <Button disabled={busy || table.length === 0} onClick={() => onStart(table[0], zip[0])}>
          Look at my data <IconArrowRight className="h-4 w-4" />
        </Button>
      </div>
    </Card>
  );
}

function ColumnRow({ proposal, choice, onChange }: {
  proposal: ColumnProposal; choice: TableColumnChoice; onChange: (c: TableColumnChoice) => void;
}) {
  const sure = certainty(proposal.confidence);
  const changed = choice.type !== proposal.proposed_type;
  return (
    <li className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center">
      <div className="flex min-w-0 flex-1 items-start gap-3">
        <TypeIcon type={choice.type} className={choice.monitor ? "bg-brand-50 text-brand-600" : ""} />
        <div className="min-w-0">
          <p className="truncate font-medium text-slate-900">{proposal.name}</p>
          {proposal.sample_values.length > 0 && (
            <p className="mt-0.5 truncate text-xs text-slate-400">e.g. {proposal.sample_values.slice(0, 3).join(" · ")}</p>
          )}
          <div className="mt-1.5">
            <Details summary="Why we suggested this">
              <p>{proposal.reason}</p>
              <p className="text-slate-400">
                {Object.entries(proposal.evidence).map(([k, v]) => `${k.replace(/_/g, " ")}: ${v}`).join(" · ")}
              </p>
            </Details>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3 sm:justify-end">
        {!changed && <Badge tone={sure.tone}>{sure.label}</Badge>}
        {changed && <Badge tone="brand">Changed by you</Badge>}
        <select
          aria-label={`Type of ${proposal.name}`}
          className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-4 focus:ring-brand-100"
          value={choice.type}
          onChange={(e) => {
            const type = e.target.value as TableColumnType;
            onChange({ ...choice, type, monitor: type !== "ignore" });
          }}
        >
          {TYPES.map((t) => <option key={t} value={t}>{COLUMN_TYPES[t].label}</option>)}
        </select>
        <div className="flex items-center gap-2">
          <Toggle checked={choice.monitor} disabled={choice.type === "ignore"} label={`Watch ${proposal.name}`}
                  onChange={(monitor) => onChange({ ...choice, monitor })} />
          <span className="w-10 text-xs text-slate-500">{choice.monitor ? "Watch" : "Skip"}</span>
        </div>
      </div>
    </li>
  );
}

function ColumnsStep({ profile, choices, setChoices, onNext, onBack }: {
  profile: TableProfile; choices: Record<string, TableColumnChoice>;
  setChoices: (f: (prev: Record<string, TableColumnChoice>) => Record<string, TableColumnChoice>) => void;
  onNext: () => void; onBack: () => void;
}) {
  const counts = TYPES.map((t) => [t, Object.values(choices).filter((c) => c.type === t).length] as const).filter(([, n]) => n);
  const watching = Object.values(choices).filter((c) => c.monitor).length;
  const rejected = profile.image_zip ? Object.entries(profile.image_zip.rejected) : [];
  return (
    <Card className="p-6 sm:p-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">We looked at your {plural(profile.n_rows, "row")}</h2>
          <p className="mt-1 text-sm text-slate-500">
            Here's what we think each column holds. Change anything that looks wrong — you know your data best.
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {counts.map(([t, n]) => <Badge key={t}>{n} {COLUMN_TYPES[t].label.toLowerCase()}</Badge>)}
        </div>
      </div>
      {rejected.length > 0 && (
        <p className="mt-4 rounded-xl bg-warn-50 px-4 py-3 text-sm text-warn-700">
          Some files in your ZIP were skipped: {rejected.map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`).join(", ")}.
        </p>
      )}
      <ul className="mt-4 divide-y divide-slate-100">
        {profile.columns.map((p) => (
          <ColumnRow key={p.name} proposal={p} choice={choices[p.name]}
                     onChange={(c) => setChoices((prev) => ({ ...prev, [p.name]: c }))} />
        ))}
      </ul>
      <div className="mt-6 flex items-center justify-between gap-4 border-t border-slate-100 pt-6">
        <Button variant="ghost" onClick={onBack}>Start over</Button>
        <Button disabled={watching === 0} onClick={onNext}>
          Next: connections <IconArrowRight className="h-4 w-4" />
        </Button>
      </div>
    </Card>
  );
}

function ConnectionsStep({ proposals, columns, choices, setChoices, busy, onSave, onBack }: {
  proposals: RelationshipProposal[]; columns: Record<string, TableColumnChoice>;
  choices: Record<string, TableRelationshipChoice>; setChoices: (next: Record<string, TableRelationshipChoice>) => void;
  busy: boolean; onSave: () => void; onBack: () => void;
}) {
  const eligible = Object.values(columns).filter((c) => c.monitor && c.type !== "ignore").map((c) => c.name);
  const usable = Object.entries(choices).filter(([, r]) => eligible.includes(r.col_a) && eligible.includes(r.col_b));
  const byKey = Object.fromEntries(proposals.map((p) => [pairKey(p.col_a, p.col_b), p]));
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const select = (value: string, set: (v: string) => void) => (
    <select className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm" value={value} onChange={(e) => set(e.target.value)}>
      <option value="">Pick a column…</option>
      {eligible.map((c) => <option key={c} value={c}>{c}</option>)}
    </select>
  );
  const suggested = usable.filter(([k]) => byKey[k]?.proposed);
  const others = usable.filter(([k]) => !byKey[k]?.proposed);

  const row = ([key, r]: [string, TableRelationshipChoice]) => {
    const p = byKey[key];
    return (
      <li key={key} className={`flex items-start gap-4 rounded-xl border px-4 py-3 transition-colors ${
        r.monitor ? "border-brand-200 bg-brand-50/40" : "border-slate-200 bg-white"}`}>
        <div className="mt-0.5"><Toggle checked={r.monitor} label={`Watch ${r.col_a} and ${r.col_b}`}
                                        onChange={(monitor) => setChoices({ ...choices, [key]: { ...r, monitor } })} /></div>
        <div className="min-w-0">
          <p className="font-medium text-slate-900">{r.col_a} <span className="text-slate-400">↔</span> {r.col_b}</p>
          <p className="text-sm text-slate-500">{p ? connectionSentence(p.kind, r.col_a, r.col_b) : "Added by you"}</p>
          {p && <div className="mt-1"><Details summary="Why we suggested this"><p>{p.reason}</p></Details></div>}
        </div>
      </li>
    );
  };

  return (
    <Card className="p-6 sm:p-8">
      <div className="flex items-start gap-3">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-brand-50 text-brand-600"><IconLink /></span>
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Which columns should keep going together?</h2>
          <p className="mt-1 text-sm leading-relaxed text-slate-500">
            Sometimes each column looks fine on its own, but the way they fit together changes — like bigger homes no
            longer costing more. We found these pairs that clearly go together in your data.
          </p>
        </div>
      </div>

      {suggested.length > 0 ? (
        <ul className="mt-6 space-y-2.5">{suggested.map(row)}</ul>
      ) : (
        <p className="mt-6 rounded-xl bg-slate-50 px-4 py-3 text-sm text-slate-500">
          We didn't find strongly connected columns. That's fine — you can still add a pair below.
        </p>
      )}

      {others.length > 0 && (
        <div className="mt-6">
          <Details summary={`Weaker pairs (${others.length})`}>
            <ul className="mt-2 space-y-2.5">{others.map(row)}</ul>
          </Details>
        </div>
      )}

      <div className="mt-6 flex flex-wrap items-center gap-2 rounded-xl bg-slate-50 px-4 py-3">
        <span className="text-sm text-slate-600">Add your own:</span>
        {select(a, setA)}<span className="text-slate-400">↔</span>{select(b, setB)}
        <Button variant="outline" size="sm" disabled={!a || !b || a === b}
                onClick={() => { setChoices({ ...choices, [pairKey(a, b)]: { col_a: a, col_b: b, monitor: true } }); setA(""); setB(""); }}>
          <IconPlus className="h-4 w-4" /> Add
        </Button>
      </div>

      <div className="mt-8 flex items-center justify-between gap-4 border-t border-slate-100 pt-6">
        <Button variant="ghost" onClick={onBack}>Back</Button>
        <Button disabled={busy} onClick={onSave}>Save baseline <IconCheckCircle className="h-4 w-4" /></Button>
      </div>
    </Card>
  );
}

function DoneStep({ result, onCheck }: { result: TableFitResult; onCheck: () => void }) {
  const watched = Object.values(result.monitored).flat();
  return (
    <Card className="overflow-hidden">
      <div className="bg-gradient-to-br from-ok-50 via-white to-white p-8">
        <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-ok-100 text-ok-600"><IconCheckCircle /></span>
        <h2 className="mt-4 text-xl font-bold text-slate-900">Your baseline is saved</h2>
        <p className="mt-1 text-sm text-slate-500">
          We're now watching {plural(watched.length, "column")}
          {result.duplicate_rows_dropped ? ` (and removed ${plural(result.duplicate_rows_dropped, "duplicate row")})` : ""}.
          Whenever you have new data, check it here.
        </p>
        <div className="mt-5 flex flex-wrap gap-1.5">
          {watched.map((c) => <Badge key={c} tone="brand">{c}</Badge>)}
        </div>
        <Button className="mt-6" onClick={onCheck}>Check new data <IconArrowRight className="h-4 w-4" /></Button>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------- report

const ISSUE_TEXT: Record<string, string> = {
  missing_column: "This column is missing from the new data.",
  unexpected_column: "This column is new — it wasn't in your baseline, so it wasn't checked.",
  dtype_change: "Some values are the wrong kind, like words in a number column.",
  null_rate: "There are more blank values than usual.",
  unseen_categories: "New categories appeared that weren't in your baseline.",
  constant_column: "Every value is the same.",
  invalid_images: "Some photos were missing or couldn't be opened.",
  invalid_text: "Some text values were empty.",
  insufficient_samples: "There weren't enough rows to check this.",
};

function columnSentence(r: ColumnDrift): string {
  if (r.status === "NOT_TESTED") return r.reason ?? "We couldn't check this column in this batch.";
  if (r.status === "STABLE") return "Looks like usual.";
  return {
    numeric: "The values have shifted compared with usual.",
    categorical: "The mix of categories has changed.",
    text: "The kind of text being written has changed.",
    image: "The photos look different from usual.",
    ignore: "",
  }[r.type];
}

function Verdict({ report }: { report: TableReport }) {
  const changed = report.overall.triggered_by;
  const issues = Object.values(report.schema_report).flat().length;
  const status = report.overall.status;
  const tone = status === "DRIFT" ? "alert" : status === "DATA_ISSUES" ? "warn" : "ok";
  const styles = {
    alert: "from-alert-50 border-alert-100 [&_.ic]:bg-alert-100 [&_.ic]:text-alert-600",
    warn: "from-warn-50 border-warn-100 [&_.ic]:bg-warn-100 [&_.ic]:text-warn-600",
    ok: "from-ok-50 border-ok-100 [&_.ic]:bg-ok-100 [&_.ic]:text-ok-600",
  }[tone];
  const title = status === "DRIFT"
    ? `We found changes in ${plural(changed.length, "place")}`
    : status === "DATA_ISSUES" ? "No changes, but some data needs attention" : "Everything looks normal";
  const body = status === "DRIFT"
    ? `Compared with your baseline, these look different: ${changed.map(pairLabel).join(", ")}.`
    : status === "DATA_ISSUES" ? "The data itself looks like usual, but some values were missing or malformed — see below."
      : "This batch looks like the data your model learned from.";
  return (
    <Card className={`animate-fade-up border bg-gradient-to-r to-white p-6 ${styles}`}>
      <div className="flex items-start gap-4">
        <span className="ic flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl">
          {tone === "ok" ? <IconCheckCircle /> : <IconAlert />}
        </span>
        <div>
          <h2 className="text-xl font-bold text-slate-900">{title}</h2>
          <p className="mt-1 text-[15px] leading-relaxed text-slate-600">{body}</p>
          <p className="mt-2 text-xs text-slate-500">
            {plural(report.n_rows, "row")} checked against baseline version {report.baseline_version}
            {issues > 0 && ` · ${plural(issues, "data issue")}`}
          </p>
        </div>
      </div>
    </Card>
  );
}

function ColumnCard({ name, r }: { name: string; r: ColumnDrift }) {
  const tone = r.status === "DRIFT" ? "alert" : r.status === "STABLE" ? "ok" : "slate";
  return (
    <Card className={`p-4 ${r.status === "DRIFT" ? "border-alert-100" : ""}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <TypeIcon type={r.type} />
          <p className="truncate font-medium text-slate-900">{name}</p>
        </div>
        <StatusPill tone={tone}>{r.status === "DRIFT" ? "Changed" : r.status === "STABLE" ? "Normal" : "Not checked"}</StatusPill>
      </div>
      <p className="mt-3 text-sm text-slate-600">{columnSentence(r)}</p>
      <div className="mt-2">
        <Details>
          <p>Test: {r.test}{r.statistic !== undefined ? ` · statistic ${r.statistic.toFixed(3)}` : ""}</p>
          {r.p_value_adjusted != null && <p>Adjusted p-value: {r.p_value_adjusted.toPrecision(3)}</p>}
          {r.threshold_used && <p>Rule: {r.threshold_used}</p>}
          <p>{r.in_family ? "Counted in the batch-wide false-alarm control." : "Not counted in the batch-wide false-alarm control."}</p>
        </Details>
      </div>
    </Card>
  );
}

function ConnectionCard({ name, r }: { name: string; r: RelationshipDrift }) {
  const tone = r.status === "DRIFT" ? (r.report_only ? "warn" : "alert") : r.status === "STABLE" ? "ok" : "slate";
  const label = r.status === "DRIFT" ? (r.report_only ? "Worth a look" : "Changed") : r.status === "STABLE" ? "Normal" : "Not checked";
  const before = r.reference_value != null && r.current_value != null;
  return (
    <Card className={`p-4 ${r.status === "DRIFT" && !r.report_only ? "border-alert-100" : ""}`}>
      <div className="flex items-start justify-between gap-3">
        <p className="font-medium text-slate-900">{pairLabel(name)}</p>
        <StatusPill tone={tone}>{label}</StatusPill>
      </div>
      <p className="mt-1 text-sm text-slate-500">{connectionSentence(r.kind, ...(name.split("<->") as [string, string]))}</p>
      {r.status === "DRIFT" && (
        <p className="mt-3 text-sm text-slate-600">
          {r.report_only ? "This might have changed. It's shown for information and doesn't raise an alert." :
            "These columns no longer go together the way they used to."}
        </p>
      )}
      {r.status === "NOT_TESTED" && <p className="mt-3 text-sm text-slate-600">{r.reason}</p>}
      {before && (
        <div className="mt-3 flex items-center gap-3 text-sm">
          <span className="rounded-lg bg-slate-100 px-2 py-1 font-mono text-slate-600">{r.reference_value!.toFixed(2)}</span>
          <IconArrowRight className="h-4 w-4 text-slate-400" />
          <span className={`rounded-lg px-2 py-1 font-mono ${r.status === "DRIFT" ? "bg-alert-50 text-alert-700" : "bg-slate-100 text-slate-600"}`}>
            {r.current_value!.toFixed(2)}
          </span>
          <span className="text-xs text-slate-400">connection strength, before → now</span>
        </div>
      )}
      {r.confounded_by && r.confounded_by.length > 0 && (
        <p className="mt-2 text-xs text-warn-700">{r.confounded_by.join(", ")} also changed on its own, which may explain this.</p>
      )}
      <div className="mt-2">
        <Details>
          <p>Measured as: {r.statistic_name}</p>
          {r.explanation && <p>{r.explanation}</p>}
          {r.p_value_adjusted != null && <p>Adjusted p-value: {r.p_value_adjusted.toPrecision(3)}</p>}
          {r.p_value != null && r.p_value_adjusted == null && <p>p-value: {r.p_value.toPrecision(3)}</p>}
        </Details>
      </div>
    </Card>
  );
}

function Report({ report }: { report: TableReport }) {
  const columns = Object.entries(report.column_drift).sort(([, a], [, b]) => Number(b.status === "DRIFT") - Number(a.status === "DRIFT"));
  const connections = Object.entries(report.relationship_drift).sort(([, a], [, b]) => Number(b.status === "DRIFT") - Number(a.status === "DRIFT"));
  const issues = Object.entries(report.schema_report).flatMap(([col, list]) => list.map((i) => ({ col, ...i })));
  const emerged = report.screening?.emerged_dependencies ?? [];
  return (
    <div className="space-y-8">
      <Verdict report={report} />

      <section>
        <h3 className="mb-3 font-semibold text-slate-900">Columns</h3>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {columns.map(([name, r]) => <ColumnCard key={name} name={name} r={r} />)}
        </div>
      </section>

      {connections.length > 0 && (
        <section>
          <h3 className="mb-3 font-semibold text-slate-900">Connections between columns</h3>
          <div className="grid gap-3 lg:grid-cols-2">
            {connections.map(([name, r]) => <ConnectionCard key={name} name={name} r={r} />)}
          </div>
        </section>
      )}

      {issues.length > 0 && (
        <section>
          <h3 className="mb-3 font-semibold text-slate-900">Data that needs attention</h3>
          <Card className="divide-y divide-slate-100">
            {issues.map((i, n) => (
              <div key={n} className="flex items-start gap-3 px-5 py-3">
                <IconAlert className={`mt-0.5 h-4 w-4 shrink-0 ${i.severity === "alert" ? "text-alert-600" : "text-warn-600"}`} />
                <p className="text-sm text-slate-700"><span className="font-medium">{i.col}:</span> {ISSUE_TEXT[i.issue] ?? i.issue.replace(/_/g, " ")}</p>
              </div>
            ))}
          </Card>
        </section>
      )}

      {emerged.length > 0 && (
        <Card className="flex items-start gap-3 border-brand-100 bg-brand-50/50 p-5">
          <IconSparkles className="mt-0.5 h-5 w-5 shrink-0 text-brand-600" />
          <p className="text-sm text-slate-700">
            <span className="font-medium">New pattern spotted:</span>{" "}
            {emerged.map((d) => `${d.col_a} and ${d.col_b}`).join(", ")} now go together, though they didn't before.
            If that's expected, update your baseline to start watching it.
          </p>
        </Card>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- page

export function TableWorkflowPage() {
  const { projectId = "" } = useParams();
  const queryClient = useQueryClient();
  const baseline = useQuery({ queryKey: ["baseline", projectId], queryFn: () => api.getBaseline(projectId), retry: false });
  const hasTableBaseline = baseline.data?.modality === "table";

  const [mode, setMode] = useState<"check" | "setup">("setup");
  useEffect(() => { if (hasTableBaseline) setMode("check"); }, [hasTableBaseline]);

  const [step, setStep] = useState(0);
  const [profile, setProfile] = useState<TableProfile | null>(null);
  const [choices, setChoices] = useState<Record<string, TableColumnChoice>>({});
  const [relChoices, setRelChoices] = useState<Record<string, TableRelationshipChoice>>({});
  const [fitResult, setFitResult] = useState<TableFitResult | null>(null);
  const [batch, setBatch] = useState<File[]>([]);
  const [batchZip, setBatchZip] = useState<File[]>([]);
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

  const startProfile = (table: File, zip?: File) => run(async () => {
    const accepted = await api.stageTable(projectId, table, zip);
    const result = await waitForJob<TableProfile>(accepted.job_id, setJob);
    setProfile(result);
    setChoices(Object.fromEntries(result.columns.map((c) => [c.name, { name: c.name, type: c.proposed_type, monitor: c.proposed_monitor }])));
    setRelChoices(Object.fromEntries(result.relationships.map((r) => [pairKey(r.col_a, r.col_b), { col_a: r.col_a, col_b: r.col_b, monitor: r.proposed }])));
    setStep(1);
  });

  const save = () => run(async () => {
    if (!profile) return;
    // Any pair of watched columns may be sent; the server rejects unsupported combinations (e.g. text ↔ text).
    const watched = (name: string) => choices[name]?.monitor && choices[name].type !== "ignore";
    const relationships = Object.values(relChoices).filter((r) => r.monitor && watched(r.col_a) && watched(r.col_b));
    const accepted = await api.fitTable(projectId, profile.stage_id, Object.values(choices), relationships);
    setFitResult(await waitForJob<TableFitResult>(accepted.job_id, setJob));
    setStep(3);
    queryClient.invalidateQueries({ queryKey: ["baseline", projectId] });
    queryClient.invalidateQueries({ queryKey: ["tableBaseline", projectId] });
  });

  const analyze = () => run(async () => {
    setReport(null);
    const accepted = await api.analyzeTable(projectId, batch[0], batchZip[0]);
    setReport(await waitForJob<TableReport>(accepted.job_id, setJob));
    queryClient.invalidateQueries({ queryKey: ["history", projectId] });
  });

  const restart = () => { setStep(0); setProfile(null); setFitResult(null); };
  const progressText = useMemo(() => (mode === "check" ? "Comparing with your baseline" : step === 2 ? "Saving your baseline" : "Reading your data"), [mode, step]);

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Check my data"
        title={mode === "check" ? "Check new data" : hasTableBaseline ? "Update your baseline" : "Set up this project"}
        subtitle={mode === "check"
          ? "Upload a recent batch. We'll compare it with your baseline and tell you, in plain words, what changed."
          : "Show us what normal looks like. It takes a few minutes and you can change anything we suggest."}
        actions={hasTableBaseline ? (
          <Tabs value={mode} onChange={(m) => { setMode(m); setError(null); }}
                options={[{ value: "check", label: "Check new data" }, { value: "setup", label: "Update baseline" }]} />
        ) : undefined}
      />

      {error && <ErrorBanner message={error} />}

      {mode === "setup" && (
        <>
          <Card className="px-6 py-4"><Stepper steps={SETUP_STEPS} current={step} /></Card>
          {busy && <Progress job={job} fallback={progressText} />}
          {!busy && step === 0 && <UploadStep busy={busy} onStart={startProfile} />}
          {!busy && step === 1 && profile && (
            <ColumnsStep profile={profile} choices={choices} setChoices={setChoices} onNext={() => setStep(2)} onBack={restart} />
          )}
          {!busy && step === 2 && profile && (
            <ConnectionsStep proposals={profile.relationships} columns={choices} choices={relChoices}
                             setChoices={setRelChoices} busy={busy} onSave={save} onBack={() => setStep(1)} />
          )}
          {!busy && step === 3 && fitResult && <DoneStep result={fitResult} onCheck={() => { setMode("check"); restart(); }} />}
        </>
      )}

      {mode === "check" && (
        <>
          <Card className="p-6 sm:p-8">
            <div className="grid gap-6 lg:grid-cols-2">
              <div>
                <p className="mb-2 text-sm font-medium text-slate-700">New data</p>
                <FileInput accept={TABULAR_ACCEPT} files={batch} onFiles={setBatch} prompt="Choose a spreadsheet" />
                <p className="mt-2 text-xs text-slate-500">Same columns as your baseline. A few hundred rows or more gives the clearest answer.</p>
              </div>
              <div>
                <p className="mb-2 text-sm font-medium text-slate-700">Photos <span className="font-normal text-slate-400">(optional)</span></p>
                <FileInput accept=".zip" files={batchZip} onFiles={setBatchZip} prompt="Choose a ZIP of photos" />
                <p className="mt-2 text-xs text-slate-500">Only if your data has a photo column.</p>
              </div>
            </div>
            <div className="mt-6 flex justify-end border-t border-slate-100 pt-6">
              <Button disabled={busy || batch.length === 0} onClick={analyze}>Run the check <IconArrowRight className="h-4 w-4" /></Button>
            </div>
          </Card>
          {busy && <Progress job={job} fallback={progressText} />}
          {report && !busy && <Report report={report} />}
        </>
      )}
    </div>
  );
}

