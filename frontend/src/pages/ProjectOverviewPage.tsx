import { useQuery } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import { Badge, Button, Card, EmptyState, ErrorBanner, PageHeader, Spinner, StatCard } from "../components/ui";
import { IconAlert, IconArrowRight, IconCheckCircle, IconClock, IconLayers, IconLink, IconTable, IconUpload } from "../components/icons";
import { COLUMN_TYPES, TypeIcon, connectionSentence } from "../components/vocab";
import { api, ApiError } from "../lib/api";
import { plural, timeAgo } from "../lib/format";
import type { HistoryRun } from "../lib/types";

function drifted(run: HistoryRun): string[] {
  const pick = (m: Record<string, { drift_detected: boolean }> = {}) =>
    Object.entries(m).filter(([, v]) => v.drift_detected).map(([k]) => k.replace("<->", " ↔ "));
  return [...pick(run.feature_metrics), ...pick(run.relationship_metrics)];
}

function LatestCheck({ run, onCheck }: { run?: HistoryRun; onCheck: () => void }) {
  if (!run) {
    return (
      <Card className="flex flex-col gap-5 p-6 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-4">
          <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-brand-50 text-brand-600"><IconUpload /></span>
          <div>
            <p className="text-lg font-semibold text-slate-900">Ready for its first check</p>
            <p className="mt-1 text-sm text-slate-500">Upload a recent batch of data to see whether anything has changed.</p>
          </div>
        </div>
        <Button onClick={onCheck}>Check new data <IconArrowRight className="h-4 w-4" /></Button>
      </Card>
    );
  }
  const changed = drifted(run);
  const alert = run.system_alert;
  return (
    <Card className={`overflow-hidden ${alert ? "border-alert-100" : "border-ok-100"}`}>
      <div className={`flex flex-col gap-5 p-6 sm:flex-row sm:items-center sm:justify-between ${
        alert ? "bg-gradient-to-r from-alert-50 to-white" : "bg-gradient-to-r from-ok-50 to-white"}`}>
        <div className="flex items-start gap-4">
          <span className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl ${
            alert ? "bg-alert-100 text-alert-600" : "bg-ok-100 text-ok-600"}`}>
            {alert ? <IconAlert /> : <IconCheckCircle />}
          </span>
          <div>
            <p className="text-lg font-semibold text-slate-900">
              {alert ? "Your last check found changes" : "Your last check looked normal"}
            </p>
            <p className="mt-1 text-sm text-slate-500">
              {timeAgo(run.ts)} · {plural(run.batch_size ?? 0, "row")} checked
              {alert && changed.length > 0 && <> · changed: <span className="font-medium text-slate-700">{changed.slice(0, 4).join(", ")}{changed.length > 4 ? "…" : ""}</span></>}
            </p>
          </div>
        </div>
        <Button onClick={onCheck}>Check new data <IconArrowRight className="h-4 w-4" /></Button>
      </div>
    </Card>
  );
}

export function ProjectOverviewPage() {
  const { projectId = "" } = useParams();
  const navigate = useNavigate();
  const go = (path: string) => navigate(`/projects/${encodeURIComponent(projectId)}/${path}`);

  const baseline = useQuery({ queryKey: ["baseline", projectId], queryFn: () => api.getBaseline(projectId), retry: false });
  const isTable = baseline.data?.modality === "table";
  const table = useQuery({ queryKey: ["tableBaseline", projectId], queryFn: () => api.getTableBaseline(projectId),
                           enabled: isTable, retry: false });
  const history = useQuery({ queryKey: ["history", projectId, "latest"], queryFn: () => api.getHistory(projectId, { limit: 1 }),
                             enabled: baseline.isSuccess, retry: false });

  const notSetUp = baseline.error instanceof ApiError && baseline.error.status === 404;
  const watched = table.data?.schema.filter((c) => c.final_monitor) ?? [];
  const connections = table.data?.relationships.filter((r) => r.final_monitor) ?? [];

  return (
    <div className="space-y-8">
      <PageHeader eyebrow="Project" title={decodeURIComponent(projectId)}
                  subtitle="A quick look at what this project watches and how your latest data compared." />

      {baseline.isLoading && <div className="flex justify-center py-16 text-slate-400"><Spinner className="h-6 w-6" /></div>}
      {baseline.error && !notSetUp && (
        <ErrorBanner message={baseline.error instanceof ApiError ? baseline.error.detail : "We couldn't load this project."} />
      )}

      {notSetUp && (
        <EmptyState
          icon={<IconTable />}
          title="This project isn't set up yet"
          description="Show us the data your model learned from. We'll suggest what to watch, you confirm, and you're ready to check new data."
          action={<Button onClick={() => go("table")}>Set up this project <IconArrowRight className="h-4 w-4" /></Button>}
        />
      )}

      {baseline.data && (
        <>
          <LatestCheck run={history.data?.runs[0]} onCheck={() => go("table")} />

          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard label="Columns watched" value={isTable ? watched.length : Object.keys(baseline.data.feature_types).length}
                      icon={<IconTable className="h-[18px] w-[18px]" />} />
            <StatCard label="Connections watched" value={isTable ? connections.length : "—"}
                      icon={<IconLink className="h-[18px] w-[18px]" />} hint={!isTable ? "Available in table projects" : undefined} />
            <StatCard label="Checks so far" value={history.data?.total ?? "—"} icon={<IconClock className="h-[18px] w-[18px]" />} />
            <StatCard label="Baseline" value={table.data ? `Version ${table.data.version}` : "Saved"}
                      hint={table.data ? `${plural(table.data.reference_rows, "row")} of usual data` : undefined}
                      icon={<IconLayers className="h-[18px] w-[18px]" />} />
          </div>

          {isTable && table.data && (
            <div className="grid gap-6 lg:grid-cols-5">
              <Card className="p-6 lg:col-span-3">
                <h2 className="font-semibold text-slate-900">What we're watching</h2>
                <p className="mt-1 text-sm text-slate-500">Each column is compared with how it usually looks.</p>
                <ul className="mt-5 divide-y divide-slate-100">
                  {watched.map((c) => (
                    <li key={c.column_name} className="flex items-center gap-3 py-2.5">
                      <TypeIcon type={c.final_type} />
                      <span className="flex-1 truncate text-sm font-medium text-slate-800">{c.column_name}</span>
                      <Badge>{COLUMN_TYPES[c.final_type].label}</Badge>
                    </li>
                  ))}
                </ul>
              </Card>
              <Card className="p-6 lg:col-span-2">
                <h2 className="font-semibold text-slate-900">Connections we're watching</h2>
                <p className="mt-1 text-sm text-slate-500">Pairs of columns that should keep going together.</p>
                {connections.length === 0 ? (
                  <p className="mt-5 text-sm text-slate-400">None yet — you can add some the next time you update the baseline.</p>
                ) : (
                  <ul className="mt-5 space-y-3">
                    {connections.map((r) => (
                      <li key={`${r.col_a}-${r.col_b}`} className="rounded-xl bg-slate-50 px-3.5 py-2.5">
                        <p className="text-sm font-medium text-slate-800">{r.col_a} ↔ {r.col_b}</p>
                        <p className="text-xs text-slate-500">{connectionSentence(r.kind, r.col_a, r.col_b)}</p>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
            </div>
          )}

          {!isTable && (
            <Card className="p-6">
              <h2 className="font-semibold text-slate-900">This project uses an older setup</h2>
              <p className="mt-1 text-sm text-slate-500">
                It watches <span className="font-medium capitalize">{baseline.data.modality}</span> data with the older tools
                (under “Advanced tools” in the menu). Setting it up again with “Check my data” adds connections between columns
                and a friendlier report.
              </p>
              <Button variant="outline" className="mt-4" onClick={() => go("table")}>Set up with the new workflow</Button>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
