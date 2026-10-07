import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Card, ErrorBanner, PageHeader, Spinner, StatusPill, TextInput } from "../components/ui";
import { IconArrowRight, IconCheckCircle, IconLayers, IconPlus, IconTrash, IconUpload } from "../components/icons";
import { api, ApiError } from "../lib/api";
import { useAuth } from "../lib/auth";
import { greeting, timeAgo } from "../lib/format";

function HowItWorks() {
  const steps: [ReactNode, string, string][] = [
    [<IconUpload key="u" />, "Show us your usual data", "Upload the data your model learned from — a spreadsheet, plus photos if you have them."],
    [<IconCheckCircle key="c" />, "Confirm what each column is", "We suggest whether each column is numbers, categories, text or images. You approve."],
    [<IconLayers key="l" />, "Check new data any time", "Upload a fresh batch and get a plain-language report of anything that changed."],
  ];
  return (
    <div className="grid gap-4 sm:grid-cols-3">
      {steps.map(([icon, title, body], i) => (
        <Card key={title} className="p-5">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-50 text-brand-600 [&>svg]:h-[18px] [&>svg]:w-[18px]">
              {icon}
            </span>
            <span className="text-xs font-semibold text-slate-400">Step {i + 1}</span>
          </div>
          <p className="mt-3 font-semibold text-slate-900">{title}</p>
          <p className="mt-1 text-sm leading-relaxed text-slate-500">{body}</p>
        </Card>
      ))}
    </div>
  );
}

function ProjectStatus({ projectId }: { projectId: string }) {
  const baseline = useQuery({ queryKey: ["baseline", projectId], queryFn: () => api.getBaseline(projectId), retry: false });
  const history = useQuery({
    queryKey: ["history", projectId, "latest"],
    queryFn: () => api.getHistory(projectId, { limit: 1 }),
    enabled: baseline.isSuccess,
    retry: false,
  });

  if (baseline.isLoading) return <p className="text-xs text-slate-400">Checking…</p>;
  if (baseline.isError) return <StatusPill tone="slate">Not set up yet</StatusPill>;
  const last = history.data?.runs[0];
  if (!last) return <StatusPill tone="brand">Ready for its first check</StatusPill>;
  return (
    <div className="flex flex-wrap items-center gap-2">
      {last.system_alert ? <StatusPill tone="alert">Changes found</StatusPill> : <StatusPill tone="ok">Looked normal</StatusPill>}
      <span className="text-xs text-slate-400">last check {timeAgo(last.ts)}</span>
    </div>
  );
}

export function ProjectsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { session } = useAuth();
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);

  const { data, isLoading, error } = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const deleteMutation = useMutation({
    mutationFn: (projectId: string) => api.deleteProject(projectId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["projects"] }),
  });

  const create = () => {
    const id = name.trim().replace(/\s+/g, "-");
    if (id) navigate(`/projects/${encodeURIComponent(id)}/table`);
  };
  const firstName = session?.name?.split(" ")[0];
  const hasProjects = !!data && data.projects.length > 0;

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow={`${greeting()}${firstName ? `, ${firstName}` : ""}`}
        title="Your projects"
        subtitle="Each project watches one dataset — for example, the inputs to one model — and tells you when new data stops looking like it."
        actions={<Button onClick={() => setCreating(true)}><IconPlus className="h-4 w-4" /> New project</Button>}
      />

      {creating && (
        <Card className="animate-fade-up p-6">
          <p className="font-semibold text-slate-900">Name your project</p>
          <p className="mt-1 text-sm text-slate-500">Something you'll recognise later, like “house-prices” or “support-tickets”.</p>
          <div className="mt-4 flex flex-col gap-2 sm:flex-row">
            <TextInput autoFocus placeholder="e.g. house-prices" value={name}
                       onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && create()} />
            <div className="flex gap-2">
              <Button onClick={create} disabled={!name.trim()}>Create <IconArrowRight className="h-4 w-4" /></Button>
              <Button variant="ghost" onClick={() => setCreating(false)}>Cancel</Button>
            </div>
          </div>
        </Card>
      )}

      {isLoading && <div className="flex justify-center py-16 text-slate-400"><Spinner className="h-6 w-6" /></div>}
      {error && <ErrorBanner message={error instanceof ApiError ? error.detail : "We couldn't load your projects."} />}

      {hasProjects && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data!.projects.map((pid) => (
            <Card key={pid} role="button" tabIndex={0}
                  onClick={() => navigate(`/projects/${encodeURIComponent(pid)}`)}
                  onKeyDown={(e) => e.key === "Enter" && navigate(`/projects/${encodeURIComponent(pid)}`)}
                  className="group cursor-pointer p-5 transition-all hover:-translate-y-0.5 hover:border-brand-200 hover:shadow-[var(--shadow-lift)]">
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-center gap-3">
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-brand-50 to-brand-100 font-display text-sm font-bold uppercase text-brand-700">
                    {pid.slice(0, 2)}
                  </span>
                  <p className="truncate font-semibold text-slate-900">{pid}</p>
                </div>
                <button
                  className="rounded-lg p-1.5 text-slate-300 opacity-0 transition-opacity hover:bg-alert-50 hover:text-alert-600 group-hover:opacity-100 focus:opacity-100"
                  aria-label={`Delete ${pid}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (confirm(`Delete “${pid}”? Its saved baselines and history will be removed for good.`)) {
                      deleteMutation.mutate(pid);
                    }
                  }}
                >
                  <IconTrash className="h-4 w-4" />
                </button>
              </div>
              <div className="mt-4"><ProjectStatus projectId={pid} /></div>
              <p className="mt-4 flex items-center gap-1 text-sm font-medium text-brand-600 opacity-0 transition-opacity group-hover:opacity-100">
                Open project <IconArrowRight className="h-4 w-4" />
              </p>
            </Card>
          ))}
        </div>
      )}

      {data && !hasProjects && (
        <Card className="overflow-hidden">
          <div className="bg-gradient-to-br from-brand-600 to-brand-700 px-8 py-10 text-white">
            <h2 className="text-2xl font-bold">Let's set up your first project</h2>
            <p className="mt-2 max-w-xl text-brand-100">
              It takes a few minutes. You'll need the data your model was trained on — a CSV or Excel file is perfect.
            </p>
            <Button variant="outline" className="mt-6 border-white/30 bg-white text-brand-700 hover:bg-brand-50"
                    onClick={() => setCreating(true)}>
              <IconPlus className="h-4 w-4" /> Create a project
            </Button>
          </div>
        </Card>
      )}

      <section>
        <h2 className="mb-4 text-lg font-semibold text-slate-900">How it works</h2>
        <HowItWorks />
      </section>
    </div>
  );
}
