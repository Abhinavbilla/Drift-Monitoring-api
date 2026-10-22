import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import type { ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { CodeBlock, CopyButton } from "../components/CodeBlock";
import { IconAlert, IconArrowRight, IconBell, IconKey, IconLayers, IconUpload } from "../components/icons";
import { Badge, Button, Card, ErrorBanner, PageHeader, Spinner, TextInput } from "../components/ui";
import { ApiError, PUBLIC_API_URL, api, type AccessToken } from "../lib/api";
import { timeAgo } from "../lib/format";

const REPO_CLIENT = "git+https://github.com/Abhinavbilla/Drift-Monitoring-api.git#subdirectory=clients/python";

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <section className="relative pl-12">
      <span className="absolute left-0 top-0 flex h-8 w-8 items-center justify-center rounded-full bg-brand-600 text-sm font-bold text-white">
        {n}
      </span>
      <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
      <div className="mt-3 space-y-4 text-[15px] leading-relaxed text-slate-600">{children}</div>
    </section>
  );
}

function FlowCard({ icon, when, title, body }: { icon: ReactNode; when: string; title: string; body: string }) {
  return (
    <Card className="p-5">
      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-50 text-brand-600 [&>svg]:h-[18px] [&>svg]:w-[18px]">{icon}</span>
      <p className="mt-4 text-xs font-semibold uppercase tracking-wider text-slate-400">{when}</p>
      <p className="mt-1 font-semibold text-slate-900">{title}</p>
      <p className="mt-1.5 text-sm leading-relaxed text-slate-500">{body}</p>
    </Card>
  );
}

function scopeLabel(t: AccessToken) {
  return !t.project_scope || t.project_scope.includes("*") ? "All models" : t.project_scope.join(", ");
}

function tokenState(t: AccessToken): { label: string; tone: "ok" | "slate" | "alert" } {
  if (t.revoked) return { label: "Revoked", tone: "slate" };
  if (t.expires_at && new Date(t.expires_at) <= new Date()) return { label: "Expired", tone: "alert" };
  return { label: "Active", tone: "ok" };
}

function TokenManager({ projects, project }: { projects: string[]; project: string }) {
  const queryClient = useQueryClient();
  const tokens = useQuery({ queryKey: ["tokens"], queryFn: api.listTokens });
  const [name, setName] = useState("");
  const [scope, setScope] = useState<"all" | "project">("all");
  const [days, setDays] = useState(90);
  const [created, setCreated] = useState<string | null>(null);
  const canScope = projects.includes(project);

  const create = useMutation({
    mutationFn: () => api.createToken(name.trim(), days, scope === "project" && canScope ? [project] : undefined),
    onSuccess: (resp) => {
      setCreated(resp.token);
      setName("");
      queryClient.invalidateQueries({ queryKey: ["tokens"] });
    },
  });
  const revoke = useMutation({
    mutationFn: (id: string) => api.revokeToken(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["tokens"] }),
  });
  const list = tokens.data?.tokens ?? [];

  return (
    <Card className="overflow-hidden">
      <form className="grid gap-3 p-5 sm:grid-cols-[1fr_auto_auto_auto] sm:items-end"
            onSubmit={(e) => { e.preventDefault(); setCreated(null); create.mutate(); }}>
        <label className="block">
          <span className="mb-1.5 block text-sm font-medium text-slate-700">Token name</span>
          <TextInput value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. nightly-pipeline" maxLength={60} />
        </label>
        <label className="block">
          <span className="mb-1.5 block text-sm font-medium text-slate-700">Can access</span>
          <select value={canScope ? scope : "all"} onChange={(e) => setScope(e.target.value as "all" | "project")}
                  className="h-[42px] rounded-xl border border-slate-300 bg-white px-3 text-sm text-slate-800 focus:border-brand-500 focus:outline-none focus:ring-4 focus:ring-brand-100">
            <option value="all">All my models</option>
            {canScope && <option value="project">Only “{project}”</option>}
          </select>
        </label>
        <label className="block">
          <span className="mb-1.5 block text-sm font-medium text-slate-700">Expires in</span>
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}
                  className="h-[42px] rounded-xl border border-slate-300 bg-white px-3 text-sm text-slate-800 focus:border-brand-500 focus:outline-none focus:ring-4 focus:ring-brand-100">
            <option value={30}>30 days</option>
            <option value={90}>90 days</option>
            <option value={365}>1 year</option>
          </select>
        </label>
        <Button type="submit" disabled={!name.trim() || create.isPending}>
          {create.isPending ? <Spinner className="h-4 w-4" /> : <IconKey className="h-4 w-4" />} Create token
        </Button>
      </form>

      {create.error && (
        <div className="px-5 pb-4">
          <ErrorBanner message={create.error instanceof ApiError ? create.error.detail : "We couldn't create the token."} />
        </div>
      )}

      {created && (
        <div className="border-t border-ok-100 bg-ok-50 px-5 py-4">
          <p className="text-sm font-semibold text-ok-700">Your new token. Copy it now — it won't be shown again.</p>
          <div className="mt-2 flex items-center gap-2 rounded-lg border border-ok-100 bg-white px-3 py-2">
            <code className="min-w-0 flex-1 truncate font-mono text-[13px] text-slate-800">{created}</code>
            <CopyButton text={created} className="text-slate-500 hover:bg-slate-100 hover:text-slate-800" />
          </div>
          <p className="mt-2 text-xs text-ok-700">
            Keep it somewhere secret, like an environment variable named <code className="font-mono">DRIFT_TOKEN</code> — never in code you commit.
          </p>
        </div>
      )}

      <div className="border-t border-slate-100">
        {tokens.isLoading && <div className="flex justify-center py-6 text-slate-400"><Spinner className="h-5 w-5" /></div>}
        {tokens.data && list.length === 0 && (
          <p className="px-5 py-5 text-sm text-slate-500">No tokens yet. Create one above for each script or service that sends data.</p>
        )}
        {list.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="px-5 py-2.5 font-semibold">Name</th>
                  <th className="px-3 py-2.5 font-semibold">Access</th>
                  <th className="px-3 py-2.5 font-semibold">Last used</th>
                  <th className="px-3 py-2.5 font-semibold">Expires</th>
                  <th className="px-3 py-2.5 font-semibold">Status</th>
                  <th className="px-5 py-2.5" />
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {list.map((t) => {
                  const state = tokenState(t);
                  return (
                    <tr key={t.id} className={state.label === "Active" ? "" : "text-slate-400"}>
                      <td className="px-5 py-3">
                        <p className="font-medium text-slate-800">{t.name}</p>
                        <p className="font-mono text-xs text-slate-400">dm_{t.prefix}_…</p>
                      </td>
                      <td className="px-3 py-3">{scopeLabel(t)}</td>
                      <td className="px-3 py-3">{t.last_used_at ? timeAgo(t.last_used_at) : "Never"}</td>
                      <td className="px-3 py-3">{t.expires_at ? new Date(t.expires_at).toLocaleDateString() : "Never"}</td>
                      <td className="px-3 py-3"><Badge tone={state.tone}>{state.label}</Badge></td>
                      <td className="px-5 py-3 text-right">
                        {state.label === "Active" && (
                          <button type="button" disabled={revoke.isPending}
                                  onClick={() => confirm(`Revoke “${t.name}”? Anything using it will stop working immediately.`) && revoke.mutate(t.id)}
                                  className="text-sm font-medium text-alert-600 hover:text-alert-700 disabled:opacity-50">
                            Revoke
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Card>
  );
}

export function ConnectPage() {
  const [params, setParams] = useSearchParams();
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const list = projects.data?.projects ?? [];
  // What's typed stays as typed (even empty); the snippets fall back to a sensible name.
  const [typed, setTyped] = useState(params.get("project") ?? "");
  const project = typed || list[0] || "my-model";
  const type = (value: string) => {
    const id = value.replace(/\s+/g, "-");  // same rule as "Add model"
    setTyped(id);
    setParams(id ? { project: id } : {}, { replace: true });
  };
  const today = new Date().toLocaleDateString("en-CA");  // local YYYY-MM-DD

  const python = {
    install: `pip install "${REPO_CLIENT}"`,
    baseline: `import os
import pandas as pd
from drift_monitor_client import DriftClient

client = DriftClient("${PUBLIC_API_URL}", token=os.environ["DRIFT_TOKEN"])

# The data your model was trained on
train = pd.read_csv("training_data.csv")

# We look at every column and propose how to watch it
profile = client.profile_table("${project}", train)
for col in profile["columns"]:
    print(col["name"], "->", col["proposed_type"], "|", col["reason"])

# Accept the proposals and lock the baseline
client.fit_table("${project}", profile)`,
    batch: `def check_batch(batch: pd.DataFrame, batch_id: str) -> dict:
    """Call this wherever your model receives new data: a nightly job,
    an Airflow task, or right after a batch of predictions."""
    report = client.analyze_table("${project}", batch, idempotency_key=batch_id)

    overall = report["overall"]
    if overall["alert"]:
        print("Data has changed in:", ", ".join(overall["triggered_by"]))
    return report


check_batch(pd.read_parquet("todays_inputs.parquet"), batch_id="${today}")`,
  };

  const curl = {
    baseline: `export DRIFT_TOKEN="dm_..."   # from step 1
API="${PUBLIC_API_URL}"

# 1. Upload the training table -> {"job_id": "...", "stage_id": "..."}
curl -s -H "Authorization: Bearer $DRIFT_TOKEN" \\
     -F file=@training_data.csv "$API/tables/${project}/stage"

# 2. Poll until "status" is "succeeded"; "result" proposes a type per column
curl -s -H "Authorization: Bearer $DRIFT_TOKEN" "$API/jobs/<job_id>"

# 3. Lock the baseline. List every column; use "ignore" for IDs.
curl -s -H "Authorization: Bearer $DRIFT_TOKEN" -H "Content-Type: application/json" \\
     -d '{"stage_id": "<stage_id>",
          "columns": [{"name": "age",     "type": "numeric",     "monitor": true},
                      {"name": "country", "type": "categorical", "monitor": true},
                      {"name": "user_id", "type": "ignore",      "monitor": false}],
          "relationships": []}' \\
     "$API/tables/${project}/fit"`,
    batch: `# Send a batch -> {"job_id": "..."}. The Idempotency-Key makes retries safe.
curl -s -H "Authorization: Bearer $DRIFT_TOKEN" -H "Idempotency-Key: ${today}" \\
     -F file=@todays_inputs.csv "$API/tables/${project}/analyze"

# Then poll the job; when it succeeds, "result" is the full report
curl -s -H "Authorization: Bearer $DRIFT_TOKEN" "$API/jobs/<job_id>"`,
  };

  const report = `{
  "overall": { "status": "DRIFT", "alert": true, "triggered_by": ["country"] },
  "column_drift": {
    "age":     { "type": "numeric",     "status": "STABLE", "p_value_adjusted": 0.82 },
    "country": { "type": "categorical", "status": "DRIFT",  "p_value_adjusted": 0.001 }
  },
  "relationship_drift": { ... },
  "schema_report": { ... }
}`;

  return (
    <div className="space-y-10">
      <PageHeader
        eyebrow="For developers"
        title="Connect your model"
        subtitle="This dashboard is where you see results. In production, your own code sends the data automatically — here's how to set that up in about ten minutes."
      />

      <div className="grid gap-4 md:grid-cols-3">
        <FlowCard icon={<IconLayers />} when="Once" title="Send your training data"
                  body="We learn what normal looks like for every column, and for pairs of columns that move together." />
        <FlowCard icon={<IconUpload />} when="Every batch" title="Send new inputs"
                  body="Each time your model gets new data — hourly, daily, per release — your pipeline sends a copy here." />
        <FlowCard icon={<IconBell />} when="When it changes" title="Get told, then look"
                  body="A webhook or email fires when the data drifts. The full report and history appear in this dashboard." />
      </div>

      <Card className="flex flex-wrap items-center gap-x-6 gap-y-3 p-5">
        <label className="flex items-center gap-3 text-sm">
          <span className="font-medium text-slate-700">Model</span>
          <TextInput list="connect-projects" value={typed} onChange={(e) => type(e.target.value)}
                     className="w-56" placeholder={list[0] || "my-model"} />
          <datalist id="connect-projects">{list.map((p) => <option key={p} value={p} />)}</datalist>
        </label>
        <div className="flex min-w-0 items-center gap-2 text-sm">
          <span className="font-medium text-slate-700">API address</span>
          <code className="truncate rounded-lg bg-slate-100 px-2.5 py-1 font-mono text-[13px] text-slate-700">{PUBLIC_API_URL}</code>
          <CopyButton text={PUBLIC_API_URL} className="text-slate-500 hover:bg-slate-100 hover:text-slate-800" />
        </div>
        <p className="w-full text-xs text-slate-400">The examples below fill in this name and address for you; in the API it's the <code className="font-mono">project_id</code>. A new name is created on its first upload.</p>
      </Card>

      <div className="space-y-10">
        <Step n={1} title="Create an access token">
          <p>
            Scripts can't sign in with Google or a password, so they use a token instead. Make one per script or
            service — if one leaks, revoke just that one.
          </p>
          <TokenManager projects={list} project={project} />
        </Step>

        <Step n={2} title="Install the Python client">
          <p>It handles uploads, waiting for results and retries. Prefer plain HTTP? Every step has a <b>curl</b> tab too.</p>
          <CodeBlock snippets={[{ label: "Terminal", code: python.install }]} />
        </Step>

        <Step n={3} title="Set the baseline (once)">
          <p>
            Send the data your model was trained on. Do this again whenever you retrain — older baselines are kept
            under <i>Saved baselines</i>.
          </p>
          <CodeBlock snippets={[{ label: "Python", code: python.baseline }, { label: "curl", code: curl.baseline }]} />
          <p className="text-sm text-slate-500">
            Want to review the column types by eye first? Do this step in the dashboard instead, with{" "}
            {list.includes(project)
              ? <Link className="font-medium text-brand-600 hover:text-brand-700" to={`/projects/${encodeURIComponent(project)}/table`}>Check my data</Link>
              : <b>Check my data</b>}
            , then automate step 4 only.
          </p>
        </Step>

        <Step n={4} title="Check every new batch">
          <p>
            We never fetch data on our own: each call is one batch, sent whenever your code decides. It's compared
            with the baseline and saved to the model's history.
          </p>
          <dl className="divide-y divide-slate-100 overflow-hidden rounded-xl border border-slate-200 bg-white text-sm">
            {[
              ["Batch scoring", "e.g. a nightly job", "Send the same inputs right after the job scores them."],
              ["Live API", "one request at a time", "Log each request's inputs; a scheduled job sends them hourly or daily. One row can't show drift."],
              ["Events", "releases, data deliveries", "Send once per event, alongside your other release checks."],
            ].map(([how, example, when]) => (
              <div key={how} className="grid gap-1 px-4 py-3 sm:grid-cols-[180px_1fr] sm:gap-4">
                <dt>
                  <span className="font-medium text-slate-800">{how}</span>
                  <span className="block text-xs text-slate-400">{example}</span>
                </dt>
                <dd className="text-slate-600">{when}</dd>
              </div>
            ))}
          </dl>
          <CodeBlock snippets={[{ label: "Python", code: python.batch }, { label: "curl", code: curl.batch }]} />
          <div className="flex gap-2.5 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
            <IconAlert className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
            <span>
              Let batch size set the schedule: aim for a few hundred rows (300+ if a column has many categories) — if an
              hour brings 50 rows, send daily. Using the date as <code className="font-mono text-[13px]">batch_id</code> makes
              reruns safe. Send the columns you trained on; extra or missing ones are flagged in the report.
            </span>
          </div>
        </Step>

        <Step n={5} title="React to the result">
          <p>
            <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-sm">overall.alert</code> is the one field to act on. The rest
            explains what changed:
          </p>
          <CodeBlock snippets={[{ label: "Report (abridged)", code: report }]} />
          <p>
            To be told without polling, add a webhook: we call your URL (Slack, PagerDuty, your own service) when an
            alert opens, on each drifting batch while it stays open, and once more when the data is back to normal.{" "}
            {list.includes(project) && (
              <Link className="inline-flex items-center gap-1 font-medium text-brand-600 hover:text-brand-700"
                    to={`/projects/${encodeURIComponent(project)}/webhooks`}>
                Set up alerts for {project} <IconArrowRight className="h-4 w-4" />
              </Link>
            )}
          </p>
        </Step>
      </div>

      <Card className="flex flex-wrap items-center justify-between gap-4 p-5">
        <div>
          <p className="font-semibold text-slate-900">Full API reference</p>
          <p className="mt-0.5 text-sm text-slate-500">Every endpoint and field, with a "try it out" button.</p>
        </div>
        <a href={`${PUBLIC_API_URL}/docs`} target="_blank" rel="noreferrer">
          <Button variant="outline">Open API docs <IconArrowRight className="h-4 w-4" /></Button>
        </a>
      </Card>
    </div>
  );
}
