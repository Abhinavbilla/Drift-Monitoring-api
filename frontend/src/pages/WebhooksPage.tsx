import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router-dom";
import { Badge, Button, Card, EmptyState, ErrorBanner, Field, PageHeader, Spinner, SuccessBanner, TextInput } from "../components/ui";
import { api, ApiError } from "../lib/api";

const EVENTS = ["opened", "resolved", "still_open"];

export function WebhooksPage() {
  const { projectId = "" } = useParams();
  const queryClient = useQueryClient();
  const [url, setUrl] = useState("");
  const [events, setEvents] = useState<string[]>(["opened", "resolved"]);
  const [newSecret, setNewSecret] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["webhooks", projectId],
    queryFn: () => api.listWebhooks(projectId),
    retry: false,
  });

  const registerMutation = useMutation({
    mutationFn: () => api.registerWebhook(projectId, url, events),
    onSuccess: (resp) => {
      setNewSecret(resp.secret ?? null);
      setUrl("");
      queryClient.invalidateQueries({ queryKey: ["webhooks", projectId] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (webhookId: string) => api.deleteWebhook(projectId, webhookId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["webhooks", projectId] }),
  });

  const toggleEvent = (ev: string) => {
    setEvents((prev) => (prev.includes(ev) ? prev.filter((e) => e !== ev) : [...prev, ev]));
  };

  return (
    <div>
      <PageHeader title="Webhooks" subtitle="Get an HMAC-signed POST whenever this project's alert state transitions." />

      <Card className="mb-6 max-w-xl p-6 space-y-4">
        <Field label="Endpoint URL" hint="Must be publicly reachable -- loopback/private/link-local addresses are rejected.">
          <TextInput value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://example.com/drift-webhook" />
        </Field>
        <div>
          <p className="mb-2 text-sm font-medium text-slate-700">Fire on</p>
          <div className="flex gap-3">
            {EVENTS.map((ev) => (
              <label key={ev} className="flex items-center gap-1.5 text-sm text-slate-600">
                <input type="checkbox" checked={events.includes(ev)} onChange={() => toggleEvent(ev)} />
                {ev}
              </label>
            ))}
          </div>
        </div>
        <Button onClick={() => registerMutation.mutate()} disabled={!url || events.length === 0 || registerMutation.isPending}>
          {registerMutation.isPending ? "Registering..." : "Register Webhook"}
        </Button>
        {registerMutation.error && (
          <ErrorBanner message={registerMutation.error instanceof ApiError ? registerMutation.error.detail : "Failed to register."} />
        )}
        {newSecret && (
          <div className="space-y-1">
            <SuccessBanner message="Webhook registered. Copy the signing secret now -- it won't be shown again." />
            <code className="block overflow-x-auto rounded-lg bg-slate-900 px-4 py-3 text-sm text-slate-100">{newSecret}</code>
          </div>
        )}
      </Card>

      {isLoading && (
        <div className="flex items-center justify-center py-16 text-slate-400">
          <Spinner className="h-6 w-6" />
        </div>
      )}

      {error && <ErrorBanner message={error instanceof ApiError ? error.detail : "Failed to load webhooks."} />}

      {data && data.webhooks.length === 0 && <EmptyState title="No webhooks registered yet" />}

      {data && data.webhooks.length > 0 && (
        <Card className="overflow-hidden">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-5 py-3">URL</th>
                <th className="px-5 py-3">Events</th>
                <th className="px-5 py-3">Status</th>
                <th className="px-5 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.webhooks.map((wh) => (
                <tr key={wh.id}>
                  <td className="px-5 py-2.5 max-w-xs truncate font-mono text-xs text-slate-600">{wh.url}</td>
                  <td className="px-5 py-2.5">
                    <div className="flex gap-1">
                      {wh.event_filter.map((e) => (
                        <Badge key={e} tone="slate">
                          {e}
                        </Badge>
                      ))}
                    </div>
                  </td>
                  <td className="px-5 py-2.5">
                    {wh.enabled ? <Badge tone="ok">Enabled</Badge> : <Badge tone="slate">Disabled</Badge>}
                  </td>
                  <td className="px-5 py-2.5 text-right">
                    <Button variant="danger" className="px-3 py-1 text-xs" onClick={() => deleteMutation.mutate(wh.id)}>
                      Delete
                    </Button>
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
