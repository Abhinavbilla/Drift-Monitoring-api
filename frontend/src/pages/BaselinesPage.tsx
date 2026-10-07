import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { Badge, Button, Card, EmptyState, ErrorBanner, PageHeader, Spinner } from "../components/ui";
import { api, ApiError } from "../lib/api";

export function BaselinesPage() {
  const { projectId = "" } = useParams();
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ["baselineVersions", projectId],
    queryFn: () => api.listBaselineVersions(projectId),
    retry: false,
  });

  const activateMutation = useMutation({
    mutationFn: (version: number) => api.activateBaselineVersion(projectId, version),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["baselineVersions", projectId] });
      queryClient.invalidateQueries({ queryKey: ["baseline", projectId] });
    },
  });

  return (
    <div>
      <PageHeader eyebrow="Saved baselines" title="Your saved baselines" subtitle="Each time you update what “normal” looks like, we keep the old version too — so you can switch back at any time." />

      {isLoading && (
        <div className="flex items-center justify-center py-16 text-slate-400">
          <Spinner className="h-6 w-6" />
        </div>
      )}

      {error && <ErrorBanner message={error instanceof ApiError ? error.detail : "Failed to load baseline versions."} />}

      {data && data.versions.length === 0 && <EmptyState title="No baseline versions yet" />}

      {data && data.versions.length > 0 && (
        <Card className="overflow-hidden">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-5 py-3">Version</th>
                <th className="px-5 py-3">Created</th>
                <th className="px-5 py-3">Name</th>
                <th className="px-5 py-3">Kind</th>
                <th className="px-5 py-3">Status</th>
                <th className="px-5 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.versions.map((v) => (
                <tr key={v.version}>
                  <td className="px-5 py-2.5 font-medium text-slate-700">v{v.version}</td>
                  <td className="px-5 py-2.5 text-slate-600">{new Date(v.created_at).toLocaleString()}</td>
                  <td className="px-5 py-2.5 text-slate-600">{v.model_version_label ?? "—"}</td>
                  <td className="px-5 py-2.5 text-slate-600 capitalize">{v.modality}</td>
                  <td className="px-5 py-2.5">
                    {v.active ? <Badge tone="brand">In use</Badge> : <Badge tone="slate">Saved</Badge>}
                  </td>
                  <td className="px-5 py-2.5 text-right">
                    {!v.active && (
                      <Button
                        variant="secondary"
                        className="px-3 py-1 text-xs"
                        onClick={() => activateMutation.mutate(v.version)}
                        disabled={activateMutation.isPending}
                      >
                        Use this one
                      </Button>
                    )}
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
