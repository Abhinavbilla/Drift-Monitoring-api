import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { Badge, Card, EmptyState, ErrorBanner, PageHeader, Spinner } from "../components/ui";
import { api, ApiError } from "../lib/api";

export function ProjectOverviewPage() {
  const { projectId = "" } = useParams();

  const { data, isLoading, error } = useQuery({
    queryKey: ["baseline", projectId],
    queryFn: () => api.getBaseline(projectId),
    retry: false,
  });

  const notFound = error instanceof ApiError && error.status === 404;

  return (
    <div>
      <PageHeader title={decodeURIComponent(projectId)} subtitle="Project overview" />

      {isLoading && (
        <div className="flex items-center justify-center py-16 text-slate-400">
          <Spinner className="h-6 w-6" />
        </div>
      )}

      {notFound && (
        <EmptyState
          title="No baseline locked yet"
          description="Head to the Fit Baseline tab to lock a reference distribution for this project."
        />
      )}

      {error && !notFound && <ErrorBanner message={error instanceof ApiError ? error.detail : "Failed to load."} />}

      {data && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Card className="p-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Modality</p>
            <p className="mt-1 text-lg font-bold text-slate-900 capitalize">{data.modality}</p>
          </Card>
          <Card className="p-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Monitored features</p>
            <p className="mt-1 text-lg font-bold text-slate-900">{Object.keys(data.feature_types).length}</p>
          </Card>
          <Card className="p-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Status</p>
            <Badge tone="ok">Baseline locked</Badge>
          </Card>

          {Object.keys(data.feature_types).length > 0 && (
            <Card className="col-span-full overflow-hidden">
              <table className="w-full text-sm">
                <thead className="border-b border-slate-200 bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="px-5 py-3">Feature</th>
                    <th className="px-5 py-3">Type</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {Object.entries(data.feature_types).map(([name, type]) => (
                    <tr key={name}>
                      <td className="px-5 py-2.5 font-medium text-slate-700">{name}</td>
                      <td className="px-5 py-2.5">
                        <Badge tone={type === "continuous" ? "brand" : "slate"}>{type}</Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
