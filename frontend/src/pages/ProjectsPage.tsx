import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Badge, Button, Card, EmptyState, ErrorBanner, PageHeader, Spinner } from "../components/ui";
import { api, ApiError } from "../lib/api";

export function ProjectsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [newProjectId, setNewProjectId] = useState("");
  const [showNewProject, setShowNewProject] = useState(false);

  const { data, isLoading, error } = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });

  const deleteMutation = useMutation({
    mutationFn: (projectId: string) => api.deleteProject(projectId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["projects"] }),
  });

  const handleCreate = () => {
    const id = newProjectId.trim();
    if (!id) return;
    navigate(`/projects/${encodeURIComponent(id)}/fit`);
  };

  return (
    <div>
      <PageHeader
        title="Projects"
        subtitle="Monitor data drift across tabular, text, image, and joint-modality projects."
        actions={
          <Button onClick={() => setShowNewProject((v) => !v)}>
            + New Project
          </Button>
        }
      />

      {showNewProject && (
        <Card className="mb-6 p-5">
          <p className="mb-3 text-sm font-medium text-slate-700">Name your new project, then lock a baseline.</p>
          <div className="flex gap-2">
            <input
              className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
              placeholder="e.g. fraud-model-v3"
              value={newProjectId}
              onChange={(e) => setNewProjectId(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            />
            <Button onClick={handleCreate}>Continue</Button>
          </div>
        </Card>
      )}

      {isLoading && (
        <div className="flex items-center justify-center py-16 text-slate-400">
          <Spinner className="h-6 w-6" />
        </div>
      )}

      {error && <ErrorBanner message={error instanceof ApiError ? error.detail : "Failed to load projects."} />}

      {data && data.projects.length === 0 && (
        <EmptyState
          title="No projects yet"
          description="Create your first project to lock a baseline and start monitoring drift."
          action={<Button onClick={() => setShowNewProject(true)}>+ New Project</Button>}
        />
      )}

      {data && data.projects.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {data.projects.map((pid) => (
            <Card
              key={pid}
              className="group cursor-pointer p-5 transition-shadow hover:shadow-md"
              onClick={() => navigate(`/projects/${encodeURIComponent(pid)}`)}
            >
              <div className="flex items-start justify-between">
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-slate-900">{pid}</p>
                  <Badge tone="brand">Active</Badge>
                </div>
                <button
                  className="invisible rounded-md px-2 py-1 text-xs text-slate-400 hover:bg-alert-50 hover:text-alert-600 group-hover:visible"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (confirm(`Delete project "${pid}"? This removes its baseline and logs permanently.`)) {
                      deleteMutation.mutate(pid);
                    }
                  }}
                >
                  Delete
                </button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
