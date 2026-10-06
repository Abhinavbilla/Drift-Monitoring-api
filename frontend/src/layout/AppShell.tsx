import { NavLink, Outlet, useParams } from "react-router-dom";
import { useAuth } from "../lib/auth";

const projectTabs = [
  { to: "", label: "Overview", end: true },
  { to: "table", label: "Table Monitoring" },
  { to: "fit", label: "Fit Baseline" },
  { to: "analyze", label: "Analyze" },
  { to: "predict", label: "Predict" },
  { to: "history", label: "History" },
  { to: "baselines", label: "Baselines" },
  { to: "webhooks", label: "Webhooks" },
  { to: "logs", label: "Audit Log" },
];

function Logo() {
  return (
    <div className="flex items-center gap-2 px-6 py-5">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-white font-bold text-sm">
        DS
      </div>
      <span className="text-base font-bold text-white tracking-tight">Drift Sentinel</span>
    </div>
  );
}

export function AppShell() {
  const { session, logout } = useAuth();
  const { projectId } = useParams();

  return (
    <div className="flex min-h-screen bg-slate-50">
      <aside className="flex w-64 flex-col bg-slate-900">
        <Logo />
        <nav className="flex-1 px-3">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `mb-1 flex items-center rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                isActive ? "bg-brand-600 text-white" : "text-slate-300 hover:bg-slate-800 hover:text-white"
              }`
            }
          >
            All Projects
          </NavLink>
          {projectId && (
            <div className="mt-4 border-t border-slate-800 pt-4">
              <p className="px-3 pb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                {decodeURIComponent(projectId)}
              </p>
              {projectTabs.map((tab) => (
                <NavLink
                  key={tab.to}
                  to={`/projects/${projectId}${tab.to ? `/${tab.to}` : ""}`}
                  end={tab.end}
                  className={({ isActive }) =>
                    `mb-0.5 flex items-center rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                      isActive ? "bg-brand-600 text-white" : "text-slate-300 hover:bg-slate-800 hover:text-white"
                    }`
                  }
                >
                  {tab.label}
                </NavLink>
              ))}
            </div>
          )}
        </nav>
        <div className="border-t border-slate-800 p-4">
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-white">{session?.name}</p>
              <p className="truncate text-xs text-slate-500">{session?.email}</p>
            </div>
            <button
              onClick={logout}
              className="shrink-0 rounded-lg px-2 py-1 text-xs font-medium text-slate-400 hover:bg-slate-800 hover:text-white"
            >
              Sign out
            </button>
          </div>
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl px-8 py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
