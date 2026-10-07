import { useState } from "react";
import type { ReactNode } from "react";
import { NavLink, Outlet, useParams } from "react-router-dom";
import { Logo } from "../components/Logo";
import {
  IconBell, IconChevronDown, IconClock, IconGrid, IconHome, IconLayers, IconList, IconLogout, IconTable, IconWrench,
} from "../components/icons";
import { useAuth } from "../lib/auth";

const projectLinks = [
  { to: "", label: "Overview", icon: <IconHome />, end: true },
  { to: "table", label: "Check my data", icon: <IconTable /> },
  { to: "history", label: "Past checks", icon: <IconClock /> },
  { to: "baselines", label: "Saved baselines", icon: <IconLayers /> },
  { to: "webhooks", label: "Alerts & webhooks", icon: <IconBell /> },
  { to: "logs", label: "Activity log", icon: <IconList /> },
];

// The per-type tools from before the table workflow; kept for older projects.
const advancedLinks = [
  { to: "fit", label: "Set up (older tool)" },
  { to: "analyze", label: "Check a batch (older tool)" },
  { to: "predict", label: "Check a single row" },
];

function SideLink({ to, end, icon, children }: { to: string; end?: boolean; icon?: ReactNode; children: ReactNode }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `group flex items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition-colors ${
          isActive ? "bg-brand-50 text-brand-700" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
        }`
      }
    >
      {icon && <span className="[&>svg]:h-[18px] [&>svg]:w-[18px]">{icon}</span>}
      <span className="truncate">{children}</span>
    </NavLink>
  );
}

function initials(name?: string) {
  return (name ?? "?").split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]?.toUpperCase()).join("");
}

export function AppShell() {
  const { session, logout } = useAuth();
  const { projectId } = useParams();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const base = projectId ? `/projects/${projectId}` : "";

  return (
    <div className="flex min-h-screen">
      {/* Outer column stretches to the page height (keeps the border/background); inner part stays in view. */}
      <aside className="w-[264px] shrink-0 border-r border-slate-200 bg-white">
       <div className="sticky top-0 flex h-screen flex-col">
        <div className="px-5 pb-4 pt-5">
          <Logo />
        </div>

        <nav className="flex-1 space-y-6 overflow-y-auto px-3 pb-4">
          <div className="space-y-0.5">
            <SideLink to="/" end icon={<IconGrid />}>All projects</SideLink>
          </div>

          {projectId && (
            <div>
              <p className="mb-2 truncate px-3 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                {decodeURIComponent(projectId)}
              </p>
              <div className="space-y-0.5">
                {projectLinks.map((l) => (
                  <SideLink key={l.to} to={`${base}${l.to ? `/${l.to}` : ""}`} end={l.end} icon={l.icon}>
                    {l.label}
                  </SideLink>
                ))}
              </div>

              <button
                type="button"
                onClick={() => setShowAdvanced((v) => !v)}
                className="mt-4 flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium text-slate-500 hover:bg-slate-100"
              >
                <IconWrench className="h-[18px] w-[18px]" />
                <span className="flex-1 text-left">Advanced tools</span>
                <IconChevronDown className={`h-4 w-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
              </button>
              {showAdvanced && (
                <div className="ml-6 mt-0.5 space-y-0.5 border-l border-slate-200 pl-3">
                  {advancedLinks.map((l) => (
                    <SideLink key={l.to} to={`${base}/${l.to}`}>{l.label}</SideLink>
                  ))}
                </div>
              )}
            </div>
          )}
        </nav>

        <div className="border-t border-slate-200 p-3">
          <div className="flex items-center gap-3 rounded-xl px-2 py-2">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-brand-400 to-brand-600 text-xs font-bold text-white">
              {initials(session?.name)}
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-semibold text-slate-800">{session?.name}</p>
              <p className="truncate text-xs text-slate-500">{session?.email}</p>
            </div>
            <button onClick={logout} title="Sign out" aria-label="Sign out"
                    className="rounded-lg p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-700">
              <IconLogout className="h-[18px] w-[18px]" />
            </button>
          </div>
        </div>
       </div>
      </aside>

      <main className="min-w-0 flex-1">
        <div className="mx-auto max-w-6xl px-6 py-8 lg:px-10 lg:py-10">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
