import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./layout/AppShell";
import { AuthProvider } from "./lib/auth";
import { ComingSoonPage } from "./pages/ComingSoonPage";
import { LoginPage } from "./pages/LoginPage";
import { ProjectOverviewPage } from "./pages/ProjectOverviewPage";
import { ProjectsPage } from "./pages/ProjectsPage";
import { ProtectedRoute } from "./routes/ProtectedRoute";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route element={<ProtectedRoute />}>
              <Route element={<AppShell />}>
                <Route path="/" element={<ProjectsPage />} />
                <Route path="/projects/:projectId" element={<ProjectOverviewPage />} />
                <Route path="/projects/:projectId/fit" element={<ComingSoonPage title="Fit Baseline" />} />
                <Route path="/projects/:projectId/analyze" element={<ComingSoonPage title="Analyze" />} />
                <Route path="/projects/:projectId/predict" element={<ComingSoonPage title="Predict" />} />
                <Route path="/projects/:projectId/history" element={<ComingSoonPage title="History" />} />
                <Route path="/projects/:projectId/baselines" element={<ComingSoonPage title="Baselines" />} />
                <Route path="/projects/:projectId/webhooks" element={<ComingSoonPage title="Webhooks" />} />
                <Route path="/projects/:projectId/logs" element={<ComingSoonPage title="Audit Log" />} />
              </Route>
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
