import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./layout/AppShell";
import { AuthProvider } from "./lib/auth";
import { AnalyzePage } from "./pages/AnalyzePage";
import { BaselinesPage } from "./pages/BaselinesPage";
import { FitPage } from "./pages/FitPage";
import { HistoryPage } from "./pages/HistoryPage";
import { LoginPage } from "./pages/LoginPage";
import { LogsPage } from "./pages/LogsPage";
import { PredictPage } from "./pages/PredictPage";
import { ProjectOverviewPage } from "./pages/ProjectOverviewPage";
import { ProjectsPage } from "./pages/ProjectsPage";
import { ProtectedRoute } from "./routes/ProtectedRoute";
import { WebhooksPage } from "./pages/WebhooksPage";

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
                <Route path="/projects/:projectId/fit" element={<FitPage />} />
                <Route path="/projects/:projectId/analyze" element={<AnalyzePage />} />
                <Route path="/projects/:projectId/predict" element={<PredictPage />} />
                <Route path="/projects/:projectId/history" element={<HistoryPage />} />
                <Route path="/projects/:projectId/baselines" element={<BaselinesPage />} />
                <Route path="/projects/:projectId/webhooks" element={<WebhooksPage />} />
                <Route path="/projects/:projectId/logs" element={<LogsPage />} />
              </Route>
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
