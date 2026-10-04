import { useState } from "react";
import { Navigate } from "react-router-dom";
import { GoogleSignInButton } from "../components/GoogleSignInButton";
import { ApiError } from "../lib/api";
import { useAuth } from "../lib/auth";

export function LoginPage() {
  const { session, loginWithGoogleIdToken } = useAuth();
  const [error, setError] = useState<string | null>(null);

  if (session) return <Navigate to="/" replace />;

  const handleToken = async (idToken: string) => {
    setError(null);
    try {
      await loginWithGoogleIdToken(idToken);
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : "Sign-in failed. Please try again.");
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50">
      <div className="w-full max-w-sm rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
        <div className="mb-8 flex flex-col items-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-brand-600 text-white font-bold text-lg">
            DS
          </div>
          <h1 className="text-xl font-bold text-slate-900">Drift Sentinel</h1>
          <p className="mt-1 text-sm text-slate-500">Sign in to monitor your projects</p>
        </div>
        <div className="flex justify-center">
          <GoogleSignInButton onToken={handleToken} />
        </div>
        {error && <p className="mt-4 text-center text-sm text-alert-600">{error}</p>}
      </div>
    </div>
  );
}
