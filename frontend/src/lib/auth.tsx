import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, registerTokenGetter } from "./api";

interface Session {
  token: string;
  email: string;
  name: string;
}

interface AuthContextValue {
  session: Session | null;
  loading: boolean;
  loginWithGoogleIdToken: (idToken: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);
const STORAGE_KEY = "drift_sentinel_session";

function loadSession(): Session | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Session;
    return parsed?.token ? parsed : null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(() => loadSession());
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    registerTokenGetter(() => session?.token ?? null);
  }, [session]);

  const loginWithGoogleIdToken = async (idToken: string) => {
    setLoading(true);
    try {
      const resp = await api.loginWithGoogle(idToken);
      const next: Session = { token: resp.session_token, email: resp.email, name: resp.name };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      setSession(next);
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem(STORAGE_KEY);
    setSession(null);
  };

  const value = useMemo(
    () => ({ session, loading, loginWithGoogleIdToken, logout }),
    [session, loading],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
