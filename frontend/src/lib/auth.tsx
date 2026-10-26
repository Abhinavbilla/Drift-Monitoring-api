import { createContext, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { api, registerTokenGetter, registerUnauthorizedHandler, type SessionResponse } from "./api";

interface Session {
  token: string;
  email: string;
  name: string;
}

interface AuthContextValue {
  session: Session | null;
  /** True when the last session ran out, so the sign-in page can say why. */
  expired: boolean;
  loading: boolean;
  loginWithGoogleIdToken: (idToken: string) => Promise<void>;
  /** Finishes any sign-in method that returns a session (email + password, codes). */
  completeSignIn: (resp: SessionResponse) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);
const STORAGE_KEY = "drift_sentinel_session";

/** Session tokens are JWTs; reads `exp` so a stale one isn't used at all. */
function isExpired(token: string): boolean {
  try {
    const payload = JSON.parse(atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    return typeof payload.exp === "number" && payload.exp * 1000 <= Date.now();
  } catch {
    return false; // not decodable here; the server still decides
  }
}

function loadSession(): { session: Session | null; expired: boolean } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { session: null, expired: false };
    const parsed = JSON.parse(raw) as Session;
    if (!parsed?.token) return { session: null, expired: false };
    if (isExpired(parsed.token)) {
      localStorage.removeItem(STORAGE_KEY);
      return { session: null, expired: true };
    }
    return { session: parsed, expired: false };
  } catch {
    return { session: null, expired: false };
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [initial] = useState(loadSession);
  const [session, setSession] = useState<Session | null>(initial.session);
  const [expired, setExpired] = useState(initial.expired);
  const [loading, setLoading] = useState(false);

  // Registered during render, not in an effect: child components' queries start
  // before a parent's effects run, so an effect left the very first requests of a
  // page load (e.g. after a refresh) without a token.
  const sessionRef = useRef(session);
  sessionRef.current = session;
  registerTokenGetter(() => sessionRef.current?.token ?? null);
  registerUnauthorizedHandler(() => {
    if (!sessionRef.current) return;
    sessionRef.current = null;
    localStorage.removeItem(STORAGE_KEY);
    setExpired(true);
    setSession(null);
  });

  const completeSignIn = (resp: SessionResponse) => {
    const next: Session = { token: resp.session_token, email: resp.email, name: resp.name };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    setExpired(false);
    setSession(next);
  };

  const loginWithGoogleIdToken = async (idToken: string) => {
    setLoading(true);
    try {
      completeSignIn(await api.loginWithGoogle(idToken));
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem(STORAGE_KEY);
    setSession(null);
  };

  const value = useMemo(
    () => ({ session, expired, loading, loginWithGoogleIdToken, completeSignIn, logout }),
    [session, expired, loading],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
