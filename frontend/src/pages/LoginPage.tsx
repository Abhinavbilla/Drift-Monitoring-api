import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import type { ReactNode } from "react";
import { Navigate } from "react-router-dom";
import { EmailSignIn } from "../components/EmailSignIn";
import { GoogleSignInButton } from "../components/GoogleSignInButton";
import { Logo, LogoMark } from "../components/Logo";
import { IconAlert, IconCheckCircle, IconLink, IconShield } from "../components/icons";
import { ApiError, api } from "../lib/api";
import { useAuth } from "../lib/auth";

function HeroBackground() {
  // Soft, slowly drifting colour fields over a dot grid -- pure CSS, no images to load.
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      <div className="absolute -left-32 -top-40 h-[520px] w-[520px] rounded-full bg-brand-600/40 blur-[110px] animate-drift-slow" />
      <div className="absolute -bottom-48 right-[-120px] h-[560px] w-[560px] rounded-full bg-fuchsia-500/25 blur-[120px] animate-drift-slower" />
      <div className="absolute left-1/3 top-1/2 h-[360px] w-[360px] rounded-full bg-cyan-400/20 blur-[100px] animate-drift-slow" />
      <div className="absolute inset-0 bg-dot-grid opacity-40" />
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-ink-900/60" />
    </div>
  );
}

function InsightCard({ tone, title, body, className = "" }: {
  tone: "ok" | "alert"; title: string; body: string; className?: string;
}) {
  const ok = tone === "ok";
  return (
    <div className={`w-72 rounded-2xl border border-white/10 bg-white/[0.07] p-4 shadow-2xl backdrop-blur-md ${className}`}>
      <div className="flex items-start gap-3">
        <span className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
          ok ? "bg-ok-500/15 text-ok-500" : "bg-alert-500/15 text-alert-500"}`}>
          {ok ? <IconCheckCircle className="h-[18px] w-[18px]" /> : <IconAlert className="h-[18px] w-[18px]" />}
        </span>
        <div>
          <p className="text-sm font-semibold text-white">{title}</p>
          <p className="mt-0.5 text-xs leading-relaxed text-slate-300">{body}</p>
        </div>
      </div>
    </div>
  );
}

function Point({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <li className="flex items-start gap-3">
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-white/10 text-brand-200">
        {icon}
      </span>
      <span className="text-[15px] leading-relaxed text-slate-300">{children}</span>
    </li>
  );
}

export function LoginPage() {
  const { session, loginWithGoogleIdToken, completeSignIn } = useAuth();
  const methods = useQuery({ queryKey: ["authMethods"], queryFn: api.authMethods, retry: false });
  const [error, setError] = useState<string | null>(null);

  if (session) return <Navigate to="/" replace />;

  const handleToken = async (idToken: string) => {
    setError(null);
    try {
      await loginWithGoogleIdToken(idToken);
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : "We couldn't sign you in. Please try again.");
    }
  };

  return (
    <div className="flex min-h-screen bg-[#f6f7fb]">
      {/* Story side */}
      <section className="relative hidden w-[55%] overflow-hidden bg-ink-900 lg:flex lg:flex-col lg:justify-between">
        <HeroBackground />
        <div className="relative z-10 px-14 pt-12">
          <Logo dark />
        </div>

        <div className="relative z-10 px-14">
          <h1 className="max-w-xl text-[44px] font-extrabold leading-[1.08] text-white">
            Know the moment your data <span className="bg-gradient-to-r from-brand-300 to-fuchsia-300 bg-clip-text text-transparent">stops looking normal</span>.
          </h1>
          <p className="mt-5 max-w-lg text-lg leading-relaxed text-slate-300">
            Drift Sentinel compares the data your models see today with the data they learned from — and tells you,
            in plain words, what changed.
          </p>
          <ul className="mt-8 max-w-lg space-y-4">
            <Point icon={<IconCheckCircle className="h-4 w-4" />}>Works with numbers, categories, written text and photos — all in one table.</Point>
            <Point icon={<IconLink className="h-4 w-4" />}>Notices when columns stop going together, even if each looks fine on its own.</Point>
            <Point icon={<IconShield className="h-4 w-4" />}>Keeps statistics, not your raw data. Only you can see your projects.</Point>
          </ul>
        </div>

        <div className="relative z-10 h-56 px-14">
          <InsightCard tone="ok" title="All 14 columns look normal" body="Today's batch matches the data your model learned from."
                       className="absolute left-14 top-2 animate-float" />
          <InsightCard tone="alert" title="Price and size no longer move together"
                       body="Each column looks fine, but bigger homes aren't costing more anymore."
                       className="absolute left-[23rem] top-16 animate-float-delayed" />
        </div>
      </section>

      {/* Sign-in side */}
      <section className="relative flex flex-1 items-center justify-center px-6 py-12">
        <div className="absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-brand-100/60 to-transparent lg:hidden" aria-hidden="true" />
        <div className="relative w-full max-w-sm animate-fade-up">
          <div className="mb-10 flex justify-center lg:hidden">
            <LogoMark className="h-12 w-12" />
          </div>
          <h2 className="text-3xl font-bold text-slate-900">Welcome</h2>
          <p className="mt-2 text-[15px] text-slate-500">Sign in to see your projects.</p>

          <div className="mt-8 rounded-2xl border border-slate-200 bg-white p-6 shadow-[var(--shadow-lift)]">
            <div className="flex justify-center">
              <GoogleSignInButton onToken={handleToken} />
            </div>
            {error && <p className="mt-4 text-center text-sm text-alert-600">{error}</p>}
            {methods.data?.email && (
              <>
                <div className="my-6 flex items-center gap-3 text-xs font-medium uppercase tracking-wider text-slate-400">
                  <span className="h-px flex-1 bg-slate-200" /> or <span className="h-px flex-1 bg-slate-200" />
                </div>
                <EmailSignIn onSignedIn={completeSignIn} />
              </>
            )}
            <p className="mt-5 border-t border-slate-100 pt-4 text-center text-xs leading-relaxed text-slate-400">
              We only use your name and email to keep your projects private to you.
            </p>
          </div>

          <p className="mt-8 text-center text-xs text-slate-400">
            New here? Signing in with Google, or creating an account, sets up your workspace automatically.
          </p>
        </div>
      </section>
    </div>
  );
}
