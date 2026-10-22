import { useId, useState } from "react";
import type { FormEvent, InputHTMLAttributes, ReactNode } from "react";
import { ApiError, api, type SessionResponse } from "../lib/api";
import { GoogleSignInButton } from "./GoogleSignInButton";
import { IconAlert, IconArrowLeft, IconArrowRight, IconEye, IconEyeOff, IconLock, IconMail, IconUser } from "./icons";

type Tab = "login" | "signup";
type Step = Tab | "verify" | "forgot" | "reset";

const HEADINGS: Record<Step, [string, string]> = {
  login: ["Welcome back", "Log in to see your models."],
  signup: ["Create your account", "Start watching your data in a couple of minutes."],
  verify: ["Check your inbox", "Enter the 6-digit code we just emailed you."],
  forgot: ["Forgot your password?", "No problem. We'll email you a code to reset it."],
  reset: ["Choose a new password", "Enter the code from your email and a new password."],
};

// The label sits beside (not around) the input, so an extra control in the
// label row (e.g. "Forgot password?") doesn't become the label's target.
function Field({ label, aside, icon, trailing, hint, ...input }: InputHTMLAttributes<HTMLInputElement> & {
  label: ReactNode; aside?: ReactNode; icon: ReactNode; trailing?: ReactNode; hint?: ReactNode;
}) {
  const id = useId();
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <label htmlFor={id} className="text-sm font-medium text-slate-700">{label}</label>
        {aside}
      </div>
      <span className="group relative block">
        <span className="pointer-events-none absolute inset-y-0 left-3.5 flex items-center text-slate-400 transition-colors group-focus-within:text-brand-500">
          {icon}
        </span>
        <input
          {...input}
          id={id}
          className="h-12 w-full rounded-xl border border-slate-300 bg-white pl-11 pr-12 text-[15px] text-slate-800 placeholder:text-slate-400
                     transition-shadow focus:border-brand-500 focus:outline-none focus:ring-4 focus:ring-brand-100"
        />
        {trailing && <span className="absolute inset-y-0 right-2 flex items-center">{trailing}</span>}
      </span>
      {hint}
    </div>
  );
}

function strength(password: string): { score: number; label: string } {
  if (password.length < 8) return { score: password ? 1 : 0, label: password ? "Too short" : "" };
  let score = 2;
  if (password.length >= 12) score++;
  if (/[a-z]/.test(password) && /[A-Z]/.test(password) && /[\d\W]/.test(password)) score++;
  return { score, label: ["", "", "Okay", "Good", "Strong"][score] };
}

function StrengthMeter({ password }: { password: string }) {
  const { score, label } = strength(password);
  const colour = score <= 1 ? "bg-alert-500" : score === 2 ? "bg-amber-400" : "bg-ok-500";
  return (
    <span className="mt-2 flex items-center gap-3">
      <span className="flex flex-1 gap-1">
        {[1, 2, 3, 4].map((n) => (
          <span key={n} className={`h-1 flex-1 rounded-full transition-colors ${n <= score ? colour : "bg-slate-200"}`} />
        ))}
      </span>
      <span className="w-16 text-right text-xs text-slate-500">{label || "8+ characters"}</span>
    </span>
  );
}

export function AuthPanel({ emailEnabled, onSession, onGoogleToken }: {
  emailEnabled: boolean;
  onSession: (resp: SessionResponse) => void;
  onGoogleToken: (idToken: string) => Promise<void>;
}) {
  const [step, setStep] = useState<Step>("login");
  const [tab, setTab] = useState<Tab>("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const go = (next: Step, message: string | null = null) => {
    if (next === "login" || next === "signup") setTab(next);
    setStep(next);
    setError(null);
    setInfo(message);
    setCode("");
  };

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      if (e instanceof ApiError && e.status === 403 && step === "login") {
        go("verify", e.detail);  // signed up earlier but never confirmed the email
        return;
      }
      setError(e instanceof ApiError ? e.detail : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    run(async () => {
      if (step === "login") onSession(await api.emailLogin(email, password));
      if (step === "signup") {
        await api.emailRegister(email, password, name.trim() || undefined);
        go("verify", `We sent a code to ${email}. It expires in 15 minutes.`);
      }
      if (step === "verify") onSession(await api.emailVerify(email, code));
      if (step === "forgot") {
        await api.emailForgot(email);
        go("reset", `If there's an account for ${email}, a code is on its way.`);
      }
      if (step === "reset") onSession(await api.emailReset(email, code, password));
    });
  };

  const google = (idToken: string) =>
    run(() => onGoogleToken(idToken));

  const [title, subtitle] = HEADINGS[emailEnabled ? step : "login"];
  const isTab = step === "login" || step === "signup";
  const submitLabel: Record<Step, string> = {
    login: "Log in", signup: "Create account", verify: "Confirm email", forgot: "Email me a code", reset: "Save and log in",
  };
  const eyeToggle = (
    <button type="button" onClick={() => setShowPassword((v) => !v)} aria-label={showPassword ? "Hide password" : "Show password"}
            className="rounded-lg p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600">
      {showPassword ? <IconEyeOff className="h-[18px] w-[18px]" /> : <IconEye className="h-[18px] w-[18px]" />}
    </button>
  );
  const passwordInput = (labelText: string, autoComplete: string, aside?: ReactNode, hint?: ReactNode) => (
    <Field
      label={labelText} aside={aside} icon={<IconLock className="h-[18px] w-[18px]" />} trailing={eyeToggle} hint={hint}
      type={showPassword ? "text" : "password"} autoComplete={autoComplete} required value={password}
      onChange={(e) => setPassword(e.target.value)} placeholder="••••••••"
    />
  );

  return (
    <div>
      {!isTab && emailEnabled && (
        <button type="button" onClick={() => go(tab)}
                className="mb-6 inline-flex items-center gap-1.5 text-sm font-medium text-slate-500 hover:text-slate-800">
          <IconArrowLeft className="h-4 w-4" /> Back
        </button>
      )}
      <h2 className="font-display text-3xl font-bold tracking-tight text-slate-900">{title}</h2>
      <p className="mt-2 text-[15px] text-slate-500">{subtitle}</p>

      <div className="mt-8 rounded-2xl border border-slate-200/80 bg-white p-6 shadow-[var(--shadow-lift)] sm:p-7">
        {emailEnabled && isTab && (
          <div role="tablist" className="mb-6 grid grid-cols-2 rounded-xl bg-slate-100 p-1 text-sm font-semibold">
            {(["login", "signup"] as const).map((t) => (
              <button key={t} type="button" role="tab" aria-selected={step === t} onClick={() => go(t)}
                      className={`rounded-lg py-2 transition-all ${step === t
                        ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"}`}>
                {t === "login" ? "Log in" : "Sign up"}
              </button>
            ))}
          </div>
        )}

        {emailEnabled && (
          <form onSubmit={submit} className="space-y-4" noValidate>
            {info && <p className="rounded-xl bg-brand-50 px-4 py-3 text-sm leading-relaxed text-brand-700">{info}</p>}

            {step === "signup" && (
              <Field label="Full name" aside={<span className="text-xs text-slate-400">optional</span>}
                     icon={<IconUser className="h-[18px] w-[18px]" />} autoComplete="name" value={name}
                     onChange={(e) => setName(e.target.value)} placeholder="Ada Lovelace" />
            )}

            {(step === "login" || step === "signup" || step === "forgot") && (
              <Field label="Email" icon={<IconMail className="h-[18px] w-[18px]" />} type="email" autoComplete="email"
                     required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@company.com" />
            )}

            {(step === "verify" || step === "reset") && (
              <label className="block">
                <span className="mb-1.5 block text-sm font-medium text-slate-700">6-digit code</span>
                <input inputMode="numeric" autoComplete="one-time-code" maxLength={6} value={code} autoFocus
                       onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))} placeholder="000000"
                       className="h-14 w-full rounded-xl border border-slate-300 bg-white text-center font-mono text-2xl tracking-[0.6em]
                                  text-slate-900 placeholder:text-slate-300 focus:border-brand-500 focus:outline-none focus:ring-4 focus:ring-brand-100" />
              </label>
            )}

            {step === "login" && passwordInput("Password", "current-password",
              <button type="button" onClick={() => go("forgot")} className="text-xs font-medium text-brand-600 hover:text-brand-700">
                Forgot password?
              </button>)}
            {step === "signup" && passwordInput("Password", "new-password", null, <StrengthMeter password={password} />)}
            {step === "reset" && passwordInput("New password", "new-password", null, <StrengthMeter password={password} />)}

            {error && (
              <p role="alert" className="flex items-start gap-2 rounded-xl bg-alert-50 px-4 py-3 text-sm text-alert-700">
                <IconAlert className="mt-0.5 h-4 w-4 shrink-0" /> {error}
              </p>
            )}

            <button type="submit" disabled={busy}
                    className="group flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-brand-600 text-[15px] font-semibold text-white
                               shadow-sm shadow-brand-600/30 transition-all hover:bg-brand-700 disabled:cursor-wait disabled:opacity-60">
              {busy ? (
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/40 border-t-white" aria-label="Please wait" />
              ) : (
                <>{submitLabel[step]} <IconArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" /></>
              )}
            </button>

            {step === "verify" && (
              <p className="text-center text-sm text-slate-500">
                Didn't get it? Check spam, or{" "}
                <button type="button" className="font-medium text-brand-600 hover:text-brand-700"
                        onClick={() => run(async () => { setInfo((await api.emailResend(email)).message); })}>
                  send a new code
                </button>
              </p>
            )}
          </form>
        )}

        {isTab && (
          <>
            {emailEnabled && (
              <div className="my-6 flex items-center gap-3 text-xs font-medium text-slate-400">
                <span className="h-px flex-1 bg-slate-200" /> or continue with <span className="h-px flex-1 bg-slate-200" />
              </div>
            )}
            <GoogleSignInButton onToken={google} text={step === "signup" ? "signup_with" : "signin_with"} />
            {!emailEnabled && error && <p role="alert" className="mt-4 text-center text-sm text-alert-600">{error}</p>}
          </>
        )}
      </div>

      {isTab && emailEnabled && (
        <p className="mt-6 text-center text-sm text-slate-500">
          {step === "login" ? "New to Drift Sentinel? " : "Already have an account? "}
          <button type="button" onClick={() => go(step === "login" ? "signup" : "login")}
                  className="font-semibold text-brand-600 hover:text-brand-700">
            {step === "login" ? "Create an account" : "Log in"}
          </button>
        </p>
      )}
      <p className="mt-4 text-center text-xs leading-relaxed text-slate-400">
        We only use your name and email to keep your models private to you.
      </p>
    </div>
  );
}
