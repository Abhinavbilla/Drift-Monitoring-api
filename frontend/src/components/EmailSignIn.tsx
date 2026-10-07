import { useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { ApiError, api, type SessionResponse } from "../lib/api";
import { Button, TextInput } from "./ui";

type Mode = "signin" | "signup" | "verify" | "forgot" | "reset";

function Link({ onClick, children }: { onClick: () => void; children: ReactNode }) {
  return (
    <button type="button" onClick={onClick} className="font-medium text-brand-600 hover:text-brand-700 hover:underline">
      {children}
    </button>
  );
}

function Label({ children }: { children: ReactNode }) {
  return <span className="mb-1.5 block text-sm font-medium text-slate-700">{children}</span>;
}

export function EmailSignIn({ onSignedIn }: { onSignedIn: (resp: SessionResponse) => void }) {
  const [mode, setMode] = useState<Mode>("signin");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const go = (next: Mode, message: string | null = null) => {
    setMode(next);
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
      if (e instanceof ApiError && e.status === 403 && mode === "signin") {
        go("verify", e.detail);  // signed up earlier but never confirmed
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
      if (mode === "signin") onSignedIn(await api.emailLogin(email, password));
      if (mode === "signup") {
        await api.emailRegister(email, password, name || undefined);
        go("verify", `We sent a 6-digit code to ${email}. It expires in 15 minutes.`);
      }
      if (mode === "verify") onSignedIn(await api.emailVerify(email, code));
      if (mode === "forgot") {
        await api.emailForgot(email);
        go("reset", `If there's an account for ${email}, we've sent it a code.`);
      }
      if (mode === "reset") onSignedIn(await api.emailReset(email, code, password));
    });
  };

  const titles: Record<Mode, string> = {
    signin: "Sign in with email",
    signup: "Create your account",
    verify: "Check your email",
    forgot: "Reset your password",
    reset: "Choose a new password",
  };
  const actions: Record<Mode, string> = {
    signin: "Sign in", signup: "Create account", verify: "Confirm", forgot: "Send me a code", reset: "Save and sign in",
  };
  const needsPassword = mode === "signin" || mode === "signup" || mode === "reset";
  const needsCode = mode === "verify" || mode === "reset";

  return (
    <form onSubmit={submit} className="space-y-4" noValidate>
      <p className="text-sm font-semibold text-slate-800">{titles[mode]}</p>

      {info && <p className="rounded-xl bg-brand-50 px-3.5 py-2.5 text-sm text-brand-700">{info}</p>}

      {mode === "signup" && (
        <label className="block">
          <Label>Your name <span className="font-normal text-slate-400">(optional)</span></Label>
          <TextInput autoComplete="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Ada Lovelace" />
        </label>
      )}

      {(mode !== "verify" && mode !== "reset") && (
        <label className="block">
          <Label>Email</Label>
          <TextInput type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)}
                     placeholder="you@company.com" />
        </label>
      )}

      {needsCode && (
        <label className="block">
          <Label>6-digit code</Label>
          <TextInput inputMode="numeric" autoComplete="one-time-code" maxLength={6} value={code}
                     onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
                     className="text-center font-mono text-lg tracking-[0.5em]" placeholder="••••••" />
        </label>
      )}

      {needsPassword && (
        <label className="block">
          <Label>{mode === "reset" ? "New password" : "Password"}</Label>
          <div className="relative">
            <TextInput type={showPassword ? "text" : "password"} required value={password}
                       autoComplete={mode === "signin" ? "current-password" : "new-password"}
                       onChange={(e) => setPassword(e.target.value)} className="pr-16" />
            <button type="button" onClick={() => setShowPassword((v) => !v)}
                    className="absolute inset-y-0 right-3 text-xs font-medium text-slate-500 hover:text-slate-700">
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>
          {mode !== "signin" && <span className="mt-1.5 block text-xs text-slate-500">At least 8 characters.</span>}
        </label>
      )}

      {error && <p role="alert" className="text-sm text-alert-600">{error}</p>}

      <Button type="submit" className="w-full" disabled={busy}>{busy ? "Please wait…" : actions[mode]}</Button>

      <div className="flex flex-wrap justify-between gap-2 text-sm">
        {mode === "signin" && (
          <>
            <Link onClick={() => go("signup")}>Create an account</Link>
            <Link onClick={() => go("forgot")}>Forgot password?</Link>
          </>
        )}
        {mode === "signup" && <span className="text-slate-500">Already have one? <Link onClick={() => go("signin")}>Sign in</Link></span>}
        {mode === "verify" && (
          <>
            <Link onClick={() => run(async () => { setInfo((await api.emailResend(email)).message); })}>Send a new code</Link>
            <Link onClick={() => go("signin")}>Back to sign in</Link>
          </>
        )}
        {(mode === "forgot" || mode === "reset") && <Link onClick={() => go("signin")}>Back to sign in</Link>}
      </div>
    </form>
  );
}
