import { useEffect, useRef } from "react";

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

// Google's library is global: initialize it once and route its callback to
// whichever button is mounted, instead of re-initializing on every render.
let initialized = false;
let handleCredential: ((idToken: string) => void) | null = null;

type ButtonText = "signin_with" | "signup_with" | "continue_with";

export function GoogleSignInButton({ onToken, text = "signin_with" }: {
  onToken: (idToken: string) => void;
  text?: ButtonText;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    handleCredential = onToken;
  }, [onToken]);

  useEffect(() => {
    let cancelled = false;
    const render = () => {
      if (cancelled || !ref.current || !CLIENT_ID) return;
      if (!window.google) {
        setTimeout(render, 100);
        return;
      }
      if (!initialized) {
        window.google.accounts.id.initialize({
          client_id: CLIENT_ID,
          callback: (resp) => handleCredential?.(resp.credential),
        });
        initialized = true;
      }
      ref.current.innerHTML = "";
      window.google.accounts.id.renderButton(ref.current, {
        theme: "outline",
        size: "large",
        shape: "pill",
        text,
        logo_alignment: "center",
        width: Math.min(400, ref.current.offsetWidth || 320),  // Google caps the button at 400px
      });
    };
    render();
    return () => {
      cancelled = true;
    };
  }, [text]);

  if (!CLIENT_ID) {
    return (
      <p className="text-sm text-alert-600">
        VITE_GOOGLE_CLIENT_ID is not set -- see frontend/.env.example.
      </p>
    );
  }

  return <div ref={ref} className="flex h-11 w-full justify-center" />;
}
