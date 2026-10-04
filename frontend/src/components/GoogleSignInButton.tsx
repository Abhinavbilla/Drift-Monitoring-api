import { useEffect, useRef } from "react";

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

export function GoogleSignInButton({ onToken }: { onToken: (idToken: string) => void }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    let cancelled = false;
    const tryInit = () => {
      if (cancelled) return;
      if (!window.google) {
        setTimeout(tryInit, 100);
        return;
      }
      window.google.accounts.id.initialize({
        client_id: CLIENT_ID,
        callback: (resp) => onToken(resp.credential),
      });
      if (ref.current) {
        window.google.accounts.id.renderButton(ref.current, {
          theme: "outline",
          size: "large",
          width: 280,
          shape: "pill",
        });
      }
    };
    tryInit();

    return () => {
      cancelled = true;
    };
  }, [onToken]);

  if (!CLIENT_ID) {
    return (
      <p className="text-sm text-alert-600">
        VITE_GOOGLE_CLIENT_ID is not set -- see frontend/.env.example.
      </p>
    );
  }

  return <div ref={ref} />;
}
