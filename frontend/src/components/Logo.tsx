import { useId } from "react";

export function LogoMark({ className = "h-9 w-9" }: { className?: string }) {
  // Unique per instance: a shared id resolves to the first copy, which may be hidden (display:none).
  const gradient = `ds-mark-${useId().replace(/:/g, "")}`;
  return (
    <svg viewBox="0 0 40 40" className={className} aria-hidden="true">
      <defs>
        <linearGradient id={gradient} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#818cf8" />
          <stop offset="1" stopColor="#4f46e5" />
        </linearGradient>
      </defs>
      <rect width="40" height="40" rx="11" fill={`url(#${gradient})`} />
      {/* a steady line that bends: the moment data starts to drift */}
      <path d="M8 24h7l3-8 4 13 3-7h7" fill="none" stroke="white" strokeWidth="2.6" strokeLinecap="round"
            strokeLinejoin="round" />
    </svg>
  );
}

export function Logo({ dark = false }: { dark?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <LogoMark />
      <div className="leading-tight">
        <p className={`font-display text-[15px] font-bold ${dark ? "text-white" : "text-slate-900"}`}>Drift Sentinel</p>
        <p className={`text-[11px] ${dark ? "text-slate-400" : "text-slate-500"}`}>Know when your data changes</p>
      </div>
    </div>
  );
}
