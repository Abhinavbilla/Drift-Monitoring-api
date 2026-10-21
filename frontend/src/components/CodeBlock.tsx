import { useState } from "react";
import { IconCheck, IconCopy } from "./icons";

export function CopyButton({ text, label = "Copy", className = "" }: { text: string; label?: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard can be blocked (e.g. non-HTTPS); the text is still selectable.
    }
  };
  return (
    <button type="button" onClick={copy}
            className={`inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs font-medium transition-colors ${className}`}>
      {copied ? <IconCheck className="h-3.5 w-3.5" /> : <IconCopy className="h-3.5 w-3.5" />}
      {copied ? "Copied" : label}
    </button>
  );
}

/** Dark code panel. With several snippets, shows a tab per language and remembers nothing -- each block is independent. */
export function CodeBlock({ snippets }: { snippets: { label: string; code: string }[] }) {
  const [active, setActive] = useState(0);
  const current = snippets[Math.min(active, snippets.length - 1)];
  return (
    <div className="overflow-hidden rounded-xl border border-slate-800 bg-[#0d1117] text-slate-200 shadow-sm">
      <div className="flex items-center justify-between border-b border-white/10 px-2 py-1.5">
        <div className="flex gap-1">
          {snippets.map((s, i) => (
            <button key={s.label} type="button" onClick={() => setActive(i)}
                    className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                      i === active ? "bg-white/10 text-white" : "text-slate-400 hover:text-slate-200"}`}>
              {s.label}
            </button>
          ))}
        </div>
        <CopyButton text={current.code} className="text-slate-400 hover:bg-white/10 hover:text-white" />
      </div>
      <pre className="overflow-x-auto px-4 py-3.5 font-mono text-[12.5px] leading-relaxed"><code>{current.code}</code></pre>
    </div>
  );
}
