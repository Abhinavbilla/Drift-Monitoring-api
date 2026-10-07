import { useId, useState } from "react";
import type {
  ButtonHTMLAttributes, ChangeEvent, DragEvent, HTMLAttributes, InputHTMLAttributes, ReactNode, TextareaHTMLAttributes,
} from "react";
import { IconAlert, IconCheck, IconCheckCircle, IconChevronDown, IconFile, IconInfo, IconUpload } from "./icons";

export function Card({ className = "", children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={`rounded-2xl border border-slate-200/80 bg-white shadow-[var(--shadow-card)] ${className}`} {...rest}>
      {children}
    </div>
  );
}

type ButtonVariant = "primary" | "secondary" | "danger" | "ghost" | "outline";

export function Button({
  variant = "primary",
  size = "md",
  className = "",
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: "sm" | "md" | "lg" }) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-xl font-semibold transition-all duration-150 " +
    "disabled:cursor-not-allowed disabled:opacity-50 active:translate-y-px";
  const sizes = { sm: "px-3 py-1.5 text-xs", md: "px-4 py-2.5 text-sm", lg: "px-5 py-3 text-base" };
  const variants: Record<ButtonVariant, string> = {
    primary: "bg-brand-600 text-white shadow-sm shadow-brand-600/25 hover:bg-brand-700",
    secondary: "bg-slate-100 text-slate-700 hover:bg-slate-200",
    outline: "border border-slate-300 bg-white text-slate-700 hover:border-slate-400 hover:bg-slate-50",
    danger: "bg-alert-600 text-white hover:bg-alert-700",
    ghost: "bg-transparent text-slate-600 hover:bg-slate-100",
  };
  return (
    <button className={`${base} ${sizes[size]} ${variants[variant]} ${className}`} {...rest}>
      {children}
    </button>
  );
}

type Tone = "slate" | "ok" | "warn" | "alert" | "brand";

export function Badge({ tone = "slate", children }: { tone?: Tone; children: ReactNode }) {
  const tones: Record<Tone, string> = {
    slate: "bg-slate-100 text-slate-600 ring-slate-200",
    ok: "bg-ok-50 text-ok-700 ring-ok-100",
    warn: "bg-warn-50 text-warn-700 ring-warn-100",
    alert: "bg-alert-50 text-alert-700 ring-alert-100",
    brand: "bg-brand-50 text-brand-700 ring-brand-100",
  };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ring-inset ${tones[tone]}`}>
      {children}
    </span>
  );
}

/** A status with a coloured dot: the one way the app shows "normal / changed / not checked". */
export function StatusPill({ tone, children }: { tone: Tone; children: ReactNode }) {
  const dot: Record<Tone, string> = {
    slate: "bg-slate-400", ok: "bg-ok-500", warn: "bg-warn-500", alert: "bg-alert-500", brand: "bg-brand-500",
  };
  return (
    <Badge tone={tone}>
      <span className={`h-1.5 w-1.5 rounded-full ${dot[tone]}`} />
      {children}
    </Badge>
  );
}

export function PageHeader({ title, subtitle, actions, eyebrow }: {
  title: string; subtitle?: string; actions?: ReactNode; eyebrow?: string;
}) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-2xl">
        {eyebrow && <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-brand-600">{eyebrow}</p>}
        <h1 className="text-[26px] font-bold leading-tight text-slate-900">{title}</h1>
        {subtitle && <p className="mt-2 text-[15px] leading-relaxed text-slate-500">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none" aria-label="Loading">
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
      <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

export function EmptyState({ title, description, action, icon }: {
  title: string; description?: string; action?: ReactNode; icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-white/60 px-6 py-16 text-center">
      {icon && <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-brand-50 text-brand-600">{icon}</div>}
      <p className="text-base font-semibold text-slate-800">{title}</p>
      {description && <p className="mt-1.5 max-w-md text-sm leading-relaxed text-slate-500">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function InfoPanel({ title, items }: { title: string; items: ReactNode[] }) {
  return (
    <div className="rounded-xl border border-brand-100 bg-brand-50/60 px-4 py-3.5">
      <p className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-brand-700">
        <IconInfo className="h-4 w-4" />
        {title}
      </p>
      <ul className="space-y-1.5 text-sm text-slate-600">
        {items.map((item, i) => (
          <li key={i} className="flex gap-2">
            <span className="text-brand-400">&bull;</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div role="alert" className="flex items-start gap-2.5 rounded-xl border border-alert-100 bg-alert-50 px-4 py-3 text-sm text-alert-700">
      <IconAlert className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

export function SuccessBanner({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2.5 rounded-xl border border-ok-100 bg-ok-50 px-4 py-3 text-sm text-ok-700">
      <IconCheckCircle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

export function Tabs<T extends string>({ options, value, onChange }: {
  options: { value: T; label: string }[]; value: T; onChange: (value: T) => void;
}) {
  return (
    <div className="inline-flex rounded-xl bg-slate-100 p-1">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={`rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
            value === opt.value ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-700"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium text-slate-700">{label}</span>
      {children}
      {hint && <span className="mt-1.5 block text-xs leading-relaxed text-slate-500">{hint}</span>}
    </label>
  );
}

const inputClass =
  "w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 " +
  "transition-shadow focus:border-brand-500 focus:outline-none focus:ring-4 focus:ring-brand-100";

export function TextInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={`${inputClass} ${props.className ?? ""}`} />;
}

export function TextArea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={`${inputClass} font-mono ${props.className ?? ""}`} />;
}

/** Drag-and-drop file zone (also clickable). Same props as the old picker, so every page gets it. */
export function FileInput({ accept, multiple, files, onFiles, prompt }: {
  accept?: string; multiple?: boolean; files: File[]; onFiles: (files: File[]) => void; prompt?: string;
}) {
  const inputId = useId();
  const [dragging, setDragging] = useState(false);

  const take = (list: FileList | null) => {
    const picked = Array.from(list ?? []);
    if (picked.length) onFiles(multiple ? picked : picked.slice(0, 1));
  };
  const onDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setDragging(false);
    take(e.dataTransfer.files);
  };

  return (
    <div>
      <input id={inputId} type="file" accept={accept} multiple={multiple} className="sr-only"
             onChange={(e: ChangeEvent<HTMLInputElement>) => take(e.target.files)} />
      <label
        htmlFor={inputId}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex cursor-pointer items-center gap-4 rounded-2xl border-2 border-dashed px-5 py-4 transition-colors ${
          dragging ? "border-brand-400 bg-brand-50" : "border-slate-200 bg-slate-50/60 hover:border-brand-300 hover:bg-brand-50/50"
        }`}
      >
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white text-brand-600 shadow-sm ring-1 ring-slate-200">
          <IconUpload />
        </span>
        <span className="text-sm">
          <span className="font-semibold text-brand-700">{prompt ?? `Choose ${multiple ? "files" : "a file"}`}</span>
          <span className="text-slate-500"> or drag {multiple ? "them" : "it"} here</span>
        </span>
      </label>

      {files.length > 0 && (
        <ul className="mt-2.5 flex flex-wrap gap-2">
          {files.map((f, i) => (
            <li key={`${f.name}-${i}`}
                className="inline-flex items-center gap-2 rounded-lg bg-white px-2.5 py-1.5 text-xs text-slate-700 ring-1 ring-slate-200">
              <IconFile className="h-3.5 w-3.5 text-slate-400" />
              <span className="max-w-[16rem] truncate">{f.name}</span>
              <span className="text-slate-400">{formatBytes(f.size)}</span>
              <button type="button" onClick={() => onFiles(files.filter((_, j) => j !== i))}
                      className="text-slate-400 hover:text-alert-600" aria-label={`Remove ${f.name}`}>
                &times;
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

export function StatCard({ label, value, hint, icon }: { label: string; value: ReactNode; hint?: ReactNode; icon?: ReactNode }) {
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between">
        <p className="text-sm font-medium text-slate-500">{label}</p>
        {icon && <span className="text-slate-400">{icon}</span>}
      </div>
      <p className="mt-2 font-display text-2xl font-bold text-slate-900">{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </Card>
  );
}

/** Horizontal step indicator for multi-step flows. */
export function Stepper({ steps, current }: { steps: string[]; current: number }) {
  return (
    <ol className="flex flex-wrap items-center gap-x-2 gap-y-3">
      {steps.map((label, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <li key={label} className="flex items-center gap-2">
            <span className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-colors ${
              done ? "bg-ok-500 text-white" : active ? "bg-brand-600 text-white ring-4 ring-brand-100" : "bg-slate-200 text-slate-500"
            }`}>
              {done ? <IconCheck className="h-4 w-4" /> : i + 1}
            </span>
            <span className={`text-sm font-medium ${active ? "text-slate-900" : done ? "text-slate-600" : "text-slate-400"}`}>
              {label}
            </span>
            {i < steps.length - 1 && <span className="mx-1 hidden h-px w-8 bg-slate-200 sm:block" />}
          </li>
        );
      })}
    </ol>
  );
}

/** Collapsible area for detail that most people don't need ("For specialists"). */
export function Details({ summary = "Technical details", children }: { summary?: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button type="button" onClick={() => setOpen((v) => !v)}
              className="inline-flex items-center gap-1 text-xs font-medium text-slate-500 hover:text-brand-700">
        {summary}
        <IconChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && <div className="mt-2 space-y-1 text-xs leading-relaxed text-slate-500">{children}</div>}
    </div>
  );
}
