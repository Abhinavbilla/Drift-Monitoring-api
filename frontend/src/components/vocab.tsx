import type { ReactNode } from "react";
import type { TableColumnType } from "../lib/types";
import { IconEyeOff, IconHash, IconImage, IconTag, IconText } from "./icons";

/** Everyday names for the technical concepts, so the UI never makes people learn our vocabulary. */
export const COLUMN_TYPES: Record<TableColumnType, { label: string; help: string; icon: ReactNode }> = {
  numeric: { label: "Numbers", help: "Amounts, counts, prices, ages…", icon: <IconHash /> },
  categorical: { label: "Categories", help: "A fixed set of choices, like country or plan", icon: <IconTag /> },
  text: { label: "Written text", help: "Descriptions, reviews, messages", icon: <IconText /> },
  image: { label: "Images", help: "Photos named in the column, from your ZIP", icon: <IconImage /> },
  ignore: { label: "Skip", help: "IDs, dates, links — not worth watching", icon: <IconEyeOff /> },
};

export const CONNECTION_KINDS: Record<string, string> = {
  num_num: "These two numbers move together",
  cat_cat: "These categories tend to go together",
  num_cat: "This number depends on the category",
  probe: "The text or photo tells us about this column",
  text_image: "Photos match their descriptions",
};

export function TypeIcon({ type, className = "" }: { type: TableColumnType; className?: string }) {
  return (
    <span className={`inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-slate-500 [&>svg]:h-4 [&>svg]:w-4 ${className}`}>
      {COLUMN_TYPES[type].icon}
    </span>
  );
}

/** A sentence naming the actual columns, e.g. "Description tells us about Type". */
export function connectionSentence(kind: string, a: string, b: string): string {
  switch (kind) {
    case "num_num": return `${a} and ${b} move together`;
    case "cat_cat": return `${a} and ${b} tend to go together`;
    case "num_cat": return `${a} depends on ${b}`;
    case "probe": return `${a} tells us about ${b}`;
    case "text_image": return `The photos match the ${a.toLowerCase()}`;
    default: return CONNECTION_KINDS[kind] ?? "";
  }
}

export const pairLabel = (name: string) => name.replace("<->", " ↔ ");
