export function timeAgo(iso: string): string {
  const seconds = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  const units: [number, string][] = [[60, "minute"], [3600, "hour"], [86400, "day"], [604800, "week"]];
  let label = "";
  for (const [size, name] of units) {
    if (seconds >= size) label = `${Math.floor(seconds / size)} ${name}`;
  }
  if (seconds >= 2592000) return new Date(iso).toLocaleDateString();
  return `${label}${label.startsWith("1 ") ? "" : "s"} ago`;
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

/** Uses the viewer's local clock; after midnight still reads as evening, not "morning". */
export function greeting(): string {
  const h = new Date().getHours();
  if (h >= 5 && h < 12) return "Good morning";
  if (h >= 12 && h < 17) return "Good afternoon";
  return "Good evening";
}
