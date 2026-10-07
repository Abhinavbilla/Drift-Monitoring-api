import type { SVGProps } from "react";

// A small, consistent icon set (24px grid, 1.75 stroke), drawn in the style of
// Lucide so the app doesn't need an icon dependency for a dozen glyphs.
type IconProps = SVGProps<SVGSVGElement>;

function Icon({ children, className = "h-5 w-5", ...rest }: IconProps & { children: React.ReactNode }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round"
         strokeLinejoin="round" className={className} aria-hidden="true" {...rest}>
      {children}
    </svg>
  );
}

export const IconGrid = (p: IconProps) => (
  <Icon {...p}><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" />
    <rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></Icon>
);
export const IconHome = (p: IconProps) => (
  <Icon {...p}><path d="M3 10.5 12 3l9 7.5" /><path d="M5 9.5V20a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1V9.5" /></Icon>
);
export const IconTable = (p: IconProps) => (
  <Icon {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 10h18M9 10v10M15 10v10" /></Icon>
);
export const IconClock = (p: IconProps) => (
  <Icon {...p}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></Icon>
);
export const IconLayers = (p: IconProps) => (
  <Icon {...p}><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 13 9 5 9-5" /></Icon>
);
export const IconBell = (p: IconProps) => (
  <Icon {...p}><path d="M6 8a6 6 0 1 1 12 0c0 7 3 8 3 8H3s3-1 3-8" /><path d="M10 20a2 2 0 0 0 4 0" /></Icon>
);
export const IconList = (p: IconProps) => (
  <Icon {...p}><path d="M8 6h13M8 12h13M8 18h13" /><circle cx="3.5" cy="6" r="1" /><circle cx="3.5" cy="12" r="1" />
    <circle cx="3.5" cy="18" r="1" /></Icon>
);
export const IconWrench = (p: IconProps) => (
  <Icon {...p}><path d="M14.7 6.3a4 4 0 0 0 5 5L21 13l-8 8-3-3 8-8-1.3-1.3a4 4 0 0 0-5-5L13 5l1.7 1.3Z" />
    <path d="M3 21l6-6" /></Icon>
);
export const IconLogout = (p: IconProps) => (
  <Icon {...p}><path d="M15 3h3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-3" /><path d="M10 17l-5-5 5-5M5 12h11" /></Icon>
);
export const IconUpload = (p: IconProps) => (
  <Icon {...p}><path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></Icon>
);
export const IconCheck = (p: IconProps) => (
  <Icon {...p}><path d="m5 12.5 4.5 4.5L19 7.5" /></Icon>
);
export const IconCheckCircle = (p: IconProps) => (
  <Icon {...p}><circle cx="12" cy="12" r="9" /><path d="m8 12.5 2.8 2.8L16.5 9.5" /></Icon>
);
export const IconAlert = (p: IconProps) => (
  <Icon {...p}><path d="M12 3 2 20h20L12 3Z" /><path d="M12 10v4M12 17h.01" /></Icon>
);
export const IconInfo = (p: IconProps) => (
  <Icon {...p}><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></Icon>
);
export const IconSparkles = (p: IconProps) => (
  <Icon {...p}><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" /></Icon>
);
export const IconLink = (p: IconProps) => (
  <Icon {...p}><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1" />
    <path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" /></Icon>
);
export const IconHash = (p: IconProps) => (
  <Icon {...p}><path d="M5 9h14M5 15h14M10 3 8 21M16 3l-2 18" /></Icon>
);
export const IconTag = (p: IconProps) => (
  <Icon {...p}><path d="M3 12V4a1 1 0 0 1 1-1h8l9 9-9 9-9-9Z" /><circle cx="8" cy="8" r="1.5" /></Icon>
);
export const IconText = (p: IconProps) => (
  <Icon {...p}><path d="M4 6h16M4 12h16M4 18h10" /></Icon>
);
export const IconImage = (p: IconProps) => (
  <Icon {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="9" cy="10" r="2" /><path d="m21 16-5-5-9 9" /></Icon>
);
export const IconEyeOff = (p: IconProps) => (
  <Icon {...p}><path d="M3 3l18 18" /><path d="M10.6 6.1A9.8 9.8 0 0 1 12 6c5 0 9 6 9 6a17 17 0 0 1-3 3.6M6.6 6.6C4.3 8.2 3 12 3 12s4 6 9 6a9 9 0 0 0 3.4-.7" /></Icon>
);
export const IconArrowRight = (p: IconProps) => (
  <Icon {...p}><path d="M5 12h14M13 6l6 6-6 6" /></Icon>
);
export const IconPlus = (p: IconProps) => (
  <Icon {...p}><path d="M12 5v14M5 12h14" /></Icon>
);
export const IconChevronDown = (p: IconProps) => (
  <Icon {...p}><path d="m6 9 6 6 6-6" /></Icon>
);
export const IconTrash = (p: IconProps) => (
  <Icon {...p}><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></Icon>
);
export const IconShield = (p: IconProps) => (
  <Icon {...p}><path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6l-8-3Z" /><path d="m9 12 2 2 4-4" /></Icon>
);
export const IconPulse = (p: IconProps) => (
  <Icon {...p}><path d="M3 12h4l3-8 4 16 3-8h4" /></Icon>
);
export const IconFile = (p: IconProps) => (
  <Icon {...p}><path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8l-5-5Z" /><path d="M14 3v5h5" /></Icon>
);
