/**
 * Format a non-negative duration in seconds for TUI display, e.g. **`45s`**, **`1h 2m 3s`**,
 * **`1d 2h 30s`**. Zero-valued middle units are skipped; sub-minute values may show one fractional
 * digit when the input is not a whole second.
 */
export function formatDurationSec(sec: number | null | undefined): string {
  if (sec === null || sec === undefined || !Number.isFinite(sec)) {
    return '—';
  }
  const s = Math.max(0, sec);

  if (s < 60 && Math.abs(s - Math.round(s)) > 1e-6) {
    return `${s.toFixed(1)}s`;
  }

  let rem = Math.round(s);
  const days = Math.floor(rem / 86_400);
  rem %= 86_400;
  const hours = Math.floor(rem / 3600);
  rem %= 3600;
  const minutes = Math.floor(rem / 60);
  rem %= 60;
  const seconds = rem;

  const parts: string[] = [];
  if (days > 0) parts.push(`${String(days)}d`);
  if (hours > 0) parts.push(`${String(hours)}h`);
  if (minutes > 0) parts.push(`${String(minutes)}m`);
  if (seconds > 0 || parts.length === 0) parts.push(`${String(seconds)}s`);

  return parts.join(' ');
}
