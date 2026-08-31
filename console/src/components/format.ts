/**
 * Formatting, and the one rule it exists to enforce.
 *
 * ABSENCE IS NOT ZERO
 * ===================
 * Every function here takes `null | undefined` and returns an em-dash for it.
 * None of them has a `default` parameter, because a default is a number the
 * console invented - and a dashboard that renders "0%" where it means "we do not
 * know" is worse than one that renders nothing.
 *
 * `formatNumber(0)` returns "0.000" and `formatNumber(null)` returns an
 * em-dash. Those are different claims and they look different.
 */

export const DASH = '—';

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return value.toFixed(digits);
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatInr(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return `₹${Math.round(value).toLocaleString('en-IN')}`;
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return value.toLocaleString('en-IN');
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return DASH;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? DASH : parsed.toISOString().replace('T', ' ').slice(0, 16);
}
