/**
 * Shared display primitives.
 *
 * The pieces here exist to make three things impossible by construction:
 *
 * 1. **Rendering a number the console invented.** `Metric` takes
 *    `number | null | undefined` and renders an em-dash for absence. There is no
 *    parameter for a default.
 * 2. **Showing an error as an empty state.** `ErrorState` renders the backend's
 *    own code and message, so "no orders" and "the database is down" cannot look
 *    alike.
 * 3. **Quoting a probability without its threshold.** `RiskBadge` takes both.
 */

import type { ReactNode } from 'react';

import { ApiError } from '@/api/client';
import { formatNumber } from '@/components/format';
import type { RiskBand } from '@/types/api';

// ---------------------------------------------------------------------------
// states
// ---------------------------------------------------------------------------

export function LoadingState({ label }: { label: string }): JSX.Element {
  return (
    <div className="state state--loading" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function EmptyState({
  title,
  detail,
}: {
  title: string;
  detail?: string | undefined;
}): JSX.Element {
  return (
    <div className="state state--empty">
      <p className="state__title">{title}</p>
      {detail ? <p className="state__detail">{detail}</p> : null}
    </div>
  );
}

/**
 * A failed request, shown as the failure it is.
 *
 * The backend's error code is displayed, not just its prose: `MODEL_UNAVAILABLE`
 * and `ORDER_NOT_FOUND` mean different things to whoever has to fix it, and a
 * generic "something went wrong" throws that away.
 */
export function ErrorState({
  error,
  onRetry,
  context,
}: {
  error: ApiError;
  onRetry?: (() => void) | undefined;
  context?: string | undefined;
}): JSX.Element {
  return (
    <div className="state state--error" role="alert">
      <p className="state__title">
        <span className="error-code">{error.code}</span>
        {context ? ` while ${context}` : ''}
      </p>
      <p className="state__detail">{error.message}</p>
      {onRetry ? (
        <button type="button" className="button button--secondary" onClick={onRetry}>
          Try again
        </button>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// display
// ---------------------------------------------------------------------------

export function Metric({
  label,
  value,
  hint,
  emphasis,
}: {
  label: string;
  value: ReactNode;
  // `| undefined` explicitly: the project builds with
  // `exactOptionalPropertyTypes`, so passing an expression that may be
  // undefined differs from omitting the prop - and passing one is normal here,
  // because a hint that only sometimes applies is normal.
  hint?: string | undefined;
  emphasis?: boolean | undefined;
}): JSX.Element {
  return (
    <div className={`metric${emphasis ? ' metric--emphasis' : ''}`}>
      <dt className="metric__label">{label}</dt>
      <dd className="metric__value">{value}</dd>
      {hint ? <p className="metric__hint">{hint}</p> : null}
    </div>
  );
}

export function MetricGrid({ children }: { children: ReactNode }): JSX.Element {
  return <dl className="metric-grid">{children}</dl>;
}

const BAND_LABELS: Record<RiskBand, string> = {
  LOW: 'Low',
  ELEVATED: 'Elevated',
  HIGH: 'High',
  SEVERE: 'Severe',
};

/**
 * A band with its probability and the threshold that interpreted it.
 *
 * The threshold travels with the probability deliberately. A bare score invites
 * the reader to compare it against 0.5, which is the error this whole system
 * exists to correct: the operating point here is derived from the merchant's own
 * economics and is nowhere near 0.5.
 */
export function RiskBadge({
  band,
  probability,
  threshold,
}: {
  band: RiskBand | string;
  probability?: number | null | undefined;
  threshold?: number | null | undefined;
}): JSX.Element {
  const key = String(band).toLowerCase();
  return (
    <span className={`badge badge--${key}`}>
      <strong>{BAND_LABELS[band as RiskBand] ?? band}</strong>
      {probability !== undefined && probability !== null ? (
        <span className="badge__detail">
          p={formatNumber(probability)}
          {threshold !== undefined && threshold !== null
            ? ` vs threshold ${formatNumber(threshold)}`
            : ''}
        </span>
      ) : null}
    </span>
  );
}

const ACTION_LABELS: Record<string, string> = {
  none: 'No friction',
  prepaid_nudge: 'Prepaid nudge',
  confirmation_required: 'Confirmation required',
  prepaid_only: 'Prepaid only',
};

export function ActionLabel({ action }: { action: string }): JSX.Element {
  return <span className={`action action--${action}`}>{ACTION_LABELS[action] ?? action}</span>;
}

/**
 * Marks a figure that rests on an assumption nobody has measured.
 *
 * The intervention success and abandonment rates are the numbers every rupee
 * figure in this system is most sensitive to, and no controlled holdout has run.
 * Showing them beside measured metrics without this marker would invite a
 * merchant to treat both as equally established.
 */
export function AssumptionTag({ children }: { children?: ReactNode }): JSX.Element {
  return (
    <span className="tag tag--assumption" title="Assumed, never measured on this data">
      {children ?? 'assumption'}
    </span>
  );
}

export function Panel({
  title,
  subtitle,
  actions,
  children,
}: {
  title: string;
  subtitle?: ReactNode | undefined;
  actions?: ReactNode | undefined;
  children: ReactNode;
}): JSX.Element {
  return (
    <section className="panel">
      <header className="panel__header">
        <div>
          <h2 className="panel__title">{title}</h2>
          {subtitle ? <p className="panel__subtitle">{subtitle}</p> : null}
        </div>
        {actions ? <div className="panel__actions">{actions}</div> : null}
      </header>
      {children}
    </section>
  );
}

export function ProvenanceNote({ children }: { children: ReactNode }): JSX.Element {
  return <p className="provenance">{children}</p>;
}
