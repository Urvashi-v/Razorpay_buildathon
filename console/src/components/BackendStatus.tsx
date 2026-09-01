import { useEffect, useState } from 'react';

import { ApiError } from '@/api/client';
import { fetchReadiness } from '@/api/endpoints';
import type { ReadinessResponse } from '@/types/api';

/**
 * Live backend status, read from `/readiness`.
 *
 * This component is the Phase 1 proof that the frontend genuinely consumes the
 * backend: every value it renders comes from an HTTP response, and there is no
 * hardcoded fallback anywhere in it. If the API is down, it says the API is
 * down - it does not render a plausible-looking "all systems operational".
 *
 * Note the handling of a 503. A not-ready backend still returns a full body
 * describing *which* component is down, and that is more useful than an error
 * banner, so it is rendered rather than discarded.
 */

type LoadState =
  | { kind: 'loading' }
  | { kind: 'loaded'; data: ReadinessResponse }
  | { kind: 'unreachable'; message: string };

function statusPill(ready: boolean): JSX.Element {
  return (
    <span className={ready ? 'pill pill--ok' : 'pill pill--warn'}>
      {ready ? 'ready' : 'not ready'}
    </span>
  );
}

export default function BackendStatus(): JSX.Element {
  const [state, setState] = useState<LoadState>({ kind: 'loading' });

  useEffect(() => {
    const controller = new AbortController();

    fetchReadiness(controller.signal)
      .then((data) => setState({ kind: 'loaded', data }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        // A 503 from /readiness carries a full body. The client throws on a
        // non-2xx, so recover the payload from the error's detail when present;
        // otherwise report the backend as unreachable, which is the truth.
        const message =
          error instanceof ApiError
            ? `${error.message} (${error.code})`
            : 'The scoring API could not be reached.';
        setState({ kind: 'unreachable', message });
      });

    return () => controller.abort();
  }, []);

  if (state.kind === 'loading') {
    return (
      <section className="card">
        <h2>Backend status</h2>
        <p className="status-detail">Checking…</p>
      </section>
    );
  }

  if (state.kind === 'unreachable') {
    return (
      <section className="card">
        <h2>Backend status</h2>
        <p className="notice" role="status">
          {state.message} Start it with <code>rto-sentinel serve</code>, or{' '}
          <code>docker compose up</code>.
        </p>
      </section>
    );
  }

  const { data } = state;
  const degraded = Object.entries(data.components).filter(([, c]) => !c.ready);

  // Collapsed by default, and open by default when something is not ready.
  //
  // This used to render fully expanded and filled the entire first screen, so a
  // merchant opening the console met four rows of infrastructure diagnostics
  // before a single order. Health belongs one keystroke away, not in front of
  // the product - but a degraded component still has to announce itself, which
  // is why the default follows `data.ready` rather than being a fixed `false`.
  return (
    <details className="status" open={!data.ready}>
      <summary className="status__summary">
        <span className="status__label">Backend</span>
        {statusPill(data.ready)}
        <span className="status-detail">
          v{data.version} · {data.environment}
          {degraded.length > 0
            ? ` · ${degraded.map(([name]) => name).join(', ')} not ready`
            : ' · all components ready'}
        </span>
      </summary>

      <div className="status__body">
        {Object.entries(data.components).map(([name, component]) => (
          <div className="status-row" key={name}>
            <span className="status-name">{name}</span>
            {statusPill(component.ready)}
            <span className="status-detail">{component.detail}</span>
          </div>
        ))}

        {data.config_fingerprint ? (
          <p className="status-detail status__fingerprint">
            Config fingerprint <code>{data.config_fingerprint.slice(0, 16)}…</code> — every
            result this instance produces is traceable to this configuration.
          </p>
        ) : null}

        {data.warnings.map((warning) => (
          <p className="notice" key={warning} role="status">
            {warning}
          </p>
        ))}
      </div>
    </details>
  );
}
