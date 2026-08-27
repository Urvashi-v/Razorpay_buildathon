import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import BackendStatus from '@/components/BackendStatus';

/**
 * The status panel renders what the backend said, and nothing else.
 *
 * The last test is the important one: with the API unreachable, the component
 * must NOT render a reassuring default. A console that shows "ready" when it
 * cannot reach the backend is worse than one that shows nothing.
 */

function mockReadiness(body: unknown, status = 200): void {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      text: async () => JSON.stringify(body),
    }),
  );
}

const READY_BODY = {
  ready: true,
  version: '0.1.0',
  environment: 'test',
  config_fingerprint: 'bd87157f2b7730e2c72f58ff686495d99da7970eef3126dd42d74c0aec4d416c',
  components: {
    configuration: { ready: true, detail: 'All configuration files parsed and validated.' },
    model: { ready: true, detail: 'Loaded from artifacts/models/rung4.pkl' },
    agents: { ready: false, detail: 'Unavailable (ANTHROPIC_API_KEY is not set).' },
  },
  warnings: ['Language layer unavailable - explanations will be reason codes only.'],
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('BackendStatus', () => {
  it('renders each component reported by the backend', async () => {
    mockReadiness(READY_BODY);
    render(<BackendStatus />);

    await waitFor(() => expect(screen.getByText('configuration')).toBeInTheDocument());
    expect(screen.getByText('model')).toBeInTheDocument();
    expect(screen.getByText('agents')).toBeInTheDocument();
    expect(
      screen.getByText('All configuration files parsed and validated.'),
    ).toBeInTheDocument();
  });

  it('surfaces the config fingerprint so results stay traceable', async () => {
    mockReadiness(READY_BODY);
    render(<BackendStatus />);
    await waitFor(() => expect(screen.getByText(/bd87157f2b7730e2/)).toBeInTheDocument());
  });

  it('shows the degraded-language-layer warning verbatim', async () => {
    mockReadiness(READY_BODY);
    render(<BackendStatus />);
    await waitFor(() =>
      expect(
        screen.getByText(/explanations will be reason codes only/i),
      ).toBeInTheDocument(),
    );
  });

  it('renders the full report when the backend answers 503 not-ready', async () => {
    // A not-ready instance returns 503 WITH a complete body. Treating that as a
    // generic failure would hide the single most useful fact on the page: which
    // component is down and why.
    mockReadiness(
      {
        ...READY_BODY,
        ready: false,
        components: {
          ...READY_BODY.components,
          model: {
            ready: false,
            detail: 'No model artefact configured. Scoring is unavailable.',
          },
        },
      },
      503,
    );
    render(<BackendStatus />);

    await waitFor(() =>
      expect(screen.getByText(/no model artefact configured/i)).toBeInTheDocument(),
    );
    // Three not-ready pills: the overall verdict, plus model and agents.
    expect(screen.getAllByText('not ready')).toHaveLength(3);
    expect(screen.getByText(/all configuration files parsed/i)).toBeInTheDocument();
  });

  it('reports an unreachable backend instead of a reassuring default', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('connection refused')));
    render(<BackendStatus />);

    // The message is interleaved with <code> elements, so match on the status
    // region's full text rather than a single text node.
    await waitFor(() =>
      expect(screen.getByRole('status')).toHaveTextContent(/could not reach the api/i),
    );
    expect(screen.queryByText('ready')).not.toBeInTheDocument();
    expect(screen.queryByText('not ready')).not.toBeInTheDocument();
  });
});
