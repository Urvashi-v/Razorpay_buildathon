/**
 * The fairness and drift screen, against stubbed HTTP.
 *
 * `fetch` is stubbed rather than the endpoint module, so URL construction is
 * under test too — the split query parameter is the thing most likely to be
 * silently wrong, and stubbing `endpoints.ts` would hide it.
 *
 * The assertions that matter most are the negative ones: a thin cohort must not
 * render like a solid one, and a drift report with no labels must not render as
 * an all-clear.
 */

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import Responsible from '@/pages/Responsible';

const COHORT = {
  cohort: 'pincode_tier',
  group: 'tier_3',
  n_orders: 582,
  flag_rate: 0.28,
  precision: 0.515,
  recall: 0.613,
  net_inr_per_1000: 9482,
  rto_rate: 0.235,
  n_positives: 137,
  n_flagged: 163,
  flag_rate_ci: [0.25, 0.32] as [number, number],
  precision_ci: [0.44, 0.59] as [number, number],
  rto_rate_ci: [0.2, 0.27] as [number, number],
  sufficient: true,
  insufficient_reason: '',
};

const THIN_COHORT = {
  ...COHORT,
  group: 'tier_9',
  n_orders: 12,
  n_flagged: 4,
  sufficient: false,
  insufficient_reason: 'only 12 orders, below the minimum support of 100',
};

const FAIRNESS = {
  generated_at: '2026-08-31T12:00:00Z',
  dataset_run_id: 'run-1',
  split: 'validation',
  model_version: 'a0d780424b79',
  threshold: 0.3481,
  disclaimer: 'Controlled benchmark experiment on synthetic data.',
  audit: {
    slices: [COHORT, THIN_COHORT],
    max_flag_rate_ratio: 2.21,
    worst_precision_drop: 0.036,
    triggered: false,
    narrative: 'This does not trip the configured review.',
    cohorts_examined: ['pincode_tier'],
    groups_below_support: ['pincode_tier=tier_9 (12 orders)'],
    min_support: 100,
    most_flagged_group: 'pincode_tier=tier_3',
    least_flagged_group: 'pincode_tier=tier_1',
  },
  slices: [COHORT, THIN_COHORT],
};

const SHIFT = {
  generated_at: '2026-08-31T12:00:00Z',
  model_version: 'a0d780424b79',
  feature_version: '1.0.0',
  generator_version: '1.0.0',
  threshold: 0.3481,
  environments: [
    { name: 'reference', description: 'the unshifted world', overrides: {}, seed: 1, n_orders: 9000 },
    {
      name: 'rto_base_rate_up',
      description: 'the COD RTO base rate rises',
      overrides: { 'base_rates.rto_given_cod': 0.38 },
      seed: 2,
      n_orders: 9000,
    },
  ],
  results: [
    {
      environment: 'reference',
      description: 'the unshifted world',
      n_orders: 8766,
      observed_rto_rate: 0.167,
      pr_auc: 0.43,
      pr_auc_lift: 2.57,
      roc_auc: 0.78,
      brier_score: 0.12,
      expected_calibration_error: 0.025,
      threshold: 0.3481,
      flag_rate: 0.181,
      precision: 0.41,
      recall: 0.44,
      net_inr_per_1000: 2276,
      pr_auc_delta: null,
      pr_auc_lift_delta: null,
      net_delta: null,
      ece_delta: null,
    },
    {
      environment: 'rto_base_rate_up',
      description: 'the COD RTO base rate rises',
      n_orders: 8767,
      observed_rto_rate: 0.237,
      pr_auc: 0.56,
      pr_auc_lift: 2.37,
      roc_auc: 0.78,
      brier_score: 0.15,
      expected_calibration_error: 0.038,
      threshold: 0.3481,
      flag_rate: 0.216,
      precision: 0.548,
      recall: 0.5,
      net_inr_per_1000: 8730,
      pr_auc_delta: 0.13,
      pr_auc_lift_delta: -0.2,
      net_delta: 6454,
      ece_delta: 0.013,
    },
  ],
  findings: ['rto_base_rate_up: ranking lift fell by 0.20x to 2.37x'],
  data_provenance: 'Controlled benchmark experiment.',
};

function driftReport(labelsAvailable: boolean) {
  return {
    generated_at: '2026-08-31T12:00:00Z',
    baseline: {
      label: 'baseline',
      n_orders: 1220,
      start: null,
      end: null,
      n_matured: 1220,
    },
    current: {
      label: 'current',
      n_orders: 814,
      start: null,
      end: null,
      n_matured: labelsAvailable ? 814 : 0,
    },
    signals: [
      {
        name: 'discount_depth',
        kind: 'feature',
        statistic: 'psi',
        distance: 0.5474,
        severity: 'investigate',
        baseline_value: 0.21,
        current_value: 0.34,
        baseline_n: 1220,
        current_n: 814,
        sufficient: true,
        note: '',
      },
    ],
    performance: labelsAvailable
      ? [
          {
            metric: 'pr_auc',
            baseline: 0.4966,
            current: 0.4664,
            delta: -0.0302,
            n_baseline_matured: 1220,
            n_current_matured: 814,
            sufficient: true,
            note: '',
          },
        ]
      : [],
    warnings: labelsAvailable
      ? ['2 input feature(s) moved between the windows.']
      : ['No matured outcomes in the current window, so no measurement was possible.'],
    model_version: 'a0d780424b79',
    feature_version: '1.0.0',
    labels_available: labelsAvailable,
    data_provenance: 'Synthetic benchmark data.',
  };
}

function ok(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    headers: new Headers({ 'content-type': 'application/json' }),
    text: async () => JSON.stringify(body),
  } as unknown as Response;
}

function notImplemented(message: string): Response {
  return {
    ok: false,
    status: 501,
    headers: new Headers({ 'content-type': 'application/json' }),
    text: async () =>
      JSON.stringify({ error: { code: 'NOT_IMPLEMENTED', message, detail: null } }),
  } as unknown as Response;
}

let calls: string[] = [];

function stub(options: { labels?: boolean; fairness?: 'ok' | '501' } = {}): void {
  calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      calls.push(url);
      if (url.includes('/evaluation/fairness')) {
        return options.fairness === '501'
          ? notImplemented('The cohort and fairness audit has not been run.')
          : ok(FAIRNESS);
      }
      if (url.includes('/evaluation/shift')) return ok(SHIFT);
      if (url.includes('/monitoring/drift')) return ok(driftReport(options.labels ?? true));
      throw new Error(`unexpected request: ${url}`);
    }),
  );
}

beforeEach(() => stub());
afterEach(() => vi.unstubAllGlobals());

describe('fairness and drift screen', () => {
  it('renders the cohort audit the backend returned', async () => {
    render(<Responsible />);

    await waitFor(() => expect(screen.getByText('tier_3')).toBeInTheDocument());
    expect(screen.getByText('not triggered.', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('2.21')).toBeInTheDocument();
    expect(screen.getByText(/does not trip the configured review/)).toBeInTheDocument();
  });

  it('requests the split the user selected', async () => {
    render(<Responsible />);
    await waitFor(() => expect(screen.getByText('tier_3')).toBeInTheDocument());
    expect(calls.some((url) => url.includes('split=validation'))).toBe(true);

    await userEvent.selectOptions(screen.getByLabelText('Split'), 'test');

    await waitFor(() => expect(calls.some((url) => url.includes('split=test'))).toBe(true));
  });

  it('marks a thin cohort so it cannot be mistaken for evidence', async () => {
    render(<Responsible />);

    await waitFor(() => expect(screen.getByText('tier_9')).toBeInTheDocument());
    const row = screen.getByText('tier_9').closest('tr');

    expect(row).toHaveClass('row--thin');
    expect(within(row!).getByText('too thin to read')).toBeInTheDocument();
  });

  it('shows Wilson intervals next to the rates they qualify', async () => {
    render(<Responsible />);

    await waitFor(() => expect(screen.getByText('tier_3')).toBeInTheDocument());
    const row = screen.getByText('tier_3').closest('tr');

    expect(within(row!).getByText('[0.25, 0.32]')).toBeInTheDocument();
    expect(within(row!).getByText('[0.44, 0.59]')).toBeInTheDocument();
  });

  it('shows the backend reason when the audit has not been run', async () => {
    stub({ fairness: '501' });
    render(<Responsible />);

    await waitFor(() =>
      expect(screen.getByText(/has not been run/)).toBeInTheDocument(),
    );
    // The empty table would read as "we checked and found nothing".
    expect(screen.queryByText('tier_3')).not.toBeInTheDocument();
  });

  it('leads the shift table with lift rather than raw PR-AUC', async () => {
    render(<Responsible />);

    // The environment name appears twice on the page - once in the results table
    // and once in the definition list below it - so the query is scoped to the
    // table rather than made ambiguous.
    await waitFor(() =>
      expect(
        screen.getByRole('table', { name: /performance in each shifted environment/i }),
      ).toBeInTheDocument(),
    );
    const table = screen.getByRole('table', {
      name: /performance in each shifted environment/i,
    });
    const row = within(table).getByText('rto_base_rate_up').closest('tr');

    // Raw PR-AUC rose while lift fell. Both are shown; the lift delta is the
    // one that reflects what actually happened to the model.
    expect(within(row!).getByText('0.560')).toBeInTheDocument();
    expect(within(row!).getByText('2.37×')).toBeInTheDocument();
    expect(within(row!).getByText('-0.20×')).toBeInTheDocument();
    expect(
      screen.getByText(/Read the lift column, not the raw PR-AUC column/),
    ).toBeInTheDocument();
  });

  it('lists the parameter overrides that define each environment', async () => {
    render(<Responsible />);

    await waitFor(() =>
      expect(screen.getByText('base_rates.rto_given_cod=0.38')).toBeInTheDocument(),
    );
    expect(screen.getByText(/the control, no overrides/)).toBeInTheDocument();
  });

  it('renders drift distances separately from labelled comparisons', async () => {
    render(<Responsible />);

    await waitFor(() => expect(screen.getByText('discount_depth')).toBeInTheDocument());
    expect(screen.getByText('Distances — what moved')).toBeInTheDocument();
    expect(screen.getByText('Labelled comparisons — what actually changed')).toBeInTheDocument();
    expect(screen.getByText('pr_auc')).toBeInTheDocument();
  });

  it('does not show an all-clear when there are no labels', async () => {
    stub({ labels: false });
    render(<Responsible />);

    // The phrase appears in the banner and again in the backend's own warning
    // list, which is the point - it is stated twice on purpose.
    await waitFor(() =>
      expect(
        screen.getAllByText(/No matured outcomes in the current window/).length,
      ).toBeGreaterThan(0),
    );
    expect(
      screen.getByText(/absence of alarms here as an absence of information/),
    ).toBeInTheDocument();
    expect(screen.getByText(/not a passing result/)).toBeInTheDocument();
    expect(screen.getByText('not possible')).toBeInTheDocument();
  });

  it('never labels a drift severity as a pass or a failure', async () => {
    render(<Responsible />);

    await waitFor(() => expect(screen.getByText('investigate')).toBeInTheDocument());
    for (const forbidden of ['FAIL', 'PASS', 'model has failed']) {
      expect(screen.queryByText(forbidden)).not.toBeInTheDocument();
    }
  });
});
