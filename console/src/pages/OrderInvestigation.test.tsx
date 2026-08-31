/**
 * The investigation screen renders what the risk endpoint returned, and nothing else.
 *
 * WHAT THESE TESTS DEFEND
 * =======================
 * The whole argument of this console is that every displayed number came from
 * the backend. That is only checkable if the tests assert on values the
 * component could not have produced itself - so the fixtures here use distinctive
 * numbers, and the assertions look for exactly those.
 *
 * The negative cases matter more than the positive one. A console that shows a
 * plausible screen when the model is unavailable is worse than one that crashes,
 * because nobody finds out.
 *
 * `fetch` is stubbed rather than the endpoint module: the request URL, the query
 * parameters and the response parsing are all part of what is under test. Mocking
 * `fetchRiskAssessment` would test the component against a function this suite
 * wrote, which proves very little.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import OrderInvestigation from '@/pages/OrderInvestigation';
import type { RiskAssessmentResponse } from '@/types/api';

/**
 * A response shaped exactly like the backend's, with values chosen to be
 * recognisable: 0.5719 is not a number a component would invent.
 */
const ASSESSMENT: RiskAssessmentResponse = {
  order: {
    order_id: 'ORD-00008874',
    merchant_id: 'M-DEMO-001',
    customer_hash: '546e7a386f62c4f41bc0d00dc4c45bb5',
    ordered_at: '2026-02-25T17:00:20Z',
    payment_method: 'cod',
    is_cod: true,
    order_value_inr: 3902,
    discount_inr: 120,
    item_count: 1,
    category: 'electronics',
    courier_partner: 'BlueDart',
    split: 'validation',
    dataset_run_id: 'd1efe6b75393a8f95dac4c6a',
    is_rto: false,
    outcome: 'delivered',
    resolved_at: '2026-02-26T21:00:20Z',
  },
  probability: 0.5719,
  raw_score: 0.5327,
  threshold: 0.3481,
  band: 'HIGH',
  action: 'confirmation_required',
  flagged: true,
  reason_codes: ['ORDER_IS_COD', 'HISTORY_VALUE_VS_PRIOR_MEAN'],
  expected_value_inr: 45.31,
  appeal_available: true,
  human_review_required: false,
  is_control_holdout: false,
  contributions: [
    { feature: 'order_is_cod', family: 'order_shape', value: true, contribution: 1.1611 },
    {
      feature: 'cust_value_vs_prior_mean',
      family: 'customer_history',
      value: 3.85,
      contribution: 0.6019,
    },
  ],
  model: {
    model_name: 'lightgbm_platt',
    model_version: 'a0d780424b79',
    calibration_method: 'platt',
    calibration_fitted_on: 'validation',
    feature_version: '1.0.0',
    feature_fingerprint: '798aef57ad3cefe9',
    dataset_run_id: 'd1efe6b75393a8f95dac4c6a',
    generator_version: '1.0.0',
    trained_at: '2026-08-29T02:17:52Z',
    training_rows: 23058,
    n_features: 54,
    selection_manifest_id: '4f17cd1f1279d897d589',
  },
  features: {
    feature_version: '1.0.0',
    feature_fingerprint: '798aef57ad3cefe9',
    n_features: 54,
    null_features: ['geo_pincode_rto_rate_smoothed'],
    context_rows: 8874,
  },
  economics: {
    cost_profile: 'mid_margin_d2c',
    rto_cost_inr: 220,
    contribution_margin_inr: 250,
    friction_support_cost_inr: 8,
    abandonment_on_friction: 0.25,
    intervention_success_rate: 0.6,
    cost_false_positive_inr: 70.5,
    saving_true_positive_inr: 132,
    threshold_formula: 'threshold = C_fp / (C_fp + S_tp)',
    band_intervention_success_rate: 0.6,
    band_abandonment_rate: 0.25,
  },
  engine_version: '1.0.0',
  scored_at: '2026-08-30T13:45:47Z',
  latency_ms: 128.4,
  outcome_is_known: true,
  data_provenance: 'Synthetic benchmark data. Labels are simulated, not real-world ground truth.',
};

const AGENT_OFF = {
  available: false,
  reason: 'RTO_AGENTS_ENABLED is false',
  provider: 'anthropic',
  model: 'unavailable',
  required_environment_variable: 'ANTHROPIC_API_KEY',
  enable_switch: 'RTO_AGENTS_ENABLED',
  tools: ['get_order'],
  note: 'The risk system does not depend on this layer.',
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** Route by URL, so the test exercises the real URL construction. */
function stubFetch(handler: (url: string) => Response): void {
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL) => Promise.resolve(handler(String(input)))),
  );
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('order investigation', () => {
  it('renders the risk the backend returned', async () => {
    stubFetch((url) => {
      if (url.includes('/v1/orders/ORD-00008874/risk')) return jsonResponse(ASSESSMENT);
      if (url.includes('/v1/explanations/status')) return jsonResponse(AGENT_OFF);
      throw new Error(`unexpected request: ${url}`);
    });

    render(<OrderInvestigation orderId="ORD-00008874" onOrderIdChange={() => {}} />);

    // The probability, band and threshold are the backend's, not the component's.
    expect(await screen.findByText('0.5719')).toBeInTheDocument();
    expect(screen.getByText('0.3481')).toBeInTheDocument();
    expect(screen.getAllByText(/High/).length).toBeGreaterThan(0);
    expect(screen.getByText('Confirmation required')).toBeInTheDocument();

    // Provenance travels with the number.
    expect(screen.getByText('a0d780424b79')).toBeInTheDocument();
    expect(screen.getByText('lightgbm_platt')).toBeInTheDocument();

    // Reason codes and attributions came from the model.
    expect(screen.getByText('ORDER_IS_COD')).toBeInTheDocument();
    expect(screen.getByText('order_is_cod')).toBeInTheDocument();
    expect(screen.getByText('+1.1611')).toBeInTheDocument();
  });

  it('requests the right URL with contributions included', async () => {
    const calls: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        calls.push(url);
        if (url.includes('/risk')) return Promise.resolve(jsonResponse(ASSESSMENT));
        return Promise.resolve(jsonResponse(AGENT_OFF));
      }),
    );

    render(<OrderInvestigation orderId="ORD-00008874" onOrderIdChange={() => {}} />);
    await screen.findByText('0.5719');

    const riskCall = calls.find((url) => url.includes('/risk'));
    expect(riskCall).toContain('/v1/orders/ORD-00008874/risk');
    expect(riskCall).toContain('include_contributions=true');
  });

  it('labels the economic assumptions as assumptions', async () => {
    stubFetch((url) =>
      url.includes('/risk') ? jsonResponse(ASSESSMENT) : jsonResponse(AGENT_OFF),
    );

    render(<OrderInvestigation orderId="ORD-00008874" onOrderIdChange={() => {}} />);
    await screen.findByText('0.5719');

    // Both unmeasured rates are tagged, so a merchant cannot read them as facts.
    expect(screen.getAllByText('assumption').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText('threshold = C_fp / (C_fp + S_tp)')).toBeInTheDocument();
  });

  it('shows a real loading state before the response arrives', () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => {})));

    render(<OrderInvestigation orderId="ORD-00008874" onOrderIdChange={() => {}} />);

    expect(screen.getByRole('status')).toHaveTextContent(/Scoring the order/i);
    // Nothing is rendered in place of the missing data.
    expect(screen.queryByText('0.5719')).not.toBeInTheDocument();
  });

  it('shows the backend error rather than a plausible screen', async () => {
    stubFetch((url) => {
      if (url.includes('/risk')) {
        return jsonResponse(
          {
            error: {
              code: 'MODEL_UNAVAILABLE',
              message: 'no calibrated model artefact under artifacts/models.',
            },
          },
          503,
        );
      }
      return jsonResponse(AGENT_OFF);
    });

    render(<OrderInvestigation orderId="ORD-00008874" onOrderIdChange={() => {}} />);

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('MODEL_UNAVAILABLE');
    expect(alert).toHaveTextContent(/no calibrated model artefact/);
    // The critical assertion: no invented risk anywhere on the page.
    expect(screen.queryByText(/0\.\d{4}/)).not.toBeInTheDocument();
  });

  it('reports a missing order as not found', async () => {
    stubFetch((url) => {
      if (url.includes('/risk')) {
        return jsonResponse(
          { error: { code: 'ORDER_NOT_FOUND', message: "no order 'ORD-99999999' exists." } },
          404,
        );
      }
      return jsonResponse(AGENT_OFF);
    });

    render(<OrderInvestigation orderId="ORD-99999999" onOrderIdChange={() => {}} />);

    expect(await screen.findByRole('alert')).toHaveTextContent('ORDER_NOT_FOUND');
  });

  it('scores nothing until an order is chosen', () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);

    render(<OrderInvestigation orderId={null} onOrderIdChange={() => {}} />);

    expect(screen.getByText(/Nothing is scored until you ask/i)).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('distinguishes an unresolved outcome from a delivered one', async () => {
    const immature = {
      ...ASSESSMENT,
      order: { ...ASSESSMENT.order, is_rto: null, outcome: 'pending', resolved_at: null },
      outcome_is_known: false,
    };
    stubFetch((url) => (url.includes('/risk') ? jsonResponse(immature) : jsonResponse(AGENT_OFF)));

    render(<OrderInvestigation orderId="ORD-00008874" onOrderIdChange={() => {}} />);

    expect(await screen.findByText('not yet known')).toBeInTheDocument();
    expect(screen.getByText(/Immature — not the same as delivered/)).toBeInTheDocument();
  });
});

describe('the agent panel when the language layer is off', () => {
  it('says so, and does not fabricate an explanation', async () => {
    stubFetch((url) => (url.includes('/risk') ? jsonResponse(ASSESSMENT) : jsonResponse(AGENT_OFF)));

    render(<OrderInvestigation orderId="ORD-00008874" onOrderIdChange={() => {}} />);

    expect(await screen.findByText(/language layer is not configured/i)).toBeInTheDocument();
    expect(screen.getByText('ANTHROPIC_API_KEY')).toBeInTheDocument();

    // The "Ask" control is disabled rather than producing something.
    const ask = screen.getByRole('button', { name: 'Ask' });
    expect(ask).toBeDisabled();
  });

  it('surfaces an agent failure instead of showing an answer', async () => {
    const user = userEvent.setup();
    stubFetch((url) => {
      if (url.includes('/risk')) return jsonResponse(ASSESSMENT);
      if (url.includes('/status')) return jsonResponse({ ...AGENT_OFF, available: true });
      if (url.includes('/investigate')) {
        return jsonResponse(
          {
            error: {
              code: 'AGENT_UNAVAILABLE',
              message: 'the Anthropic API call failed (APITimeoutError).',
            },
          },
          503,
        );
      }
      throw new Error(`unexpected request: ${url}`);
    });

    render(<OrderInvestigation orderId="ORD-00008874" onOrderIdChange={() => {}} />);
    await screen.findByText('0.5719');

    await user.click(await screen.findByRole('button', { name: 'Ask' }));

    await waitFor(() => {
      expect(screen.getAllByRole('alert').some((node) => node.textContent?.includes('AGENT_UNAVAILABLE'))).toBe(true);
    });
    expect(screen.getByText(/No explanation is shown because none was produced/i)).toBeInTheDocument();
  });
});
