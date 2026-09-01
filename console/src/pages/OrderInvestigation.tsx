/**
 * One order, scored end to end by the backend.
 *
 * Requesting this screen runs the whole chain server-side: the database row, the
 * feature pipeline, the trained artefact, the calibrator, and the deterministic
 * decision engine. The console displays what comes back and computes none of it.
 *
 * WHAT IS SHOWN TOGETHER, AND WHY
 * ===============================
 * The probability never appears without the threshold that interpreted it. A
 * bare 0.57 invites comparison against 0.5; the operating point here is derived
 * from the merchant's own economics and is 0.348, and the two numbers only mean
 * something side by side.
 *
 * The economic assumptions are shown with the decision rather than hidden behind
 * a link, and the two rates nobody has measured are tagged as assumptions
 * wherever they appear.
 */

import { useState } from 'react';

import { fetchRiskAssessment } from '@/api/endpoints';
import AgentPanel from '@/components/AgentPanel';
import {
  ActionLabel,
  AssumptionTag,
  ErrorState,
  LoadingState,
  Metric,
  MetricGrid,
  Panel,
  ProvenanceNote,
  RiskBadge,
} from '@/components/primitives';
import {
  DASH,
  formatCount,
  formatDate,
  formatInr,
  formatNumber,
  formatPercent,
} from '@/components/format';
import { useApi } from '@/hooks/useApi';
import type { FeatureContribution, RiskAssessmentResponse } from '@/types/api';

interface Props {
  orderId: string | null;
  onOrderIdChange: (orderId: string) => void;
}

export default function OrderInvestigation({ orderId, onOrderIdChange }: Props): JSX.Element {
  const [draft, setDraft] = useState(orderId ?? '');

  const assessment = useApi(
    (signal) =>
      fetchRiskAssessment(orderId ?? '', { includeContributions: true }, signal),
    [orderId],
    { enabled: Boolean(orderId) },
  );

  return (
    <div className="page">
      <Panel
        title="Order investigation"
        subtitle="Scoring runs the real pipeline on request: database → features → model → calibration → decision engine."
      >
        <form
          className="filters"
          onSubmit={(event) => {
            event.preventDefault();
            if (draft.trim()) onOrderIdChange(draft.trim());
          }}
        >
          <div className="field field--grow">
            <label htmlFor="order-id">Order ID</label>
            <input
              id="order-id"
              type="text"
              value={draft}
              placeholder="ORD-00043224"
              onChange={(event) => setDraft(event.target.value)}
            />
          </div>
          <button type="submit" className="button" disabled={!draft.trim()}>
            Assess
          </button>
        </form>

        {!orderId ? (
          <p className="state__detail">
            Enter an order ID, or choose one from the queue. Nothing is scored until you ask.
          </p>
        ) : null}

        {assessment.status === 'loading' ? (
          <LoadingState label="Scoring the order — this runs the full pipeline and can take a few seconds on a cold model." />
        ) : null}

        {assessment.status === 'error' ? (
          <ErrorState
            error={assessment.error}
            onRetry={assessment.reload}
            context={`assessing ${orderId}`}
          />
        ) : null}

        {assessment.status === 'success' ? <Assessment data={assessment.data} /> : null}
      </Panel>

      {assessment.status === 'success' ? <AgentPanel orderId={assessment.data.order.order_id} /> : null}
    </div>
  );
}

function Assessment({ data }: { data: RiskAssessmentResponse }): JSX.Element {
  const { order, economics, model, features } = data;

  return (
    <>
      <div className="decision-banner">
        <RiskBadge band={data.band} probability={data.probability} threshold={data.threshold} />
        <ActionLabel action={data.action} />
        {data.human_review_required ? <span className="tag tag--review">Human review</span> : null}
        {data.is_control_holdout ? (
          <span className="tag tag--holdout" title="No friction applied, so precision stays measurable">
            Control holdout
          </span>
        ) : null}
        {data.appeal_available ? <span className="tag tag--appeal">Appeal available</span> : null}
      </div>

      <h3 className="subheading">Order</h3>
      <MetricGrid>
        <Metric label="Order" value={<code>{order.order_id}</code>} />
        <Metric label="Placed" value={formatDate(order.ordered_at)} />
        <Metric label="Value" value={formatInr(order.order_value_inr)} />
        <Metric label="Discount" value={formatInr(order.discount_inr)} />
        <Metric label="Items" value={formatCount(order.item_count)} />
        <Metric label="Category" value={order.category} />
        <Metric label="Payment" value={order.payment_method} />
        <Metric label="Courier" value={order.courier_partner ?? DASH} />
        <Metric label="Customer" value={<code>{order.customer_hash.slice(0, 16)}…</code>} />
        <Metric
          label="Outcome"
          value={
            order.is_rto === null ? (
              <span className="chip chip--pending">not yet known</span>
            ) : (
              <span className={`chip chip--${order.is_rto ? 'rto' : 'delivered'}`}>
                {order.outcome}
              </span>
            )
          }
          hint={order.is_rto === null ? 'Immature — not the same as delivered' : undefined}
        />
      </MetricGrid>

      <h3 className="subheading">Risk</h3>
      <MetricGrid>
        <Metric
          label="Calibrated probability"
          value={formatNumber(data.probability, 4)}
          hint="P(RTO | information available at order time)"
          emphasis
        />
        <Metric
          label="Operating threshold"
          value={formatNumber(data.threshold, 4)}
          hint="Derived from merchant economics — not 0.5"
          emphasis
        />
        <Metric label="Raw score" value={formatNumber(data.raw_score, 4)} hint="Pre-calibration" />
        <Metric label="Band" value={data.band} />
        <Metric
          label="Expected value of acting"
          value={formatInr(data.expected_value_inr)}
          hint="Rests on assumed rates"
        />
        <Metric label="Latency" value={data.latency_ms ? `${Math.round(data.latency_ms)} ms` : DASH} />
      </MetricGrid>

      <h3 className="subheading">Why — reason codes</h3>
      {data.reason_codes.length > 0 ? (
        <ul className="code-list">
          {data.reason_codes.map((code) => (
            <li key={code}>
              <code>{code}</code>
            </li>
          ))}
        </ul>
      ) : (
        <p className="state__detail">
          No reason codes: this order was not flagged, so no friction was justified.
        </p>
      )}

      <h3 className="subheading">Feature contributions</h3>
      {data.contributions.length > 0 ? (
        <ContributionTable contributions={data.contributions} />
      ) : (
        <p className="state__detail">
          The model produced no per-feature attributions for this order.
        </p>
      )}

      <h3 className="subheading">Economic decision</h3>
      <MetricGrid>
        <Metric label="Cost profile" value={economics.cost_profile} />
        <Metric label="RTO cost" value={formatInr(economics.rto_cost_inr)} hint="Merchant input" />
        <Metric
          label="Contribution margin"
          value={formatInr(economics.contribution_margin_inr)}
          hint="Merchant input"
        />
        <Metric
          label="Cost of a false positive"
          value={formatInr(economics.cost_false_positive_inr)}
          hint="C_fp"
        />
        <Metric
          label="Saving per true positive"
          value={formatInr(economics.saving_true_positive_inr)}
          hint="S_tp"
        />
        <Metric
          label="Intervention success"
          value={
            <>
              {formatPercent(economics.intervention_success_rate)} <AssumptionTag />
            </>
          }
          hint="Never measured on this data"
        />
        <Metric
          label="Abandonment on friction"
          value={
            <>
              {formatPercent(economics.abandonment_on_friction)} <AssumptionTag />
            </>
          }
          hint="Never measured on this data"
        />
      </MetricGrid>
      <p className="formula">
        <code>{economics.threshold_formula}</code>
      </p>

      <h3 className="subheading">Provenance</h3>
      <MetricGrid>
        <Metric label="Model" value={model.model_name} />
        <Metric label="Model version" value={<code>{model.model_version}</code>} />
        <Metric
          label="Calibration"
          value={model.calibration_method ?? DASH}
          hint={`Fitted on ${model.calibration_fitted_on ?? DASH}`}
        />
        <Metric label="Feature version" value={features.feature_version} />
        <Metric
          label="Features used"
          value={formatCount(features.n_features)}
          hint={`${features.null_features.length} null for this order`}
        />
        <Metric
          label="History depth"
          value={formatCount(features.context_rows)}
          hint="Rows the aggregates were computed over"
        />
        <Metric label="Engine" value={<code>{data.engine_version}</code>} />
        <Metric label="Scored at" value={formatDate(data.scored_at)} />
      </MetricGrid>

      {features.null_features.length > 0 ? (
        <details className="details">
          <summary>{features.null_features.length} features had no value for this order</summary>
          <p className="state__detail">
            Not an error. A first-time customer genuinely has no prior RTO rate, and the model
            handles missingness natively — but a score built largely from nulls deserves less
            confidence than one that is not.
          </p>
          <ul className="code-list">
            {features.null_features.map((name) => (
              <li key={name}>
                <code>{name}</code>
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      <ProvenanceNote>{data.data_provenance}</ProvenanceNote>
    </>
  );
}

function ContributionTable({
  contributions,
}: {
  contributions: FeatureContribution[];
}): JSX.Element {
  const largest = Math.max(...contributions.map((entry) => Math.abs(entry.contribution)), 1);

  return (
    <table className="table">
      <caption className="visually-hidden">
        Per-feature SHAP contributions computed by the model
      </caption>
      <thead>
        <tr>
          <th scope="col">Feature</th>
          <th scope="col">Family</th>
          <th scope="col">Value</th>
          <th scope="col">Contribution</th>
          <th scope="col">
            <span className="visually-hidden">Magnitude</span>
          </th>
        </tr>
      </thead>
      <tbody>
        {contributions.map((entry) => (
          <tr key={entry.feature}>
            <th scope="row">
              <code>{entry.feature}</code>
            </th>
            <td>{entry.family}</td>
            <td className="numeric">{String(entry.value)}</td>
            <td className="numeric">
              {entry.contribution > 0 ? '+' : ''}
              {formatNumber(entry.contribution, 4)}
            </td>
            <td>
              <div
                className={`bar bar--${entry.contribution > 0 ? 'up' : 'down'}`}
                style={{ width: `${(Math.abs(entry.contribution) / largest) * 100}%` }}
                aria-hidden="true"
              />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
