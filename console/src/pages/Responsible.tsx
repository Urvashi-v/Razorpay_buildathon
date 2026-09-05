/**
 * Fairness, robustness and drift — the three things a risk system should be
 * asked about and usually is not.
 *
 * THE RULES THIS SCREEN HOLDS
 * ===========================
 * **A thin cohort never looks like a solid one.** Groups below the minimum
 * support are shown, because suppressing them would hide exactly what an audit
 * exists to look at, but they carry a visible marker and their numbers are
 * de-emphasised. A reader scanning a column must not mistake a precision
 * computed on nine flagged orders for one computed on nine hundred.
 *
 * **Drift is not failure.** The distance table and the labelled comparisons are
 * separate sections with separate headings, and when no labels exist the screen
 * says the question is unanswered rather than showing a reassuring row of green.
 *
 * **Lift, not raw PR-AUC.** In the shift table the raw PR-AUC column is present
 * but de-emphasised, because it rises when the base rate rises. Leading with it
 * would report the arithmetic of the base rate as a property of the model.
 *
 * Every number here comes from `/v1/evaluation/fairness`, `/v1/evaluation/shift`
 * and `/v1/monitoring/drift`. When an experiment has not been run the endpoint
 * returns 501 with its reason and this screen shows that reason — it does not
 * render an empty table, which would read as "we checked and found nothing".
 */

import { useState } from 'react';

import {
  fetchAblation,
  fetchDriftReport,
  fetchFairness,
  fetchShiftStudy,
} from '@/api/endpoints';
import {
  ErrorState,
  LoadingState,
  Metric,
  MetricGrid,
  Panel,
  ProvenanceNote,
} from '@/components/primitives';
import { DASH, formatCount, formatInr, formatNumber, formatPercent } from '@/components/format';
import { useApi } from '@/hooks/useApi';
import type {
  AblationArm,
  AblationStudy,
  CohortResult,
  DriftReport,
  DriftSignal,
  FairnessResponse,
  ShiftResult,
  ShiftStudy,
} from '@/types/api';

export default function Responsible(): JSX.Element {
  const [split, setSplit] = useState<'validation' | 'test'>('validation');
  const fairness = useApi((signal) => fetchFairness(split, signal), [split]);
  const shift = useApi((signal) => fetchShiftStudy(signal), []);
  const drift = useApi((signal) => fetchDriftReport(signal), []);
  const ablation = useApi((signal) => fetchAblation(signal), []);

  return (
    <div className="page">
      <Panel
        title="Cohort fairness"
        subtitle="Operational cohorts only. No sensitive characteristic is recorded, examined or inferred."
      >
        <div className="field">
          <label htmlFor="fairness-split">Split</label>
          <select
            id="fairness-split"
            value={split}
            onChange={(event) => setSplit(event.target.value as 'validation' | 'test')}
          >
            <option value="validation">validation</option>
            <option value="test">sealed test</option>
          </select>
        </div>

        <p className="state__detail">
          The cohorts are delivery-area tier, order-value band, customer-history depth and
          payment method — every one a fact recorded on the order. There is no gender,
          religion, caste, ethnicity, age or income field in this data, and none is derived
          from names or addresses. The audit refuses by name any cohort that looks like one.
        </p>

        {fairness.status === 'loading' ? <LoadingState label="Reading the audit…" /> : null}
        {fairness.status === 'error' ? (
          <ErrorState
            error={fairness.error}
            onRetry={fairness.reload}
            context={`reading the ${split} fairness audit`}
          />
        ) : null}
        {fairness.status === 'success' ? <Fairness data={fairness.data} /> : null}
      </Panel>

      <Panel
        title="What each feature family is worth"
        subtitle="Leave-one-family-out, retrained per arm, measured in net rupees rather than AUC."
      >
        {ablation.status === 'loading' ? <LoadingState label="Reading the ablation…" /> : null}
        {ablation.status === 'error' ? (
          <ErrorState
            error={ablation.error}
            onRetry={ablation.reload}
            context="reading the ablation study"
          />
        ) : null}
        {ablation.status === 'success' ? <Ablation data={ablation.data} /> : null}
      </Panel>

      <Panel
        title="Distribution shift"
        subtitle="The frozen model, unretrained, facing deliberately perturbed worlds."
      >
        {shift.status === 'loading' ? <LoadingState label="Reading the shift study…" /> : null}
        {shift.status === 'error' ? (
          <ErrorState
            error={shift.error}
            onRetry={shift.reload}
            context="reading the distribution-shift study"
          />
        ) : null}
        {shift.status === 'success' ? <Shift data={shift.data} /> : null}
      </Panel>

      <Panel
        title="Drift monitoring"
        subtitle="Baseline period versus current period. Drift is not failure."
      >
        {drift.status === 'loading' ? <LoadingState label="Reading the drift report…" /> : null}
        {drift.status === 'error' ? (
          <ErrorState
            error={drift.error}
            onRetry={drift.reload}
            context="reading the drift report"
          />
        ) : null}
        {drift.status === 'success' ? <Drift data={drift.data} /> : null}
      </Panel>
    </div>
  );
}

// ---------------------------------------------------------------------------
// fairness
// ---------------------------------------------------------------------------

function Fairness({ data }: { data: FairnessResponse }): JSX.Element {
  const { audit } = data;
  const cohorts = [...new Set(data.slices.map((entry) => entry.cohort))];

  return (
    <>
      <div className={`notice ${audit.triggered ? 'notice--warning' : ''}`} role="status">
        <p>
          <strong>
            Disparity review: {audit.triggered ? 'TRIGGERED' : 'not triggered'}.
          </strong>{' '}
          {audit.narrative}
        </p>
      </div>

      <MetricGrid>
        <Metric
          label="Max flag-rate ratio"
          value={formatNumber(audit.max_flag_rate_ratio, 2)}
          hint={
            audit.most_flagged_group
              ? `${audit.most_flagged_group} vs ${audit.least_flagged_group}`
              : undefined
          }
          emphasis
        />
        <Metric
          label="Worst precision drop"
          value={formatNumber(audit.worst_precision_drop, 3)}
          hint="Within a cohort, never across two"
          emphasis
        />
        <Metric label="Minimum support" value={formatCount(audit.min_support)} hint="Orders" />
        <Metric
          label="Groups below support"
          value={formatCount(audit.groups_below_support.length)}
          hint="Shown, but excluded from the comparison"
        />
        <Metric label="Model" value={<code>{data.model_version}</code>} />
        <Metric label="Threshold" value={formatNumber(data.threshold, 4)} />
      </MetricGrid>

      <p className="state__detail">
        The trigger is a conjunction, and that is deliberate: a group flagged more often is
        not by itself a finding, because a group that returns more parcels should be flagged
        more. The finding is a group flagged materially more often <em>while</em> the model
        is materially worse at being right about it.
      </p>

      {cohorts.map((cohort) => (
        <CohortTable
          key={cohort}
          cohort={cohort}
          rows={data.slices.filter((entry) => entry.cohort === cohort)}
        />
      ))}

      <ProvenanceNote>{data.disclaimer}</ProvenanceNote>
    </>
  );
}

function CohortTable({ cohort, rows }: { cohort: string; rows: CohortResult[] }): JSX.Element {
  return (
    <>
      <h3 className="subheading">{cohort.replace(/_/g, ' ')}</h3>
      <table className="table">
        <caption className="visually-hidden">
          Cohort breakdown for {cohort}, with Wilson intervals
        </caption>
        <thead>
          <tr>
            <th scope="col">Group</th>
            <th scope="col">Orders</th>
            <th scope="col">RTO rate</th>
            <th scope="col">Flag rate</th>
            <th scope="col">Precision</th>
            <th scope="col">Recall</th>
            <th scope="col">Net ₹/1k</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.group} className={row.sufficient ? undefined : 'row--thin'}>
              <th scope="row">
                {row.group}
                {row.sufficient ? null : (
                  <span className="tag tag--thin" title={row.insufficient_reason}>
                    too thin to read
                  </span>
                )}
              </th>
              <td className="numeric">{formatCount(row.n_orders)}</td>
              <td className="numeric">
                {formatPercent(row.rto_rate)} <Interval bounds={row.rto_rate_ci} />
              </td>
              <td className="numeric">
                {formatPercent(row.flag_rate)} <Interval bounds={row.flag_rate_ci} />
              </td>
              <td className="numeric">
                {formatNumber(row.precision, 3)} <Interval bounds={row.precision_ci} />
              </td>
              <td className="numeric">{formatNumber(row.recall, 3)}</td>
              <td className="numeric">{formatInr(row.net_inr_per_1000)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/**
 * A Wilson interval, rendered next to its estimate.
 *
 * Shown on every rate so a reader sees the precision of the estimate at the same
 * moment they see the estimate. A cohort table without intervals invites
 * comparing 0.44 against 0.47 as though the difference were real.
 */
function Interval({ bounds }: { bounds: [number, number] | null }): JSX.Element | null {
  if (!bounds) return null;
  return (
    <span className="interval">
      [{formatNumber(bounds[0], 2)}, {formatNumber(bounds[1], 2)}]
    </span>
  );
}

// ---------------------------------------------------------------------------
// ablation
// ---------------------------------------------------------------------------

/**
 * What each feature family contributes, in money.
 *
 * The verdict column comes from the backend, not from a comparison done here.
 * "not established" means the bootstrap interval spans zero — the data cannot
 * say the family mattered — and that is deliberately worded so it cannot be read
 * as "contributes nothing". Overlapping families hide each other's value, and a
 * console that rendered a red X next to `customer_history` would be asserting
 * something this study cannot support.
 */
function Ablation({ data }: { data: AblationStudy }): JSX.Element {
  return (
    <>
      <MetricGrid>
        <Metric
          label="Full model"
          value={formatInr(data.full_model.net_inr_per_1000)}
          hint={`${formatCount(data.full_model.n_features)} features — the reference arm`}
          emphasis
        />
        <Metric label="Families ablated" value={formatCount(data.arms.length)} />
        <Metric label="Split" value={data.split} hint="Never the sealed test set" />
        <Metric label="Cost profile" value={data.cost_profile} />
      </MetricGrid>

      <div className="notice" role="status">
        <p>
          <strong>An interval that spans zero has not been shown to matter.</strong> It does
          not mean the family contributes nothing — leave-one-out measures what a family adds{' '}
          <em>once every other family is present</em>, so overlapping signal hides individual
          value.
        </p>
      </div>

      <table className="table">
        <caption className="visually-hidden">
          Net rupee contribution of each feature family
        </caption>
        <thead>
          <tr>
            <th scope="col">Family removed</th>
            <th scope="col">Features</th>
            <th scope="col">Net ₹/1k</th>
            <th scope="col">Δ vs full</th>
            <th scope="col">95% interval</th>
            <th scope="col" className="muted">
              PR-AUC
            </th>
            <th scope="col">Reading</th>
          </tr>
        </thead>
        <tbody>
          <tr className="row--reference">
            <th scope="row">
              <em>(full model)</em>
            </th>
            <td className="numeric">{formatCount(data.full_model.n_features)}</td>
            <td className="numeric">{formatInr(data.full_model.net_inr_per_1000)}</td>
            <td className="numeric">{DASH}</td>
            <td className="numeric">{DASH}</td>
            <td className="numeric muted">{formatNumber(data.full_model.pr_auc, 3)}</td>
            <td>reference</td>
          </tr>
          {data.arms.map((arm) => (
            <AblationRow key={arm.family_removed} arm={arm} />
          ))}
        </tbody>
      </table>

      <h3 className="subheading">Findings</h3>
      <ul className="caveats">
        {data.findings.map((finding) => (
          <li key={finding}>{finding}</li>
        ))}
      </ul>

      <ProvenanceNote>{data.data_provenance}</ProvenanceNote>
    </>
  );
}

function AblationRow({ arm }: { arm: AblationArm }): JSX.Element {
  const established = arm.verdict === 'earns its place' || arm.verdict === 'costs money';
  return (
    <tr className={established ? undefined : 'row--thin'}>
      <th scope="row">
        <code>{arm.family_removed}</code>
      </th>
      <td className="numeric">{formatCount(arm.n_features)}</td>
      <td className="numeric">{formatInr(arm.net_inr_per_1000)}</td>
      <td className={`numeric ${arm.delta_vs_full < 0 && established ? 'bad' : ''}`}>
        {arm.delta_vs_full >= 0 ? '+' : ''}
        {formatInr(arm.delta_vs_full)}
      </td>
      <td className="numeric">
        <span className="interval">
          [{formatInr(arm.delta_ci_low)}, {formatInr(arm.delta_ci_high)}]
        </span>
      </td>
      <td className="numeric muted">{formatNumber(arm.pr_auc, 3)}</td>
      <td>{arm.verdict}</td>
    </tr>
  );
}

// ---------------------------------------------------------------------------
// distribution shift
// ---------------------------------------------------------------------------

function Shift({ data }: { data: ShiftStudy }): JSX.Element {
  return (
    <>
      <MetricGrid>
        <Metric label="Environments" value={formatCount(data.results.length)} />
        <Metric
          label="Threshold"
          value={formatNumber(data.threshold, 4)}
          hint="Held fixed across every environment"
        />
        <Metric label="Model" value={<code>{data.model_version}</code>} hint="Not retrained" />
        <Metric label="Generator" value={data.generator_version} />
      </MetricGrid>

      <div className="notice" role="status">
        <p>
          <strong>Read the lift column, not the raw PR-AUC column.</strong> A random ranker
          scores PR-AUC equal to the positive rate, so an environment whose base rate moved
          hands the model a different floor for free. Raw PR-AUC rises when the world gets
          riskier; lift divides that floor out.
        </p>
      </div>

      <table className="table">
        <caption className="visually-hidden">
          Model performance in each shifted environment
        </caption>
        <thead>
          <tr>
            <th scope="col">Environment</th>
            <th scope="col">Orders</th>
            <th scope="col">RTO rate</th>
            <th scope="col">Lift</th>
            <th scope="col">Δ Lift</th>
            <th scope="col" className="muted">
              PR-AUC
            </th>
            <th scope="col">ECE</th>
            <th scope="col">Flag rate</th>
            <th scope="col">Net ₹/1k</th>
          </tr>
        </thead>
        <tbody>
          {data.results.map((row) => (
            <ShiftRow key={row.environment} row={row} />
          ))}
        </tbody>
      </table>

      <h3 className="subheading">What changed in each environment</h3>
      <dl className="definitions">
        {data.environments.map((spec) => (
          <div key={spec.name}>
            <dt>
              <code>{spec.name}</code>
            </dt>
            <dd>
              {spec.description}
              {Object.keys(spec.overrides).length > 0 ? (
                <ul className="code-list">
                  {Object.entries(spec.overrides).map(([key, value]) => (
                    <li key={key}>
                      <code>
                        {key}={value}
                      </code>
                    </li>
                  ))}
                </ul>
              ) : (
                <em> — the control, no overrides</em>
              )}
            </dd>
          </div>
        ))}
      </dl>

      <h3 className="subheading">Findings</h3>
      <ul className="caveats">
        {data.findings.map((finding) => (
          <li key={finding}>{finding}</li>
        ))}
      </ul>

      <ProvenanceNote>{data.data_provenance}</ProvenanceNote>
    </>
  );
}

function ShiftRow({ row }: { row: ShiftResult }): JSX.Element {
  // Mirrors `MATERIAL_LIFT_DROP` in `eval/shift.py`, which the backend uses to
  // decide what goes in `findings`. Duplicated deliberately and narrowly: this
  // only tints a cell, and shipping a `degraded` flag through the API to drive a
  // colour would put a presentation decision in the contract. If the backend
  // constant moves, move this with it - the findings list is the source of truth
  // either way, and it is rendered below the table.
  const MATERIAL_LIFT_DROP = 0.15;
  const degraded =
    row.pr_auc_lift_delta !== null && row.pr_auc_lift_delta < -MATERIAL_LIFT_DROP;
  const losing = row.net_inr_per_1000 <= 0;

  return (
    <tr className={row.environment === 'reference' ? 'row--reference' : undefined}>
      <th scope="row">
        <code>{row.environment}</code>
      </th>
      <td className="numeric">{formatCount(row.n_orders)}</td>
      <td className="numeric">{formatPercent(row.observed_rto_rate)}</td>
      <td className="numeric">{formatNumber(row.pr_auc_lift, 2)}×</td>
      <td className={`numeric ${degraded ? 'bad' : ''}`}>
        {row.pr_auc_lift_delta === null
          ? DASH
          : `${row.pr_auc_lift_delta >= 0 ? '+' : ''}${formatNumber(row.pr_auc_lift_delta, 2)}×`}
      </td>
      <td className="numeric muted">{formatNumber(row.pr_auc, 3)}</td>
      <td className="numeric">{formatNumber(row.expected_calibration_error, 3)}</td>
      <td className="numeric">{formatPercent(row.flag_rate)}</td>
      <td className={`numeric ${losing ? 'bad' : ''}`}>{formatInr(row.net_inr_per_1000)}</td>
    </tr>
  );
}

// ---------------------------------------------------------------------------
// drift
// ---------------------------------------------------------------------------

function Drift({ data }: { data: DriftReport }): JSX.Element {
  return (
    <>
      {!data.labels_available ? (
        <div className="notice notice--warning" role="alert">
          <p>
            <strong>No matured outcomes in the current window.</strong> Nothing below is a
            measurement of model quality — every row describes a distribution that moved.
          </p>
          <p>
            Treat the absence of alarms here as an absence of information, not as evidence
            the model is fine.
          </p>
        </div>
      ) : null}

      <MetricGrid>
        <Metric
          label="Baseline"
          value={formatCount(data.baseline.n_orders)}
          hint={`${formatCount(data.baseline.n_matured)} with a known outcome`}
        />
        <Metric
          label="Current"
          value={formatCount(data.current.n_orders)}
          hint={`${formatCount(data.current.n_matured)} with a known outcome`}
        />
        <Metric
          label="Labelled comparison"
          value={data.labels_available ? 'possible' : 'not possible'}
          emphasis
        />
        <Metric label="Model" value={<code>{data.model_version}</code>} />
      </MetricGrid>

      <h3 className="subheading">Distances — what moved</h3>
      <p className="state__detail">
        These say how far a distribution travelled and nothing about whether that is bad.
        Input drift is expected in a seasonal business: COD share, order values and category
        mix all move during a festive peak without the model getting worse.
      </p>
      <table className="table">
        <caption className="visually-hidden">Drift signals between the two windows</caption>
        <thead>
          <tr>
            <th scope="col">Kind</th>
            <th scope="col">Quantity</th>
            <th scope="col">Statistic</th>
            <th scope="col">Baseline</th>
            <th scope="col">Current</th>
            <th scope="col">Distance</th>
            <th scope="col">Reading</th>
          </tr>
        </thead>
        <tbody>
          {data.signals.map((signal) => (
            <SignalRow key={`${signal.kind}-${signal.name}-${signal.statistic}`} signal={signal} />
          ))}
        </tbody>
      </table>

      <h3 className="subheading">Labelled comparisons — what actually changed</h3>
      {data.performance.length > 0 ? (
        <table className="table">
          <caption className="visually-hidden">
            Measured model-quality changes between the two windows
          </caption>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col">Baseline</th>
              <th scope="col">Current</th>
              <th scope="col">Δ</th>
              <th scope="col">Evidence</th>
            </tr>
          </thead>
          <tbody>
            {data.performance.map((delta) => (
              <tr key={delta.metric}>
                <th scope="row">{delta.metric}</th>
                <td className="numeric">{formatNumber(delta.baseline, 4)}</td>
                <td className="numeric">{formatNumber(delta.current, 4)}</td>
                <td className="numeric">
                  {delta.delta >= 0 ? '+' : ''}
                  {formatNumber(delta.delta, 4)}
                </td>
                <td>
                  {delta.sufficient ? (
                    `${formatCount(delta.n_current_matured)} matured orders`
                  ) : (
                    <span className="tag tag--thin">{delta.note}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="state__detail">
          None. No labelled comparison was possible, so no claim about model quality is made
          here — this is the honest empty state, not a passing result.
        </p>
      )}

      <h3 className="subheading">What this means</h3>
      <ul className="caveats">
        {data.warnings.map((warning) => (
          <li key={warning}>{warning}</li>
        ))}
      </ul>

      <ProvenanceNote>{data.data_provenance}</ProvenanceNote>
    </>
  );
}

function SignalRow({ signal }: { signal: DriftSignal }): JSX.Element {
  return (
    <tr className={signal.sufficient ? undefined : 'row--thin'}>
      <td>{signal.kind.replace(/_/g, ' ')}</td>
      <th scope="row">
        <code>{signal.name}</code>
      </th>
      <td>{signal.statistic.replace(/_/g, ' ')}</td>
      <td className="numeric">{formatNumber(signal.baseline_value, 4)}</td>
      <td className="numeric">{formatNumber(signal.current_value, 4)}</td>
      <td className="numeric">{formatNumber(signal.distance, 4)}</td>
      <td>
        <span className={`badge badge--drift-${signal.severity}`}>{signal.severity}</span>
        {signal.sufficient ? null : (
          <span className="tag tag--thin" title={signal.note}>
            too thin to read
          </span>
        )}
      </td>
    </tr>
  );
}
