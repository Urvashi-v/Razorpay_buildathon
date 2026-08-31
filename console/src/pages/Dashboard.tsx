/**
 * The dashboard: what is in the database, what model is loaded, what it costs.
 *
 * Every figure here comes from `/v1/monitoring/*` and `/v1/evaluation/*`. There
 * is no arithmetic in this file beyond turning a rate into a percentage for
 * display, and no chart draws a value the backend did not send.
 *
 * The RTO rate is computed by the backend over MATURED orders only. Dividing by
 * every order would count "not yet resolved" as "did not return" and understate
 * it - the same mistake the label pipeline refuses to make.
 */

import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';

import { fetchDataStatus, fetchFinalModel, fetchModelStatus } from '@/api/endpoints';
import {
  ErrorState,
  LoadingState,
  Metric,
  MetricGrid,
  Panel,
  ProvenanceNote,
} from '@/components/primitives';
import {
  formatCount,
  formatInr,
  formatNumber,
  formatPercent,
} from '@/components/format';
import { useApi } from '@/hooks/useApi';

const BAND_COLOURS: Record<string, string> = {
  train: '#60a5fa',
  validation: '#a78bfa',
  test: '#f472b6',
  excluded_immature: '#64748b',
  excluded_group_protocol: '#475569',
};

export default function Dashboard(): JSX.Element {
  const data = useApi((signal) => fetchDataStatus(undefined, signal), []);
  const model = useApi((signal) => fetchModelStatus(signal), []);
  const test = useApi((signal) => fetchFinalModel('test', signal), []);

  return (
    <div className="page">
      <Panel
        title="Order book"
        subtitle="Read from PostgreSQL. Counts are rows, not estimates."
      >
        {data.status === 'loading' ? <LoadingState label="Querying the database…" /> : null}
        {data.status === 'error' ? (
          <ErrorState error={data.error} onRetry={data.reload} context="reading the order book" />
        ) : null}
        {data.status === 'success' ? (
          <>
            <MetricGrid>
              <Metric label="Total orders" value={formatCount(data.data.total_orders)} emphasis />
              <Metric
                label="Matured"
                value={formatCount(data.data.matured_orders)}
                hint="Outcome is known"
              />
              <Metric
                label="Immature"
                value={formatCount(data.data.immature_orders)}
                hint="No outcome yet — never counted as delivered"
              />
              <Metric
                label="Observed RTO rate"
                value={formatPercent(data.data.observed_rto_rate)}
                hint="Over matured orders only"
                emphasis
              />
            </MetricGrid>

            <div className="split-layout">
              <div>
                <h3 className="subheading">Orders by split</h3>
                <SplitChart counts={data.data.orders_by_split} />
              </div>
              <div>
                <h3 className="subheading">Orders by payment method</h3>
                <table className="table">
                  <caption className="visually-hidden">Order counts by payment method</caption>
                  <thead>
                    <tr>
                      <th scope="col">Method</th>
                      <th scope="col">Orders</th>
                      <th scope="col">Share</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.data.orders_by_payment_method).map(([method, count]) => (
                      <tr key={method}>
                        <th scope="row">{method}</th>
                        <td className="numeric">{formatCount(count)}</td>
                        <td className="numeric">
                          {formatPercent(count / Math.max(data.data.total_orders, 1))}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <h3 className="subheading">Dataset runs</h3>
            <table className="table">
              <caption className="visually-hidden">Benchmark dataset runs in the database</caption>
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Generator</th>
                  <th scope="col">Orders</th>
                  <th scope="col">Created</th>
                </tr>
              </thead>
              <tbody>
                {data.data.dataset_runs.map((run) => (
                  <tr key={run.run_id}>
                    <th scope="row">
                      <code>{run.run_id.slice(0, 12)}…</code>
                    </th>
                    <td>{run.generator_version}</td>
                    <td className="numeric">{formatCount(run.n_orders)}</td>
                    <td>{run.created_at.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : null}
      </Panel>

      <Panel title="Model in service" subtitle="The artefact this API actually loaded">
        {model.status === 'loading' ? <LoadingState label="Reading the model registry…" /> : null}
        {model.status === 'error' ? (
          <ErrorState error={model.error} onRetry={model.reload} context="reading model status" />
        ) : null}
        {model.status === 'success' ? (
          model.data.available ? (
            <MetricGrid>
              <Metric label="Model" value={model.data.model_name ?? '—'} emphasis />
              <Metric
                label="Version"
                value={<code>{model.data.model_version ?? '—'}</code>}
              />
              <Metric
                label="Calibration"
                value={model.data.calibration_method ?? '—'}
                hint={`Fitted on ${model.data.calibration_fitted_on ?? '—'}`}
              />
              <Metric label="Feature version" value={model.data.feature_version ?? '—'} />
              <Metric label="Features" value={formatCount(model.data.n_features)} />
              <Metric label="Training rows" value={formatCount(model.data.training_rows)} />
            </MetricGrid>
          ) : (
            <div className="state state--error" role="alert">
              <p className="state__title">No model is loaded</p>
              <p className="state__detail">{model.data.reason}</p>
            </div>
          )
        ) : null}
      </Panel>

      <Panel
        title="Economics on the sealed test set"
        subtitle="The honest measurement. Validation figures are selection-contaminated and live on the Evaluation page."
      >
        {test.status === 'loading' ? <LoadingState label="Reading the test evaluation…" /> : null}
        {test.status === 'error' ? (
          <ErrorState
            error={test.error}
            onRetry={test.reload}
            context="reading the sealed-set evaluation"
          />
        ) : null}
        {test.status === 'success' ? (
          <>
            <MetricGrid>
              <Metric
                label="Net saving per 1,000 orders"
                value={formatInr(test.data.net_inr_per_1000_orders)}
                hint={`95% interval ${formatInr(test.data.net_ci_low)} to ${formatInr(test.data.net_ci_high)}`}
                emphasis
              />
              <Metric
                label="Do-nothing loss per 1,000"
                value={formatInr(Math.abs(test.data.do_nothing_loss_per_1000_orders))}
                hint="What the merchant absorbs today"
              />
              <Metric
                label="Flag rate"
                value={formatPercent(test.data.flag_rate)}
                hint={`At threshold ${formatNumber(test.data.threshold, 4)}`}
              />
              <Metric label="Precision" value={formatNumber(test.data.precision)} />
              <Metric
                label="False-positive cost"
                value={formatInr(test.data.false_positive_cost_inr)}
                hint="Reported separately, never netted away"
              />
              <Metric label="PR-AUC" value={formatNumber(test.data.pr_auc)} />
            </MetricGrid>
            {test.data.net_ci_low <= 0 && test.data.net_ci_high >= 0 ? (
              <p className="notice notice--warning">
                <strong>The interval crosses zero.</strong> On {formatCount(test.data.n_rows)}{' '}
                sealed orders this measurement cannot distinguish the model from doing nothing. The
                point estimate is positive; the evidence does not establish it.
              </p>
            ) : null}
            <ProvenanceNote>{test.data.data_provenance}</ProvenanceNote>
          </>
        ) : null}
      </Panel>
    </div>
  );
}

function SplitChart({ counts }: { counts: Record<string, number> }): JSX.Element {
  const entries = Object.entries(counts).map(([name, value]) => ({ name, value }));
  if (entries.length === 0) {
    return <p className="state__detail">No split assignments recorded.</p>;
  }
  return (
    <div className="chart" role="img" aria-label="Orders by dataset split">
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie data={entries} dataKey="value" nameKey="name" outerRadius={80} innerRadius={45}>
            {entries.map((entry) => (
              <Cell key={entry.name} fill={BAND_COLOURS[entry.name] ?? '#94a3b8'} />
            ))}
          </Pie>
          <Tooltip formatter={(value: number) => formatCount(value)} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
