/**
 * Measured results, with validation and test kept visibly apart.
 *
 * WHY THE TWO COLUMNS ARE LABELLED SO INSISTENTLY
 * ===============================================
 * The validation figures are selection-contaminated: hyperparameters were chosen
 * on that split and the shipped calibrator was refitted on it. The test figures
 * were measured once, after every choice was frozen. Putting them side by side
 * without saying which is which would invite a reader to take the better numbers
 * as the result, and the better numbers are the contaminated ones.
 *
 * So validation carries a warning wherever it appears, and the sealed-set column
 * carries the reason it was opened.
 */

import { fetchFinalModel, fetchLadder } from '@/api/endpoints';
import {
  ErrorState,
  LoadingState,
  Panel,
  ProvenanceNote,
} from '@/components/primitives';
import {
  DASH,
  formatCount,
  formatInr,
  formatNumber,
  formatPercent,
} from '@/components/format';
import { useApi } from '@/hooks/useApi';
import type { FinalModelResponse } from '@/types/api';

export default function Evaluation(): JSX.Element {
  const validation = useApi((signal) => fetchFinalModel('validation', signal), []);
  const test = useApi((signal) => fetchFinalModel('test', signal), []);
  const ladder = useApi((signal) => fetchLadder('validation', signal), []);

  return (
    <div className="page">
      <Panel
        title="Final model"
        subtitle="Every figure computed by the backend from real predictions against held-out labels."
      >
        {(validation.status === 'loading' || test.status === 'loading') ? (
          <LoadingState label="Reading evaluation artefacts…" />
        ) : null}

        {validation.status === 'error' && test.status === 'error' ? (
          <ErrorState
            error={test.error}
            onRetry={() => {
              validation.reload();
              test.reload();
            }}
            context="reading evaluations"
          />
        ) : null}

        {validation.status === 'success' || test.status === 'success' ? (
          <>
            <div className="notice notice--warning">
              <p>
                <strong>The two columns are not equally trustworthy.</strong> Hyperparameters were
                selected on validation and the shipped calibrator was refitted on it, so those
                figures describe data the model was tuned against. The sealed test set was opened
                once, after every choice was frozen.
              </p>
            </div>

            <ComparisonTable
              validation={validation.status === 'success' ? validation.data : null}
              test={test.status === 'success' ? test.data : null}
            />

            {test.status === 'success' && test.data.unseal_reason ? (
              <details className="details">
                <summary>Why the sealed set was opened</summary>
                <p className="state__detail">{test.data.unseal_reason}</p>
              </details>
            ) : null}

            {test.status === 'success' ? (
              <ProvenanceNote>{test.data.data_provenance}</ProvenanceNote>
            ) : null}
          </>
        ) : null}
      </Panel>

      <Panel
        title="The baseline ladder"
        subtitle="Every rung on the same validation split, at the same cost-derived threshold. If a simpler rung wins on money, it ships."
      >
        {ladder.status === 'loading' ? <LoadingState label="Reading ladder results…" /> : null}
        {ladder.status === 'error' ? (
          <ErrorState error={ladder.error} onRetry={ladder.reload} context="reading the ladder" />
        ) : null}
        {ladder.status === 'success' ? (
          <>
            <p className="state__detail">
              Threshold {formatNumber(ladder.data.threshold, 4)} — {ladder.data.threshold_source}
            </p>
            <table className="table">
              <caption className="visually-hidden">Baseline ladder results</caption>
              <thead>
                <tr>
                  <th scope="col">Rung</th>
                  <th scope="col">Model</th>
                  <th scope="col">PR-AUC</th>
                  <th scope="col">Train−val</th>
                  <th scope="col">Flag rate</th>
                  <th scope="col">Precision</th>
                  <th scope="col">Recall</th>
                  <th scope="col">Net ₹/1k</th>
                </tr>
              </thead>
              <tbody>
                {ladder.data.rungs.map((rung) => (
                  <tr key={rung.model_name}>
                    <td className="numeric">{rung.rung_id}</td>
                    <th scope="row">
                      <code>{rung.model_name}</code>
                    </th>
                    <td className="numeric">
                      {formatNumber(rung.pr_auc)}
                      <span className="interval">
                        [{formatNumber(rung.pr_auc_ci_low)}, {formatNumber(rung.pr_auc_ci_high)}]
                      </span>
                    </td>
                    <td className="numeric">
                      {rung.overfit_gap === null ? (
                        DASH
                      ) : (
                        <span className={rung.overfit_gap > 0.2 ? 'value--warning' : undefined}>
                          {rung.overfit_gap > 0 ? '+' : ''}
                          {formatNumber(rung.overfit_gap)}
                        </span>
                      )}
                    </td>
                    <td className="numeric">{formatNumber(rung.flag_rate)}</td>
                    <td className="numeric">{formatNumber(rung.precision)}</td>
                    <td className="numeric">{formatNumber(rung.recall)}</td>
                    <td className="numeric">{formatInr(rung.net_inr_per_1000_orders)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="state__detail">
              A large positive train−val gap means the rung memorised the training window; its
              validation score describes an overfitting configuration rather than a ceiling.
            </p>
          </>
        ) : null}
      </Panel>
    </div>
  );
}

const ROWS: {
  label: string;
  render: (data: FinalModelResponse) => string;
  hint?: string;
}[] = [
  { label: 'Rows', render: (d) => formatCount(d.n_rows) },
  { label: 'Positive rate', render: (d) => formatNumber(d.positive_rate, 4) },
  {
    label: 'PR-AUC',
    render: (d) =>
      `${formatNumber(d.pr_auc)} [${formatNumber(d.pr_auc_ci_low)}, ${formatNumber(d.pr_auc_ci_high)}]`,
    hint: 'Leads, because it is not inflated by the large negative class',
  },
  { label: 'ROC-AUC', render: (d) => formatNumber(d.roc_auc) },
  { label: 'Recall @ precision 80%', render: (d) => formatNumber(d.recall_at_precision_80) },
  { label: 'Recall @ precision 90%', render: (d) => formatNumber(d.recall_at_precision_90) },
  { label: 'Brier score', render: (d) => formatNumber(d.brier_score, 4) },
  {
    label: 'Brier, uncalibrated',
    render: (d) => formatNumber(d.brier_score_uncalibrated, 4),
  },
  {
    label: 'Expected calibration error',
    render: (d) => formatNumber(d.expected_calibration_error, 4),
  },
  {
    label: 'ECE, uncalibrated',
    render: (d) => formatNumber(d.expected_calibration_error_uncalibrated, 4),
  },
  { label: 'Operating threshold', render: (d) => formatNumber(d.threshold, 4) },
  { label: 'Flag rate', render: (d) => formatPercent(d.flag_rate) },
  { label: 'Precision', render: (d) => formatNumber(d.precision) },
  { label: 'Recall', render: (d) => formatNumber(d.recall) },
  { label: 'F1', render: (d) => formatNumber(d.f1) },
  {
    label: 'Confusion (TP/FP/FN/TN)',
    render: (d) =>
      `${formatCount(d.true_positives)} / ${formatCount(d.false_positives)} / ${formatCount(d.false_negatives)} / ${formatCount(d.true_negatives)}`,
  },
  {
    label: 'Net ₹ per 1,000 orders',
    render: (d) =>
      `${formatInr(d.net_inr_per_1000_orders)} [${formatInr(d.net_ci_low)}, ${formatInr(d.net_ci_high)}]`,
    hint: 'The headline. Measured against doing nothing.',
  },
  { label: 'False-positive cost', render: (d) => formatInr(d.false_positive_cost_inr) },
];

function ComparisonTable({
  validation,
  test,
}: {
  validation: FinalModelResponse | null;
  test: FinalModelResponse | null;
}): JSX.Element {
  return (
    <table className="table table--comparison">
      <caption className="visually-hidden">
        Final model metrics on the validation and sealed test splits
      </caption>
      <thead>
        <tr>
          <th scope="col">Metric</th>
          <th scope="col">
            Validation
            <span className="column-warning">selection-contaminated</span>
          </th>
          <th scope="col">
            Sealed test
            <span className="column-good">the honest read</span>
          </th>
        </tr>
      </thead>
      <tbody>
        {ROWS.map((row) => (
          <tr key={row.label}>
            <th scope="row">
              {row.label}
              {row.hint ? <span className="row-hint">{row.hint}</span> : null}
            </th>
            <td className="numeric">{validation ? row.render(validation) : DASH}</td>
            <td className="numeric numeric--emphasis">{test ? row.render(test) : DASH}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
