/**
 * The economic simulator.
 *
 * THE RULE: NO ECONOMIC ARITHMETIC HAPPENS IN THIS FILE
 * =====================================================
 * Moving a slider sends the new cost inputs to `POST /v1/economics/simulate`.
 * The backend re-derives the threshold, re-resolves every band boundary,
 * re-assigns every order in the scored book, and re-prices the result. This
 * component renders what comes back.
 *
 * It would be easy - and much snappier - to compute `C_fp / (C_fp + S_tp)` in
 * JavaScript as the user drags. It would also mean two implementations of the
 * decision rule, and the one on screen would be the one nobody tested. There is
 * exactly one implementation, and it is the one that serves production traffic.
 *
 * The visible consequence is latency: the panel shows a real loading state while
 * the server recomputes. That is the honest trade, and the alternative is a
 * number that agrees with nothing.
 */

import { useEffect, useState } from 'react';

import { fetchCostProfiles, simulateEconomics } from '@/api/endpoints';
import {
  AssumptionTag,
  ErrorState,
  LoadingState,
  Metric,
  MetricGrid,
  Panel,
} from '@/components/primitives';
import {
  formatCount,
  formatInr,
  formatNumber,
  formatPercent,
} from '@/components/format';
import { useAction, useApi } from '@/hooks/useApi';
import type { CostInputs, SimulationResult } from '@/types/api';

interface SliderSpec {
  key: keyof CostInputs;
  label: string;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
  assumption?: boolean;
  hint: string;
}

const SLIDERS: SliderSpec[] = [
  {
    key: 'contribution_margin_inr',
    label: 'Contribution margin',
    min: 0,
    max: 2000,
    step: 10,
    format: formatInr,
    hint: 'What a false positive costs: the margin lost if a good customer walks away.',
  },
  {
    key: 'rto_cost_inr',
    label: 'RTO cost',
    min: 0,
    max: 1000,
    step: 10,
    format: formatInr,
    hint: 'Full cost of one return: freight both ways, repack, QC, support.',
  },
  {
    key: 'intervention_success_rate',
    label: 'Intervention success rate',
    min: 0,
    max: 1,
    step: 0.01,
    format: (value) => formatPercent(value),
    assumption: true,
    hint: 'P(risky order saved | friction applied). Never measured on this data.',
  },
  {
    key: 'abandonment_on_friction',
    label: 'Abandonment on friction',
    min: 0,
    max: 1,
    step: 0.01,
    format: (value) => formatPercent(value),
    assumption: true,
    hint: 'P(good customer abandons | frictioned). Never measured on this data.',
  },
  {
    key: 'friction_support_cost_inr',
    label: 'Support cost per friction',
    min: 0,
    max: 200,
    step: 1,
    format: formatInr,
    hint: 'Ops and messaging cost each time friction is applied.',
  },
];

export default function EconomicSimulator(): JSX.Element {
  const profiles = useApi((signal) => fetchCostProfiles(signal), []);
  const [inputs, setInputs] = useState<CostInputs | null>(null);
  const [baseProfile, setBaseProfile] = useState<string>('');
  const simulation = useAction((values: CostInputs, profile: string, signal: AbortSignal) =>
    simulateEconomics(values, profile, signal),
  );

  // Seed the sliders from the configured default profile once it arrives.
  useEffect(() => {
    if (profiles.status === 'success' && inputs === null) {
      const chosen =
        profiles.data.profiles.find((entry) => entry.key === profiles.data.default_profile) ??
        profiles.data.profiles[0];
      if (chosen) {
        setInputs(chosen.inputs);
        setBaseProfile(chosen.key);
      }
    }
  }, [profiles, inputs]);

  if (profiles.status === 'loading') {
    return (
      <div className="page">
        <Panel title="Economic simulator">
          <LoadingState label="Loading cost profiles…" />
        </Panel>
      </div>
    );
  }

  if (profiles.status === 'error') {
    return (
      <div className="page">
        <Panel title="Economic simulator">
          <ErrorState
            error={profiles.error}
            onRetry={profiles.reload}
            context="loading cost profiles"
          />
        </Panel>
      </div>
    );
  }

  if (profiles.status !== 'success' || inputs === null) {
    return (
      <div className="page">
        <Panel title="Economic simulator">
          <LoadingState label="Preparing…" />
        </Panel>
      </div>
    );
  }

  const bounds = profiles.data.bounds;

  return (
    <div className="page">
      <Panel
        title="Economic simulator"
        subtitle="Change the economics; the backend recomputes the threshold, the bands, every order's assignment and the rupee totals. Nothing is calculated in the browser."
      >
        <div className="field">
          <label htmlFor="profile">Start from a profile</label>
          <select
            id="profile"
            value={baseProfile}
            onChange={(event) => {
              const chosen = profiles.data.profiles.find(
                (entry) => entry.key === event.target.value,
              );
              if (chosen) {
                setBaseProfile(chosen.key);
                setInputs(chosen.inputs);
                simulation.reset();
              }
            }}
          >
            {profiles.data.profiles.map((profile) => (
              <option key={profile.key} value={profile.key}>
                {profile.label}
              </option>
            ))}
          </select>
        </div>

        <div className="sliders">
          {SLIDERS.map((slider) => {
            const bound = bounds[slider.key];
            const min = bound?.min ?? slider.min;
            const max = bound?.max ?? slider.max;
            return (
              <div className="slider" key={slider.key}>
                <label htmlFor={`slider-${slider.key}`}>
                  {slider.label}
                  {slider.assumption ? <AssumptionTag /> : null}
                  <output htmlFor={`slider-${slider.key}`} className="slider__value">
                    {slider.format(inputs[slider.key])}
                  </output>
                </label>
                <input
                  id={`slider-${slider.key}`}
                  type="range"
                  min={Math.max(min, slider.min)}
                  max={Math.min(max, slider.max)}
                  step={slider.step}
                  value={inputs[slider.key]}
                  aria-describedby={`hint-${slider.key}`}
                  onChange={(event) =>
                    setInputs({ ...inputs, [slider.key]: Number(event.target.value) })
                  }
                />
                <p className="slider__hint" id={`hint-${slider.key}`}>
                  {slider.hint}
                </p>
              </div>
            );
          })}
        </div>

        <div className="actions">
          <button
            type="button"
            className="button"
            disabled={simulation.state.status === 'loading'}
            onClick={() => void simulation.run(inputs, baseProfile)}
          >
            {simulation.state.status === 'loading' ? 'Recomputing…' : 'Recompute on the server'}
          </button>
          <p className="state__detail">
            The recomputation reruns the decision policy over the whole scored validation book.
          </p>
        </div>

        <p className="notice">{profiles.data.assumption_warning}</p>
      </Panel>

      {simulation.state.status === 'loading' ? (
        <Panel title="Result">
          <LoadingState label="The server is re-pricing the book…" />
        </Panel>
      ) : null}

      {simulation.state.status === 'error' ? (
        <Panel title="Result">
          <ErrorState error={simulation.state.error} context="running the simulation" />
        </Panel>
      ) : null}

      {simulation.state.status === 'success' ? <Result data={simulation.state.data} /> : null}
    </div>
  );
}

function Result({ data }: { data: SimulationResult }): JSX.Element {
  const { economics, threshold } = data;
  const delta =
    data.baseline_net_inr_per_1000_orders === null
      ? null
      : economics.expected_net_inr_per_1000_orders - data.baseline_net_inr_per_1000_orders;

  return (
    <>
      <Panel title="Derived operating point" subtitle="Computed by the backend from the inputs above">
        <MetricGrid>
          <Metric
            label="Threshold"
            value={formatNumber(threshold.threshold, 4)}
            hint="Not 0.5 — derived from these economics"
            emphasis
          />
          <Metric label="C_fp" value={formatInr(threshold.cost_false_positive_inr)} />
          <Metric label="S_tp" value={formatInr(threshold.saving_true_positive_inr)} />
          {data.baseline_threshold !== null ? (
            <Metric
              label="Baseline threshold"
              value={formatNumber(data.baseline_threshold, 4)}
              hint="The profile you started from"
            />
          ) : null}
        </MetricGrid>
        <p className="formula">
          <code>{threshold.formula}</code>
        </p>
      </Panel>

      <Panel title="What it costs" subtitle={`Priced over ${formatCount(economics.n_orders)} scored orders on the ${economics.split} split`}>
        <MetricGrid>
          <Metric
            label="Net per 1,000 orders"
            value={formatInr(economics.expected_net_inr_per_1000_orders)}
            hint={delta === null ? undefined : `${delta >= 0 ? '+' : ''}${formatInr(delta)} vs baseline`}
            emphasis
          />
          <Metric label="Flag rate" value={formatPercent(economics.flag_rate)} />
          <Metric
            label="Orders affected"
            value={formatCount(economics.expected_orders_affected)}
          />
          <Metric label="Expected savings" value={formatInr(economics.expected_savings_inr)} />
          <Metric
            label="False-positive cost"
            value={formatInr(economics.expected_false_positive_cost_inr)}
            hint="Never netted away"
          />
          <Metric
            label="Residual RTO loss"
            value={formatInr(economics.expected_false_negative_loss_inr)}
          />
          <Metric
            label="Do-nothing loss per 1,000"
            value={formatInr(Math.abs(economics.do_nothing_loss_inr_per_1000_orders))}
          />
          <Metric
            label="Net after control holdout"
            value={formatInr(economics.net_inr_per_1000_after_holdout)}
            hint={`${formatPercent(economics.holdout_fraction_of_flagged, 0)} of flagged orders receive no friction`}
          />
        </MetricGrid>
      </Panel>

      <Panel title="The friction ladder at this threshold">
        <table className="table">
          <caption className="visually-hidden">Intervention ladder bands</caption>
          <thead>
            <tr>
              <th scope="col">Band</th>
              <th scope="col">Action</th>
              <th scope="col">Range</th>
              <th scope="col">Orders</th>
              <th scope="col">Share</th>
              <th scope="col">Net</th>
            </tr>
          </thead>
          <tbody>
            {data.ladder.map((rung) => (
              <tr key={rung.band}>
                <th scope="row">
                  <span className={`badge badge--${rung.band.toLowerCase()}`}>{rung.band}</span>
                </th>
                <td>{rung.action}</td>
                <td className="numeric">
                  [{formatNumber(rung.lower_bound, 4)},{' '}
                  {rung.upper_bound === null ? '1.0' : formatNumber(rung.upper_bound, 4)})
                </td>
                <td className="numeric">{formatCount(rung.n_orders)}</td>
                <td className="numeric">{formatPercent(rung.share_of_book)}</td>
                <td className="numeric">{formatInr(rung.expected_net_inr)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {data.collapsed_bands.length > 0 ? (
          <div className="notice notice--warning">
            <p>
              <strong>Some bands cannot fire at this threshold.</strong> Your economics leave no
              probability space above them:
            </p>
            <ul>
              {data.collapsed_bands.map((entry) => (
                <li key={entry}>{entry}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </Panel>
    </>
  );
}
