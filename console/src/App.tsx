import BackendStatus from '@/components/BackendStatus';

/**
 * The console shell.
 *
 * Phase 1 deliberately ships the shell and the backend connection, and nothing
 * more. The order queue, the threshold sliders, the reliability diagram and the
 * fairness breakdown are Phase 4 - and building their UI before the endpoints
 * that feed them exist would mean populating charts with invented numbers, which
 * is the failure mode this whole project argues against.
 *
 * So what is here is real: a live readiness panel that reads the API and reports
 * what it finds, including when what it finds is "no model loaded".
 */

const PLANNED = [
  { phase: 'Phase 1', label: 'Architecture, contracts, config, API skeleton', done: true },
  { phase: 'Phase 2', label: 'Synthetic generator, splits, cost model, threshold derivation' },
  { phase: 'Phase 3', label: 'Baseline ladder rungs 0-5, isotonic calibration' },
  { phase: 'Phase 4', label: 'Evaluation harness, fairness audit, order queue, sliders' },
  { phase: 'Phase 5', label: 'Reason codes, confirmations, digest, address repair' },
  { phase: 'Phase 6', label: 'Drift monitoring, outcome loop, control holdout' },
];

export default function App(): JSX.Element {
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>RTO Sentinel</h1>
        <p>
          A cost-calibrated return-risk scorer for Indian COD commerce — that knows what a false
          positive costs.
        </p>
      </header>

      <BackendStatus />

      <section className="card">
        <h2>What is built</h2>
        <ul className="phase-list">
          {PLANNED.map((item) => (
            <li key={item.phase} className={item.done ? 'done' : undefined}>
              <strong>{item.phase}</strong> — {item.label}
              {item.done ? ' ✓' : ''}
            </li>
          ))}
        </ul>
      </section>

      <p className="notice notice--provenance">
        <strong>Data provenance.</strong> Models in this project are trained on synthetic data
        generated from published Indian RTO base rates. Absolute metric values are not a claim
        about production performance. Every figure this console displays is read from the backend;
        none is computed or held here.
      </p>
    </div>
  );
}
