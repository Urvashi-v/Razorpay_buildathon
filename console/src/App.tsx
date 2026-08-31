/**
 * The console shell.
 *
 * Six screens, each a client of the real backend. Nothing in this application
 * computes a risk score, a threshold, a band or a rupee figure: every number
 * displayed arrived in a response, and when a request fails the screen says so
 * rather than showing a plausible substitute.
 *
 * Navigation is local state rather than a router. There is one router-shaped
 * requirement here - deep-linking to an order - and it is met by the order id
 * living in this component and being passed down. Adding a routing dependency
 * for that would be more moving parts than the problem has.
 */

import { useState } from 'react';

import BackendStatus from '@/components/BackendStatus';
import Dashboard from '@/pages/Dashboard';
import EconomicSimulator from '@/pages/EconomicSimulator';
import Evaluation from '@/pages/Evaluation';
import OrderInvestigation from '@/pages/OrderInvestigation';
import OrderQueue from '@/pages/OrderQueue';

type Screen = 'dashboard' | 'queue' | 'investigation' | 'simulator' | 'evaluation';

const NAV: { id: Screen; label: string; hint: string }[] = [
  { id: 'dashboard', label: 'Dashboard', hint: 'Order book, model in service, economics' },
  { id: 'queue', label: 'Order queue', hint: 'Real orders, filtered server-side' },
  { id: 'investigation', label: 'Investigate', hint: 'Score one order end to end' },
  { id: 'simulator', label: 'Simulator', hint: 'Change the economics, recompute the policy' },
  { id: 'evaluation', label: 'Evaluation', hint: 'Measured metrics, validation vs sealed test' },
];

export default function App(): JSX.Element {
  const [screen, setScreen] = useState<Screen>('dashboard');
  const [orderId, setOrderId] = useState<string | null>(null);

  function investigate(id: string): void {
    setOrderId(id);
    setScreen('investigation');
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>RTO Sentinel</h1>
          <p className="app-header__tagline">
            A cost-calibrated return-risk scorer for Indian COD commerce — that knows what a false
            positive costs.
          </p>
        </div>
        <BackendStatus />
      </header>

      <nav className="app-nav" aria-label="Console sections">
        <ul>
          {NAV.map((item) => (
            <li key={item.id}>
              <button
                type="button"
                className={`nav-button${screen === item.id ? ' nav-button--active' : ''}`}
                aria-current={screen === item.id ? 'page' : undefined}
                onClick={() => setScreen(item.id)}
              >
                <span className="nav-button__label">{item.label}</span>
                <span className="nav-button__hint">{item.hint}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <main>
        {screen === 'dashboard' ? <Dashboard /> : null}
        {screen === 'queue' ? <OrderQueue onSelect={investigate} /> : null}
        {screen === 'investigation' ? (
          <OrderInvestigation orderId={orderId} onOrderIdChange={setOrderId} />
        ) : null}
        {screen === 'simulator' ? <EconomicSimulator /> : null}
        {screen === 'evaluation' ? <Evaluation /> : null}
      </main>

      <footer className="app-footer">
        <p>
          <strong>Data provenance.</strong> Models in this project are trained on synthetic data
          generated from published Indian RTO base rates. Absolute metric values are not a claim
          about production performance. Every figure this console displays is read from the
          backend; none is computed or held here.
        </p>
      </footer>
    </div>
  );
}
