/**
 * The order queue: real rows, filtered server-side.
 *
 * WHY FILTERING IS A REQUEST AND NOT AN ARRAY FILTER
 * ==================================================
 * The obvious implementation fetches a page and filters it in JavaScript. It is
 * also wrong: the user would be filtering the fifty rows that happened to
 * arrive, while the pagination total still described the unfiltered book. The
 * numbers on screen would disagree with each other and nobody would know which
 * to believe.
 *
 * So every filter is a query parameter, the backend returns a page and a total
 * computed under the same filters, and "showing 50 of 12,431" is true.
 *
 * RISK BAND IS DELIBERATELY NOT A FILTER HERE
 * ===========================================
 * A band is not stored on an order - it is derived by scoring, and scoring runs
 * the whole feature pipeline per order. Filtering a page by band would mean
 * scoring every order in the database on every keystroke. The band is shown on
 * the investigation screen, where scoring one order is the point.
 */

import { useState } from 'react';

import { fetchOrders, type OrderFilters } from '@/api/endpoints';
import {
  EmptyState,
  ErrorState,
  LoadingState,
  Panel,
} from '@/components/primitives';
import {
  formatCount,
  formatDate,
  formatInr,
} from '@/components/format';
import { useApi } from '@/hooks/useApi';
import type { OrderSummary } from '@/types/api';

const PAGE_SIZE = 25;

interface Props {
  onSelect: (orderId: string) => void;
}

export default function OrderQueue({ onSelect }: Props): JSX.Element {
  const [split, setSplit] = useState('');
  const [paymentMethod, setPaymentMethod] = useState('');
  const [merchantId, setMerchantId] = useState('');
  const [offset, setOffset] = useState(0);

  const filters: OrderFilters = {
    limit: PAGE_SIZE,
    offset,
    ...(split ? { split } : {}),
    ...(paymentMethod ? { payment_method: paymentMethod } : {}),
    ...(merchantId ? { merchant_id: merchantId } : {}),
  };

  const page = useApi(
    (signal) => fetchOrders(filters, signal),
    [split, paymentMethod, merchantId, offset],
  );

  function updateFilter(setter: (value: string) => void) {
    return (event: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
      setter(event.target.value);
      setOffset(0); // a new filter starts at the first page, not page four
    };
  }

  const total = page.status === 'success' ? page.data.total : null;

  return (
    <div className="page">
      <Panel
        title="Order queue"
        subtitle="Rows from the database. Filters are applied server-side, so the total below is the filtered total."
      >
        <form className="filters" onSubmit={(event) => event.preventDefault()}>
          <div className="field">
            <label htmlFor="filter-split">Split</label>
            <select id="filter-split" value={split} onChange={updateFilter(setSplit)}>
              <option value="">All splits</option>
              <option value="train">train</option>
              <option value="validation">validation</option>
              <option value="test">test</option>
              <option value="excluded_immature">excluded — immature</option>
              <option value="excluded_group_protocol">excluded — group protocol</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="filter-payment">Payment</label>
            <select
              id="filter-payment"
              value={paymentMethod}
              onChange={updateFilter(setPaymentMethod)}
            >
              <option value="">COD and prepaid</option>
              <option value="cod">COD only</option>
              <option value="prepaid">Prepaid only</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="filter-merchant">Merchant</label>
            <input
              id="filter-merchant"
              type="text"
              value={merchantId}
              placeholder="e.g. M-DEMO-001"
              onChange={updateFilter(setMerchantId)}
            />
          </div>

          <p className="filters__summary" aria-live="polite">
            {total === null
              ? 'Loading…'
              : `Showing ${formatCount(Math.min(PAGE_SIZE, Math.max(total - offset, 0)))} of ${formatCount(total)}`}
          </p>
        </form>

        {page.status === 'loading' ? <LoadingState label="Fetching orders…" /> : null}
        {page.status === 'error' ? (
          <ErrorState error={page.error} onRetry={page.reload} context="listing orders" />
        ) : null}
        {page.status === 'success' && page.data.orders.length === 0 ? (
          <EmptyState
            title="No orders match these filters"
            detail="This is an empty result, not a failure. Widen the filters to see more."
          />
        ) : null}

        {page.status === 'success' && page.data.orders.length > 0 ? (
          <>
            <table className="table table--interactive">
              <caption className="visually-hidden">Orders matching the current filters</caption>
              <thead>
                <tr>
                  <th scope="col">Order</th>
                  <th scope="col">Placed</th>
                  <th scope="col">Value</th>
                  <th scope="col">Payment</th>
                  <th scope="col">Category</th>
                  <th scope="col">Split</th>
                  <th scope="col">Outcome</th>
                  <th scope="col">
                    <span className="visually-hidden">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {page.data.orders.map((order) => (
                  <OrderRow key={order.order_id} order={order} onSelect={onSelect} />
                ))}
              </tbody>
            </table>

            <nav className="pagination" aria-label="Order queue pages">
              <button
                type="button"
                className="button button--secondary"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(offset - PAGE_SIZE, 0))}
              >
                Previous
              </button>
              <span aria-live="polite">
                {formatCount(offset + 1)}–{formatCount(offset + page.data.orders.length)}
              </span>
              <button
                type="button"
                className="button button--secondary"
                disabled={offset + PAGE_SIZE >= page.data.total}
                onClick={() => setOffset(offset + PAGE_SIZE)}
              >
                Next
              </button>
            </nav>
          </>
        ) : null}
      </Panel>
    </div>
  );
}

function OrderRow({
  order,
  onSelect,
}: {
  order: OrderSummary;
  onSelect: (orderId: string) => void;
}): JSX.Element {
  return (
    <tr>
      <th scope="row">
        <code>{order.order_id}</code>
      </th>
      <td>{formatDate(order.ordered_at)}</td>
      <td className="numeric">{formatInr(order.order_value_inr)}</td>
      <td>
        <span className={`chip chip--${order.payment_method}`}>{order.payment_method}</span>
      </td>
      <td>{order.category}</td>
      <td>{order.split}</td>
      <td>
        <OutcomeCell order={order} />
      </td>
      <td>
        <button
          type="button"
          className="button button--small"
          onClick={() => onSelect(order.order_id)}
        >
          Investigate
        </button>
      </td>
    </tr>
  );
}

/**
 * The outcome, with "not yet known" kept distinct from "delivered".
 *
 * `is_rto: null` means the order has not matured. Rendering that as "delivered"
 * is the single most effective way to make a risk console optimistic, so it gets
 * its own visibly different treatment.
 */
function OutcomeCell({ order }: { order: OrderSummary }): JSX.Element {
  if (order.is_rto === null) {
    return (
      <span className="chip chip--pending" title="No outcome yet. Not the same as delivered.">
        {order.outcome ?? 'pending'}
      </span>
    );
  }
  return (
    <span className={`chip chip--${order.is_rto ? 'rto' : 'delivered'}`}>
      {order.outcome ?? (order.is_rto ? 'rto' : 'delivered')}
    </span>
  );
}

export { OutcomeCell };
