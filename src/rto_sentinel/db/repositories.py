"""Repositories - the only place that reads or writes the database.

Every query in this application lives here. Route handlers do not build queries,
the decision engine does not touch a session, and the agent layer gets read-only
accessors and nothing else. Two reasons this boundary earns its keep:

* **The decision log is append-only in practice, not just in intent.** There is
  no ``update_decision``. A changed decision is a new row, which is what an audit
  trail requires. The absence of the method is the enforcement.
* **The agent layer cannot write.** :class:`ReadOnlyRepository` is the interface
  the agent toolset receives. It has no write methods to call, so "the LLM must
  not modify a decision" is a fact about the type it holds rather than a rule
  someone has to remember.

WHY THE DATASET LOADER USES CORE INSERTS
----------------------------------------
:class:`DatasetRepository` bulk-loads a generated dataset with SQLAlchemy Core
``insert()`` statements in chunks rather than ORM objects. Building 120,000 ORM
instances to insert them once is slow and pointless - there is no object graph to
maintain afterwards. The chunking keeps the parameter count inside the driver's
limit, which is the thing that actually breaks on a naive single-statement load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd
from sqlalchemy import Integer, delete, func, insert, select
from sqlalchemy.orm import selectinload

from rto_sentinel.data import schema as cols
from rto_sentinel.db.models import (
    Address,
    Customer,
    DatasetRun,
    DeliveryEvent,
    Order,
    OrderOutcomeRecord,
    SimulationLatent,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.orm import Session

    from rto_sentinel.contracts.decision import Decision as DecisionContract
    from rto_sentinel.contracts.decision import OpsOverride
    from rto_sentinel.contracts.orders import OrderOutcomeUpdate, OrderPayload
    from rto_sentinel.data.generator import GenerationResult
    from rto_sentinel.db.models import Decision, OpsOverrideRecord

#: Rows per INSERT batch. Chosen so that (rows x columns) stays well under the
#: 65,535-parameter ceiling that psycopg enforces per statement.
INSERT_CHUNK_SIZE = 1000

#: `decisions.reason_codes` is a TEXT column, so the tuple is joined on write and
#: split on read. Reason codes are `[A-Z0-9_]` by construction (see
#: `decision.reason_codes.code_for`), so a comma cannot appear inside one and the
#: round trip is lossless. Stored flat rather than as JSON because these are
#: filtered and counted in SQL by operations dashboards, and a JSON column would
#: make the common query the awkward one.
REASON_CODE_SEPARATOR = ","


def split_reason_codes(stored: str | None) -> list[str]:
    """Read the stored reason codes back into a list."""
    return [code for code in (stored or "").split(REASON_CODE_SEPARATOR) if code]


def _nan_to_none(value: Any) -> Any:
    """Convert pandas' NaN/NaT sentinels into SQL NULL.

    Necessary and not merely tidy: ``float('nan')`` inserted into a nullable
    column is *not* NULL, it is the float NaN, and it would quietly turn "this
    customer has no history" into "this customer has an unrepresentable history".
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    # numpy scalars do not adapt cleanly to every driver.
    if hasattr(value, "item") and not isinstance(value, (str, bytes, datetime)):
        return value.item()
    return value


def _records(frame: pd.DataFrame, columns: dict[str, str]) -> list[dict[str, Any]]:
    """Turn a frame into insert dicts, mapping frame columns to table columns."""
    subset = frame[list(columns)]
    renamed = subset.rename(columns=columns)
    return [
        {str(key): _nan_to_none(value) for key, value in record.items()}
        for record in renamed.to_dict(orient="records")
    ]


def _chunked(
    records: list[dict[str, Any]], size: int = INSERT_CHUNK_SIZE
) -> list[list[dict[str, Any]]]:
    return [records[i : i + size] for i in range(0, len(records), size)]


class ReadOnlyRepository(Protocol):
    """The read surface. This is what the agent layer is handed."""

    def get_order(self, order_id: str) -> Order | None: ...

    def get_latest_decision(self, order_id: str) -> Decision | None: ...

    def list_review_queue(self, merchant_id: str, limit: int = 50) -> list[Decision]: ...

    def digest_figures(
        self, merchant_id: str, period_start: datetime, period_end: datetime
    ) -> dict[str, float]:
        """Aggregate rupee figures for the weekly digest, computed in SQL.

        The digest writer receives the output of this method as the complete set
        of numbers it may mention. It does not compute anything itself.
        """
        ...


class DatasetRepository:
    """Loads a generated benchmark dataset into the database, and reads it back.

    The load is transactional at the caller's level: everything is inserted inside
    the session the caller supplies, so a failure part-way leaves no half-loaded
    dataset behind.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # --- write ------------------------------------------------------------

    def load(self, result: GenerationResult, *, replace: bool = True) -> dict[str, int]:
        """Insert a whole generated dataset. Returns row counts per table.

        ``replace=True`` deletes any existing rows for the same ``run_id`` first.
        That makes re-running the seed script idempotent: the ``run_id`` is a
        deterministic digest of the generator parameters, so regenerating the same
        dataset replaces it rather than accumulating a second near-identical copy
        that would silently double every count.
        """
        metadata = result.metadata
        if replace:
            self.delete_run(metadata.run_id)

        self._session.execute(
            insert(DatasetRun).values(
                run_id=metadata.run_id,
                generator_version=metadata.generator_version,
                seed=metadata.seed,
                config_fingerprint=metadata.config_fingerprint,
                config_snapshot=metadata.config_snapshot,
                n_customers=metadata.n_customers,
                n_orders=metadata.n_orders,
                start_date=metadata.start_date,
                end_date=metadata.end_date,
                realised_rto_rate_cod=_nan_to_none(metadata.realised_rto_rate_cod),
                realised_rto_rate_prepaid=_nan_to_none(metadata.realised_rto_rate_prepaid),
                realised_cod_share=_nan_to_none(metadata.realised_cod_share),
                n_mature=metadata.n_mature,
                n_immature=metadata.n_immature,
            )
        )

        counts: dict[str, int] = {"dataset_runs": 1}

        # --- customers ----------------------------------------------------
        customer_records = _records(
            result.customers,
            {
                cols.CUSTOMER_HASH: "customer_hash",
                "signup_at": "signup_at",
                "home_pincode": "home_pincode",
                "home_pincode_tier": "home_pincode_tier",
                "generated_order_count": "generated_order_count",
            },
        )
        for record in customer_records:
            record["dataset_run_id"] = metadata.run_id
        for chunk in _chunked(customer_records):
            self._session.execute(insert(Customer), chunk)
        counts["customers"] = len(customer_records)

        # --- addresses ----------------------------------------------------
        address_records = _records(
            result.addresses,
            {
                cols.ADDRESS_FINGERPRINT: "address_fingerprint",
                cols.ADDRESS_LINE: "address_line",
                cols.ADDRESS_CITY: "address_city",
                cols.ADDRESS_STATE: "address_state",
                cols.PINCODE: "pincode",
                cols.PINCODE_TIER: "pincode_tier",
                cols.ADDR_TOKEN_COUNT: "addr_token_count",
                cols.ADDR_HAS_HOUSE_NUMBER: "addr_has_house_number",
                cols.ADDR_HAS_FLOOR_NUMBER: "addr_has_floor_number",
                cols.ADDR_HAS_LANDMARK: "addr_has_landmark",
                cols.ADDR_PINCODE_CITY_CONSISTENT: "addr_pincode_city_consistent",
                cols.ADDR_ALLCAPS_RATIO: "addr_allcaps_ratio",
                cols.ADDR_GIBBERISH_RATIO: "addr_gibberish_ratio",
                "first_seen_at": "first_seen_at",
            },
        )
        for record in address_records:
            record["dataset_run_id"] = metadata.run_id
        for chunk in _chunked(address_records):
            self._session.execute(insert(Address), chunk)
        counts["addresses"] = len(address_records)

        self._session.flush()

        # --- resolve surrogate keys ---------------------------------------
        customer_pks: dict[str, int] = {
            str(key): int(value)
            for key, value in self._session.execute(
                select(Customer.customer_hash, Customer.id).where(
                    Customer.dataset_run_id == metadata.run_id
                )
            ).all()
        }
        address_pks: dict[str, int] = {
            str(key): int(value)
            for key, value in self._session.execute(
                select(Address.address_fingerprint, Address.id).where(
                    Address.dataset_run_id == metadata.run_id
                )
            ).all()
        }

        # --- orders --------------------------------------------------------
        order_columns = {
            cols.ORDER_ID: "order_id",
            cols.MERCHANT_ID: "merchant_id",
            cols.CUSTOMER_HASH: "customer_hash",
            cols.ORDERED_AT: "ordered_at",
            cols.DISPATCHED_AT: "dispatched_at",
            cols.FIRST_ATTEMPT_AT: "first_attempt_at",
            cols.DAY_INDEX: "day_index",
            cols.PAYMENT_METHOD: "payment_method",
            cols.IS_COD: "is_cod",
            cols.ORDER_VALUE_INR: "order_value_inr",
            cols.DISCOUNT_INR: "discount_inr",
            cols.DISCOUNT_DEPTH: "discount_depth",
            cols.ITEM_COUNT: "item_count",
            cols.CATEGORY: "category",
            cols.CART_EDITED: "cart_edited",
            cols.PRODUCT_PAGE_SECONDS: "product_page_seconds",
            cols.SESSIONS_BEFORE_PURCHASE: "sessions_before_purchase",
            cols.DEVICE_CLASS: "device_class",
            cols.HOUR_OF_DAY: "hour_of_day",
            cols.DAY_OF_WEEK: "day_of_week",
            cols.IS_LATE_NIGHT: "is_late_night",
            cols.IS_SALE_DAY: "is_sale_day",
            cols.TIME_TO_CHECKOUT_SECONDS: "time_to_checkout_seconds",
            cols.COD_AFTER_PREPAID_FAILURE: "cod_after_prepaid_failure",
            cols.COURIER_PARTNER: "courier_partner",
            cols.PRIOR_ORDER_COUNT: "prior_order_count",
            cols.PRIOR_RTO_COUNT: "prior_rto_count",
            cols.PRIOR_RTO_RATE: "prior_rto_rate",
            cols.DAYS_SINCE_LAST_ORDER: "days_since_last_order",
            cols.PREPAID_TO_COD_RATIO: "prepaid_to_cod_ratio",
            cols.MEAN_RESOLUTION_DAYS: "mean_resolution_days",
            cols.IS_NEW_CUSTOMER: "is_new_customer",
            cols.SPLIT: "split",
        }
        order_records = _records(result.orders, order_columns)
        fingerprints = result.orders[cols.ADDRESS_FINGERPRINT].tolist()
        for record, fingerprint in zip(order_records, fingerprints, strict=True):
            record["customer_pk"] = customer_pks[record["customer_hash"]]
            record["address_pk"] = address_pks[fingerprint]
            record["dataset_run_id"] = metadata.run_id
        for chunk in _chunked(order_records, size=500):
            self._session.execute(insert(Order), chunk)
        counts["orders"] = len(order_records)

        self._session.flush()
        order_pks: dict[str, int] = {
            str(key): int(value)
            for key, value in self._session.execute(
                select(Order.order_id, Order.id).where(Order.dataset_run_id == metadata.run_id)
            ).all()
        }

        # --- outcomes -------------------------------------------------------
        outcome_records = _records(
            result.orders,
            {
                cols.ORDER_ID: "order_id",
                cols.OUTCOME: "outcome",
                cols.IS_RTO: "is_rto",
                cols.RESOLVED_AT: "resolved_at",
                cols.MATURITY_DAYS: "maturity_days",
                cols.IS_MATURE: "is_mature",
            },
        )
        for record in outcome_records:
            record["order_pk"] = order_pks[record["order_id"]]
            # A pandas object column holding True/False/None round-trips as
            # numpy.bool_ or None; normalise so the driver binds a real boolean.
            record["is_rto"] = None if record["is_rto"] is None else bool(record["is_rto"])
            record["is_mature"] = bool(record["is_mature"])
        for chunk in _chunked(outcome_records):
            self._session.execute(insert(OrderOutcomeRecord), chunk)
        counts["order_outcomes"] = len(outcome_records)

        # --- delivery events -------------------------------------------------
        event_records = _records(
            result.delivery_events,
            {
                "order_id": "order_id",
                "sequence": "sequence",
                "event_type": "event_type",
                "occurred_at": "occurred_at",
            },
        )
        for record in event_records:
            record["order_pk"] = order_pks[record["order_id"]]
        for chunk in _chunked(event_records):
            self._session.execute(insert(DeliveryEvent), chunk)
        counts["delivery_events"] = len(event_records)

        # --- simulation latents ----------------------------------------------
        latent_records = _records(
            result.latents,
            {
                cols.ORDER_ID: "order_id",
                cols.TRUE_RTO_PROBABILITY: "true_rto_probability",
                cols.LATENT_LOGIT: "latent_logit",
                "customer_reliability": "customer_reliability",
                "latent_address_quality": "latent_address_quality",
                "address_deliverability": "address_deliverability",
                "pincode_effect": "pincode_effect",
                "label_flipped": "label_flipped",
            },
        )
        for record in latent_records:
            record["order_pk"] = order_pks[record["order_id"]]
            record["dataset_run_id"] = metadata.run_id
            record["label_flipped"] = bool(record["label_flipped"])
        for chunk in _chunked(latent_records):
            self._session.execute(insert(SimulationLatent), chunk)
        counts["simulation_latents"] = len(latent_records)

        return counts

    def delete_run(self, run_id: str) -> None:
        """Remove a dataset run and everything that hangs off it.

        Child rows are deleted explicitly rather than relying on ``ON DELETE
        CASCADE``, because SQLite does not enforce foreign keys by default and a
        loader that only works on PostgreSQL would leave the test suite quietly
        accumulating orphans.
        """
        order_ids = [
            row[0]
            for row in self._session.execute(
                select(Order.id).where(Order.dataset_run_id == run_id)
            ).all()
        ]
        if order_ids:
            for chunk in (order_ids[i : i + 500] for i in range(0, len(order_ids), 500)):
                self._session.execute(
                    delete(OrderOutcomeRecord).where(OrderOutcomeRecord.order_pk.in_(chunk))
                )
                self._session.execute(
                    delete(DeliveryEvent).where(DeliveryEvent.order_pk.in_(chunk))
                )
                self._session.execute(
                    delete(SimulationLatent).where(SimulationLatent.order_pk.in_(chunk))
                )
        self._session.execute(delete(Order).where(Order.dataset_run_id == run_id))
        self._session.execute(delete(Address).where(Address.dataset_run_id == run_id))
        self._session.execute(delete(Customer).where(Customer.dataset_run_id == run_id))
        self._session.execute(delete(DatasetRun).where(DatasetRun.run_id == run_id))

    # --- read -------------------------------------------------------------

    def get_run(self, run_id: str) -> DatasetRun | None:
        return self._session.get(DatasetRun, run_id)

    def list_runs(self) -> list[DatasetRun]:
        return list(
            self._session.execute(
                select(DatasetRun).order_by(DatasetRun.created_at.desc())
            ).scalars()
        )

    def table_counts(self, run_id: str) -> dict[str, int]:
        """Row counts per table for one dataset run, straight from SQL."""
        order_subquery = select(Order.id).where(Order.dataset_run_id == run_id).scalar_subquery()
        return {
            "customers": self._scalar(
                select(func.count()).select_from(Customer).where(Customer.dataset_run_id == run_id)
            ),
            "addresses": self._scalar(
                select(func.count()).select_from(Address).where(Address.dataset_run_id == run_id)
            ),
            "orders": self._scalar(
                select(func.count()).select_from(Order).where(Order.dataset_run_id == run_id)
            ),
            "order_outcomes": self._scalar(
                select(func.count())
                .select_from(OrderOutcomeRecord)
                .where(OrderOutcomeRecord.order_pk.in_(order_subquery))
            ),
            "delivery_events": self._scalar(
                select(func.count())
                .select_from(DeliveryEvent)
                .where(DeliveryEvent.order_pk.in_(order_subquery))
            ),
            "simulation_latents": self._scalar(
                select(func.count())
                .select_from(SimulationLatent)
                .where(SimulationLatent.dataset_run_id == run_id)
            ),
        }

    def split_counts(self, run_id: str) -> dict[str, int]:
        rows = self._session.execute(
            select(Order.split, func.count())
            .where(Order.dataset_run_id == run_id)
            .group_by(Order.split)
        ).all()
        return {str(split): int(count) for split, count in rows}

    def outcome_counts(self, run_id: str) -> dict[str, int]:
        rows = self._session.execute(
            select(OrderOutcomeRecord.outcome, func.count())
            .join(Order, Order.id == OrderOutcomeRecord.order_pk)
            .where(Order.dataset_run_id == run_id)
            .group_by(OrderOutcomeRecord.outcome)
        ).all()
        return {str(outcome): int(count) for outcome, count in rows}

    def rto_rate_by_payment_method(self, run_id: str) -> dict[str, float]:
        """Observed RTO rate per payment method, over MATURE orders only.

        The maturity filter is the point. Including pending orders would divide by
        a denominator containing rows whose outcome is unknown, which understates
        the rate - the exact direction that would make a benchmark look easier
        than it is.
        """
        rows = self._session.execute(
            select(
                Order.payment_method,
                func.count(),
                # Cast before summing: PostgreSQL and SQLite disagree about
                # whether a boolean can be summed directly.
                func.sum(func.cast(OrderOutcomeRecord.is_rto, Integer)),
            )
            .join(OrderOutcomeRecord, OrderOutcomeRecord.order_pk == Order.id)
            .where(Order.dataset_run_id == run_id)
            .where(OrderOutcomeRecord.is_mature.is_(True))
            .group_by(Order.payment_method)
        ).all()
        return {
            str(method): (float(rto or 0) / total if total else float("nan"))
            for method, total, rto in rows
        }

    def _scalar(self, statement: Any) -> int:
        return int(self._session.execute(statement).scalar_one())


class OrderRepository:
    """Reads and writes individual orders and their outcomes (serving path)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, payload: OrderPayload) -> Order:
        raise NotImplementedError("Single-order persistence lands in Phase 4 (serving path).")

    def get_order(self, order_id: str) -> Order | None:
        return self._session.execute(
            select(Order).where(Order.order_id == order_id)
        ).scalar_one_or_none()

    def record_outcome(self, update: OrderOutcomeUpdate) -> None:
        """Record a terminal delivery state.

        Rejects an outcome whose ``resolved_at`` precedes the order's
        ``ordered_at``: that is not a late label, it is corrupt data, and letting
        it through would poison every as-of aggregate computed afterwards.
        """
        raise NotImplementedError("Outcome ingestion lands in Phase 4 (serving path).")


class DecisionRepository:
    """Append-only decision log. Note the absence of an update method."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def append(self, decision: DecisionContract, *, config_fingerprint: str) -> Decision:
        raise NotImplementedError("Decision logging lands in Phase 4.")

    def get_latest_decision(self, order_id: str) -> Decision | None:
        raise NotImplementedError("Decision logging lands in Phase 4.")

    def list_review_queue(self, merchant_id: str, limit: int = 50) -> list[Decision]:
        """SEVERE-band decisions awaiting a human, oldest first.

        Oldest first on purpose: a queue sorted by risk score leaves the least
        risky appeals waiting forever, and those are disproportionately the false
        positives - the customers who did nothing wrong.
        """
        raise NotImplementedError("Review queue lands in Phase 4.")


class OverrideRepository:
    """Ops overrides. Always available, always logged."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def append(self, override: OpsOverride) -> None:
        raise NotImplementedError("Override logging lands in Phase 4.")


# ---------------------------------------------------------------------------
# the serving path
# ---------------------------------------------------------------------------

#: Columns the feature pipeline needs that live on the orders table itself.
_ORDER_FEATURE_COLUMNS = (
    Order.order_id,
    Order.merchant_id,
    Order.customer_hash,
    Order.ordered_at,
    Order.payment_method,
    Order.is_cod,
    Order.order_value_inr,
    Order.discount_inr,
    Order.discount_depth,
    Order.item_count,
    Order.category,
    Order.hour_of_day,
    Order.day_of_week,
    Order.is_late_night,
    Order.is_sale_day,
    Order.cart_edited,
    Order.device_class,
    Order.sessions_before_purchase,
    Order.time_to_checkout_seconds,
    Order.product_page_seconds,
    Order.cod_after_prepaid_failure,
    Order.courier_partner,
    Order.split,
    Order.day_index,
    Order.dataset_run_id,
)


@dataclass(frozen=True, slots=True)
class OrderPage:
    """One page of orders, with the total so a client can paginate honestly."""

    orders: list[Order]
    total: int
    limit: int
    offset: int


class ServingRepository:
    """Reads the serving path needs: one order, a page of orders, a context frame.

    Kept apart from :class:`DatasetRepository`, which is a bulk loader. The
    queries here are the ones a live request makes, and they are written once,
    here, rather than assembled inside route handlers - a handler that builds SQL
    is a handler that will eventually build a slightly different query for the
    same question.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- single order ---------------------------------------------------

    def get_order(self, order_id: str, *, dataset_run_id: str | None = None) -> Order | None:
        """One order by its id, disambiguated by dataset run.

        ORDER IDS ARE UNIQUE WITHIN A RUN, NOT ACROSS THE DATABASE
        ---------------------------------------------------------
        Every generator run numbers its orders from ``ORD-00000001``, which is
        why migration ``4f1c2a7d8e30`` scoped the uniqueness constraint to
        ``(dataset_run_id, order_id)``. A database holding two benchmark runs
        therefore holds two orders called ``ORD-00000042``, and they are
        different orders belonging to different synthetic universes.

        Without ``dataset_run_id`` this returns the one from the most recently
        created run - a defined answer rather than an arbitrary one - and the API
        exposes the parameter so a caller can be explicit.
        """
        statement = select(Order).where(Order.order_id == order_id)
        if dataset_run_id is not None:
            statement = statement.where(Order.dataset_run_id == dataset_run_id)
        else:
            statement = statement.join(
                DatasetRun, Order.dataset_run_id == DatasetRun.run_id, isouter=True
            ).order_by(DatasetRun.created_at.desc().nullslast())
        return self._session.execute(statement.limit(1)).scalar_one_or_none()

    def get_outcome(self, order: Order) -> OrderOutcomeRecord | None:
        """The label, where it has matured. NULL is a legitimate answer.

        Keyed on the order's surrogate primary key rather than its ``order_id``,
        because the id alone does not identify a row once the database holds more
        than one dataset run.
        """
        return self._session.execute(
            select(OrderOutcomeRecord).where(OrderOutcomeRecord.order_pk == order.id).limit(1)
        ).scalar_one_or_none()

    def get_address(self, address_pk: int) -> Address | None:
        return self._session.get(Address, address_pk)

    # -- listing --------------------------------------------------------

    def list_orders(
        self,
        *,
        merchant_id: str | None = None,
        customer_hash: str | None = None,
        split: str | None = None,
        payment_method: str | None = None,
        dataset_run_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> OrderPage:
        """A page of orders, newest first, with the unpaginated total.

        Newest first because an operations queue is read from the top. The total
        is computed with the same filters so a client can show "showing 50 of
        12,431" without a second endpoint that could disagree.
        """
        filters = []
        if merchant_id is not None:
            filters.append(Order.merchant_id == merchant_id)
        if customer_hash is not None:
            filters.append(Order.customer_hash == customer_hash)
        if split is not None:
            filters.append(Order.split == split)
        if payment_method is not None:
            filters.append(Order.payment_method == payment_method)
        if dataset_run_id is not None:
            filters.append(Order.dataset_run_id == dataset_run_id)

        total = self._session.execute(
            select(func.count()).select_from(Order).where(*filters)
        ).scalar_one()
        rows = (
            self._session.execute(
                select(Order)
                # Eager-loaded, deliberately. The list response reports each
                # order's outcome, and a lazy relationship would issue one query
                # per row - 200 round trips for a page that should cost two.
                .options(selectinload(Order.outcome))
                .where(*filters)
                .order_by(Order.ordered_at.desc(), Order.order_id.desc())
                .limit(limit)
                .offset(offset)
            )
            .scalars()
            .all()
        )
        return OrderPage(orders=list(rows), total=int(total), limit=limit, offset=offset)

    # -- the context frame ----------------------------------------------

    def context_frame(self, order: Order, *, limit: int = 20000) -> pd.DataFrame:
        """Every row the feature pipeline needs to score ``order`` correctly.

        WHY A FRAME AND NOT A ROW
        -------------------------
        The feature families recompute history from the data rather than reading
        the ``prior_*`` snapshot stored on the row. That is the right design for
        training - the snapshot could be stale or wrong and the as-of machinery
        is the thing under test - but it means a single order cannot be scored in
        isolation. Customer history needs that customer's earlier orders;
        geography needs enough of the merchant's book for a pincode aggregate to
        clear its minimum support.

        So this returns the order plus its context: everything for the same
        merchant placed at or before this order's ``ordered_at``. The as-of
        machinery then masks anything that had not *resolved* by then, which is
        what makes including extra rows safe rather than leaky - the guarantee
        the leakage suite verifies rather than assumes.

        ``limit`` bounds the query. A production system would precompute the
        geography aggregates into a feature store and read one row; this reads
        the book, which is honest about what the features actually require and
        is fast enough for a benchmark of this size. When the cap truncates the
        history, the geography features degrade towards their prior rather than
        becoming wrong - the minimum-support guard sees less evidence and shrinks
        harder.
        """
        statement = (
            select(
                *_ORDER_FEATURE_COLUMNS,
                Address.address_fingerprint,
                Address.pincode,
                Address.pincode_tier,
                Address.addr_token_count,
                Address.addr_has_house_number,
                Address.addr_has_floor_number,
                Address.addr_has_landmark,
                Address.addr_pincode_city_consistent,
                Address.addr_allcaps_ratio,
                Address.addr_gibberish_ratio,
                Customer.signup_at,
                OrderOutcomeRecord.is_rto,
                OrderOutcomeRecord.resolved_at,
                OrderOutcomeRecord.maturity_days,
                OrderOutcomeRecord.is_mature,
                OrderOutcomeRecord.outcome,
            )
            .join(Address, Order.address_pk == Address.id)
            .join(Customer, Order.customer_pk == Customer.id)
            .join(OrderOutcomeRecord, OrderOutcomeRecord.order_pk == Order.id, isouter=True)
            .where(
                # Scoped to the order's OWN dataset run. Two benchmark runs share
                # merchant ids and pincodes while being entirely independent
                # universes; blending them would compute this customer's history
                # and this pincode's RTO rate from orders that never existed
                # alongside each other.
                Order.dataset_run_id == order.dataset_run_id,
                Order.merchant_id == order.merchant_id,
                Order.ordered_at <= order.ordered_at,
            )
            .order_by(Order.ordered_at.desc())
            .limit(limit)
        )
        result = self._session.execute(statement)
        frame = pd.DataFrame(result.mappings().all())
        if frame.empty:  # pragma: no cover - the target order is always present
            return frame

        # The target order must be in the frame even if the cap would have cut
        # it: it is the row being scored, and a truncated context is survivable
        # while a missing subject is not.
        if order.order_id not in set(frame[cols.ORDER_ID]):  # pragma: no cover - cap edge case
            msg = f"context frame for {order.order_id} did not contain the order itself"
            raise ValueError(msg)

        return frame.sort_values(cols.ORDERED_AT).reset_index(drop=True)


class DecisionLogRepository:
    """Append-only decision log. Note the absence of an update method.

    A decision is an event, not a record to be corrected. When an operator
    disagrees, an override is appended alongside; the original stays exactly as
    it was, because an audit trail that can be edited is not an audit trail.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def append(
        self,
        decision: DecisionContract,
        *,
        order_pk: int,
        model_name: str,
        model_version: str,
        config_fingerprint: str,
    ) -> Decision:
        from rto_sentinel.db.models import Decision as DecisionRow

        # The decision row records `model_version`, not `feature_version`: the
        # model card is the authority on which feature set produced a given
        # model, so storing both would create two places for that fact to
        # disagree. The API response carries the feature version, read from the
        # card at score time.
        row = DecisionRow(
            order_pk=order_pk,
            order_id=decision.order_id,
            probability=decision.probability,
            threshold=decision.threshold,
            band=decision.band.value,
            action=decision.action.value,
            flagged=decision.flagged,
            reason_codes=REASON_CODE_SEPARATOR.join(decision.reason_codes),
            expected_value_inr=decision.expected_value_inr,
            appeal_available=decision.appeal_available,
            human_review_required=decision.human_review_required,
            is_control_holdout=decision.is_control_holdout,
            model_name=model_name,
            model_version=model_version,
            engine_version=decision.engine_version,
            config_fingerprint=config_fingerprint,
            decided_at=decision.decided_at,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def get_latest_decision(self, order_id: str) -> Decision | None:
        from rto_sentinel.db.models import Decision as DecisionRow

        return self._session.execute(
            select(DecisionRow)
            .where(DecisionRow.order_id == order_id)
            .order_by(DecisionRow.decided_at.desc(), DecisionRow.id.desc())
            .limit(1)
        ).scalar_one_or_none()

    def list_for_order(self, order_id: str, *, limit: int = 20) -> list[Decision]:
        from rto_sentinel.db.models import Decision as DecisionRow

        rows = (
            self._session.execute(
                select(DecisionRow)
                .where(DecisionRow.order_id == order_id)
                .order_by(DecisionRow.decided_at.desc(), DecisionRow.id.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return list(rows)

    def list_review_queue(self, merchant_id: str, limit: int = 50) -> list[Decision]:
        """SEVERE-band decisions awaiting a human, oldest first.

        Oldest first on purpose: a queue sorted by risk score leaves the least
        risky appeals waiting forever, and those are disproportionately the false
        positives - the customers who did nothing wrong.
        """
        from rto_sentinel.db.models import Decision as DecisionRow

        rows = (
            self._session.execute(
                select(DecisionRow)
                .join(Order, DecisionRow.order_pk == Order.id)
                .where(
                    Order.merchant_id == merchant_id,
                    DecisionRow.human_review_required.is_(True),
                )
                .order_by(DecisionRow.decided_at.asc(), DecisionRow.id.asc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return list(rows)

    def band_counts(self, *, merchant_id: str | None = None) -> dict[str, int]:
        """How many decisions landed in each band. Operational, not a metric."""
        from rto_sentinel.db.models import Decision as DecisionRow

        statement = select(DecisionRow.band, func.count()).group_by(DecisionRow.band)
        if merchant_id is not None:
            statement = statement.join(Order, DecisionRow.order_pk == Order.id).where(
                Order.merchant_id == merchant_id
            )
        return {band: int(count) for band, count in self._session.execute(statement).all()}


class OpsOverrideRepository:
    """Ops overrides. Always available, always logged, never destructive."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def append(self, override: OpsOverride, *, decision_pk: int) -> OpsOverrideRecord:
        from rto_sentinel.db.models import OpsOverrideRecord as OverrideRow

        row = OverrideRow(
            decision_pk=decision_pk,
            order_id=override.order_id,
            original_band=override.original_band.value,
            override_band=override.override_band.value,
            direction=override.direction.value,
            operator_id=override.operator_id,
            note=override.note,
            created_at=override.created_at,
        )
        self._session.add(row)
        self._session.flush()
        return row

    def list_for_order(self, order_id: str) -> list[OpsOverrideRecord]:
        from rto_sentinel.db.models import OpsOverrideRecord as OverrideRow

        rows = (
            self._session.execute(
                select(OverrideRow)
                .where(OverrideRow.order_id == order_id)
                .order_by(OverrideRow.created_at.asc(), OverrideRow.id.asc())
            )
            .scalars()
            .all()
        )
        return list(rows)

    def direction_counts(self) -> dict[str, int]:
        """Escalations against relaxations - the shape of human disagreement."""
        from rto_sentinel.db.models import OpsOverrideRecord as OverrideRow

        return {
            direction: int(count)
            for direction, count in self._session.execute(
                select(OverrideRow.direction, func.count()).group_by(OverrideRow.direction)
            ).all()
        }
