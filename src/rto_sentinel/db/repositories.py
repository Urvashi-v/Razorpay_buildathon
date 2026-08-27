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
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd
from sqlalchemy import Integer, delete, func, insert, select

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
    from rto_sentinel.db.models import Decision

#: Rows per INSERT batch. Chosen so that (rows x columns) stays well under the
#: 65,535-parameter ceiling that psycopg enforces per statement.
INSERT_CHUNK_SIZE = 1000


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
