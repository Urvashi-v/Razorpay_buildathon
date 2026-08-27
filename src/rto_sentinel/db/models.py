"""Database tables.

WHAT IS STORED, AND THE RULE THAT DECIDES IT
============================================
Enough raw information to **reconstruct** a prediction, never just the prediction
itself. A decision log that records only "score 0.44, band HIGH" is unauditable:
six months later nobody can tell whether the score was right, whether the feature
that drove it was computed correctly, or what the customer's history actually
looked like at that instant. So the order-time feature snapshot, every timestamp
in the fulfilment chain, and the full event trail are all persisted.

THE TABLES
==========

``customers``
    The customer dimension. A hashed identifier, a signup timestamp, and a home
    pincode. No name, phone, email, gender or age column exists - not nullable
    ones, none - because a column that does not exist cannot be populated later
    by a well-meaning integration.

``addresses``
    Address fingerprints with their observable quality signals. Deduplicated, so
    a customer reusing one home address appears once. The raw text is retained
    because an ops associate reviewing a SEVERE decision needs to see what the
    customer actually typed; it never leaves the merchant boundary.

``orders``
    One row per order, carrying the order-time feature snapshot. Those columns
    were computed during simulation from orders that had already **resolved**
    before this order was placed, so what is stored is what was genuinely
    knowable at that instant.

``order_outcomes``
    The terminal delivery state and its timestamps, including label maturity.
    Separate from ``orders`` because it *is the label*. If ``outcome`` were a
    column on ``orders`` the convenient query would be the leaky one; here the
    convenient query has to reference ``resolved_at``, so the time constraint
    becomes the obvious thing to write.

``delivery_events``
    The ordered event trail: placed, dispatched, attempted, terminal. This is
    what makes "what was known when" answerable rather than assumed.

``dataset_runs``
    Provenance for a generated dataset: seed, generator version, configuration
    snapshot and fingerprint, creation timestamp, realised base rates. Everything
    needed to regenerate it exactly.

``simulation_latents``
    The simulator's own ground truth - the true per-order RTO probability and the
    latent variables behind it. **Never a feature.** It exists so calibration can
    be measured against a known truth, which is one of the few things synthetic
    data can honestly offer. Every column here is in
    ``data.schema.FORBIDDEN_IN_FEATURES`` and the table is excluded from the ML
    export by default.

``decisions`` / ``ops_overrides`` / ``model_runs``
    The Phase 1 audit tables.

PRIVACY NOTE
------------
``customer_hash`` is an opaque digest. This database never receives, stores, or
is capable of storing a pre-image, because in this project there is no real
identity behind it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rto_sentinel.contracts.enums import (
    DatasetSplit,
    DeviceClass,
    InterventionAction,
    OrderOutcome,
    OverrideDirection,
    PaymentMethod,
    PincodeTier,
    RiskBand,
)
from rto_sentinel.db.base import Base, TimestampMixin


class DatasetRun(Base, TimestampMixin):
    """Provenance for one generated dataset.

    Everything needed to reproduce it: the seed, the generator version, the exact
    configuration and its fingerprint. ``run_id`` is a deterministic digest of the
    parameters, so regenerating the same dataset is an upsert rather than a second
    near-identical copy sitting alongside the first.
    """

    __tablename__ = "dataset_runs"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)

    generator_version: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    config_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    n_customers: Mapped[int] = mapped_column(Integer, nullable=False)
    n_orders: Mapped[int] = mapped_column(Integer, nullable=False)
    start_date: Mapped[datetime] = mapped_column(nullable=False)
    end_date: Mapped[datetime] = mapped_column(nullable=False)

    realised_rto_rate_cod: Mapped[float | None] = mapped_column(Float)
    realised_rto_rate_prepaid: Mapped[float | None] = mapped_column(Float)
    realised_cod_share: Mapped[float | None] = mapped_column(Float)
    n_mature: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_immature: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Stated on the row itself, not only in the README. Anyone querying this
    # database should meet the caveat where they meet the data.
    data_provenance: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default=(
            "Synthetic benchmark data. Labels are simulated outcomes of the documented "
            "process in docs/simulator.md, not real-world ground truth."
        ),
    )

    __table_args__ = (
        CheckConstraint("n_orders > 0", name="run_has_orders"),
        CheckConstraint("end_date >= start_date", name="run_dates_ordered"),
    )


class Customer(Base, TimestampMixin):
    """The customer dimension. Identity is a hash and nothing else."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    signup_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    home_pincode: Mapped[str] = mapped_column(String(6), nullable=False, index=True)
    home_pincode_tier: Mapped[PincodeTier] = mapped_column(String(16), nullable=False)
    generated_order_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    dataset_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_runs.run_id", ondelete="CASCADE"), index=True
    )

    orders: Mapped[list[Order]] = relationship("Order", back_populates="customer")


class Address(Base, TimestampMixin):
    """Address fingerprints and their observable quality signals.

    The quality columns are measurements of the text a customer typed - structural
    completeness, not fluency. See ``data/address.py`` for why that distinction
    matters and how it is maintained.
    """

    __tablename__ = "addresses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    address_fingerprint: Mapped[str] = mapped_column(
        String(32), unique=True, nullable=False, index=True
    )

    address_line: Mapped[str] = mapped_column(Text, nullable=False)
    address_city: Mapped[str] = mapped_column(String(128), nullable=False)
    address_state: Mapped[str] = mapped_column(String(128), nullable=False)
    pincode: Mapped[str] = mapped_column(String(6), nullable=False, index=True)
    pincode_tier: Mapped[PincodeTier] = mapped_column(String(16), nullable=False, index=True)

    addr_token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    addr_has_house_number: Mapped[bool] = mapped_column(Boolean, nullable=False)
    addr_has_floor_number: Mapped[bool] = mapped_column(Boolean, nullable=False)
    addr_has_landmark: Mapped[bool] = mapped_column(Boolean, nullable=False)
    addr_pincode_city_consistent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    addr_allcaps_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    addr_gibberish_ratio: Mapped[float] = mapped_column(Float, nullable=False)

    first_seen_at: Mapped[datetime] = mapped_column(nullable=False)
    dataset_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_runs.run_id", ondelete="CASCADE"), index=True
    )

    __table_args__ = (
        CheckConstraint("addr_token_count >= 0", name="token_count_non_negative"),
        CheckConstraint(
            "addr_allcaps_ratio >= 0 AND addr_allcaps_ratio <= 1", name="allcaps_ratio_in_range"
        ),
        CheckConstraint(
            "addr_gibberish_ratio >= 0 AND addr_gibberish_ratio <= 1",
            name="gibberish_ratio_in_range",
        ),
    )


class Order(Base, TimestampMixin):
    """An order, with the feature snapshot that was knowable when it was placed.

    The ``prior_*`` columns are the as-of history: aggregates over this customer's
    orders that had already **resolved** before ``ordered_at``. Storing the
    snapshot rather than recomputing it later is what makes a past prediction
    reconstructible - a recomputation months afterwards would see a different
    history and quietly produce a different answer.
    """

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    merchant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    customer_pk: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    customer_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    address_pk: Mapped[int] = mapped_column(
        ForeignKey("addresses.id", ondelete="RESTRICT"), nullable=False, index=True
    )

    # --- timestamps for temporal reasoning ----------------------------------
    ordered_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    dispatched_at: Mapped[datetime | None] = mapped_column()
    first_attempt_at: Mapped[datetime | None] = mapped_column()
    day_index: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # --- order shape --------------------------------------------------------
    payment_method: Mapped[PaymentMethod] = mapped_column(String(16), nullable=False, index=True)
    is_cod: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    order_value_inr: Mapped[float] = mapped_column(Float, nullable=False)
    discount_inr: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    discount_depth: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    item_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    cart_edited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- session and timing -------------------------------------------------
    product_page_seconds: Mapped[float | None] = mapped_column(Float)
    sessions_before_purchase: Mapped[int | None] = mapped_column(Integer)
    device_class: Mapped[DeviceClass] = mapped_column(String(16), nullable=False)
    hour_of_day: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)
    is_late_night: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_sale_day: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    time_to_checkout_seconds: Mapped[float | None] = mapped_column(Float)
    cod_after_prepaid_failure: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- route --------------------------------------------------------------
    courier_partner: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # --- as-of customer history (the order-time snapshot) -------------------
    prior_order_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prior_rto_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prior_rto_rate: Mapped[float | None] = mapped_column(Float)
    days_since_last_order: Mapped[float | None] = mapped_column(Float)
    prepaid_to_cod_ratio: Mapped[float | None] = mapped_column(Float)
    mean_resolution_days: Mapped[float | None] = mapped_column(Float)
    is_new_customer: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    split: Mapped[DatasetSplit] = mapped_column(String(32), nullable=False, index=True)
    dataset_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_runs.run_id", ondelete="CASCADE"), index=True
    )

    customer: Mapped[Customer] = relationship("Customer", back_populates="orders")
    address: Mapped[Address] = relationship("Address")
    outcome: Mapped[OrderOutcomeRecord | None] = relationship(
        "OrderOutcomeRecord", back_populates="order", uselist=False
    )
    events: Mapped[list[DeliveryEvent]] = relationship("DeliveryEvent", back_populates="order")
    decisions: Mapped[list[Decision]] = relationship("Decision", back_populates="order")

    __table_args__ = (
        CheckConstraint("order_value_inr > 0", name="order_value_positive"),
        CheckConstraint("discount_inr >= 0", name="discount_non_negative"),
        CheckConstraint("item_count >= 1", name="item_count_at_least_one"),
        CheckConstraint("hour_of_day >= 0 AND hour_of_day <= 23", name="hour_in_range"),
        CheckConstraint(
            "prior_rto_count <= prior_order_count", name="prior_rtos_within_prior_orders"
        ),
        # Dispatch cannot precede the order. A database-level guarantee, because
        # every piece of temporal reasoning downstream depends on it.
        CheckConstraint(
            "dispatched_at IS NULL OR dispatched_at >= ordered_at",
            name="dispatch_after_order",
        ),
        CheckConstraint(
            "first_attempt_at IS NULL OR dispatched_at IS NULL "
            "OR first_attempt_at >= dispatched_at",
            name="attempt_after_dispatch",
        ),
        Index("ix_orders_merchant_ordered_at", "merchant_id", "ordered_at"),
        Index("ix_orders_customer_ordered_at", "customer_hash", "ordered_at"),
    )


class OrderOutcomeRecord(Base, TimestampMixin):
    """The terminal delivery state. This is the label.

    ``is_rto`` is deliberately nullable: an immature order has no known outcome,
    and NULL is the only honest representation of that. Defaulting it to False
    would be the single most common way a benchmark manufactures optimism.
    """

    __tablename__ = "order_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_pk: Mapped[int] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    order_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    outcome: Mapped[OrderOutcome] = mapped_column(String(16), nullable=False, index=True)
    is_rto: Mapped[bool | None] = mapped_column(Boolean, index=True)

    resolved_at: Mapped[datetime | None] = mapped_column(index=True)
    maturity_days: Mapped[float | None] = mapped_column(Float)
    is_mature: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)

    order: Mapped[Order] = relationship("Order", back_populates="outcome")

    __table_args__ = (
        # An immature outcome must carry a NULL label, and a mature one must not.
        # Written as a pure boolean expression so it is valid on both PostgreSQL
        # and SQLite - comparing a boolean column to an integer literal is a type
        # error on PostgreSQL, which is how the first attempt at this failed.
        CheckConstraint(
            "(is_mature AND is_rto IS NOT NULL) OR (NOT is_mature AND is_rto IS NULL)",
            name="label_matches_maturity",
        ),
        CheckConstraint(
            "maturity_days IS NULL OR maturity_days >= 0", name="maturity_days_non_negative"
        ),
    )


class DeliveryEvent(Base, TimestampMixin):
    """One step in an order's fulfilment trail.

    The trail is what makes a past prediction reconstructible. Without it, "what
    did we know at the moment we decided?" is a question with no answer.
    """

    __tablename__ = "delivery_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_pk: Mapped[int] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True
    )
    order_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    occurred_at: Mapped[datetime] = mapped_column(nullable=False, index=True)

    order: Mapped[Order] = relationship("Order", back_populates="events")

    __table_args__ = (
        UniqueConstraint("order_pk", "sequence", name="event_sequence_unique"),
        CheckConstraint("sequence >= 1", name="sequence_positive"),
        Index("ix_delivery_events_order_sequence", "order_id", "sequence"),
    )


class SimulationLatent(Base, TimestampMixin):
    """The simulator's own ground truth. NEVER A FEATURE.

    Synthetic data can honestly offer one thing real data cannot: the true
    probability behind each label. That makes calibration measurable against a
    known target rather than only against observed frequencies, which is worth
    having.

    It is also perfect leakage if it ever reaches a model, so it lives in its own
    table, every column is listed in ``data.schema.FORBIDDEN_IN_FEATURES``, and the
    ML export excludes it unless it is explicitly asked for.
    """

    __tablename__ = "simulation_latents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_pk: Mapped[int] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    order_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    true_rto_probability: Mapped[float] = mapped_column(Float, nullable=False)
    latent_logit: Mapped[float] = mapped_column(Float, nullable=False)
    customer_reliability: Mapped[float] = mapped_column(Float, nullable=False)
    latent_address_quality: Mapped[float] = mapped_column(Float, nullable=False)
    address_deliverability: Mapped[float] = mapped_column(Float, nullable=False)
    pincode_effect: Mapped[float] = mapped_column(Float, nullable=False)
    label_flipped: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    dataset_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_runs.run_id", ondelete="CASCADE"), index=True
    )

    __table_args__ = (
        CheckConstraint(
            "true_rto_probability >= 0 AND true_rto_probability <= 1",
            name="true_probability_is_a_probability",
        ),
    )


class Decision(Base, TimestampMixin):
    """Append-only decision log. Retained for audit.

    ``engine_version`` and ``config_fingerprint`` make a logged decision
    replayable: given the same score and cost inputs, the engine that produced this
    row can be re-run and the result compared. That is what "auditable" means in
    practice, and it is only possible because the engine is deterministic and no
    LLM touches it.
    """

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_pk: Mapped[int] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True
    )
    order_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    band: Mapped[RiskBand] = mapped_column(String(16), nullable=False, index=True)
    action: Mapped[InterventionAction] = mapped_column(String(32), nullable=False)
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)

    reason_codes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    expected_value_inr: Mapped[float] = mapped_column(Float, nullable=False)

    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    engine_version: Mapped[str] = mapped_column(String(32), nullable=False)
    config_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)

    appeal_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    human_review_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_control_holdout: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    decided_at: Mapped[datetime] = mapped_column(nullable=False, index=True)

    order: Mapped[Order] = relationship("Order", back_populates="decisions")
    overrides: Mapped[list[OpsOverrideRecord]] = relationship(
        "OpsOverrideRecord", back_populates="decision"
    )

    __table_args__ = (
        CheckConstraint(
            "probability >= 0 AND probability <= 1", name="probability_is_a_probability"
        ),
        CheckConstraint("threshold >= 0 AND threshold <= 1", name="threshold_in_range"),
        # The appeal path is a database-level guarantee, not only a Python one.
        # A bare column reference is the portable way to assert a boolean is true.
        CheckConstraint("appeal_available", name="appeal_always_available"),
        Index("ix_decisions_band_decided_at", "band", "decided_at"),
    )


class OpsOverrideRecord(Base, TimestampMixin):
    """A human changing the engine's recommendation. Always logged.

    SPEC section 02, step 5: overrides are logged as counterfactual evidence.
    ``operator_id`` is a hashed identity - the audit trail needs to distinguish
    operators consistently, not to identify them by name.
    """

    __tablename__ = "ops_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decision_pk: Mapped[int] = mapped_column(
        ForeignKey("decisions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    order_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    original_band: Mapped[RiskBand] = mapped_column(String(16), nullable=False)
    override_band: Mapped[RiskBand] = mapped_column(String(16), nullable=False)
    direction: Mapped[OverrideDirection] = mapped_column(String(16), nullable=False, index=True)

    operator_id: Mapped[str] = mapped_column(String(64), nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")

    decision: Mapped[Decision] = relationship("Decision", back_populates="overrides")

    __table_args__ = (
        CheckConstraint("original_band <> override_band", name="override_changes_band"),
    )


class ModelRun(Base, TimestampMixin):
    """Provenance for one training or evaluation run.

    ``test_set_scored`` exists so the "scored exactly once" seal is queryable
    rather than only present as a file on someone's laptop.
    """

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    rung_id: Mapped[int] = mapped_column(Integer, nullable=False)

    config_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_runs.run_id", ondelete="SET NULL"), index=True
    )

    n_train: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_validation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_test: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    test_set_scored: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    __table_args__ = (UniqueConstraint("model_name", "model_version", name="model_version_unique"),)
