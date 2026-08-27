"""Database tables.

FIVE TABLES, AND THE REASON EACH EXISTS
=======================================

``orders``
    What was presented for scoring. Order metadata only. There is no name
    column, no phone column and no email column - not nullable ones, none at
    all - because a column that does not exist cannot be populated by a
    well-meaning integration later.

``order_outcomes``
    The terminal delivery state, arriving 5-9 days after the order. Separate
    from ``orders`` because it is *the label*, and keeping it in its own table
    with its own timestamp is what makes an as-of join natural and a leaky join
    unnatural.

``decisions``
    Append-only log of every decision the engine made, with the probability, the
    threshold, the band, the reason codes and the engine version. Retained for
    audit (SPEC section 09). Nothing updates a row here; a changed decision is a
    new row.

``ops_overrides``
    Every human override, with direction and operator. These are counterfactual
    evidence: an ops associate relaxing a SEVERE band is telling the system
    something the model did not know, and that signal is worth more than it
    looks.

``model_runs``
    Provenance for each trained artefact and evaluation run - config
    fingerprint, seed, split sizes. This is what makes a number in REPORT.md
    traceable back to the state that produced it.

PRIVACY NOTE
------------
``customer_hash`` is a salted digest supplied by the caller. This database never
receives, stores, or is capable of storing the pre-image. ``address_line`` is
stored because address quality is auditable and an ops associate reviewing a
SEVERE decision needs to see what the customer actually typed - it is inside the
merchant boundary, and it never leaves it through an API response or an LLM
prompt.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
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
    InterventionAction,
    OrderOutcome,
    OverrideDirection,
    PaymentMethod,
    PincodeTier,
    RiskBand,
)
from rto_sentinel.db.base import Base, TimestampMixin


class Order(Base, TimestampMixin):
    """An order presented for scoring."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    merchant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    customer_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    ordered_at: Mapped[datetime] = mapped_column(nullable=False, index=True)

    payment_method: Mapped[PaymentMethod] = mapped_column(String(16), nullable=False)
    order_value_inr: Mapped[float] = mapped_column(Float, nullable=False)
    discount_inr: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    item_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    category: Mapped[str | None] = mapped_column(String(64))

    address_line: Mapped[str] = mapped_column(Text, nullable=False)
    address_city: Mapped[str] = mapped_column(String(128), nullable=False)
    address_state: Mapped[str] = mapped_column(String(128), nullable=False)
    pincode: Mapped[str] = mapped_column(String(6), nullable=False, index=True)
    pincode_tier: Mapped[PincodeTier] = mapped_column(String(16), default=PincodeTier.UNKNOWN)

    courier_partner: Mapped[str | None] = mapped_column(String(64))

    outcome: Mapped[OrderOutcomeRecord | None] = relationship(
        "OrderOutcomeRecord", back_populates="order", uselist=False
    )
    decisions: Mapped[list[Decision]] = relationship("Decision", back_populates="order")

    __table_args__ = (
        CheckConstraint("order_value_inr > 0", name="order_value_positive"),
        CheckConstraint("discount_inr >= 0", name="discount_non_negative"),
        Index("ix_orders_merchant_ordered_at", "merchant_id", "ordered_at"),
    )


class OrderOutcomeRecord(Base, TimestampMixin):
    """The terminal delivery state. This is the label.

    Kept in its own table with its own ``resolved_at`` so that "what was known at
    order time" is expressible in SQL. An as-of join reads
    ``resolved_at < ordered_at``; there is no convenient query shape that
    accidentally reads the future.
    """

    __tablename__ = "order_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_pk: Mapped[int] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    outcome: Mapped[OrderOutcome] = mapped_column(String(16), nullable=False, index=True)
    resolved_at: Mapped[datetime] = mapped_column(nullable=False, index=True)

    order: Mapped[Order] = relationship("Order", back_populates="outcome")


class Decision(Base, TimestampMixin):
    """Append-only decision log. Retained for audit.

    ``engine_version`` and ``config_fingerprint`` make a logged decision
    replayable: given the same score and cost inputs, the engine that produced
    this row can be re-run and the result compared. That is what "auditable"
    means in practice, and it is only possible because the engine is
    deterministic and no LLM touches it.
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
        CheckConstraint(
            "appeal_available = 1 OR appeal_available = true",
            name="appeal_always_available",
        ),
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

    n_train: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_validation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_test: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    test_set_scored: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    __table_args__ = (UniqueConstraint("model_name", "model_version", name="model_version_unique"),)
