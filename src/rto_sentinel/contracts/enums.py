"""Enumerations shared across the API, the decision engine and the database.

Defined once, here, so that the string ``"SEVERE"`` means exactly the same thing
in a JSON response, in a policy config and in a Postgres column.
"""

from __future__ import annotations

from enum import StrEnum


class PaymentMethod(StrEnum):
    """How the order is being paid for. The entire problem lives in COD."""

    COD = "cod"
    PREPAID = "prepaid"


class DeviceClass(StrEnum):
    MOBILE_WEB = "mobile_web"
    MOBILE_APP = "mobile_app"
    DESKTOP = "desktop"
    UNKNOWN = "unknown"


class PincodeTier(StrEnum):
    """Tier is a fairness-audit dimension, never a standalone risk feature."""

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    UNKNOWN = "unknown"


class RiskBand(StrEnum):
    """The four rungs of the friction ladder. Ordered; see :func:`band_rank`."""

    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


BAND_ORDER: tuple[RiskBand, ...] = (
    RiskBand.LOW,
    RiskBand.ELEVATED,
    RiskBand.HIGH,
    RiskBand.SEVERE,
)


def band_rank(band: RiskBand) -> int:
    """Position of a band on the ladder, 0 (LOW) to 3 (SEVERE)."""
    return BAND_ORDER.index(band)


class InterventionAction(StrEnum):
    """What the merchant actually does. No value here is a silent hard block."""

    NONE = "none"
    PREPAID_NUDGE = "prepaid_nudge"
    CONFIRMATION_REQUIRED = "confirmation_required"
    PREPAID_ONLY = "prepaid_only"


class OrderOutcome(StrEnum):
    """Terminal delivery state. Only a resolved order carries a usable label."""

    PENDING = "pending"
    DELIVERED = "delivered"
    RTO = "rto"
    CANCELLED = "cancelled"


class DatasetSplit(StrEnum):
    """Which part of the experiment a row belongs to.

    The two exclusion values are distinct on purpose. ``EXCLUDED_IMMATURE`` means
    the outcome is not yet known; ``EXCLUDED_GROUP_PROTOCOL`` means the outcome is
    known but the row was dropped to keep customers disjoint across splits. They
    have different causes and different remedies, and collapsing them would hide
    how much data each rule actually costs.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    EXCLUDED_IMMATURE = "excluded_immature"
    EXCLUDED_GROUP_PROTOCOL = "excluded_group_protocol"


class OverrideDirection(StrEnum):
    """Direction of a human override, logged as counterfactual evidence."""

    RELAXED = "relaxed"  # ops reduced the friction the engine recommended
    ESCALATED = "escalated"  # ops increased it
