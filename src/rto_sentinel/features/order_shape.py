"""Order-shape features: value, basket, discount, payment mode.

SPEC section 04. The simplest family in the project - every value here comes
straight off the order payload, so there is no as-of question to get wrong. That
makes it the family where the *fairness* notes matter more than the leakage ones.

DEEP DISCOUNTS ARE PARTLY THE MERCHANT'S OWN DOING
==================================================
Discount depth is genuinely predictive: a 60%-off impulse purchase returns more
often than a full-price considered one. But the merchant set that discount. A
model that learns "deep discount means risky" and then frictions the customer has
charged the customer for the merchant's promotion strategy.

The feature stays, because it predicts. What changes is the reporting: the
evaluation surfaces discount depth as a *merchant insight* - "your 60%-off
campaign has a 34% RTO rate" - alongside its use as a customer-level signal. That
is a presentation decision made here, at the point the feature is defined, rather
than left to whoever writes the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rto_sentinel.data import schema as cols
from rto_sentinel.features.base import FeatureFamily
from rto_sentinel.features.spec import (
    Availability,
    FeatureSet,
    FeatureSpec,
    ObservationPoint,
)

FAMILY = "order_shape"


def _payload_spec(
    name: str,
    dtype: str,
    description: str,
    source: tuple[str, ...],
    risk_note: str,
    monotonic: str | None = None,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        family=FAMILY,
        dtype=dtype,  # type: ignore[arg-type]
        description=description,
        source_columns=source,
        observation_point=ObservationPoint.ORDER_PAYLOAD,
        availability=Availability.AT_ORDER_TIME,
        risk_note=risk_note,
        monotonic=monotonic,  # type: ignore[arg-type]
    )


class OrderShapeFamily(FeatureFamily):
    """What was bought, for how much, and how it was paid for."""

    name = FAMILY

    @property
    def feature_set(self) -> FeatureSet:
        return FeatureSet(
            (
                _payload_spec(
                    "order_value_inr",
                    "float",
                    "Net order value in rupees, after discount.",
                    (cols.ORDER_VALUE_INR,),
                    "Low leakage risk. Reported by order-value quartile in the fairness "
                    "audit, because friction on a small order costs a customer "
                    "proportionally more than on a large one.",
                ),
                _payload_spec(
                    "order_log_value",
                    "float",
                    "Natural log of (1 + order value). Compresses a long right tail.",
                    (cols.ORDER_VALUE_INR,),
                    "Low risk. A monotone transform of order value; both are kept because "
                    "trees split on raw scale while the log is easier to read in SHAP.",
                ),
                _payload_spec(
                    "order_is_cod",
                    "bool",
                    "True when the order is cash on delivery.",
                    (cols.IS_COD,),
                    "The single strongest split in the problem - 26% against under 2%. Not "
                    "a leak: payment method is chosen at checkout, before scoring.",
                ),
                _payload_spec(
                    "order_discount_depth",
                    "float",
                    "Discount as a fraction of gross order value.",
                    (cols.DISCOUNT_DEPTH,),
                    "Predictive, and partly the merchant's own doing. Surfaced as a merchant "
                    "insight in the evaluation, not only as a customer-level penalty. See "
                    "the module docstring.",
                    monotonic="increasing",
                ),
                _payload_spec(
                    "order_discount_inr",
                    "float",
                    "Absolute discount in rupees.",
                    (cols.DISCOUNT_INR,),
                    "Low risk. Kept alongside depth because a large absolute discount on an "
                    "expensive item is a different situation from a deep one on a cheap item.",
                ),
                _payload_spec(
                    "order_item_count",
                    "int",
                    "Total units in the basket.",
                    (cols.ITEM_COUNT,),
                    "Low risk. Multi-item baskets fail delivery differently from single ones.",
                ),
                _payload_spec(
                    "order_value_per_item",
                    "float",
                    "Net order value divided by unit count.",
                    (cols.ORDER_VALUE_INR, cols.ITEM_COUNT),
                    "Low risk. Separates 'one expensive thing' from 'many cheap things', "
                    "which behave differently at the doorstep.",
                ),
                _payload_spec(
                    "order_category",
                    "category",
                    "Product category of the order.",
                    (cols.CATEGORY,),
                    "Fashion returns more than electronics, which is a property of the goods "
                    "rather than of the customer. Low fairness risk, and useful for merchant "
                    "reporting.",
                ),
                _payload_spec(
                    "order_cart_edited",
                    "bool",
                    "True when the basket was modified before checkout.",
                    (cols.CART_EDITED,),
                    "Weak signal. Included because hesitation before purchase is a plausible "
                    "intent proxy; expected to earn little and be a candidate for removal in "
                    "the ablation study.",
                ),
            )
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        value = frame[cols.ORDER_VALUE_INR].astype("float64")
        items = frame[cols.ITEM_COUNT].astype("float64")

        out["order_value_inr"] = value
        out["order_log_value"] = np.log1p(value)
        out["order_is_cod"] = frame[cols.IS_COD].astype(bool)
        out["order_discount_depth"] = frame[cols.DISCOUNT_DEPTH].astype("float64")
        out["order_discount_inr"] = frame[cols.DISCOUNT_INR].astype("float64")
        out["order_item_count"] = frame[cols.ITEM_COUNT].astype("int64")
        with np.errstate(invalid="ignore", divide="ignore"):
            out["order_value_per_item"] = np.where(items > 0, value / items, np.nan)
        out["order_category"] = frame[cols.CATEGORY].astype("category")
        out["order_cart_edited"] = frame[cols.CART_EDITED].astype(bool)

        return out[list(self.feature_set.names)]
