"""Session and intent features.

SPEC section 04: "Weak individually. Included because low-intent late-night
impulse purchases are a documented RTO driver."

That sentence is the honest framing and it is repeated here because it sets the
expectation this family should be held to. Every feature here comes straight off
the order payload, so there is no as-of question; the interesting question is
whether they earn their place at all. The leave-one-family-out ablation has now
answered it: removing `session_intent` costs INR 234 per 1,000 orders on
validation with an interval of [-1,063, +575], which spans zero. Its
contribution is **not established** - and notably, removing it slightly
*improved* PR-AUC (+0.009) while costing money, which is precisely why this
project ranks families on rupees rather than on AUC.

A NOTE ON `hour_of_day`
=======================
Encoded as a plain integer rather than cyclically (sin/cos). Gradient-boosted
trees split on ordered values and handle the 23-to-0 wrap by making two splits,
so a cyclic encoding buys nothing and costs interpretability - "hour >= 23" reads
directly in a SHAP explanation, where "hour_sin <= -0.97" does not. This matters
because these values end up in reason codes an ops associate reads.
"""

from __future__ import annotations

import pandas as pd

from rto_sentinel.data import schema as cols
from rto_sentinel.features.base import FeatureFamily
from rto_sentinel.features.spec import (
    Availability,
    FeatureSet,
    FeatureSpec,
    ObservationPoint,
)

FAMILY = "session_intent"


def _payload_spec(
    name: str, dtype: str, description: str, source: tuple[str, ...], risk_note: str
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
    )


class SessionIntentFamily(FeatureFamily):
    """How the order was placed, and when."""

    name = FAMILY

    @property
    def feature_set(self) -> FeatureSet:
        return FeatureSet(
            (
                _payload_spec(
                    "session_product_page_seconds",
                    "float",
                    "Seconds spent on the product page before adding to cart.",
                    (cols.PRODUCT_PAGE_SECONDS,),
                    "Weak. A considered purchase and a distracted browser look the same. "
                    "Low fairness risk - it describes a session, not a person.",
                ),
                _payload_spec(
                    "session_sessions_before_purchase",
                    "int",
                    "Distinct sessions before this order was placed.",
                    (cols.SESSIONS_BEFORE_PURCHASE,),
                    "Weak. More sessions usually means more deliberation, but it also means "
                    "a slower connection or a shared device.",
                ),
                _payload_spec(
                    "session_time_to_checkout_seconds",
                    "float",
                    "Seconds from cart creation to order placement.",
                    (cols.TIME_TO_CHECKOUT_SECONDS,),
                    "The clearest impulse proxy in this family. Still weak on its own.",
                ),
                _payload_spec(
                    "session_device_class",
                    "category",
                    "Device the order was placed from.",
                    (cols.DEVICE_CLASS,),
                    "WATCH THIS ONE. Device class correlates with income in India, so it is "
                    "a plausible proxy for something this project has no business predicting "
                    "on. Kept for now because it is a documented delivery-context signal, and "
                    "flagged as a removal candidate if the fairness audit trips.",
                ),
                _payload_spec(
                    "session_hour_of_day",
                    "int",
                    "Hour the order was placed, 0-23, UTC.",
                    (cols.HOUR_OF_DAY,),
                    "Low fairness risk. Encoded as an integer rather than cyclically so it "
                    "reads directly in a reason code - see the module docstring.",
                ),
                _payload_spec(
                    "session_day_of_week",
                    "int",
                    "Day of week, Monday=0.",
                    (cols.DAY_OF_WEEK,),
                    "Low risk. Weekend orders deliver differently from weekday ones.",
                ),
                _payload_spec(
                    "session_is_late_night",
                    "bool",
                    "True when the order was placed between 23:00 and 04:59.",
                    (cols.IS_LATE_NIGHT,),
                    "The documented low-intent impulse window. A blunt encoding of "
                    "hour_of_day, kept because it is what the reason code will say.",
                ),
                _payload_spec(
                    "session_is_sale_day",
                    "bool",
                    "True when the order fell on a promotional day.",
                    (cols.IS_SALE_DAY,),
                    "A merchant-controlled variable, like discount depth. Surfaced as a "
                    "merchant insight, not only as a customer-level signal.",
                ),
                _payload_spec(
                    "session_cod_after_prepaid_failure",
                    "bool",
                    "True when COD was chosen after a prepaid attempt failed.",
                    (cols.COD_AFTER_PREPAID_FAILURE,),
                    "Known at checkout - the failure happened seconds earlier in the same "
                    "session, so this is payload, not history. Rare, which limits how much "
                    "the model can learn from it.",
                ),
            )
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)

        out["session_product_page_seconds"] = frame[cols.PRODUCT_PAGE_SECONDS].astype("float64")
        out["session_sessions_before_purchase"] = frame[cols.SESSIONS_BEFORE_PURCHASE].astype(
            "int64"
        )
        out["session_time_to_checkout_seconds"] = frame[cols.TIME_TO_CHECKOUT_SECONDS].astype(
            "float64"
        )
        out["session_device_class"] = frame[cols.DEVICE_CLASS].astype("category")
        out["session_hour_of_day"] = frame[cols.HOUR_OF_DAY].astype("int64")
        out["session_day_of_week"] = frame[cols.DAY_OF_WEEK].astype("int64")
        out["session_is_late_night"] = frame[cols.IS_LATE_NIGHT].astype(bool)
        out["session_is_sale_day"] = frame[cols.IS_SALE_DAY].astype(bool)
        out["session_cod_after_prepaid_failure"] = frame[cols.COD_AFTER_PREPAID_FAILURE].astype(
            bool
        )

        return out[list(self.feature_set.names)]
