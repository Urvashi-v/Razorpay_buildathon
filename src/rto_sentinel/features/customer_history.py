"""Customer-history features, computed strictly as-of order time.

SPEC section 04. This is the family with the highest leakage risk in the project,
because the strongest honest signal - "this customer has returned things before" -
is one careless groupby away from becoming "this customer will return this".

THE DISTINCTION THAT MATTERS
============================
Two clocks, and every feature here declares which one it runs on.

**Placed.** ``prior_order_count`` counts orders the customer placed before this
one. The merchant watched those happen; the count is knowable instantly.

**Resolved.** ``prior_rto_count`` counts orders that came *back* before this one
was placed. An order placed on day 40 that returns on day 47 contributes nothing
to an order placed on day 42 - on day 42 nobody knew.

Both are legitimate features. Computing the second on the first clock is the leak.

WHY MISSING HISTORY IS NaN, NOT ZERO
====================================
A first-time customer has no return rate. Encoding that as 0.0 claims a zero
percent return rate, which is the most optimistic possible reading of "we know
nothing about them" - and it is exactly the reading that makes a model confident
about the cohort it understands least. LightGBM handles NaN natively and can
learn "no history" as its own state, which is what we want it to learn.

Counts are different: "zero prior orders" is a fact, so ``prior_order_count`` is
0 rather than NaN. The rule is stated once in ``data.asof``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rto_sentinel.data import schema as cols
from rto_sentinel.data.asof import as_of_aggregate
from rto_sentinel.features.base import FeatureFamily
from rto_sentinel.features.spec import (
    ALL_HISTORY_PLACED,
    ALL_HISTORY_RESOLVED,
    Availability,
    FeatureSet,
    FeatureSpec,
    ObservationPoint,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

FAMILY = "customer_history"

#: Beta prior for the smoothed personal return rate. Two pseudo-observations at
#: the population base rate, so a customer with one return out of one order does
#: not read as a 100% returner. Deliberately weak - real history should dominate
#: quickly - but non-zero, because a single order is not evidence.
PRIOR_STRENGTH = 2.0
PRIOR_RATE = 0.20


class CustomerHistoryFamily(FeatureFamily):
    """Everything the merchant knows about this customer, as of this instant."""

    name = FAMILY

    @property
    def feature_set(self) -> FeatureSet:
        return FeatureSet(
            (
                FeatureSpec(
                    name="cust_account_age_days",
                    family=FAMILY,
                    dtype="float",
                    description="Days between the customer's signup and this order.",
                    source_columns=("signup_at", cols.ORDERED_AT),
                    observation_point=ObservationPoint.CUSTOMER_RECORD,
                    availability=Availability.AT_ORDER_TIME,
                    risk_note=(
                        "Low risk. Account age is a legitimate tenure signal and is known "
                        "at checkout. Correlated with order count, so the two are not "
                        "independent evidence."
                    ),
                ),
                FeatureSpec(
                    name="cust_prior_order_count",
                    family=FAMILY,
                    dtype="int",
                    description="Orders this customer placed before this one.",
                    source_columns=(cols.CUSTOMER_HASH, cols.ORDERED_AT),
                    observation_point=ObservationPoint.PRIOR_ORDERS_PLACED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_PLACED,
                    risk_note=(
                        "Placed-clock, so knowable instantly. Zero for a first-time "
                        "customer - a fact, not a missing value."
                    ),
                ),
                FeatureSpec(
                    name="cust_prior_cod_count",
                    family=FAMILY,
                    dtype="int",
                    description="Prior orders paid cash on delivery.",
                    source_columns=(cols.CUSTOMER_HASH, cols.ORDERED_AT, cols.IS_COD),
                    observation_point=ObservationPoint.PRIOR_ORDERS_PLACED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_PLACED,
                    risk_note="Payment method is chosen at checkout, so this is placed-clock.",
                ),
                FeatureSpec(
                    name="cust_prepaid_share",
                    family=FAMILY,
                    dtype="float",
                    description="Share of prior orders that were prepaid.",
                    source_columns=(cols.CUSTOMER_HASH, cols.ORDERED_AT, cols.IS_COD),
                    observation_point=ObservationPoint.PRIOR_ORDERS_PLACED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_PLACED,
                    expected_null_share=0.3,
                    risk_note=(
                        "A customer who usually prepays and suddenly picks COD is a "
                        "documented pattern. NaN for first-time customers."
                    ),
                ),
                FeatureSpec(
                    name="cust_prior_resolved_count",
                    family=FAMILY,
                    dtype="int",
                    description="Prior orders whose delivery outcome was known by this instant.",
                    source_columns=(cols.CUSTOMER_HASH, cols.ORDERED_AT, cols.RESOLVED_AT),
                    observation_point=ObservationPoint.PRIOR_ORDERS_RESOLVED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_RESOLVED,
                    risk_note=(
                        "The denominator behind every rate below. Emitted so the model can "
                        "tell 'zero returns from one order' apart from 'zero returns from "
                        "twenty' - those are very different pieces of evidence."
                    ),
                ),
                FeatureSpec(
                    name="cust_prior_rto_count",
                    family=FAMILY,
                    dtype="int",
                    description="Prior orders that had already returned by this instant.",
                    source_columns=(
                        cols.CUSTOMER_HASH,
                        cols.ORDERED_AT,
                        cols.RESOLVED_AT,
                        cols.IS_RTO,
                    ),
                    observation_point=ObservationPoint.PRIOR_ORDERS_RESOLVED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_RESOLVED,
                    monotonic="increasing",
                    risk_note=(
                        "RESOLVED-clock. An order placed earlier that has not come back yet "
                        "contributes nothing - that is the whole point of the as-of join."
                    ),
                ),
                FeatureSpec(
                    name="cust_prior_rto_rate",
                    family=FAMILY,
                    dtype="float",
                    description="Raw share of resolved prior orders that returned.",
                    source_columns=(
                        cols.CUSTOMER_HASH,
                        cols.ORDERED_AT,
                        cols.RESOLVED_AT,
                        cols.IS_RTO,
                    ),
                    observation_point=ObservationPoint.PRIOR_ORDERS_RESOLVED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_RESOLVED,
                    monotonic="increasing",
                    expected_null_share=0.4,
                    risk_note=(
                        "Strongest honest signal in the problem. NaN when nothing has "
                        "resolved yet - never 0.0, which would claim a clean record the "
                        "merchant has no basis for."
                    ),
                ),
                FeatureSpec(
                    name="cust_prior_rto_rate_smoothed",
                    family=FAMILY,
                    dtype="float",
                    description=(
                        "Return rate shrunk toward the population base rate with a "
                        f"Beta({PRIOR_STRENGTH * PRIOR_RATE:.1f}, "
                        f"{PRIOR_STRENGTH * (1 - PRIOR_RATE):.1f}) prior."
                    ),
                    source_columns=(
                        cols.CUSTOMER_HASH,
                        cols.ORDERED_AT,
                        cols.RESOLVED_AT,
                        cols.IS_RTO,
                    ),
                    observation_point=ObservationPoint.PRIOR_ORDERS_RESOLVED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_RESOLVED,
                    monotonic="increasing",
                    risk_note=(
                        "One return out of one order is not a 100% returner. Shrinkage stops "
                        "the model treating a single unlucky delivery as a verdict on a "
                        "person - which matters because that verdict costs them friction."
                    ),
                ),
                FeatureSpec(
                    name="cust_days_since_last_order",
                    family=FAMILY,
                    dtype="float",
                    description="Days since this customer last placed an order.",
                    source_columns=(cols.CUSTOMER_HASH, cols.ORDERED_AT),
                    observation_point=ObservationPoint.PRIOR_ORDERS_PLACED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_PLACED,
                    expected_null_share=0.3,
                    risk_note=(
                        "Placed-clock: the merchant knows when someone last ordered even if "
                        "that order has not been delivered yet. NaN for first-timers."
                    ),
                ),
                FeatureSpec(
                    name="cust_prior_value_mean",
                    family=FAMILY,
                    dtype="float",
                    description="Mean value of this customer's prior orders.",
                    source_columns=(cols.CUSTOMER_HASH, cols.ORDERED_AT, cols.ORDER_VALUE_INR),
                    observation_point=ObservationPoint.PRIOR_ORDERS_PLACED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_PLACED,
                    expected_null_share=0.3,
                    risk_note="Order value is known at checkout, so placed-clock is correct.",
                ),
                FeatureSpec(
                    name="cust_value_vs_prior_mean",
                    family=FAMILY,
                    dtype="float",
                    description="This order's value divided by the customer's prior mean.",
                    source_columns=(cols.CUSTOMER_HASH, cols.ORDERED_AT, cols.ORDER_VALUE_INR),
                    observation_point=ObservationPoint.PRIOR_ORDERS_PLACED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_PLACED,
                    expected_null_share=0.3,
                    risk_note=(
                        "An order far above someone's usual basket is a documented risk "
                        "pattern. Ratio rather than z-score: most customers have too few "
                        "orders for a standard deviation to mean anything."
                    ),
                ),
                FeatureSpec(
                    name="cust_mean_resolution_days",
                    family=FAMILY,
                    dtype="float",
                    description="Mean days from order to outcome across resolved prior orders.",
                    source_columns=(
                        cols.CUSTOMER_HASH,
                        cols.ORDERED_AT,
                        cols.RESOLVED_AT,
                        cols.MATURITY_DAYS,
                    ),
                    observation_point=ObservationPoint.PRIOR_ORDERS_RESOLVED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_RESOLVED,
                    expected_null_share=0.4,
                    risk_note=(
                        "SUBTLE. Resolution time is outcome-correlated - returns take longer "
                        "than deliveries - so this is a partial proxy for past outcomes. That "
                        "is legitimate for PRIOR orders whose outcome is genuinely known, and "
                        "would be a severe leak for the current one. Resolved-clock, "
                        "strictly."
                    ),
                ),
                FeatureSpec(
                    name="cust_is_new",
                    family=FAMILY,
                    dtype="bool",
                    description="True when this is the customer's first order.",
                    source_columns=(cols.CUSTOMER_HASH, cols.ORDERED_AT),
                    observation_point=ObservationPoint.PRIOR_ORDERS_PLACED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=ALL_HISTORY_PLACED,
                    risk_note=(
                        "The cold-start cohort. Reported separately in every evaluation, "
                        "because a model that simply learns 'new equals risky' has found a "
                        "population, not a behaviour."
                    ),
                ),
            )
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        working = frame.copy()
        working["_is_rto_numeric"] = working[cols.IS_RTO].astype("float64")
        working["_is_cod_numeric"] = working[cols.IS_COD].astype("float64")

        # A synthetic "resolution" column on the PLACED clock, so the same as-of
        # machinery serves both clocks. Setting resolution = order time means "a
        # prior order counts from the moment it was placed", which is exactly what
        # the placed clock means.
        working["_placed_clock"] = working[cols.ORDERED_AT]

        def placed(value_column: str | None, aggregate: str) -> pd.Series:
            return as_of_aggregate(
                working,
                group_key=cols.CUSTOMER_HASH,
                value_column=value_column,
                order_time_column=cols.ORDERED_AT,
                resolution_time_column="_placed_clock",
                aggregate=aggregate,  # type: ignore[arg-type]
            )

        def resolved(value_column: str | None, aggregate: str) -> pd.Series:
            return as_of_aggregate(
                working,
                group_key=cols.CUSTOMER_HASH,
                value_column=value_column,
                order_time_column=cols.ORDERED_AT,
                resolution_time_column=cols.RESOLVED_AT,
                aggregate=aggregate,  # type: ignore[arg-type]
            )

        # --- placed clock ----------------------------------------------------
        prior_orders = placed(None, "count")
        out["cust_prior_order_count"] = prior_orders.astype("int64")
        out["cust_is_new"] = prior_orders == 0

        prior_cod = placed("_is_cod_numeric", "sum")
        out["cust_prior_cod_count"] = prior_cod.fillna(0).astype("int64")
        with np.errstate(invalid="ignore", divide="ignore"):
            out["cust_prepaid_share"] = np.where(
                prior_orders > 0, 1.0 - (prior_cod / prior_orders), np.nan
            )

        prior_value_mean = placed(cols.ORDER_VALUE_INR, "mean")
        out["cust_prior_value_mean"] = prior_value_mean
        with np.errstate(invalid="ignore", divide="ignore"):
            out["cust_value_vs_prior_mean"] = np.where(
                prior_value_mean > 0,
                frame[cols.ORDER_VALUE_INR] / prior_value_mean,
                np.nan,
            )

        out["cust_days_since_last_order"] = self._days_since_last_order(frame)

        # --- customer record --------------------------------------------------
        if "signup_at" in frame.columns:
            age = (frame[cols.ORDERED_AT] - frame["signup_at"]).dt.total_seconds() / 86400.0
            # Clipped at zero: a negative account age means the join brought in a
            # signup timestamp from the future, which is a data bug, not a feature.
            out["cust_account_age_days"] = age.clip(lower=0.0)
        else:
            out["cust_account_age_days"] = np.nan

        # --- resolved clock ---------------------------------------------------
        resolved_count = resolved(None, "count")
        rto_count = resolved("_is_rto_numeric", "sum")
        out["cust_prior_resolved_count"] = resolved_count.astype("int64")
        out["cust_prior_rto_count"] = rto_count.fillna(0).astype("int64")
        out["cust_prior_rto_rate"] = resolved("_is_rto_numeric", "mean")
        out["cust_mean_resolution_days"] = resolved(cols.MATURITY_DAYS, "mean")

        # Beta-shrunk personal rate. Defined for everyone, including first-timers,
        # where it collapses to the population prior - which is the honest answer
        # for someone we know nothing about.
        out["cust_prior_rto_rate_smoothed"] = (
            rto_count.fillna(0.0) + PRIOR_STRENGTH * PRIOR_RATE
        ) / (resolved_count.fillna(0.0) + PRIOR_STRENGTH)

        return out[list(self.feature_set.names)]

    @staticmethod
    def _days_since_last_order(frame: pd.DataFrame) -> pd.Series:
        """Days since the customer's previous order, on the placed clock.

        A shift within customer, on a frame sorted by time. Uses ``ordered_at``
        only, so it never touches an outcome.
        """
        ordered = frame[[cols.CUSTOMER_HASH, cols.ORDERED_AT]].sort_values(
            [cols.CUSTOMER_HASH, cols.ORDERED_AT], kind="stable"
        )
        previous = ordered.groupby(cols.CUSTOMER_HASH, sort=False)[cols.ORDERED_AT].shift(1)
        gap = (ordered[cols.ORDERED_AT] - previous).dt.total_seconds() / 86400.0
        return gap.reindex(frame.index)
