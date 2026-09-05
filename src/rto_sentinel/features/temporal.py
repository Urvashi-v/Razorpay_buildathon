"""Temporal features: bounded lookback windows and recent behaviour change.

Why this family exists separately from ``customer_history``: an all-time
aggregate and a 30-day aggregate answer different questions. "Has this customer
ever returned anything" is a character judgement that follows someone forever;
"has this customer returned anything recently" decays, which is both fairer and
usually more predictive. A customer who had a bad month two years ago should not
be paying for it at every checkout.

THE TWO CLOCKS, AGAIN
=====================
Order-frequency windows run on the **placed** clock - the merchant watched those
orders happen and knows about them instantly.

Return windows run on the **resolved** clock. An order placed 5 days ago that has
not come back yet contributes nothing to a 30-day return count, because on this
date nobody knows how it ended.

Both windows are half-open on the left and strictly bounded on the right at the
current order's timestamp: ``(t - window, t)``. The strict right bound is the
as-of rule; the left bound is what makes it a window rather than all history.

THE BEHAVIOUR-CHANGE FEATURE
============================
``temporal_order_burst`` compares recent order frequency to the customer's own
longer-run baseline. A sudden burst of orders is a documented risk pattern - and
it is the one feature here whose fairness note is not "low risk", because a
legitimate seasonal shopper and an impulsive one look identical over a short
window. It earns its place only if an ablation shows it pays. The ablation has been
run and `temporal` was not among the families it ablated - the study covers the
five in `config/evaluation.yaml` - so this family's contribution remains
unmeasured.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from rto_sentinel.data import schema as cols
from rto_sentinel.features.base import FeatureFamily
from rto_sentinel.features.spec import (
    Availability,
    FeatureSet,
    FeatureSpec,
    LookbackWindow,
    ObservationPoint,
)

FAMILY = "temporal"

#: Windows in days. 7 catches a burst, 30 a month's behaviour, 90 a season.
PLACED_WINDOWS = (7, 30, 90)
RESOLVED_WINDOWS = (30, 90)


class TemporalFamily(FeatureFamily):
    """Bounded-window behaviour, on whichever clock each window belongs to."""

    name = FAMILY

    @property
    def feature_set(self) -> FeatureSet:
        specs: list[FeatureSpec] = []

        for days in PLACED_WINDOWS:
            specs.append(
                FeatureSpec(
                    name=f"temporal_orders_last_{days}d",
                    family=FAMILY,
                    dtype="int",
                    description=f"Orders this customer placed in the {days} days before this one.",
                    source_columns=(cols.CUSTOMER_HASH, cols.ORDERED_AT),
                    observation_point=ObservationPoint.PRIOR_ORDERS_PLACED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=LookbackWindow(days=days, clock="placed"),
                    risk_note=(
                        "Placed-clock, so knowable instantly. Zero is a fact here, not a "
                        "missing value: the customer genuinely placed no orders."
                    ),
                )
            )

        for days in RESOLVED_WINDOWS:
            specs.append(
                FeatureSpec(
                    name=f"temporal_rto_count_last_{days}d",
                    family=FAMILY,
                    dtype="int",
                    description=(
                        f"Orders that returned in the {days} days before this one was placed."
                    ),
                    source_columns=(
                        cols.CUSTOMER_HASH,
                        cols.ORDERED_AT,
                        cols.RESOLVED_AT,
                        cols.IS_RTO,
                    ),
                    observation_point=ObservationPoint.PRIOR_ORDERS_RESOLVED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=LookbackWindow(days=days, clock="resolved"),
                    monotonic="increasing",
                    risk_note=(
                        "RESOLVED-clock. An order placed inside the window that has not come "
                        "back yet contributes nothing, because on this date nobody knows how "
                        "it ended."
                    ),
                )
            )
            specs.append(
                FeatureSpec(
                    name=f"temporal_rto_rate_last_{days}d",
                    family=FAMILY,
                    dtype="float",
                    description=f"Share of orders resolving in the last {days} days that returned.",
                    source_columns=(
                        cols.CUSTOMER_HASH,
                        cols.ORDERED_AT,
                        cols.RESOLVED_AT,
                        cols.IS_RTO,
                    ),
                    observation_point=ObservationPoint.PRIOR_ORDERS_RESOLVED,
                    availability=Availability.AT_ORDER_TIME,
                    lookback=LookbackWindow(days=days, clock="resolved"),
                    monotonic="increasing",
                    # Measured: 56% at the 30-day window, 38% at 90 days. Set to
                    # the wider window's value; the 30-day one runs higher.
                    expected_null_share=0.45,
                    risk_note=(
                        "Mostly NaN, and correctly so - most customers have nothing resolving "
                        "in any given window. A recency-weighted rate is fairer than an "
                        "all-time one: a bad month two years ago should decay."
                    ),
                )
            )

        specs.append(
            FeatureSpec(
                name="temporal_days_since_last_rto",
                family=FAMILY,
                dtype="float",
                description="Days since this customer's most recent known return.",
                source_columns=(cols.CUSTOMER_HASH, cols.ORDERED_AT, cols.RESOLVED_AT, cols.IS_RTO),
                observation_point=ObservationPoint.PRIOR_ORDERS_RESOLVED,
                availability=Availability.AT_ORDER_TIME,
                lookback=LookbackWindow(days=None, clock="resolved"),
                monotonic="decreasing",
                expected_null_share=0.78,  # measured
                risk_note=(
                    "Recency of the last return, which decays naturally. NaN for the large "
                    "majority who have never had one - and NaN is right, because 'never' is "
                    "not a very large number of days, it is a different state."
                ),
            )
        )
        specs.append(
            FeatureSpec(
                name="temporal_order_burst",
                family=FAMILY,
                dtype="float",
                description=(
                    "Orders in the last 7 days relative to the customer's 90-day daily rate."
                ),
                source_columns=(cols.CUSTOMER_HASH, cols.ORDERED_AT),
                observation_point=ObservationPoint.PRIOR_ORDERS_PLACED,
                availability=Availability.AT_ORDER_TIME,
                lookback=LookbackWindow(days=90, clock="placed"),
                expected_null_share=0.36,  # measured
                risk_note=(
                    "HIGHEST FAIRNESS RISK IN THIS FAMILY. A legitimate seasonal shopper and "
                    "an impulsive one look identical over a short window. Kept only if the "
                    "ablation study shows it pays for itself in rupees, not in AUC."
                ),
            )
        )
        return FeatureSet(tuple(specs))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)

        # Work in integer nanoseconds throughout. A timezone-aware pandas column
        # comes back from `.to_numpy()` as an object array of Timestamps, which
        # cannot be subtracted from a timedelta64 - and silently converting to
        # naive datetimes would drop the offset, which is exactly the class of
        # bug the timestamptz columns exist to prevent.
        order_ns, _ = _epoch_ns(frame[cols.ORDERED_AT])
        resolved_ns, resolved_valid = _epoch_ns(frame[cols.RESOLVED_AT])
        labels = frame[cols.IS_RTO].astype("float64").to_numpy()
        groups = frame[cols.CUSTOMER_HASH].to_numpy()
        order_valid = np.ones(len(frame), dtype=bool)

        for days in PLACED_WINDOWS:
            out[f"temporal_orders_last_{days}d"] = _windowed_count(
                groups, order_ns, order_ns, order_valid, window_days=days
            ).astype("int64")

        for days in RESOLVED_WINDOWS:
            counts, positives = _windowed_count_and_sum(
                groups, order_ns, resolved_ns, resolved_valid, labels, window_days=days
            )
            out[f"temporal_rto_count_last_{days}d"] = np.nan_to_num(positives).astype("int64")
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"temporal_rto_rate_last_{days}d"] = np.where(
                    counts > 0, positives / counts, np.nan
                )

        out["temporal_days_since_last_rto"] = _days_since_last_positive(
            groups, order_ns, resolved_ns, resolved_valid, labels
        )

        orders_7 = out["temporal_orders_last_7d"].to_numpy(dtype="float64")
        orders_90 = out["temporal_orders_last_90d"].to_numpy(dtype="float64")
        # Baseline daily rate over 90 days, projected onto a 7-day window. NaN
        # when there is no 90-day history to compare against - a ratio against an
        # empty baseline is not a small number, it is undefined.
        with np.errstate(invalid="ignore", divide="ignore"):
            baseline_7d = orders_90 * (7.0 / 90.0)
            out["temporal_order_burst"] = np.where(
                orders_90 > 0, orders_7 / np.maximum(baseline_7d, 1e-9), np.nan
            )

        return out[list(self.feature_set.names)]


# ---------------------------------------------------------------------------
# Windowed aggregation primitives
# ---------------------------------------------------------------------------
#
# Written as explicit per-group scans rather than a rolling groupby. Two reasons:
# the two clocks mean the "window column" and the "anchor column" differ, which
# pandas' rolling API does not express; and an explicit scan with a visible
# comparison is something a reviewer can check, which matters more here than
# raw speed. ``tests/unit/test_temporal_features.py`` checks each of them
# against hand-computed fixtures.

NANOS_PER_DAY = 86_400_000_000_000


def _epoch_ns(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Nanoseconds since epoch, plus a validity mask for NaT.

    Returns integers rather than datetimes so every comparison below is plain
    arithmetic. The validity mask is computed from the original series, because
    NaT has no meaningful integer representation.
    """
    valid = series.notna().to_numpy()
    values = series.astype("int64").to_numpy()
    return values, valid


def _group_slices(groups: np.ndarray) -> dict[object, np.ndarray]:
    """Row positions per group, in the frame's existing order."""
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1], True])
    return {sorted_groups[start]: order[start:end] for start, end in pairwise(boundaries)}


def _windowed_count(
    groups: np.ndarray,
    anchor_ns: np.ndarray,
    event_ns: np.ndarray,
    event_valid: np.ndarray,
    *,
    window_days: int,
) -> np.ndarray:
    """Count of events in ``[anchor - window, anchor)`` within each group.

    Strictly before the anchor: an event at the exact anchor instant is the row
    itself, or information that has not propagated yet.
    """
    window = window_days * NANOS_PER_DAY
    result = np.zeros(len(groups), dtype="float64")

    for positions in _group_slices(groups).values():
        anchors = anchor_ns[positions]
        events = event_ns[positions][event_valid[positions]]
        if events.size == 0:
            continue
        ordered = np.sort(events)
        upper = np.searchsorted(ordered, anchors, side="left")
        lower = np.searchsorted(ordered, anchors - window, side="left")
        result[positions] = upper - lower
    return result


def _windowed_count_and_sum(
    groups: np.ndarray,
    anchor_ns: np.ndarray,
    event_ns: np.ndarray,
    event_valid: np.ndarray,
    values: np.ndarray,
    *,
    window_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count and value-sum of events in ``[anchor - window, anchor)`` per group."""
    window = window_days * NANOS_PER_DAY
    counts = np.zeros(len(groups), dtype="float64")
    sums = np.zeros(len(groups), dtype="float64")

    for positions in _group_slices(groups).values():
        anchors = anchor_ns[positions]
        valid = event_valid[positions]
        if not valid.any():
            continue

        events = event_ns[positions][valid]
        group_values = np.nan_to_num(values[positions][valid])
        order = np.argsort(events, kind="stable")
        ordered_events = events[order]
        cumulative = np.r_[0.0, np.cumsum(group_values[order])]

        upper = np.searchsorted(ordered_events, anchors, side="left")
        lower = np.searchsorted(ordered_events, anchors - window, side="left")
        counts[positions] = upper - lower
        sums[positions] = cumulative[upper] - cumulative[lower]
    return counts, sums


def _days_since_last_positive(
    groups: np.ndarray,
    anchor_ns: np.ndarray,
    event_ns: np.ndarray,
    event_valid: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Days since the most recent positive event that resolved before the anchor."""
    result = np.full(len(groups), np.nan, dtype="float64")

    for positions in _group_slices(groups).values():
        anchors = anchor_ns[positions]
        positive = event_valid[positions] & (np.nan_to_num(values[positions]) > 0)
        if not positive.any():
            continue

        ordered = np.sort(event_ns[positions][positive])
        index = np.searchsorted(ordered, anchors, side="left") - 1
        has_previous = index >= 0
        if not has_previous.any():
            continue

        gaps = np.full(len(anchors), np.nan, dtype="float64")
        previous = ordered[index[has_previous]]
        gaps[has_previous] = (anchors[has_previous] - previous) / NANOS_PER_DAY
        result[positions] = gaps
    return result
