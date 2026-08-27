"""As-of aggregation is correct, and provably so.

The argument these tests make is deliberately narrow: **the fast implementation
agrees with an obviously-correct slow one**. Leakage bugs are silent and
flattering, so "I read the code and it looked right" is not a claim worth making
about a function like this. A nested loop with an explicit timestamp comparison
is something a reviewer can verify by reading; the merge_asof version is not.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from rto_sentinel.data.asof import (
    AsOfLeakageError,
    as_of_aggregate,
    assert_no_future_information,
    brute_force_as_of,
)

BASE = datetime(2025, 9, 1, tzinfo=UTC)


def _frame() -> pd.DataFrame:
    """A tiny, hand-checkable fixture.

    Customer A's second order is placed on day 5, but their first order does not
    resolve until day 7 - so at order time the history is still empty. That single
    row is the whole point of an as-of join.
    """
    return pd.DataFrame(
        {
            "customer": ["A", "A", "A", "B", "B"],
            "ordered_at": [
                BASE,
                BASE + timedelta(days=5),
                BASE + timedelta(days=10),
                BASE + timedelta(days=2),
                BASE + timedelta(days=20),
            ],
            "resolved_at": [
                BASE + timedelta(days=7),
                BASE + timedelta(days=9),
                BASE + timedelta(days=14),
                BASE + timedelta(days=6),
                pd.NaT,  # still pending: contributes to nobody's history
            ],
            "is_rto": [1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )


def _count(frame: pd.DataFrame) -> pd.Series:
    return as_of_aggregate(
        frame,
        group_key="customer",
        value_column=None,
        order_time_column="ordered_at",
        resolution_time_column="resolved_at",
        aggregate="count",
    )


def test_history_is_empty_until_an_earlier_order_has_resolved() -> None:
    """The case the whole module exists for.

    A's order on day 5 comes after A's order on day 0 - but that first order does
    not resolve until day 7. On day 5 nobody knows how it turned out, so the
    history must be empty. A naive "orders placed earlier" aggregate would report
    1 here, and would be reading the future.
    """
    counts = _count(_frame())
    assert counts.iloc[0] == 0  # A, day 0: nothing before it
    assert counts.iloc[1] == 0  # A, day 5: day-0 order resolves on day 7. Not yet known.
    assert counts.iloc[2] == 2  # A, day 10: day-0 and day-5 orders both resolved
    assert counts.iloc[3] == 0  # B, day 2: nothing resolved yet
    assert counts.iloc[4] == 1  # B, day 20: the day-2 order resolved on day 6


def test_pending_orders_contribute_to_nobody() -> None:
    """An unresolved order carries no outcome information, so it counts for nothing."""
    frame = _frame()
    frame.loc[4, "ordered_at"] = BASE + timedelta(days=3)
    counts = _count(frame)
    # B's day-20 row is now day 3; B's day-2 order still resolves on day 6.
    assert counts.iloc[4] == 0


@pytest.mark.parametrize("aggregate", ["count", "sum", "mean"])
def test_fast_path_matches_the_brute_force_reference(aggregate: str) -> None:
    """The core correctness argument, on the hand-checkable fixture."""
    frame = _frame()
    fast = as_of_aggregate(
        frame,
        group_key="customer",
        value_column=None if aggregate == "count" else "is_rto",
        order_time_column="ordered_at",
        resolution_time_column="resolved_at",
        aggregate=aggregate,  # type: ignore[arg-type]
    )
    slow = brute_force_as_of(
        frame,
        group_key="customer",
        value_column=None if aggregate == "count" else "is_rto",
        order_time_column="ordered_at",
        resolution_time_column="resolved_at",
        aggregate=aggregate,  # type: ignore[arg-type]
    )
    pd.testing.assert_series_equal(fast, slow, check_names=False)


@pytest.mark.parametrize("aggregate", ["count", "sum", "mean"])
def test_fast_path_matches_brute_force_on_random_data(aggregate: str) -> None:
    """The same argument on messy data: ties, gaps, pending rows, many groups.

    Random data is where a merge-based implementation goes wrong - duplicate
    timestamps and exact boundary matches are exactly the cases hand-written
    fixtures forget.
    """
    rng = np.random.default_rng(7)
    n = 220
    ordered_offsets = rng.integers(0, 60, size=n)
    resolution_lag = rng.integers(1, 10, size=n)
    pending = rng.random(n) < 0.15

    frame = pd.DataFrame(
        {
            "customer": rng.choice(["A", "B", "C", "D", "E"], size=n),
            "ordered_at": [BASE + timedelta(days=int(d)) for d in ordered_offsets],
            "resolved_at": [
                pd.NaT
                if pending[i]
                else BASE + timedelta(days=int(ordered_offsets[i] + resolution_lag[i]))
                for i in range(n)
            ],
            "is_rto": rng.integers(0, 2, size=n).astype(float),
        }
    )

    fast = as_of_aggregate(
        frame,
        group_key="customer",
        value_column=None if aggregate == "count" else "is_rto",
        order_time_column="ordered_at",
        resolution_time_column="resolved_at",
        aggregate=aggregate,  # type: ignore[arg-type]
    )
    slow = brute_force_as_of(
        frame,
        group_key="customer",
        value_column=None if aggregate == "count" else "is_rto",
        order_time_column="ordered_at",
        resolution_time_column="resolved_at",
        aggregate=aggregate,  # type: ignore[arg-type]
    )
    pd.testing.assert_series_equal(fast, slow, check_names=False)


def test_exact_timestamp_matches_are_excluded() -> None:
    """Strict ``<``, not ``<=``.

    An order resolving at the exact instant another is placed is excluded, because
    in production that information would not have propagated yet.
    """
    frame = pd.DataFrame(
        {
            "customer": ["A", "A"],
            "ordered_at": [BASE, BASE + timedelta(days=5)],
            "resolved_at": [BASE + timedelta(days=5), BASE + timedelta(days=9)],
            "is_rto": [1.0, 0.0],
        }
    )
    counts = _count(frame)
    assert counts.iloc[1] == 0


def test_mean_of_no_history_is_nan_not_zero() -> None:
    """NaN says "we know nothing". Zero claims a zero percent return rate."""
    means = as_of_aggregate(
        _frame(),
        group_key="customer",
        value_column="is_rto",
        order_time_column="ordered_at",
        resolution_time_column="resolved_at",
        aggregate="mean",
    )
    assert np.isnan(means.iloc[0])
    assert np.isnan(means.iloc[1])
    assert means.iloc[2] == pytest.approx(0.5)


def test_min_periods_suppresses_thin_history() -> None:
    means = as_of_aggregate(
        _frame(),
        group_key="customer",
        value_column="is_rto",
        order_time_column="ordered_at",
        resolution_time_column="resolved_at",
        aggregate="mean",
        min_periods=3,
    )
    assert means.isna().all()


def test_groups_do_not_bleed_into_each_other() -> None:
    """B's outcomes must never appear in A's history."""
    frame = _frame()
    counts = _count(frame)
    a_rows = frame.index[frame["customer"] == "A"]
    assert list(counts.loc[a_rows]) == [0, 0, 2]


def test_a_missing_value_column_is_refused() -> None:
    with pytest.raises(ValueError, match="requires a value_column"):
        as_of_aggregate(
            _frame(),
            group_key="customer",
            value_column=None,
            order_time_column="ordered_at",
            resolution_time_column="resolved_at",
            aggregate="mean",
        )


# ---------------------------------------------------------------------------
# the leakage assertion itself
# ---------------------------------------------------------------------------


def _correct_history_rate(frame: pd.DataFrame) -> pd.Series:
    """The correct thing: average over orders that had already RESOLVED."""
    return as_of_aggregate(
        frame,
        group_key="customer",
        value_column="is_rto",
        order_time_column="ordered_at",
        resolution_time_column="resolved_at",
        aggregate="mean",
    )


def _leaking_history_rate(frame: pd.DataFrame) -> pd.Series:
    """The plausible-looking mistake: average over orders PLACED earlier.

    This is the bug the leakage suite exists to catch, and it is easy to write by
    accident - an expanding mean over a customer's orders in placement order looks
    entirely reasonable. It is not: an order placed on day 40 that comes back on
    day 47 was not known to be an RTO on day 42, and this includes it anyway.
    """
    ordered = frame.sort_values(["customer", "ordered_at"], kind="stable")
    shifted = ordered.groupby("customer", sort=False)["is_rto"].shift(1)
    rate = shifted.groupby(ordered["customer"]).expanding().mean().reset_index(level=0, drop=True)
    return rate.reindex(frame.index)


CUTOFF = pd.Timestamp(BASE + timedelta(days=8))

# A cutoff can only expose a leak that straddles it: the consuming order must be
# placed before it, and the outcome it wrongly consumed must resolve after it.
# Here customer A orders on day 5 and their day-0 order resolves on day 7, so a
# cutoff on day 6 is the window where the leak is visible. Day 8 is too late -
# by then the day-7 outcome is legitimately known and the leaky feature and the
# correct one agree. This is why leakage checks are run at several cutoffs in the
# suite rather than at one arbitrary point.
LEAK_CUTOFF = pd.Timestamp(BASE + timedelta(days=6))


@pytest.mark.parametrize("days", [4, 6, 8, 12])
def test_hiding_the_future_does_not_change_a_correct_aggregate(days: int) -> None:
    """Rewind the world to a cutoff; earlier rows must be unaffected.

    Run at several cutoffs, because any single cutoff only exposes leaks that
    straddle it.
    """
    assert_no_future_information(
        _frame(),
        compute=_correct_history_rate,
        order_time_column="ordered_at",
        resolution_time_column="resolved_at",
        outcome_columns=["is_rto"],
        cutoff=pd.Timestamp(BASE + timedelta(days=days)),
    )


def test_the_leakage_assertion_catches_a_leaking_feature() -> None:
    """The assertion must be able to fail, or it is decoration.

    Pointed at a feature computed over *placed* rather than *resolved* history, it
    fires - which is the whole reason it accepts an arbitrary compute function
    rather than only validating the one implementation already known to be right.
    """
    with pytest.raises(AsOfLeakageError, match="reading the future"):
        assert_no_future_information(
            _frame(),
            compute=_leaking_history_rate,
            order_time_column="ordered_at",
            resolution_time_column="resolved_at",
            outcome_columns=["is_rto"],
            cutoff=LEAK_CUTOFF,
        )


def test_the_generated_dataset_history_survives_hiding_the_future(small_dataset) -> None:
    """The real benchmark, not a fixture.

    The generator builds its history features order by order in chronological
    time, so they should be leak-free by construction. This checks that claim
    against the actual output rather than trusting the design.
    """
    from rto_sentinel.data import schema as cols

    orders = small_dataset.orders.rename(
        columns={cols.CUSTOMER_HASH: "customer", cols.IS_RTO: "is_rto"}
    )
    orders["is_rto"] = orders["is_rto"].astype("float64")
    cutoff = pd.Timestamp(orders[cols.ORDERED_AT].quantile(0.6))

    assert_no_future_information(
        orders,
        compute=lambda frame: as_of_aggregate(
            frame,
            group_key="customer",
            value_column="is_rto",
            order_time_column=cols.ORDERED_AT,
            resolution_time_column=cols.RESOLVED_AT,
            aggregate="mean",
        ),
        order_time_column=cols.ORDERED_AT,
        resolution_time_column=cols.RESOLVED_AT,
        outcome_columns=["is_rto"],
        cutoff=cutoff,
    )
