"""As-of joins - the mechanism that keeps historical aggregates honest.

SPEC section 03, rule 4: a customer's historical return rate must use only orders
that had already **resolved** before the current order was placed. Not orders that
were merely placed earlier - resolved earlier. An order placed on day 40 that
comes back on day 47 was not known to be an RTO on day 42, and a feature that
pretends otherwise is leakage wearing a plausible disguise.

The generator already builds its history features this way, order by order, in
chronological time. This module exists for everything computed *afterwards* -
the Phase 3 feature families, and any recomputation over a loaded dataset - so
the rule has exactly one reviewed implementation rather than being re-derived
five times.

TWO IMPLEMENTATIONS, ON PURPOSE
-------------------------------
:func:`as_of_aggregate` is the fast one: a cumulative-then-merge_asof approach
that runs in O(n log n).

:func:`brute_force_as_of` is an obviously-correct O(n^2) reference. It exists so
the fast one can be *tested against it* rather than merely believed. Leakage bugs
are silent and flattering, and "the clever version agrees with the obvious
version" is the only argument worth making about a function like this.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

AggregateFn = Literal["count", "sum", "mean"]


class AsOfLeakageError(AssertionError):
    """Raised when an aggregate is shown to depend on information from the future."""


def as_of_aggregate(
    frame: pd.DataFrame,
    *,
    group_key: str,
    value_column: str | None,
    order_time_column: str,
    resolution_time_column: str,
    aggregate: AggregateFn,
    min_periods: int = 0,
) -> pd.Series:
    """Aggregate ``value_column`` over each group's *resolved* prior history.

    For every row ``r``, the result includes only rows ``h`` where::

        h[group_key] == r[group_key]
        h[resolution_time_column] < r[order_time_column]

    Note the strict ``<``: an order resolving at the exact instant another is
    placed is excluded, because in production that information would not have
    propagated yet.

    THE EMPTY-HISTORY RULE, STATED ONCE
    -----------------------------------
    ``count`` and ``sum`` return **0** for a customer with no qualifying history.
    Both have a genuine zero: "zero prior orders" and "zero prior RTOs" are facts,
    not absences.

    ``mean`` returns **NaN**. There is no such thing as an average over nothing,
    and 0 would silently claim the customer has a zero percent return rate - the
    most optimistic possible reading of "we know nothing about them". LightGBM
    handles missingness natively and can learn "no history" as its own state,
    which is what we want it to learn.

    ``min_periods`` overrides all of that: below the threshold every aggregate
    returns NaN, because a rate computed from two orders is noise wearing a
    number's clothing.

    Rows whose ``resolution_time_column`` is null (still pending) contribute to
    nobody's history, which is exactly right: an unresolved order carries no
    outcome information.
    """
    if aggregate != "count" and value_column is None:
        msg = f"aggregate {aggregate!r} requires a value_column"
        raise ValueError(msg)

    left = frame[[group_key, order_time_column]].copy()
    left["_row"] = np.arange(len(frame))
    left = left.sort_values([order_time_column, "_row"], kind="stable")

    right_columns = [group_key, resolution_time_column]
    if value_column is not None:
        right_columns.append(value_column)
    right = frame[right_columns].copy()
    right = right[right[resolution_time_column].notna()]
    right = right.sort_values([resolution_time_column], kind="stable")

    # Running count and sum within each group, ordered by resolution time. Read
    # off at a given instant, these give the totals over everything resolved
    # strictly before it - which is the as-of aggregate.
    right["_count"] = right.groupby(group_key, sort=False).cumcount() + 1
    if value_column is not None:
        right["_sum"] = right.groupby(group_key, sort=False)[value_column].cumsum()

    merge_columns = [resolution_time_column, group_key, "_count"]
    if value_column is not None:
        merge_columns.append("_sum")

    merged = pd.merge_asof(
        left,
        right[merge_columns],
        left_on=order_time_column,
        right_on=resolution_time_column,
        by=group_key,
        direction="backward",
        allow_exact_matches=False,  # strict <, not <=
    )

    counts = merged["_count"].to_numpy(dtype="float64", na_value=0.0)
    if aggregate == "count":
        values = counts
    else:
        sums = merged["_sum"].to_numpy(dtype="float64", na_value=0.0)
        if aggregate == "sum":
            values = sums
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                values = np.where(counts > 0, sums / counts, np.nan)

    # Apply the empty-history rule, then min_periods on top of it.
    if aggregate == "mean":
        values = np.where(counts > 0, values, np.nan)
    if min_periods > 0:
        values = np.where(counts >= min_periods, values, np.nan)

    result = pd.Series(values, index=merged["_row"].to_numpy())
    return result.sort_index().set_axis(frame.index)


def brute_force_as_of(
    frame: pd.DataFrame,
    *,
    group_key: str,
    value_column: str | None,
    order_time_column: str,
    resolution_time_column: str,
    aggregate: AggregateFn,
    min_periods: int = 0,
) -> pd.Series:
    """Obviously-correct reference implementation. O(n^2); for tests and small data.

    Deliberately written the slow, boring way - a nested scan with an explicit
    comparison - so that reading it is enough to be convinced it is right. The
    fast path above is validated against this.

    Obeys the same empty-history rule: ``count`` and ``sum`` are 0 over an empty
    history, ``mean`` is NaN, and ``min_periods`` overrides both.
    """
    group_values = frame[group_key].to_numpy()
    order_times = frame[order_time_column].to_numpy()
    resolution_times = frame[resolution_time_column].to_numpy()
    values = frame[value_column].to_numpy() if value_column is not None else None

    out = np.full(len(frame), np.nan, dtype="float64")
    for i in range(len(frame)):
        total = 0.0
        count = 0
        for j in range(len(frame)):
            if i == j:
                continue
            if group_values[j] != group_values[i]:
                continue
            if pd.isna(resolution_times[j]):
                continue
            if not resolution_times[j] < order_times[i]:
                continue
            count += 1
            if values is not None:
                total += float(values[j])

        if min_periods > 0 and count < min_periods:
            out[i] = np.nan
        elif aggregate == "count":
            out[i] = count
        elif aggregate == "sum":
            out[i] = total
        elif count == 0:
            out[i] = np.nan  # no mean over an empty history
        else:
            out[i] = total / count
    return pd.Series(out, index=frame.index)


def assert_no_future_information(
    frame: pd.DataFrame,
    *,
    compute: Callable[[pd.DataFrame], pd.Series],
    order_time_column: str,
    resolution_time_column: str,
    outcome_columns: Sequence[str],
    cutoff: pd.Timestamp,
) -> None:
    """Fail loudly if a computed feature changes when the future is hidden.

    ``compute`` is any function that turns an order table into a feature series -
    an :func:`as_of_aggregate` call, or a whole Phase 3 feature family. That
    generality is the point: this guard is only worth having if it can be pointed
    at code that has *not* already been proven correct.

    THE TEST
    --------
    Compute the feature on the full frame. Then build a **rewound** view of the
    world as it looked at ``cutoff`` - every order still present, but any order
    that had not yet resolved has its resolution timestamp and its outcome columns
    blanked - and compute it again. For rows ordered before ``cutoff``, the two
    results must be identical.

    WHY REWIND RATHER THAN DELETE
    -----------------------------
    An earlier version of this function *removed* unresolved rows instead of
    blanking their outcomes. That silently excluded from the comparison exactly
    the rows most likely to leak: an order placed on day 5 that resolves on day 9
    was dropped from the day-8 view entirely, so its feature value was never
    checked. Rewinding keeps every order visible and hides only what genuinely was
    not known yet, which is what "as-of" means.

    ``outcome_columns`` names the columns derived from the outcome - the label
    and anything computed from it. They must be blanked too, because a feature
    that reads the label directly would sail past a check that only hid the
    resolution timestamp.
    """
    full = compute(frame)

    rewound = frame.copy()
    unresolved = rewound[resolution_time_column].isna() | (
        rewound[resolution_time_column] >= cutoff
    )
    rewound.loc[unresolved, resolution_time_column] = pd.NaT
    for column in outcome_columns:
        if column in rewound.columns:
            rewound.loc[unresolved, column] = np.nan

    truncated = compute(rewound)

    past_rows = frame.index[frame[order_time_column] < cutoff]
    common = past_rows.intersection(truncated.index)
    if len(common) == 0:  # pragma: no cover - a cutoff with nothing before it
        return

    left = full.loc[common].astype("float64")
    right = truncated.loc[common].astype("float64")

    both_null = left.isna() & right.isna()
    close = np.isclose(left.fillna(0.0), right.fillna(0.0), rtol=1e-9, atol=1e-9)
    mismatched = ~(both_null | close)

    if bool(mismatched.any()):
        offenders = list(common[mismatched.to_numpy()][:5])
        msg = (
            f"a feature changed for {int(mismatched.sum())} row(s) when outcomes at or after "
            f"{cutoff} were hidden. It is reading the future. "
            f"First offending rows: {offenders}"
        )
        raise AsOfLeakageError(msg)
