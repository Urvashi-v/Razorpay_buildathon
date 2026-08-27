"""As-of joins - the mechanism that keeps historical aggregates honest.

SPEC section 03, rule 4: a customer's historical return rate must use only orders
that had already *resolved* before the current order was placed. Not orders that
were merely placed earlier - resolved earlier. An order placed on day 40 that
comes back on day 47 was not known to be an RTO on day 42, and a feature that
pretends otherwise is leakage wearing a plausible disguise.

This module exists so that rule is enforced by one reviewed implementation rather
than re-derived in every feature family. ``tests/leakage/test_no_future_
aggregates.py`` tests this module directly.

STATUS: Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

AggregateFn = Literal["count", "sum", "mean"]


def as_of_aggregate(
    frame: pd.DataFrame,
    *,
    group_key: str,
    value_column: str | None,
    order_time_column: str,
    resolution_time_column: str,
    aggregate: AggregateFn,
    output_column: str,
    min_periods: int = 0,
) -> pd.Series:
    """Aggregate ``value_column`` over each group's *resolved* prior history.

    For every row ``r``, the result includes only rows ``h`` where::

        h[group_key] == r[group_key]
        h[resolution_time_column] < r[order_time_column]
        h is not r

    Note the strict ``<``: an order resolving at the exact instant another is
    placed is excluded, because in production that information would not have
    propagated yet.

    Rows with fewer than ``min_periods`` qualifying predecessors return NaN
    rather than 0. This matters: LightGBM handles missingness natively and can
    learn "no history" as its own state, whereas a 0 silently claims a customer
    has a zero percent return rate.
    """
    raise NotImplementedError("As-of aggregation lands in Phase 2.")


def assert_no_future_information(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    order_time_column: str,
    resolution_time_column: str,
) -> None:
    """Fail loudly if any aggregate feature could only be known in the future.

    The Phase 2 implementation recomputes each aggregate under a deliberately
    shifted clock and asserts the values are unchanged. A feature that moves when
    the future is hidden was reading the future.
    """
    raise NotImplementedError("Leakage assertion lands in Phase 2.")
