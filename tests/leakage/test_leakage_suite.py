"""The leakage suite. SPEC section 03.

    "Judges can run pytest and see them pass. That is a more persuasive claim
    about rigour than any number on a slide."

The four tests the specification names, running against the real generated
dataset. They were placeholders in Phase 1, skipped and labelled as such; the
pipeline they test now exists, so they run.

Each one targets a specific way this dataset could be quietly wrong in a
direction that flatters a model. That is the selection criterion - a leakage test
that catches only obvious corruption is not worth the file it lives in.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rto_sentinel.configuration.schemas import GeneratorConfig, SplitsConfig
from rto_sentinel.data import schema as cols
from rto_sentinel.data.asof import as_of_aggregate, assert_no_future_information
from rto_sentinel.data.generator import GenerationResult
from rto_sentinel.data.splits import assign_splits, customers_are_disjoint

pytestmark = pytest.mark.leakage

MODELLING = ("train", "validation", "test")


@pytest.fixture
def split_orders(small_dataset: GenerationResult, splits_config: SplitsConfig) -> pd.DataFrame:
    orders = small_dataset.orders.copy()
    orders[cols.SPLIT] = assign_splits(orders, splits_config).labels
    return orders


# ---------------------------------------------------------------------------
# 1. test_no_future_aggregates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantile", [0.3, 0.5, 0.7])
def test_no_future_aggregates(small_dataset: GenerationResult, quantile: float) -> None:
    """No feature may use information that post-dates the order.

    The world is rewound to a cutoff - every order still present, but any order
    that had not yet resolved has its outcome and resolution timestamp hidden -
    and the customer-history aggregate is recomputed. For orders placed before the
    cutoff it must be unchanged.

    The subtle case this catches: an aggregate built from orders *placed* earlier
    rather than *resolved* earlier. An order placed on day 40 that comes back on
    day 47 was not known to be an RTO on day 42.

    Run at three cutoffs, because any single cutoff only exposes leaks that
    straddle it.
    """
    orders = small_dataset.orders.copy()
    orders["_label"] = orders[cols.IS_RTO].astype("float64")
    cutoff = pd.Timestamp(orders[cols.ORDERED_AT].quantile(quantile))

    assert_no_future_information(
        orders,
        compute=lambda frame: as_of_aggregate(
            frame,
            group_key=cols.CUSTOMER_HASH,
            value_column="_label",
            order_time_column=cols.ORDERED_AT,
            resolution_time_column=cols.RESOLVED_AT,
            aggregate="mean",
        ),
        order_time_column=cols.ORDERED_AT,
        resolution_time_column=cols.RESOLVED_AT,
        outcome_columns=["_label", cols.IS_RTO, cols.OUTCOME],
        cutoff=cutoff,
    )


def test_stored_history_matches_an_independent_as_of_recomputation(
    small_dataset: GenerationResult,
) -> None:
    """The generator's own history columns agree with the as-of implementation.

    The generator builds history incrementally during simulation; ``data.asof``
    computes it afterwards from the finished table. They are entirely separate
    pieces of code, so agreement between them is real evidence rather than a
    tautology - a leak in either would show up as a mismatch.

    One rule has to be mirrored for the comparison to be meaningful: cancellations
    are excluded from delivery history. An order cancelled before dispatch was
    never presented to the customer and carries no delivery signal, so the
    generator leaves it out. Blanking its resolution timestamp here reproduces
    that exactly - a cancelled order then contributes to nobody's history.
    """
    orders = small_dataset.orders.copy()
    orders["_label"] = orders[cols.IS_RTO].astype("float64")
    orders.loc[orders[cols.OUTCOME] == "cancelled", cols.RESOLVED_AT] = pd.NaT

    recomputed = as_of_aggregate(
        orders,
        group_key=cols.CUSTOMER_HASH,
        value_column=None,
        order_time_column=cols.ORDERED_AT,
        resolution_time_column=cols.RESOLVED_AT,
        aggregate="count",
    )
    pd.testing.assert_series_equal(
        recomputed.astype("int64"),
        orders[cols.PRIOR_ORDER_COUNT].astype("int64"),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# 2. test_customer_disjoint_splits
# ---------------------------------------------------------------------------


def test_customer_disjoint_splits(split_orders: pd.DataFrame) -> None:
    """No ``customer_hash`` appears in more than one modelling split.

    Without this the model memorises individuals and reports a score it cannot
    reproduce on anyone new.
    """
    assert customers_are_disjoint(split_orders)

    modelling = split_orders[split_orders[cols.SPLIT].isin(MODELLING)]
    per_split = {
        name: set(modelling.loc[modelling[cols.SPLIT] == name, cols.CUSTOMER_HASH])
        for name in MODELLING
    }
    assert not (per_split["train"] & per_split["validation"])
    assert not (per_split["train"] & per_split["test"])
    assert not (per_split["validation"] & per_split["test"])


# ---------------------------------------------------------------------------
# 3. test_label_maturity
# ---------------------------------------------------------------------------


def test_label_maturity(small_dataset: GenerationResult, generator_config: GeneratorConfig) -> None:
    """No order is labelled before its terminal state is known.

    Three things at once: an immature order carries a NULL label rather than an
    optimistic "delivered"; a mature order resolves inside the configured window;
    and no resolution falls outside the horizon the dataset claims to cover.
    """
    orders = small_dataset.orders
    mature = orders[cols.IS_MATURE].astype(bool)

    # Immature orders are unlabelled, marked pending, and carry no resolution.
    immature = orders[~mature]
    assert immature[cols.IS_RTO].isna().all()
    assert (immature[cols.OUTCOME] == "pending").all()
    assert immature[cols.RESOLVED_AT].isna().all()

    # Mature orders are labelled and resolve inside the maturity window.
    resolved = orders[mature]
    assert resolved[cols.IS_RTO].notna().all()
    assert resolved[cols.RESOLVED_AT].notna().all()
    assert (
        resolved[cols.MATURITY_DAYS] <= generator_config.label_maturity.max_resolution_days + 1e-9
    ).all()
    assert (resolved[cols.RESOLVED_AT] > resolved[cols.ORDERED_AT]).all()

    # Nothing resolves beyond the horizon the dataset claims to cover.
    horizon_end = pd.Timestamp(small_dataset.metadata.end_date)
    assert (resolved[cols.RESOLVED_AT] <= horizon_end).all()

    # The immature tail is real: a dataset with none would not be exercising this.
    assert len(immature) > 0


def test_immature_orders_never_enter_a_modelling_split(split_orders: pd.DataFrame) -> None:
    """An unresolved outcome must not become a training example."""
    modelling = split_orders[split_orders[cols.SPLIT].isin(MODELLING)]
    assert modelling[cols.IS_MATURE].astype(bool).all()
    assert modelling[cols.IS_RTO].notna().all()


# ---------------------------------------------------------------------------
# 4. test_target_not_in_features
# ---------------------------------------------------------------------------


def test_target_not_in_features() -> None:
    """No forbidden column may reach a design matrix.

    Checked against the declared order-time column set, which is what the Phase 3
    feature pipeline builds from. The label, its timestamps, the split marker, the
    identity columns and the simulator's own latent variables are all excluded.
    """
    order_time = set(cols.ORDER_TIME_COLUMNS)
    forbidden = set(cols.FORBIDDEN_IN_FEATURES)

    overlap = order_time & forbidden
    assert not overlap, f"order-time columns overlap the forbidden set: {sorted(overlap)}"

    assert cols.TARGET_COLUMN in forbidden
    assert cols.OUTCOME in forbidden
    assert cols.RESOLVED_AT in forbidden
    assert cols.SPLIT in forbidden
    assert cols.CUSTOMER_HASH in forbidden
    assert cols.TRUE_RTO_PROBABILITY in forbidden
    assert cols.LATENT_LOGIT in forbidden


def test_post_order_timestamps_are_forbidden() -> None:
    """Knowing when an order resolved is close to knowing how it resolved.

    A model given ``dispatched_at`` and ``resolved_at`` can read the gap between
    them, and RTOs take longer than deliveries by construction. That is a leak
    that looks like a perfectly innocent date column.
    """
    for column in (cols.DISPATCHED_AT, cols.FIRST_ATTEMPT_AT, cols.RESOLVED_AT, cols.MATURITY_DAYS):
        assert column in cols.FORBIDDEN_IN_FEATURES


def test_the_simulator_ground_truth_is_not_in_the_benchmark_table(
    small_dataset: GenerationResult,
) -> None:
    """The true probability lives in its own frame and never joins ``orders``."""
    order_columns = set(small_dataset.orders.columns)
    assert cols.TRUE_RTO_PROBABILITY not in order_columns
    assert cols.LATENT_LOGIT not in order_columns
    assert cols.TRUE_RTO_PROBABILITY in small_dataset.latents.columns


def test_the_order_time_column_set_is_not_empty() -> None:
    """Guards against someone emptying the set and making these checks vacuous."""
    assert len(cols.ORDER_TIME_COLUMNS) >= 20
    assert len(cols.FORBIDDEN_IN_FEATURES) >= 12
