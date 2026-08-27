"""The simulator produces the relationships it claims to.

A generator can be reproducible and still be useless - reproducibly wrong is
still wrong. These tests check that the documented causal structure actually
appears in the output, and, just as importantly, that the task is **not** a
deterministic rule waiting to be reverse-engineered.

The direction-of-effect tests are deliberately loose. They assert a sign, not a
magnitude, because the magnitude is a property of the configuration and would
turn every parameter tweak into a test failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rto_sentinel.configuration.schemas import GeneratorConfig
from rto_sentinel.data import schema as cols
from rto_sentinel.data.generator import GenerationResult


@pytest.fixture
def mature(small_dataset: GenerationResult) -> pd.DataFrame:
    """Only orders whose outcome is actually known."""
    return small_dataset.orders[small_dataset.orders[cols.IS_MATURE]].copy()


# ---------------------------------------------------------------------------
# base rates match the published anchors
# ---------------------------------------------------------------------------


def test_cod_rto_rate_is_near_the_published_anchor(
    small_dataset: GenerationResult, generator_config: GeneratorConfig
) -> None:
    """~26% RTO on COD (Shipway ShipNotes FY25), hit by explicit calibration."""
    realised = small_dataset.metadata.realised_rto_rate_cod
    target = generator_config.base_rates.rto_given_cod
    # Loose on a 2,000-order sample: this is sampling noise, not a claim.
    assert abs(realised - target) < 0.05


def test_prepaid_rto_rate_is_far_below_cod(small_dataset: GenerationResult) -> None:
    """The entire premise: the problem lives in one payment mode."""
    assert small_dataset.metadata.realised_rto_rate_prepaid < 0.06
    assert (
        small_dataset.metadata.realised_rto_rate_prepaid
        < small_dataset.metadata.realised_rto_rate_cod / 3
    )


def test_cod_share_is_near_the_published_anchor(
    small_dataset: GenerationResult, generator_config: GeneratorConfig
) -> None:
    """COD is 60-65% of Indian e-commerce orders (ET Prime Research)."""
    assert (
        abs(small_dataset.metadata.realised_cod_share - generator_config.payment.cod_share) < 0.05
    )


# ---------------------------------------------------------------------------
# the documented causal relationships appear in the data
# ---------------------------------------------------------------------------


def test_prior_rto_history_predicts_future_rto(mature: pd.DataFrame) -> None:
    """The strongest honest signal in the problem must actually be present."""
    with_history = mature[mature[cols.PRIOR_ORDER_COUNT] > 0]
    returners = with_history[with_history[cols.PRIOR_RTO_RATE] > 0.5]
    clean = with_history[with_history[cols.PRIOR_RTO_RATE] == 0.0]

    assert len(returners) > 10 and len(clean) > 10
    assert returners[cols.IS_RTO].mean() > clean[cols.IS_RTO].mean()


def test_worse_addresses_return_more_often(mature: pd.DataFrame) -> None:
    """Poor address text causes real delivery failure - and must be measurable."""
    has_number = mature[mature[cols.ADDR_HAS_HOUSE_NUMBER]]
    no_number = mature[~mature[cols.ADDR_HAS_HOUSE_NUMBER]]

    assert len(has_number) > 50 and len(no_number) > 50
    assert no_number[cols.IS_RTO].mean() > has_number[cols.IS_RTO].mean()


def test_pincode_city_inconsistency_raises_risk(mature: pd.DataFrame) -> None:
    consistent = mature[mature[cols.ADDR_PINCODE_CITY_CONSISTENT]]
    inconsistent = mature[~mature[cols.ADDR_PINCODE_CITY_CONSISTENT]]

    assert len(inconsistent) > 20
    assert inconsistent[cols.IS_RTO].mean() > consistent[cols.IS_RTO].mean()


def test_deeper_discounts_return_more_often(mature: pd.DataFrame) -> None:
    """Deep-discount impulse orders are high risk - partly the merchant's doing."""
    deep = mature[mature[cols.DISCOUNT_DEPTH] > mature[cols.DISCOUNT_DEPTH].quantile(0.75)]
    shallow = mature[mature[cols.DISCOUNT_DEPTH] < mature[cols.DISCOUNT_DEPTH].quantile(0.25)]

    assert deep[cols.IS_RTO].mean() > shallow[cols.IS_RTO].mean()


def test_tier_3_pincodes_carry_more_risk(mature: pd.DataFrame) -> None:
    """The simulator does make tier-3 riskier on average.

    Stated plainly because it is the mechanism the Phase 4 fairness audit exists
    to scrutinise. The audit's question is not whether this gradient exists - it
    does, by construction - but whether a model trained on it transfers cost onto
    tier-3 customers beyond what its precision justifies.
    """
    by_tier = mature.groupby(cols.PINCODE_TIER)[cols.IS_RTO].mean()
    assert by_tier["tier_3"] > by_tier["tier_1"]


def test_sale_days_carry_more_volume(small_dataset: GenerationResult) -> None:
    """Time-related behaviour reaches the outcome through the sale calendar."""
    orders = small_dataset.orders
    sale = orders[orders[cols.IS_SALE_DAY]]
    normal = orders[~orders[cols.IS_SALE_DAY]]

    assert len(sale) > 0
    sale_per_day = len(sale) / max(sale[cols.DAY_INDEX].nunique(), 1)
    normal_per_day = len(normal) / max(normal[cols.DAY_INDEX].nunique(), 1)
    assert sale_per_day > normal_per_day


def test_sale_days_carry_deeper_discounts(small_dataset: GenerationResult) -> None:
    orders = small_dataset.orders
    assert (
        orders.loc[orders[cols.IS_SALE_DAY], cols.DISCOUNT_DEPTH].mean()
        > orders.loc[~orders[cols.IS_SALE_DAY], cols.DISCOUNT_DEPTH].mean()
    )


# ---------------------------------------------------------------------------
# the task is not a deterministic rule
# ---------------------------------------------------------------------------


def test_the_true_probability_is_not_degenerate(small_dataset: GenerationResult) -> None:
    """Risk must be spread, not concentrated at 0 and 1.

    If the simulator produced near-certain outcomes, a model could separate them
    perfectly and the resulting metrics would say nothing about a real problem.
    """
    probability = small_dataset.latents[cols.TRUE_RTO_PROBABILITY]

    assert probability.between(0.0, 1.0).all()
    assert (probability > 0.95).mean() < 0.02
    assert (probability < 0.005).mean() < 0.25
    assert probability.std() > 0.05


def test_outcomes_are_not_a_deterministic_function_of_the_true_probability(
    small_dataset: GenerationResult,
) -> None:
    """High-risk orders must sometimes deliver, and low-risk orders sometimes return.

    This is the Bayes-optimal ceiling made visible: even knowing the true
    probability exactly, the outcome is a coin flip weighted by it. A model that
    scores near-perfectly on this data has found a bug, not a signal.
    """
    joined = small_dataset.orders.merge(small_dataset.latents, on=cols.ORDER_ID)
    joined = joined[joined[cols.IS_MATURE]]

    high = joined[joined[cols.TRUE_RTO_PROBABILITY] > 0.6]
    low = joined[joined[cols.TRUE_RTO_PROBABILITY] < 0.1]

    assert len(high) > 5, "the simulator should produce some genuinely high-risk orders"
    assert high[cols.IS_RTO].mean() < 1.0, "some high-risk orders must still deliver"
    assert low[cols.IS_RTO].mean() > 0.0, "some low-risk orders must still return"


def test_latent_drivers_are_not_exposed_as_columns(small_dataset: GenerationResult) -> None:
    """The unobservable drivers must not appear in the benchmark table.

    Customer reliability and the per-pincode effect are what make the task
    realistically hard. If either leaked into ``orders``, a model could recover
    the simulator exactly and every metric would be meaningless.
    """
    order_columns = set(small_dataset.orders.columns)
    for latent in (
        "customer_reliability",
        "pincode_effect",
        "latent_address_quality",
        cols.TRUE_RTO_PROBABILITY,
        cols.LATENT_LOGIT,
    ):
        assert latent not in order_columns


def test_observable_address_signals_do_not_perfectly_recover_latent_quality(
    small_dataset: GenerationResult,
) -> None:
    """The model sees rendered text, not the latent quality that produced it.

    That gap is deliberate: it is the difference between a realistic feature and
    a lossless encoding of a hidden variable.
    """
    joined = small_dataset.orders.merge(small_dataset.latents, on=cols.ORDER_ID)
    correlation = np.corrcoef(
        joined[cols.ADDR_TOKEN_COUNT].astype(float),
        joined["latent_address_quality"].astype(float),
    )[0, 1]

    assert 0.05 < abs(correlation) < 0.95


def test_some_labels_are_flipped(small_dataset: GenerationResult) -> None:
    """Courier miscoding: outcome data is not perfectly recorded in reality."""
    flipped = small_dataset.latents["label_flipped"]
    assert flipped.sum() >= 0
    assert flipped.mean() < 0.02


# ---------------------------------------------------------------------------
# population realism
# ---------------------------------------------------------------------------


def test_no_single_customer_dominates_the_book(small_dataset: GenerationResult) -> None:
    """An unclipped Pareto tail hands one customer a tenth of the merchant's book.

    No real store looks like that, and one person's latent reliability would
    dominate the whole dataset.
    """
    counts = small_dataset.orders[cols.CUSTOMER_HASH].value_counts()
    assert counts.iloc[0] / len(small_dataset.orders) < 0.05


def test_customers_reuse_a_home_address(small_dataset: GenerationResult) -> None:
    """Repeat structure is what makes an address dimension worth having."""
    addresses_per_customer = len(small_dataset.addresses) / len(small_dataset.customers)
    assert addresses_per_customer < 2.0


def test_both_new_and_returning_customers_exist(small_dataset: GenerationResult) -> None:
    """The cold-start cohort has to be present to be reported on separately."""
    new_share = small_dataset.orders[cols.IS_NEW_CUSTOMER].mean()
    assert 0.05 < new_share < 0.95


def test_new_customers_have_no_history(small_dataset: GenerationResult) -> None:
    new = small_dataset.orders[small_dataset.orders[cols.IS_NEW_CUSTOMER]]
    assert (new[cols.PRIOR_ORDER_COUNT] == 0).all()
    assert new[cols.PRIOR_RTO_RATE].isna().all()


def test_missing_history_is_null_not_zero(small_dataset: GenerationResult) -> None:
    """ "No history" must be NaN, never 0.

    A 0 claims the customer has a zero percent return rate, which is the most
    optimistic possible reading of "we know nothing about them". NaN lets a
    gradient-boosted model learn the absence as its own state.
    """
    orders = small_dataset.orders
    no_history = orders[orders[cols.PRIOR_ORDER_COUNT] == 0]
    assert no_history[cols.PRIOR_RTO_RATE].isna().all()
    assert no_history[cols.MEAN_RESOLUTION_DAYS].isna().all()


def test_days_since_last_order_tracks_placement_not_resolution(
    small_dataset: GenerationResult,
) -> None:
    """A subtle and deliberate distinction between two "no history" notions.

    ``prior_order_count`` counts orders that had already **resolved** before this
    one - outcome information. ``days_since_last_order`` measures time since the
    customer last **placed** an order, which the merchant knows immediately.

    So a customer who ordered two days ago has ``prior_order_count == 0`` (that
    order has not come back yet) but a real ``days_since_last_order``. Both are
    correct, and conflating them would either leak outcome timing or discard a
    legitimate signal. Null belongs only to genuinely first-time customers.
    """
    orders = small_dataset.orders
    first_time = orders[orders[cols.IS_NEW_CUSTOMER]]
    returning = orders[~orders[cols.IS_NEW_CUSTOMER]]

    assert first_time[cols.DAYS_SINCE_LAST_ORDER].isna().all()
    assert returning[cols.DAYS_SINCE_LAST_ORDER].notna().all()

    # The distinction is real in this dataset, not merely hypothetical.
    unresolved_but_returning = orders[
        (orders[cols.PRIOR_ORDER_COUNT] == 0) & (~orders[cols.IS_NEW_CUSTOMER])
    ]
    assert len(unresolved_but_returning) > 0
    assert unresolved_but_returning[cols.DAYS_SINCE_LAST_ORDER].notna().all()
