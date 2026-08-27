"""The split protocol holds: temporal, grouped, sealed - and unbiased.

The last of those is the one worth dwelling on. An earlier implementation
satisfied "temporal" and "grouped" perfectly and was still wrong, because it
selected validation and test customers by a rule correlated with their
behaviour: assign each customer to their earliest split and drop their later
orders, and validation ends up almost entirely cold-start. Two tests below
(``test_split_composition_is_not_selection_biased`` and its companion) exist
specifically to stop that regressing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rto_sentinel.configuration.schemas import SplitsConfig
from rto_sentinel.contracts.enums import DatasetSplit
from rto_sentinel.data import schema as cols
from rto_sentinel.data.generator import GenerationResult
from rto_sentinel.data.splits import (
    SealBrokenError,
    TestSetSeal,
    assign_splits,
    customer_pool,
    customers_are_disjoint,
    drift_window_mask,
    splits_are_time_ordered,
)

MODELLING = (DatasetSplit.TRAIN.value, DatasetSplit.VALIDATION.value, DatasetSplit.TEST.value)


@pytest.fixture
def split_orders(small_dataset: GenerationResult, splits_config: SplitsConfig):
    orders = small_dataset.orders.copy()
    orders[cols.SPLIT] = assign_splits(orders, splits_config).labels
    return orders


# ---------------------------------------------------------------------------
# rule 1: temporal
# ---------------------------------------------------------------------------


def test_every_split_is_populated(split_orders, splits_config: SplitsConfig) -> None:
    counts = assign_splits(split_orders, splits_config).as_dict()
    for name in MODELLING:
        assert counts[name] > 0, f"the {name} split is empty"


def test_splits_do_not_overlap_in_time(split_orders) -> None:
    """Train ends before validation begins, and validation before test."""
    assert splits_are_time_ordered(split_orders)


def test_split_day_ranges_match_the_configuration(
    split_orders, splits_config: SplitsConfig
) -> None:
    for name, (start, end) in (
        (DatasetSplit.TRAIN.value, splits_config.temporal.train_days),
        (DatasetSplit.VALIDATION.value, splits_config.temporal.validation_days),
        (DatasetSplit.TEST.value, splits_config.temporal.test_days),
    ):
        days = split_orders.loc[split_orders[cols.SPLIT] == name, cols.DAY_INDEX]
        assert days.min() >= start
        assert days.max() <= end


# ---------------------------------------------------------------------------
# rule 2: grouped
# ---------------------------------------------------------------------------


def test_no_customer_appears_in_two_splits(split_orders) -> None:
    assert customers_are_disjoint(split_orders)


def test_pool_assignment_is_deterministic(splits_config: SplitsConfig) -> None:
    """Same identifier, same pool - on every machine and every run."""
    for identifier in ("abc123", "deadbeef", "0" * 32):
        assert customer_pool(identifier, splits_config) == customer_pool(identifier, splits_config)


def test_pool_assignment_respects_the_configured_shares(splits_config: SplitsConfig) -> None:
    """The hash must spread customers roughly in the configured proportions."""
    from collections import Counter

    counts = Counter(customer_pool(f"customer-{i:06d}", splits_config) for i in range(6000))
    for name, share in splits_config.group.pool_shares.items():
        observed = counts[name] / 6000
        assert abs(observed - share) < 0.03, f"{name}: expected ~{share}, got {observed:.3f}"


def test_pool_assignment_ignores_behaviour(
    small_dataset: GenerationResult, splits_config: SplitsConfig
) -> None:
    """Pools must depend on the identifier alone.

    If pool membership correlated with order count, the splits would differ
    systematically in how much history their customers have - which is exactly
    the bias the pooling scheme replaced.
    """
    orders = small_dataset.orders.copy()
    orders["pool"] = orders[cols.CUSTOMER_HASH].map(lambda h: customer_pool(h, splits_config))
    per_customer = orders.groupby([cols.CUSTOMER_HASH, "pool"], sort=False).size().reset_index()
    means = per_customer.groupby("pool")[0].mean()
    assert means.max() / means.min() < 1.35


# ---------------------------------------------------------------------------
# the bias regression guard
# ---------------------------------------------------------------------------


def test_split_composition_is_not_selection_biased(split_orders) -> None:
    """Validation and test must not be dominated by cold-start customers.

    The earlier "earliest split wins" scheme produced 43% first-time customers in
    train against 87% and 88% in validation and test. A threshold fitted on that
    is a cold-start threshold, and a test score measured on it is a cold-start
    score wearing a general-performance label.

    The bound here is loose because a genuine temporal effect remains and should:
    later windows naturally contain more returning customers, since the customer
    base matures over the horizon. What must not reappear is the *selection*
    effect, which pointed the other way and was several times larger.
    """
    shares = {
        name: split_orders.loc[split_orders[cols.SPLIT] == name, cols.IS_NEW_CUSTOMER].mean()
        for name in MODELLING
    }
    for name, share in shares.items():
        assert share < 0.75, f"{name} is {share:.0%} cold-start customers: {shares}"


def test_later_splits_have_more_history_not_less(split_orders) -> None:
    """The remaining difference is temporal maturation, and points the right way.

    Under the biased scheme validation customers had *less* history than training
    customers, which is impossible under an honest temporal split - later orders
    come from a more established customer base, not a fresher one.
    """
    means = {
        name: split_orders.loc[split_orders[cols.SPLIT] == name, cols.PRIOR_ORDER_COUNT].mean()
        for name in MODELLING
    }
    assert means[DatasetSplit.VALIDATION.value] >= means[DatasetSplit.TRAIN.value]
    assert means[DatasetSplit.TEST.value] >= means[DatasetSplit.TRAIN.value]


# ---------------------------------------------------------------------------
# label maturity interaction
# ---------------------------------------------------------------------------


def test_immature_rows_never_join_a_modelling_split(split_orders) -> None:
    immature = split_orders[~split_orders[cols.IS_MATURE].astype(bool)]
    assert (immature[cols.SPLIT] == DatasetSplit.EXCLUDED_IMMATURE.value).all()


def test_every_modelling_row_has_a_label(split_orders) -> None:
    modelling = split_orders[split_orders[cols.SPLIT].isin(MODELLING)]
    assert modelling[cols.IS_RTO].notna().all()


def test_the_cost_of_the_protocol_is_reported(
    small_dataset: GenerationResult, splits_config: SplitsConfig
) -> None:
    """Dropped rows must be counted, not silently absent.

    A validation set that quietly lost most of its rows is not something anyone
    should have to discover for themselves.
    """
    assignment = assign_splits(small_dataset.orders, splits_config)
    counts = assignment.as_dict()
    total = sum(counts[key] for key in counts if key != "customers_truncated")
    assert total == len(small_dataset.orders)
    assert counts["excluded_group_protocol"] > 0
    assert assignment.n_modelling < len(small_dataset.orders)


# ---------------------------------------------------------------------------
# drift window
# ---------------------------------------------------------------------------


def test_drift_window_selects_the_final_days(
    small_dataset: GenerationResult, splits_config: SplitsConfig
) -> None:
    mask = drift_window_mask(small_dataset.orders, splits_config)
    selected = small_dataset.orders[mask]
    span = selected[cols.DAY_INDEX].max() - selected[cols.DAY_INDEX].min()
    assert span < splits_config.drift_window.final_days
    assert len(selected) > 0


# ---------------------------------------------------------------------------
# rule 3: the seal
# ---------------------------------------------------------------------------


def test_the_seal_permits_exactly_one_scoring_run(tmp_path: Path) -> None:
    seal = TestSetSeal(tmp_path / "seal.json")
    assert not seal.is_broken

    seal.claim(model_name="lightgbm_isotonic", config_fingerprint="abc", dataset_run_id="run1")
    assert seal.is_broken

    with pytest.raises(SealBrokenError, match="already been scored"):
        seal.claim(model_name="lightgbm_isotonic", config_fingerprint="abc", dataset_run_id="run1")


def test_the_seal_records_what_was_scored(tmp_path: Path) -> None:
    seal = TestSetSeal(tmp_path / "seal.json")
    seal.claim(model_name="rung4", config_fingerprint="fingerprint", dataset_run_id="run-7")

    receipt = seal.read_receipt()
    assert receipt is not None
    assert receipt["model_name"] == "rung4"
    assert receipt["config_fingerprint"] == "fingerprint"
    assert receipt["dataset_run_id"] == "run-7"
    assert "scored_at" in receipt


def test_breaking_the_seal_requires_deleting_the_receipt(tmp_path: Path) -> None:
    """Re-scoring is possible, but only as a deliberate, visible act."""
    path = tmp_path / "seal.json"
    seal = TestSetSeal(path)
    seal.claim(model_name="rung4", config_fingerprint="f", dataset_run_id="r")

    path.unlink()
    assert not seal.is_broken
    seal.claim(model_name="rung4", config_fingerprint="f", dataset_run_id="r")
