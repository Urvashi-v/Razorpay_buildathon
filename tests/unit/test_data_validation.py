"""The validator catches each failure mode it claims to catch.

A validator nobody has tried to break is a validator that passes everything. So
each test here takes a *valid* dataset, introduces exactly one defect, and
asserts the validator notices - which is the only way to know the check is wired
up rather than merely written down.

Every defect below is one that would flatter a model if it went unnoticed, which
is why these particular ones exist.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from rto_sentinel.configuration.schemas import GeneratorConfig
from rto_sentinel.data import schema as cols
from rto_sentinel.data.generator import GenerationResult
from rto_sentinel.data.validation import (
    DataValidationError,
    validate_delivery_events,
    validate_orders,
)


@pytest.fixture
def orders(small_dataset: GenerationResult) -> pd.DataFrame:
    return small_dataset.orders.copy()


def _errors(frame: pd.DataFrame, config: GeneratorConfig) -> list[str]:
    return validate_orders(frame, config=config, strict=False).errors


def test_a_clean_dataset_passes(
    small_dataset: GenerationResult, generator_config: GeneratorConfig
) -> None:
    report = validate_orders(
        small_dataset.orders,
        config=generator_config,
        customers=small_dataset.customers,
        strict=False,
    )
    assert report.ok, report.render()
    assert report.checks_run >= 10


# ---------------------------------------------------------------------------
# nulls, duplicates, ranges
# ---------------------------------------------------------------------------


def test_nulls_in_a_non_nullable_column_are_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    orders.loc[orders.index[0], cols.PINCODE] = None
    assert any("null values" in error for error in _errors(orders, generator_config))


def test_duplicate_order_ids_are_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    """A duplicated order inflates a customer's history and can straddle splits."""
    orders.loc[orders.index[1], cols.ORDER_ID] = orders.loc[orders.index[0], cols.ORDER_ID]
    assert any("duplicate identifiers" in error for error in _errors(orders, generator_config))


def test_negative_order_values_are_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    orders.loc[orders.index[0], cols.ORDER_VALUE_INR] = -100.0
    assert any("at or below zero" in error for error in _errors(orders, generator_config))


def test_negative_discounts_are_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    orders.loc[orders.index[0], cols.DISCOUNT_INR] = -5.0
    assert any("negative values" in error for error in _errors(orders, generator_config))


def test_zero_item_counts_are_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    orders.loc[orders.index[0], cols.ITEM_COUNT] = 0
    assert any("fewer than one item" in error for error in _errors(orders, generator_config))


def test_impossible_prior_counts_are_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    """More prior RTOs than prior orders is arithmetically impossible."""
    orders.loc[orders.index[0], cols.PRIOR_ORDER_COUNT] = 1
    orders.loc[orders.index[0], cols.PRIOR_RTO_COUNT] = 5
    assert any("exceed prior orders" in error for error in _errors(orders, generator_config))


def test_out_of_range_hours_are_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    orders.loc[orders.index[0], cols.HOUR_OF_DAY] = 25
    assert any("outside 0-23" in error for error in _errors(orders, generator_config))


# ---------------------------------------------------------------------------
# categorical domains
# ---------------------------------------------------------------------------


def test_invalid_categorical_values_are_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    """A typo creates a level a tree model will happily split on."""
    orders.loc[orders.index[0], cols.PAYMENT_METHOD] = "cash"
    assert any("invalid values" in error for error in _errors(orders, generator_config))


def test_unknown_category_is_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    orders.loc[orders.index[0], cols.CATEGORY] = "spacecraft"
    assert any("not in the catalogue" in error for error in _errors(orders, generator_config))


def test_unknown_courier_is_caught(orders: pd.DataFrame, generator_config: GeneratorConfig) -> None:
    orders.loc[orders.index[0], cols.COURIER_PARTNER] = "courier_zzz"
    assert any("not in the courier list" in error for error in _errors(orders, generator_config))


# ---------------------------------------------------------------------------
# impossible timestamps
# ---------------------------------------------------------------------------


def test_dispatch_before_order_is_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    row = orders.index[orders[cols.DISPATCHED_AT].notna()][0]
    orders.loc[row, cols.DISPATCHED_AT] = orders.loc[row, cols.ORDERED_AT] - timedelta(days=1)
    assert any(
        "dispatched before the order" in error for error in _errors(orders, generator_config)
    )


def test_attempt_before_dispatch_is_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    row = orders.index[orders[cols.FIRST_ATTEMPT_AT].notna()][0]
    orders.loc[row, cols.FIRST_ATTEMPT_AT] = orders.loc[row, cols.DISPATCHED_AT] - timedelta(days=1)
    assert any("attempted before dispatch" in error for error in _errors(orders, generator_config))


def test_resolution_before_order_is_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    """The check that matters most: it would invert the arrow of time."""
    row = orders.index[orders[cols.RESOLVED_AT].notna()][0]
    orders.loc[row, cols.RESOLVED_AT] = orders.loc[row, cols.ORDERED_AT] - timedelta(days=2)
    assert any("resolved before the order" in error for error in _errors(orders, generator_config))


# ---------------------------------------------------------------------------
# outcome consistency and label maturity
# ---------------------------------------------------------------------------


def test_pending_order_with_a_resolution_timestamp_is_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    row = orders.index[orders[cols.OUTCOME] == "pending"][0]
    orders.loc[row, cols.RESOLVED_AT] = orders.loc[row, cols.ORDERED_AT] + timedelta(days=3)
    assert any("pending orders carry a resolution" in e for e in _errors(orders, generator_config))


def test_terminal_order_without_a_resolution_timestamp_is_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    row = orders.index[orders[cols.OUTCOME] == "delivered"][0]
    orders.loc[row, cols.RESOLVED_AT] = None
    assert any("no resolution timestamp" in error for error in _errors(orders, generator_config))


def test_label_disagreeing_with_outcome_is_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    row = orders.index[orders[cols.OUTCOME] == "delivered"][0]
    orders.loc[row, cols.IS_RTO] = True
    assert any("disagrees with outcome" in error for error in _errors(orders, generator_config))


def test_an_optimistically_labelled_immature_order_is_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    """The single most important check in this module.

    Labelling an unresolved order "delivered" is how a benchmark manufactures
    optimism: it converts every not-yet-returned order into a clean negative, and
    every model trained on it looks better than it is.
    """
    row = orders.index[~orders[cols.IS_MATURE].astype(bool)][0]
    orders.loc[row, cols.IS_RTO] = False
    assert any(
        "immature orders carry a label" in error for error in _errors(orders, generator_config)
    )


def test_a_mature_order_without_a_label_is_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    row = orders.index[orders[cols.IS_MATURE].astype(bool)][0]
    orders.loc[row, cols.IS_RTO] = None
    assert any("no label" in error for error in _errors(orders, generator_config))


def test_resolution_beyond_the_maturity_window_is_caught(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    row = orders.index[orders[cols.IS_MATURE].astype(bool)][0]
    orders.loc[row, cols.MATURITY_DAYS] = generator_config.label_maturity.max_resolution_days + 5
    assert any("later than the configured maximum" in e for e in _errors(orders, generator_config))


# ---------------------------------------------------------------------------
# customer / order relationships
# ---------------------------------------------------------------------------


def test_an_order_for_an_unknown_customer_is_caught(
    small_dataset: GenerationResult, generator_config: GeneratorConfig
) -> None:
    orders = small_dataset.orders.copy()
    orders.loc[orders.index[0], cols.CUSTOMER_HASH] = "0" * 32
    report = validate_orders(
        orders, config=generator_config, customers=small_dataset.customers, strict=False
    )
    assert any("customer that does not exist" in error for error in report.errors)


def test_orders_never_precede_their_customer(
    small_dataset: GenerationResult, generator_config: GeneratorConfig
) -> None:
    """Signup is anchored to the first order, so this should never fire."""
    report = validate_orders(
        small_dataset.orders,
        config=generator_config,
        customers=small_dataset.customers,
        strict=False,
    )
    assert not any("precede their customer" in warning for warning in report.warnings)


# ---------------------------------------------------------------------------
# base-rate drift
# ---------------------------------------------------------------------------


def test_base_rate_drift_is_an_error_in_strict_mode(
    orders: pd.DataFrame, generator_config: GeneratorConfig
) -> None:
    """A generator that has drifted off its published anchors is not the benchmark.

    Simulated by flipping every COD label, which is a caricature - but the check
    it exercises is the one that catches a real, gradual drift after a config edit.
    """
    cod_mature = orders[cols.IS_COD].astype(bool) & orders[cols.IS_MATURE].astype(bool)
    orders.loc[cod_mature, cols.IS_RTO] = True
    orders.loc[cod_mature, cols.OUTCOME] = "rto"

    strict_report = validate_orders(orders, config=generator_config, strict=True)
    assert any("exceeds tolerance" in error for error in strict_report.errors)

    lenient_report = validate_orders(orders, config=generator_config, strict=False)
    assert any("exceeds tolerance" in warning for warning in lenient_report.warnings)


# ---------------------------------------------------------------------------
# reporting behaviour
# ---------------------------------------------------------------------------


def test_missing_columns_stop_the_cascade(generator_config: GeneratorConfig) -> None:
    """One clear error beats forty downstream KeyErrors."""
    report = validate_orders(pd.DataFrame({"order_id": ["ORD-1"]}), config=generator_config)
    assert len(report.errors) == 1
    assert "missing columns" in report.errors[0]


def test_raise_for_errors_raises(orders: pd.DataFrame, generator_config: GeneratorConfig) -> None:
    orders.loc[orders.index[0], cols.ORDER_VALUE_INR] = -1.0
    report = validate_orders(orders, config=generator_config, strict=False)
    with pytest.raises(DataValidationError, match="failed validation"):
        report.raise_for_errors()


# ---------------------------------------------------------------------------
# delivery events
# ---------------------------------------------------------------------------


def test_clean_delivery_events_pass(small_dataset: GenerationResult) -> None:
    report = validate_delivery_events(small_dataset.delivery_events, small_dataset.orders)
    assert report.ok, report.render()


def test_orphaned_delivery_events_are_caught(small_dataset: GenerationResult) -> None:
    events = small_dataset.delivery_events.copy()
    events.loc[events.index[0], "order_id"] = "ORD-does-not-exist"
    report = validate_delivery_events(events, small_dataset.orders)
    assert any("unknown order" in error for error in report.errors)


def test_out_of_order_events_are_caught(small_dataset: GenerationResult) -> None:
    """An event trail that runs backwards makes reconstruction meaningless."""
    events = small_dataset.delivery_events.copy()
    target = events.index[events["sequence"] == 2][0]
    events.loc[target, "occurred_at"] = events.loc[target, "occurred_at"] - timedelta(days=30)
    report = validate_delivery_events(events, small_dataset.orders)
    assert any("before the previous event" in error for error in report.errors)


def test_an_order_without_a_placement_event_is_caught(small_dataset: GenerationResult) -> None:
    events = small_dataset.delivery_events.copy()
    events = events[~((events["order_id"] == "ORD-00000001") & (events["sequence"] == 1))]
    report = validate_delivery_events(events, small_dataset.orders)
    assert any("no order_placed event" in error for error in report.errors)
