"""Leakage tests for the feature pipeline.

The Phase 2 suite (``test_leakage_suite.py``) proves the *dataset* is honest.
This one proves the *features built from it* are, which is a different and harder
claim: a leak-free dataset can still be turned into a leaking design matrix by a
single careless aggregate.

THE STRATEGY: REWIND THE WORLD, REBUILD, COMPARE
================================================
Most tests here work the same way. Pick a cutoff. Rebuild the feature matrix on a
frame where every order that had not yet resolved by that cutoff has its outcome
and resolution timestamp blanked - the world exactly as it looked at that instant.
Then compare, for rows ordered *before* the cutoff, against the matrix built on
the full data.

Anything that changed was reading the future.

This is stronger than inspecting the code, because it tests the values a model
would actually be trained on. It also scales: it checks all 54 features at once
and will check the 55th automatically.

A NOTE ON CUTOFFS
=================
A single cutoff only exposes leaks that *straddle* it - the consuming row must be
before it and the leaked outcome after it. So every rewind test runs at several.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rto_sentinel.configuration.schemas import FeaturesConfig, GeneratorConfig, SplitsConfig
from rto_sentinel.data import schema as cols
from rto_sentinel.features import FeaturePipeline, ModelingDataset, TestSetAccessError
from rto_sentinel.features.spec import Availability, ObservationPoint

pytestmark = pytest.mark.leakage

MODELLING = ("train", "validation", "test")
CUTOFF_QUANTILES = (0.35, 0.55, 0.75)

#: Observation points whose value depends on a delivery OUTCOME. Features at
#: these points must change when outcomes are hidden; features at any other point
#: must not. Both directions are checked - see
#: ``test_feature_timestamp_integrity``.
OUTCOME_DEPENDENT = {
    ObservationPoint.PRIOR_ORDERS_RESOLVED,
    ObservationPoint.POPULATION_RESOLVED,
}


def _pipeline(
    features_config: FeaturesConfig, generator_config: GeneratorConfig
) -> FeaturePipeline:
    return FeaturePipeline(features_config, generator_config)


def _rewind(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """The order table as it looked at ``cutoff``.

    Every order is still present - the merchant knew it had been placed - but any
    order that had not resolved yet has its outcome and resolution timestamp
    blanked. Rewinding rather than deleting matters: deleting unresolved rows
    would remove from the comparison exactly the rows most likely to leak.
    """
    rewound = frame.copy()
    unresolved = rewound[cols.RESOLVED_AT].isna() | (rewound[cols.RESOLVED_AT] >= cutoff)
    rewound.loc[unresolved, cols.RESOLVED_AT] = pd.NaT
    rewound.loc[unresolved, cols.IS_RTO] = None
    rewound.loc[unresolved, cols.OUTCOME] = "pending"
    rewound.loc[unresolved, cols.MATURITY_DAYS] = np.nan
    rewound.loc[unresolved, cols.IS_MATURE] = False
    return rewound


def _compare(
    full: pd.DataFrame, rewound: pd.DataFrame, rows: pd.Index, columns: list[str]
) -> list[str]:
    """Feature names whose values differ on ``rows``. Empty means agreement."""
    changed: list[str] = []
    for column in columns:
        left = full.loc[rows, column]
        right = rewound.loc[rows, column]
        if left.dtype.name == "category" or left.dtype == object:
            if not left.astype(str).equals(right.astype(str)):
                changed.append(column)
            continue
        left_numeric = pd.to_numeric(left, errors="coerce").astype("float64")
        right_numeric = pd.to_numeric(right, errors="coerce").astype("float64")
        both_null = left_numeric.isna() & right_numeric.isna()
        close = np.isclose(
            left_numeric.fillna(0.0), right_numeric.fillna(0.0), rtol=1e-9, atol=1e-9
        )
        if not bool((both_null | close).all()):
            changed.append(column)
    return changed


# ---------------------------------------------------------------------------
# 1. test_no_future_outcome_features
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantile", CUTOFF_QUANTILES)
def test_no_future_outcome_features(
    feature_frame: pd.DataFrame,
    features_config: FeaturesConfig,
    generator_config: GeneratorConfig,
    quantile: float,
) -> None:
    """No feature may change when future outcomes are hidden.

    The headline leakage test, run over the whole design matrix at once. If any
    of the 54 features reads an outcome that had not happened yet, its value moves
    when that outcome is removed, and this fails naming the feature.
    """
    pipeline = _pipeline(features_config, generator_config)
    cutoff = pd.Timestamp(feature_frame[cols.ORDERED_AT].quantile(quantile))

    full = pipeline.build(feature_frame).matrix
    rewound = pipeline.build(_rewind(feature_frame, cutoff)).matrix

    past_rows = feature_frame.index[feature_frame[cols.ORDERED_AT] < cutoff]
    assert len(past_rows) > 100, "the cutoff should leave a meaningful number of past rows"

    changed = _compare(full, rewound, past_rows, list(full.columns))
    assert not changed, (
        f"features changed when outcomes at or after {cutoff} were hidden: {changed}. "
        "They are reading the future."
    )


def test_the_rewind_check_can_actually_fail(
    feature_frame: pd.DataFrame,
    features_config: FeaturesConfig,
    generator_config: GeneratorConfig,
) -> None:
    """The guard must be capable of failing, or it is decoration.

    A deliberately leaky feature - the customer's return rate computed over orders
    *placed* earlier rather than *resolved* earlier - is injected and the same
    comparison is applied. It must be caught.
    """
    cutoff = pd.Timestamp(feature_frame[cols.ORDERED_AT].quantile(0.55))

    def leaky(frame: pd.DataFrame) -> pd.DataFrame:
        ordered = frame.sort_values([cols.CUSTOMER_HASH, cols.ORDERED_AT], kind="stable")
        shifted = ordered.groupby(cols.CUSTOMER_HASH, sort=False)[cols.IS_RTO].shift(1)
        rate = (
            shifted.astype("float64")
            .groupby(ordered[cols.CUSTOMER_HASH])
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
        return pd.DataFrame({"leaky_prior_rate": rate.reindex(frame.index)})

    full = leaky(feature_frame)
    rewound = leaky(_rewind(feature_frame, cutoff))
    past_rows = feature_frame.index[feature_frame[cols.ORDERED_AT] < cutoff]

    changed = _compare(full, rewound, past_rows, ["leaky_prior_rate"])
    assert changed == ["leaky_prior_rate"], (
        "the rewind comparison failed to catch a known-leaky feature, so it proves "
        "nothing about the real ones"
    )


# ---------------------------------------------------------------------------
# 2. test_temporal_ordering
# ---------------------------------------------------------------------------


def test_temporal_ordering(modeling_dataset: ModelingDataset) -> None:
    """Every training order precedes every validation order, and so on.

    Checked on the built dataset rather than on the config, because a correct
    configuration applied by a buggy assignment is still a leak.

    Reads the test split's *timestamps* without unsealing it. Dates are not
    outcomes; the label is what stays sealed.
    """
    bounds = {}
    for name in MODELLING:
        mask = modeling_dataset.splits == name
        times = modeling_dataset.ordered_at.loc[mask]
        assert not times.empty, f"the {name} split is empty"
        bounds[name] = (times.min(), times.max())

    assert bounds["train"][1] < bounds["validation"][0], (
        "a training order is not strictly earlier than every validation order"
    )
    assert bounds["validation"][1] < bounds["test"][0], (
        "a validation order is not strictly earlier than every test order"
    )


def test_split_day_ranges_match_the_configuration(
    modeling_dataset: ModelingDataset, splits_config: SplitsConfig
) -> None:
    windows = {
        "train": splits_config.temporal.train_days,
        "validation": splits_config.temporal.validation_days,
        "test": splits_config.temporal.test_days,
    }
    for name, (start, end) in windows.items():
        days = modeling_dataset.day_index.loc[modeling_dataset.splits == name]
        assert days.min() >= start
        assert days.max() <= end


# ---------------------------------------------------------------------------
# 3. test_test_set_isolation
# ---------------------------------------------------------------------------


def test_test_set_isolation(
    feature_frame: pd.DataFrame,
    features_config: FeaturesConfig,
    generator_config: GeneratorConfig,
    splits_config: SplitsConfig,
) -> None:
    """No test-set information reaches a training row's features.

    The pipeline is handed the whole order table, including test rows. This test
    verifies the claim that doing so is safe rather than asserting it: features
    are rebuilt on a frame truncated to the training window only, and every
    training row must come out identical.

    If any feature aggregated globally - a target-encoded pincode rate computed
    over the whole dataset, say - the truncated build would differ and this fails.
    """
    pipeline = _pipeline(features_config, generator_config)
    train_end = splits_config.temporal.train_days[1]

    full = pipeline.build(feature_frame).matrix
    training_window = feature_frame[feature_frame[cols.DAY_INDEX] <= train_end]
    truncated = pipeline.build(training_window).matrix

    changed = _compare(full, truncated, training_window.index, list(full.columns))
    assert not changed, (
        f"training-row features changed when validation and test rows were removed: {changed}. "
        "Something is aggregating globally instead of as-of."
    )


def test_the_test_split_is_sealed(modeling_dataset: ModelingDataset) -> None:
    """Reaching the test set requires an explicit, recorded decision."""
    assert modeling_dataset.test_is_sealed
    with pytest.raises(TestSetAccessError, match="sealed"):
        _ = modeling_dataset.test


def test_unsealing_requires_a_written_reason(
    small_dataset,
    features_config: FeaturesConfig,
    generator_config: GeneratorConfig,
    splits_config: SplitsConfig,
    split_labels: pd.Series,
) -> None:
    """Built fresh, so the shared fixture stays sealed for every other test."""
    from rto_sentinel.features import build_modeling_dataset

    dataset = build_modeling_dataset(
        small_dataset,
        features_config=features_config,
        generator_config=generator_config,
        splits_config=splits_config,
        split_labels=split_labels,
    )

    with pytest.raises(ValueError, match="written reason"):
        dataset.unseal_test(reason="   ")

    dataset.unseal_test(reason="final sealed evaluation, threshold already fixed on validation")
    assert not dataset.test_is_sealed
    assert dataset.unseal_reason is not None
    assert len(dataset.test) > 0


def test_customers_do_not_cross_splits(modeling_dataset: ModelingDataset) -> None:
    """A customer in training must not reappear in validation or test."""
    per_split = {
        name: set(modeling_dataset.customer_hashes.loc[modeling_dataset.splits == name])
        for name in MODELLING
    }
    assert not (per_split["train"] & per_split["validation"])
    assert not (per_split["train"] & per_split["test"])
    assert not (per_split["validation"] & per_split["test"])


# ---------------------------------------------------------------------------
# 4. test_feature_timestamp_integrity
# ---------------------------------------------------------------------------


def test_feature_timestamp_integrity(
    feature_frame: pd.DataFrame,
    features_config: FeaturesConfig,
    generator_config: GeneratorConfig,
) -> None:
    """Every feature's declared observation point matches how it behaves.

    Checked in BOTH directions, which is what makes it more than a spelling test:

    * A feature declared independent of outcomes must NOT change when outcomes
      are hidden. If it does, the declaration is wrong and the feature leaks.
    * A feature declared outcome-dependent MUST change for at least some rows.
      If it never does, either the declaration is wrong or the feature is not
      computing what it claims - both worth knowing.
    """
    pipeline = _pipeline(features_config, generator_config)
    feature_set = pipeline.feature_set
    cutoff = pd.Timestamp(feature_frame[cols.ORDERED_AT].quantile(0.5))

    full = pipeline.build(feature_frame).matrix
    rewound = pipeline.build(_rewind(feature_frame, cutoff)).matrix

    # Compare over ALL rows here, not just past ones: a row after the cutoff
    # legitimately loses history when the world is rewound, and that is precisely
    # the movement an outcome-dependent feature should show.
    all_rows = feature_frame.index
    changed = set(_compare(full, rewound, all_rows, list(full.columns)))

    outcome_independent_that_moved = [
        spec.name
        for spec in feature_set
        if spec.observation_point not in OUTCOME_DEPENDENT and spec.name in changed
    ]
    assert not outcome_independent_that_moved, (
        "features declared independent of delivery outcomes changed when outcomes were "
        f"hidden: {outcome_independent_that_moved}. Their observation point is wrong."
    )

    outcome_dependent_that_did_not_move = [
        spec.name
        for spec in feature_set
        if spec.observation_point in OUTCOME_DEPENDENT and spec.name not in changed
    ]
    assert not outcome_dependent_that_did_not_move, (
        "features declared outcome-dependent did not change when outcomes were hidden: "
        f"{outcome_dependent_that_did_not_move}. Either the declaration is wrong or the "
        "feature is not computing what it claims."
    )


def test_every_feature_declares_availability_at_order_time(
    features_config: FeaturesConfig, generator_config: GeneratorConfig
) -> None:
    """The cheapest leak check: it needs no data and runs first."""
    feature_set = _pipeline(features_config, generator_config).feature_set
    unavailable = feature_set.unavailable_at_prediction_time()
    assert not unavailable, [spec.name for spec in unavailable]
    assert all(spec.availability is Availability.AT_ORDER_TIME for spec in feature_set)


def test_declared_source_columns_exist(
    feature_frame: pd.DataFrame, features_config: FeaturesConfig, generator_config: GeneratorConfig
) -> None:
    """A spec that names a column which does not exist is documentation rot."""
    feature_set = _pipeline(features_config, generator_config).feature_set
    available = set(feature_frame.columns)
    missing = {
        spec.name: sorted(set(spec.source_columns) - available)
        for spec in feature_set
        if set(spec.source_columns) - available
    }
    assert not missing, f"features declare source columns that do not exist: {missing}"


def test_outcome_windows_run_on_the_resolved_clock(
    features_config: FeaturesConfig, generator_config: GeneratorConfig
) -> None:
    """A window over outcomes must not be keyed on placement time.

    Enforced by ``FeatureSpec`` at construction, so this is really a test that the
    validator is wired up - but that is worth having, because it is the single
    rule whose violation is hardest to spot by reading.
    """
    feature_set = _pipeline(features_config, generator_config).feature_set
    for spec in feature_set:
        if spec.observation_point in OUTCOME_DEPENDENT:
            assert spec.lookback is not None
            assert spec.lookback.clock == "resolved", spec.name


# ---------------------------------------------------------------------------
# 5. test_duplicate_or_near_duplicate_leakage
# ---------------------------------------------------------------------------


def test_duplicate_or_near_duplicate_leakage(modeling_dataset: ModelingDataset) -> None:
    """The same order, or an indistinguishable one, must not span two splits.

    Three checks, in increasing subtlety:

    1. No repeated ``order_id`` anywhere.
    2. No exact duplicate feature row shared between two splits. With continuous
       features this is essentially impossible by chance, so a hit means the same
       order reached two splits.
    3. No repeated (customer, address, value, timestamp) tuple. This is the
       near-duplicate case: two distinct order ids describing what is effectively
       one event, which would let a model memorise it in training and be graded on
       it in test.
    """
    order_ids = modeling_dataset.order_ids
    assert not order_ids.duplicated().any(), "duplicate order ids in the modelling dataset"

    numeric = modeling_dataset.features.select_dtypes(include=["number"]).round(6)
    fingerprint = pd.util.hash_pandas_object(numeric, index=False)
    combined = pd.DataFrame({"fingerprint": fingerprint, "split": modeling_dataset.splits.values})

    per_fingerprint = combined.groupby("fingerprint")["split"].nunique()
    crossing = int((per_fingerprint > 1).sum())
    assert crossing == 0, (
        f"{crossing} identical feature rows appear in more than one split; the same order "
        "may have reached both training and evaluation"
    )

    identity = pd.DataFrame(
        {
            "customer": modeling_dataset.customer_hashes.values,
            "ordered_at": modeling_dataset.ordered_at.values,
            "value": modeling_dataset.features["order_value_inr"].round(2).values,
            "split": modeling_dataset.splits.values,
        }
    )
    near_duplicates = identity.duplicated(subset=["customer", "ordered_at", "value"]).sum()
    assert near_duplicates == 0, (
        f"{near_duplicates} near-duplicate orders (same customer, instant and value). "
        "Two rows describing one event let a model memorise in training and be graded "
        "on the same event in evaluation."
    )


def test_the_duplicate_check_can_actually_fail(modeling_dataset: ModelingDataset) -> None:
    """Inject a duplicated row and confirm the fingerprint check notices."""
    numeric = modeling_dataset.features.select_dtypes(include=["number"]).round(6)
    splits = modeling_dataset.splits.reset_index(drop=True)

    train_row = numeric.loc[splits == "train"].iloc[[0]]
    contaminated = pd.concat([numeric, train_row], ignore_index=True)
    contaminated_splits = pd.concat([splits, pd.Series(["validation"])], ignore_index=True)

    fingerprint = pd.util.hash_pandas_object(contaminated, index=False)
    frame = pd.DataFrame({"fingerprint": fingerprint, "split": contaminated_splits})
    assert int((frame.groupby("fingerprint")["split"].nunique() > 1).sum()) == 1


# ---------------------------------------------------------------------------
# 6. test_customer_history_cutoff
# ---------------------------------------------------------------------------


def test_customer_history_cutoff(
    feature_frame: pd.DataFrame,
    features_config: FeaturesConfig,
    generator_config: GeneratorConfig,
) -> None:
    """Customer history features are verified against a brute-force recomputation.

    For a sample of rows, the prior resolved-order count and return count are
    recomputed by an explicit nested scan over the raw order table - the slow,
    obvious way, with the timestamp comparison written out - and compared against
    what the pipeline produced.

    The argument is the same one made for the as-of join itself: the clever
    implementation is only trustworthy if it agrees with an obvious one.
    """
    pipeline = _pipeline(features_config, generator_config)
    matrix = pipeline.build(feature_frame).matrix

    customers = feature_frame[cols.CUSTOMER_HASH].to_numpy()
    order_times = feature_frame[cols.ORDERED_AT].to_numpy()
    resolution_times = feature_frame[cols.RESOLVED_AT].to_numpy()
    labels = feature_frame[cols.IS_RTO].astype("float64").to_numpy()

    # Bias the sample toward rows that actually have history - a sample of
    # first-time customers would pass trivially.
    with_history = feature_frame.index[matrix["cust_prior_resolved_count"] > 0]
    assert len(with_history) > 50, "the fixture should contain customers with resolved history"
    sample = list(with_history[:: max(len(with_history) // 60, 1)])[:60]

    for position, row in enumerate(feature_frame.index):
        if row not in sample:
            continue
        expected_resolved = 0
        expected_rto = 0.0
        for other in range(len(feature_frame)):
            if other == position:
                continue
            if customers[other] != customers[position]:
                continue
            if pd.isna(resolution_times[other]):
                continue
            # THE CUTOFF: strictly before this order was placed.
            if not resolution_times[other] < order_times[position]:
                continue
            expected_resolved += 1
            expected_rto += float(np.nan_to_num(labels[other]))

        assert matrix.loc[row, "cust_prior_resolved_count"] == expected_resolved, (
            f"row {row}: prior resolved count disagrees with a brute-force scan"
        )
        assert matrix.loc[row, "cust_prior_rto_count"] == expected_rto, (
            f"row {row}: prior RTO count disagrees with a brute-force scan"
        )


def test_history_excludes_orders_resolving_at_the_exact_instant(
    feature_frame: pd.DataFrame,
    features_config: FeaturesConfig,
    generator_config: GeneratorConfig,
) -> None:
    """Strictly ``<``, not ``<=``.

    An order resolving at the exact instant another is placed is excluded, because
    in production that information would not have propagated yet. Constructed as a
    two-row fixture because the boundary is too rare to hit by chance.
    """
    pipeline = _pipeline(features_config, generator_config)
    sample = feature_frame.head(2).copy().reset_index(drop=True)

    sample.loc[:, cols.CUSTOMER_HASH] = "a" * 32
    # `pd.Timedelta(5, unit="D")` rather than `pd.Timedelta(days=5)`: under
    # numpy 2.5 the keyword form builds a generic-unit timedelta and warns on
    # addition. Naming the unit explicitly avoids it.
    first_time = pd.Timestamp(sample.loc[0, cols.ORDERED_AT])
    second_time = first_time + pd.Timedelta(5, unit="D")

    sample.loc[0, cols.ORDERED_AT] = first_time
    sample.loc[0, cols.RESOLVED_AT] = second_time  # resolves exactly when row 1 is placed
    sample.loc[0, cols.IS_RTO] = True
    sample.loc[0, cols.OUTCOME] = "rto"
    sample.loc[0, cols.IS_MATURE] = True
    sample.loc[1, cols.ORDERED_AT] = second_time

    matrix = pipeline.build(sample).matrix
    assert matrix.loc[1, "cust_prior_resolved_count"] == 0
    assert matrix.loc[1, "cust_prior_rto_count"] == 0
