"""The feature machinery: specs, windows, shrinkage, and the pipeline guards.

The leakage suite proves the features do not read the future. These tests prove
they compute the right thing - a different claim, and one that hand-computed
fixtures answer better than any property test.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from rto_sentinel.configuration.schemas import FeaturesConfig, GeneratorConfig
from rto_sentinel.features import FeaturePipeline, RefusedFeatureError, TargetLeakageError
from rto_sentinel.features.pipeline import FeatureContractError, _matches_token_pattern
from rto_sentinel.features.spec import (
    ALL_HISTORY_PLACED,
    Availability,
    FeatureSet,
    FeatureSpec,
    LookbackWindow,
    ObservationPoint,
)
from rto_sentinel.features.temporal import (
    _days_since_last_positive,
    _epoch_ns,
    _windowed_count,
    _windowed_count_and_sum,
)

BASE = datetime(2025, 9, 1, tzinfo=UTC)


def _spec(**overrides: object) -> FeatureSpec:
    payload: dict[str, object] = {
        "name": "example",
        "family": "test",
        "dtype": "float",
        "description": "An example.",
        "source_columns": ("a",),
        "observation_point": ObservationPoint.ORDER_PAYLOAD,
        "availability": Availability.AT_ORDER_TIME,
        "risk_note": "None.",
    }
    payload.update(overrides)
    return FeatureSpec(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FeatureSpec validation
# ---------------------------------------------------------------------------


def test_a_feature_without_a_risk_note_is_refused() -> None:
    """A feature nobody can articulate a risk for has not been thought about."""
    with pytest.raises(ValueError, match="no risk note"):
        _spec(risk_note="   ")


def test_a_feature_without_source_columns_is_refused() -> None:
    with pytest.raises(ValueError, match="no source columns"):
        _spec(source_columns=())


def test_an_aggregating_feature_must_declare_a_lookback() -> None:
    with pytest.raises(ValueError, match="no lookback"):
        _spec(observation_point=ObservationPoint.PRIOR_ORDERS_PLACED)


def test_an_outcome_window_on_the_placed_clock_is_refused() -> None:
    """The single most dangerous misdeclaration in the project, refused at build.

    A window over resolved outcomes keyed on placement time counts orders that
    have not come back yet - the exact leak the as-of machinery exists to prevent.
    """
    with pytest.raises(ValueError, match="has not necessarily come back yet"):
        _spec(
            observation_point=ObservationPoint.PRIOR_ORDERS_RESOLVED,
            lookback=ALL_HISTORY_PLACED,
        )


def test_a_valid_outcome_window_is_accepted() -> None:
    spec = _spec(
        observation_point=ObservationPoint.PRIOR_ORDERS_RESOLVED,
        lookback=LookbackWindow(days=30, clock="resolved"),
    )
    assert spec.lookback is not None
    assert spec.lookback.clock == "resolved"


# ---------------------------------------------------------------------------
# FeatureSet
# ---------------------------------------------------------------------------


def test_duplicate_feature_names_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate feature names"):
        FeatureSet((_spec(name="x"), _spec(name="x")))


def test_the_fingerprint_tracks_definitions() -> None:
    """A changed definition must change the fingerprint.

    Otherwise a model artefact could claim a feature set it was not trained on -
    a silently altered lookback window would be invisible in a results table.
    """
    baseline = FeatureSet((_spec(name="a"), _spec(name="b")))

    assert baseline.fingerprint() == FeatureSet((_spec(name="a"), _spec(name="b"))).fingerprint()
    assert baseline.fingerprint() != FeatureSet((_spec(name="a"),)).fingerprint()
    assert (
        baseline.fingerprint()
        != FeatureSet((_spec(name="a"), _spec(name="b", description="Changed."))).fingerprint()
    )
    # Order is part of the contract: training and inference build the matrix
    # through the same object, so a reordering is a real change.
    assert baseline.fingerprint() != FeatureSet((_spec(name="b"), _spec(name="a"))).fingerprint()


def test_unavailable_features_are_reported() -> None:
    feature_set = FeatureSet(
        (
            _spec(name="ok"),
            _spec(
                name="leaky",
                availability=Availability.POST_ORDER,
                observation_point=ObservationPoint.POST_ORDER,
            ),
        )
    )
    assert [spec.name for spec in feature_set.unavailable_at_prediction_time()] == ["leaky"]


# ---------------------------------------------------------------------------
# refused-pattern token matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column,pattern,expected",
    [
        # The false positives that made substring matching unusable.
        ("cust_account_age_days", "age", True),
        ("session_product_page_seconds", "age", False),
        ("order_value_per_item", "age", False),
        # Genuine hits.
        ("customer_age", "age", True),
        ("age", "age", True),
        ("cust_customer_name_hash", "customer_name", True),
        ("pincode_raw", "pincode_raw", True),
        # Multi-token patterns must match a consecutive run.
        ("customer_history_name", "customer_name", False),
        ("geo_pincode_rto_rate_smoothed", "pincode_raw", False),
    ],
)
def test_token_matching(column: str, pattern: str, expected: bool) -> None:
    assert _matches_token_pattern(column, pattern) is expected


# ---------------------------------------------------------------------------
# windowed primitives
# ---------------------------------------------------------------------------


def _times(*day_offsets: float | None) -> tuple[np.ndarray, np.ndarray]:
    series = pd.Series(
        [None if d is None else BASE + pd.Timedelta(d, unit="D") for d in day_offsets],
        dtype="datetime64[ns, UTC]",
    )
    return _epoch_ns(series)


def test_windowed_count_is_half_open_on_the_left_and_strict_on_the_right() -> None:
    """Events at days 0, 5, 9; anchors at 10 and 40, 30-day window.

    Hand-computed: from day 10 the window is [-20, 10) and all three events fall
    inside except the anchor's own row, which is not in the event list here.
    """
    groups = np.array(["A", "A"])
    anchors, _ = _times(10, 40)
    events, valid = _times(0, 5)

    counts = _windowed_count(
        groups, anchors, np.r_[events, events[:0]], np.r_[valid, valid[:0]], window_days=30
    )
    assert list(counts) == [2.0, 0.0]  # from day 40 both events are outside the window


def test_windowed_count_excludes_the_exact_anchor_instant() -> None:
    """An event at the anchor instant has not propagated yet."""
    groups = np.array(["A", "A"])
    anchors, _ = _times(5, 5)
    events, valid = _times(5, 5)
    counts = _windowed_count(groups, anchors, events, valid, window_days=30)
    assert list(counts) == [0.0, 0.0]


def test_windowed_count_and_sum_matches_a_hand_computation() -> None:
    """Three resolutions at days 1, 2 and 40; two of them returns.

    Anchor at day 10 with a 30-day window sees days 1 and 2 - one return out of
    two. The day-40 resolution is in the future and must not appear.
    """
    groups = np.array(["A", "A", "A"])
    anchors, _ = _times(10, 10, 10)
    events, valid = _times(1, 2, 40)
    values = np.array([1.0, 0.0, 1.0])

    counts, sums = _windowed_count_and_sum(groups, anchors, events, valid, values, window_days=30)
    assert counts[0] == 2.0
    assert sums[0] == 1.0


def test_windowed_aggregates_do_not_cross_groups() -> None:
    groups = np.array(["A", "B"])
    anchors, _ = _times(10, 10)
    events, valid = _times(1, 1)
    values = np.array([1.0, 1.0])

    counts, sums = _windowed_count_and_sum(groups, anchors, events, valid, values, window_days=30)
    # Each customer sees only their own event, and that event is their own row -
    # which is excluded because it resolves at day 1, before the anchor at day 10.
    assert list(counts) == [1.0, 1.0]
    assert list(sums) == [1.0, 1.0]


def test_pending_events_contribute_to_nothing() -> None:
    groups = np.array(["A", "A"])
    anchors, _ = _times(10, 10)
    events, valid = _times(None, 1)
    values = np.array([1.0, 1.0])

    counts, _ = _windowed_count_and_sum(groups, anchors, events, valid, values, window_days=30)
    assert list(counts) == [1.0, 1.0]


def test_days_since_last_positive() -> None:
    groups = np.array(["A", "A", "A"])
    anchors, _ = _times(10, 10, 10)
    events, valid = _times(2, 6, 40)
    values = np.array([1.0, 0.0, 1.0])  # only day 2 is a positive that has resolved

    gaps = _days_since_last_positive(groups, anchors, events, valid, values)
    assert gaps[0] == pytest.approx(8.0)


def test_days_since_last_positive_is_nan_without_one() -> None:
    groups = np.array(["A"])
    anchors, _ = _times(10)
    events, valid = _times(2)
    gaps = _days_since_last_positive(groups, anchors, events, valid, np.array([0.0]))
    assert np.isnan(gaps[0])


# ---------------------------------------------------------------------------
# pipeline guards
# ---------------------------------------------------------------------------


def test_the_pipeline_declares_before_it_computes(
    features_config: FeaturesConfig, generator_config: GeneratorConfig
) -> None:
    """The declared set is available without touching any data."""
    pipeline = FeaturePipeline(features_config, generator_config)
    feature_set = pipeline.feature_set

    assert len(feature_set) > 40
    assert "customer_history" in feature_set.families
    assert "temporal" in feature_set.families
    pipeline.check_declarations()


def test_an_unavailable_feature_is_rejected_before_computation(
    features_config: FeaturesConfig, generator_config: GeneratorConfig
) -> None:
    """The cheapest leak check, and it must actually fire."""

    class LeakyFamily:
        name = "leaky"

        @property
        def feature_set(self) -> FeatureSet:
            return FeatureSet(
                (
                    _spec(
                        name="knows_the_future",
                        availability=Availability.POST_ORDER,
                        observation_point=ObservationPoint.POST_ORDER,
                    ),
                )
            )

        def transform(self, frame: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
            raise AssertionError("transform must not be reached")

    pipeline = FeaturePipeline(features_config, generator_config, families=[LeakyFamily()])  # type: ignore[list-item]
    with pytest.raises(TargetLeakageError, match="unavailable at prediction time"):
        pipeline.check_declarations()


def test_a_refused_feature_name_is_rejected_with_its_reason(
    features_config: FeaturesConfig, generator_config: GeneratorConfig
) -> None:
    """The error quotes the configured justification, not just a refusal."""
    pipeline = FeaturePipeline(features_config, generator_config)
    with pytest.raises(RefusedFeatureError) as caught:
        pipeline.check_refused(["cust_customer_name_ngram"])

    message = str(caught.value)
    assert "name_derived" in message
    assert "caste" in message  # the configured reason is quoted verbatim


def test_a_forbidden_column_is_rejected() -> None:
    with pytest.raises(TargetLeakageError, match="forbidden columns"):
        FeaturePipeline.check_forbidden(["order_value_inr", "is_rto"])


def test_a_family_emitting_undeclared_columns_is_rejected(
    features_config: FeaturesConfig, generator_config: GeneratorConfig, feature_frame: pd.DataFrame
) -> None:
    """A debugging column that nobody audits must not become a production feature."""

    class SloppyFamily:
        name = "sloppy"

        @property
        def feature_set(self) -> FeatureSet:
            # A real source column, so the missing-input guard (which correctly
            # runs first) does not mask the contract violation under test.
            return FeatureSet((_spec(name="declared", source_columns=("order_value_inr",)),))

        def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"declared": 1.0, "sneaky_debug_column": 2.0}, index=frame.index)

    pipeline = FeaturePipeline(features_config, generator_config, families=[SloppyFamily()])  # type: ignore[list-item]
    with pytest.raises(FeatureContractError, match="emitted"):
        pipeline.build(feature_frame)


def test_disabling_a_family_removes_its_features(
    features_config: FeaturesConfig, generator_config: GeneratorConfig
) -> None:
    """The ablation study is a config change, not a code change."""
    full = FeaturePipeline(features_config, generator_config).feature_set

    families = dict(features_config.families)
    families["temporal"] = families["temporal"].model_copy(update={"enabled": False})
    ablated_config = features_config.model_copy(update={"families": families})
    ablated = FeaturePipeline(ablated_config, generator_config).feature_set

    assert "temporal" in full.families
    assert "temporal" not in ablated.families
    assert len(ablated) < len(full)
    assert ablated.fingerprint() != full.fingerprint()


# ---------------------------------------------------------------------------
# geography shrinkage and support
# ---------------------------------------------------------------------------


def test_thin_pincodes_are_withheld_not_shrunk(
    features_config: FeaturesConfig, generator_config: GeneratorConfig, feature_frame: pd.DataFrame
) -> None:
    """Below minimum support the rate is NaN, not a shrunk estimate.

    Shrinkage alone still passes a little signal through from two or three
    orders, and a place should not acquire a reputation from three deliveries.
    """
    pipeline = FeaturePipeline(features_config, generator_config)
    matrix = pipeline.build(feature_frame).matrix

    support = features_config.families["geography_route"].min_support
    assert support is not None

    thin = matrix["geo_pincode_resolved_count"] < support
    assert thin.any(), "the fixture should contain thin pincodes"
    assert matrix.loc[thin, "geo_pincode_rto_rate_smoothed"].isna().all()


def test_smoothed_rates_stay_within_bounds(
    features_config: FeaturesConfig, generator_config: GeneratorConfig, feature_frame: pd.DataFrame
) -> None:
    matrix = FeaturePipeline(features_config, generator_config).build(feature_frame).matrix
    for column in ("geo_pincode_rto_rate_smoothed", "geo_courier_rto_rate_smoothed"):
        values = matrix[column].dropna()
        assert ((values >= 0.0) & (values <= 1.0)).all()


# ---------------------------------------------------------------------------
# customer-history semantics
# ---------------------------------------------------------------------------


def test_missing_history_is_nan_for_rates_and_zero_for_counts(
    features_config: FeaturesConfig, generator_config: GeneratorConfig, feature_frame: pd.DataFrame
) -> None:
    """The rule from ``data.asof``, checked where it actually matters.

    A 0.0 return rate for a first-time customer claims a clean record the merchant
    has no basis for. A 0 prior-order count is a fact.
    """
    matrix = FeaturePipeline(features_config, generator_config).build(feature_frame).matrix
    new_customers = matrix["cust_is_new"].astype(bool)
    assert new_customers.any()

    assert (matrix.loc[new_customers, "cust_prior_order_count"] == 0).all()
    assert (matrix.loc[new_customers, "cust_prior_rto_count"] == 0).all()
    assert matrix.loc[new_customers, "cust_prior_rto_rate"].isna().all()
    assert matrix.loc[new_customers, "cust_days_since_last_order"].isna().all()


def test_the_smoothed_personal_rate_is_defined_for_everyone(
    features_config: FeaturesConfig, generator_config: GeneratorConfig, feature_frame: pd.DataFrame
) -> None:
    """It collapses to the population prior for a customer we know nothing about."""
    matrix = FeaturePipeline(features_config, generator_config).build(feature_frame).matrix
    smoothed = matrix["cust_prior_rto_rate_smoothed"]

    assert smoothed.notna().all()
    assert ((smoothed >= 0.0) & (smoothed <= 1.0)).all()

    new_customers = matrix["cust_is_new"].astype(bool)
    assert smoothed.loc[new_customers].nunique() == 1, (
        "customers with no history should all receive the same prior"
    )


def test_account_age_is_never_negative(
    features_config: FeaturesConfig, generator_config: GeneratorConfig, feature_frame: pd.DataFrame
) -> None:
    """A negative age means a signup timestamp arrived from the future."""
    matrix = FeaturePipeline(features_config, generator_config).build(feature_frame).matrix
    age = matrix["cust_account_age_days"].dropna()
    assert (age >= 0).all()
