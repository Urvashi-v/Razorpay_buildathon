"""The refused-feature list is real and reachable from code.

SPEC section 04, "What I am refusing to use". Name-derived features, raw pincode
categoricals, protected attributes, and cross-merchant signals.

Phase 1 can assert that the refusal is *configured and enumerable* - that the
patterns exist, that each carries a stated reason, and that nothing in the
declared feature schema already violates it. The pipeline check that rejects a
matching column at build time lands with the pipeline in Phase 2, and
``tests/leakage`` holds the placeholder for it.
"""

from __future__ import annotations

import pytest

from rto_sentinel.configuration import load_features_config
from rto_sentinel.data import schema
from rto_sentinel.settings import Settings

REQUIRED_REFUSAL_GROUPS = {
    "name_derived",
    "raw_pincode_categorical",
    "protected_attributes",
    "cross_merchant",
}


def test_all_four_refusal_groups_are_present(settings: Settings) -> None:
    config = load_features_config(settings)
    assert {group.id for group in config.refused} == REQUIRED_REFUSAL_GROUPS


def test_every_refusal_states_a_reason(settings: Settings) -> None:
    """A refusal without a reason is a rule nobody can evaluate or challenge."""
    config = load_features_config(settings)
    for group in config.refused:
        assert group.reason.strip(), f"refusal group {group.id} has no stated reason"
        assert group.patterns, f"refusal group {group.id} lists no patterns"


def test_name_derived_features_are_refused(settings: Settings) -> None:
    """Religion, caste and region inference from names is a live harm."""
    patterns = load_features_config(settings).refused_patterns
    for name in ("customer_name", "first_name", "surname", "name_ngram", "name_embedding"):
        assert name in patterns


def test_protected_attributes_are_refused(settings: Settings) -> None:
    patterns = load_features_config(settings).refused_patterns
    for name in ("gender", "age", "religion", "caste", "language_inferred"):
        assert name in patterns


def test_raw_pincode_is_refused_as_a_categorical(settings: Settings) -> None:
    """With enough trees a raw pincode categorical becomes a redlining machine."""
    patterns = load_features_config(settings).refused_patterns
    assert "pincode_raw" in patterns
    assert "pincode_target_encoded" in patterns

    # Only the smoothed aggregate is permitted, and geography must declare it.
    geography = load_features_config(settings).families["geography_route"]
    assert geography.shrinkage == "bayesian"
    assert geography.min_support is not None and geography.min_support > 0
    assert "pincode_rto_rate_smoothed" in geography.signals


def test_no_declared_signal_matches_a_refused_pattern(settings: Settings) -> None:
    """The families as configured do not already violate the refusal list."""
    config = load_features_config(settings)
    refused = config.refused_patterns
    offenders = [
        (family_name, signal)
        for family_name, family in config.families.items()
        for signal in family.signals
        if signal.lower() in refused
    ]
    assert not offenders, f"declared signals matching a refused pattern: {offenders}"


def test_raw_pincode_is_forbidden_in_the_design_matrix() -> None:
    """The schema's forbidden set backs up the config's refusal list."""
    assert schema.PINCODE in schema.FORBIDDEN_IN_FEATURES
    assert schema.CUSTOMER_HASH in schema.FORBIDDEN_IN_FEATURES
    assert schema.ADDRESS_LINE in schema.FORBIDDEN_IN_FEATURES


def test_the_target_and_its_timestamp_are_forbidden_in_features() -> None:
    """The label leaks in two ways, and both are closed.

    ``is_rto`` is the obvious one. ``resolved_at`` is the subtle one: knowing when
    an order resolved is close to knowing how it resolved, and a model given it
    will find that out.
    """
    assert schema.TARGET_COLUMN in schema.FORBIDDEN_IN_FEATURES
    assert schema.OUTCOME in schema.FORBIDDEN_IN_FEATURES
    assert schema.RESOLVED_AT in schema.FORBIDDEN_IN_FEATURES
    assert schema.SPLIT in schema.FORBIDDEN_IN_FEATURES


@pytest.mark.parametrize("column", sorted(schema.FORBIDDEN_IN_FEATURES))
def test_forbidden_columns_are_real_columns(column: str) -> None:
    """Guards against a typo silently disabling a forbidden-column check."""
    assert column in schema.RAW_COLUMNS
