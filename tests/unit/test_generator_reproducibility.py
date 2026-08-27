"""Reproducibility: the same configuration and seed produce the same data.

This is the property the whole benchmark rests on. If a dataset cannot be
regenerated exactly, then no result measured on it can be checked by anyone else,
and "we ran it and got 0.61" becomes a claim rather than an experiment.

The tests below check it from both directions - identical inputs give identical
output, and any change to an input changes the output - because a generator that
ignored its seed would trivially pass the first test alone.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from rto_sentinel.configuration.schemas import GeneratorConfig
from rto_sentinel.data.generator import (
    SUPPORTED_GENERATOR_VERSIONS,
    ConfiguredOrderGenerator,
    GenerationResult,
    GeneratorParams,
    UnsupportedGeneratorVersionError,
)

# Small enough to run several times in one test module.
PARAMS = GeneratorParams(
    seed=99,
    generator_version="1.0.0",
    n_customers=250,
    n_orders=700,
    start_date=datetime(2025, 9, 1, tzinfo=UTC),
    end_date=datetime(2026, 2, 27, tzinfo=UTC),
)


def _generate(config: GeneratorConfig, params: GeneratorParams) -> GenerationResult:
    return ConfiguredOrderGenerator().generate(config, params)


def test_same_seed_produces_identical_orders(generator_config: GeneratorConfig) -> None:
    """The headline reproducibility guarantee, checked on the full frame."""
    first = _generate(generator_config, PARAMS)
    second = _generate(generator_config, PARAMS)

    pd.testing.assert_frame_equal(first.orders, second.orders)


def test_same_seed_produces_identical_dimensions_and_events(
    generator_config: GeneratorConfig,
) -> None:
    """Every frame, not only the headline one."""
    first = _generate(generator_config, PARAMS)
    second = _generate(generator_config, PARAMS)

    pd.testing.assert_frame_equal(first.customers, second.customers)
    pd.testing.assert_frame_equal(first.addresses, second.addresses)
    pd.testing.assert_frame_equal(first.delivery_events, second.delivery_events)
    pd.testing.assert_frame_equal(first.latents, second.latents)


def test_same_seed_produces_identical_metadata(generator_config: GeneratorConfig) -> None:
    """Provenance is reproducible too, apart from the wall-clock creation time."""
    first = _generate(generator_config, PARAMS).metadata
    second = _generate(generator_config, PARAMS).metadata

    assert first.run_id == second.run_id
    assert first.seed == second.seed
    assert first.generator_version == second.generator_version
    assert first.config_fingerprint == second.config_fingerprint
    assert first.realised_rto_rate_cod == second.realised_rto_rate_cod
    assert first.realised_rto_rate_prepaid == second.realised_rto_rate_prepaid
    assert first.n_mature == second.n_mature
    # created_at is a wall-clock timestamp and is deliberately NOT compared: it
    # records when the dataset was built, not what it contains.


def test_a_different_seed_produces_different_data(generator_config: GeneratorConfig) -> None:
    """Guards against a generator that quietly ignores its seed.

    Without this, a hardcoded RNG would pass every other test in this file.
    """
    first = _generate(generator_config, PARAMS)
    second = _generate(generator_config, replace(PARAMS, seed=PARAMS.seed + 1))

    assert first.metadata.run_id != second.metadata.run_id
    assert not first.orders["order_value_inr"].equals(second.orders["order_value_inr"])
    assert not first.orders["is_rto"].equals(second.orders["is_rto"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_customers", 300),
        ("n_orders", 800),
        ("seed", 12345),
    ],
)
def test_every_parameter_changes_the_run_id(
    generator_config: GeneratorConfig, field: str, value: int
) -> None:
    """The run id is a digest of the parameters, so each one must move it.

    A run id that ignored a parameter would let two genuinely different datasets
    collide in the database and silently overwrite each other.
    """
    baseline = _generate(generator_config, PARAMS).metadata.run_id
    changed = _generate(generator_config, replace(PARAMS, **{field: value})).metadata.run_id
    assert baseline != changed


def test_dates_change_the_run_id(generator_config: GeneratorConfig) -> None:
    shifted = replace(PARAMS, end_date=datetime(2026, 2, 20, tzinfo=UTC))
    assert _generate(generator_config, PARAMS).metadata.run_id != (
        _generate(generator_config, shifted).metadata.run_id
    )


def test_config_fingerprint_tracks_the_configuration(generator_config: GeneratorConfig) -> None:
    """Editing the configuration must change the recorded fingerprint.

    This is what makes "the split protocol was fixed before modelling" checkable
    rather than merely asserted: a configuration edited after results exist has a
    different fingerprint from the one stamped on those results.
    """
    baseline = _generate(generator_config, PARAMS).metadata.config_fingerprint

    tweaked = generator_config.model_copy(
        update={
            "noise": generator_config.noise.model_copy(
                update={"logit_sigma": generator_config.noise.logit_sigma + 0.1}
            )
        }
    )
    assert _generate(tweaked, PARAMS).metadata.config_fingerprint != baseline


# ---------------------------------------------------------------------------
# generator versioning
# ---------------------------------------------------------------------------


def test_unknown_generator_version_is_refused() -> None:
    """An unimplemented process version fails loudly rather than silently running 1.0.0.

    Silently falling back would mean a dataset labelled "2.0.0" that was actually
    produced by a different process - the sort of provenance error that is
    undetectable afterwards.
    """
    with pytest.raises(UnsupportedGeneratorVersionError, match="not implemented"):
        GeneratorParams(
            seed=1,
            generator_version="9.9.9",
            n_customers=10,
            n_orders=10,
            start_date=datetime(2025, 9, 1, tzinfo=UTC),
            end_date=datetime(2025, 9, 30, tzinfo=UTC),
        )


def test_supported_versions_are_declared() -> None:
    assert "1.0.0" in SUPPORTED_GENERATOR_VERSIONS


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="end_date must not precede"):
        GeneratorParams(
            seed=1,
            generator_version="1.0.0",
            n_customers=10,
            n_orders=10,
            start_date=datetime(2025, 9, 30, tzinfo=UTC),
            end_date=datetime(2025, 9, 1, tzinfo=UTC),
        )

    with pytest.raises(ValueError, match="must both be positive"):
        GeneratorParams(
            seed=1,
            generator_version="1.0.0",
            n_customers=0,
            n_orders=10,
            start_date=datetime(2025, 9, 1, tzinfo=UTC),
            end_date=datetime(2025, 9, 30, tzinfo=UTC),
        )
