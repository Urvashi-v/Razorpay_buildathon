"""The generated feature dictionary stays in sync with the declarations.

``docs/features.md`` is generated from :class:`FeatureSpec` objects rather than
written by hand, because hand-written feature documentation is wrong within a
month - someone adds a column, someone else changes a lookback window, and the
document quietly becomes fiction a reviewer nonetheless trusts.

This test is what makes that guarantee real: a stale committed file fails CI.
"""

from __future__ import annotations

from pathlib import Path

from rto_sentinel.configuration.schemas import FeaturesConfig, GeneratorConfig
from rto_sentinel.features import FEATURE_VERSION, FeaturePipeline
from rto_sentinel.features.catalogue import render_markdown


def test_the_committed_feature_dictionary_is_current(
    repo_root: Path, features_config: FeaturesConfig, generator_config: GeneratorConfig
) -> None:
    """Regenerate with: rto-sentinel features docs"""
    pipeline = FeaturePipeline(features_config, generator_config)
    feature_set = pipeline.feature_set

    expected = render_markdown(
        feature_set, feature_version=FEATURE_VERSION, fingerprint=feature_set.fingerprint()
    )
    committed = (repo_root / "docs" / "features.md").read_text(encoding="utf-8")

    assert committed == expected, (
        "docs/features.md is out of date with the feature declarations. "
        "Regenerate it with `rto-sentinel features docs`."
    )


def test_every_feature_appears_in_the_dictionary(
    repo_root: Path, features_config: FeaturesConfig, generator_config: GeneratorConfig
) -> None:
    committed = (repo_root / "docs" / "features.md").read_text(encoding="utf-8")
    feature_set = FeaturePipeline(features_config, generator_config).feature_set
    missing = [spec.name for spec in feature_set if f"`{spec.name}`" not in committed]
    assert not missing, f"features absent from the dictionary: {missing}"


def test_the_dictionary_records_the_fingerprint(
    repo_root: Path, features_config: FeaturesConfig, generator_config: GeneratorConfig
) -> None:
    """So a document and a model artefact can be matched to the same feature set."""
    committed = (repo_root / "docs" / "features.md").read_text(encoding="utf-8")
    feature_set = FeaturePipeline(features_config, generator_config).feature_set
    assert feature_set.fingerprint() in committed


def test_every_feature_prefix_maps_to_its_family(features_config, generator_config) -> None:
    """`FAMILY_BY_PREFIX` must agree with the live feature set, not with a memory.

    A trained artefact carries feature *names* and not the FeatureSet that
    produced them, so a model asked for attributions at serving time resolves the
    family from the name alone. If that table drifts from reality, every reason
    code silently loses its operational prefix - which is exactly what happened
    before the serving path started using attributions.
    """
    from rto_sentinel.features.pipeline import FAMILY_BY_PREFIX, FeaturePipeline, family_of

    pipeline = FeaturePipeline(features_config, generator_config=generator_config)

    observed: dict[str, set[str]] = {}
    for spec in pipeline.feature_set:
        observed.setdefault(spec.name.split("_", 1)[0], set()).add(spec.family)

    ambiguous = {prefix: sorted(f) for prefix, f in observed.items() if len(f) > 1}
    assert not ambiguous, (
        f"a feature-name prefix maps to more than one family, so the family cannot be "
        f"recovered from a name: {ambiguous}"
    )

    for prefix, families in observed.items():
        family = next(iter(families))
        assert FAMILY_BY_PREFIX.get(prefix) == family, (
            f"prefix {prefix!r} belongs to family {family!r} but the table says "
            f"{FAMILY_BY_PREFIX.get(prefix)!r}"
        )

    for spec in pipeline.feature_set:
        assert family_of(spec.name) == spec.family
