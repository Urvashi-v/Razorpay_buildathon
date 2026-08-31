"""Distribution-shift experiments: overrides, reproducibility, and honest comparison.

The reproducibility test is the load-bearing one. A robustness study that cannot
be re-run to the same numbers is an anecdote, and the whole argument for these
environments is that a reviewer can regenerate them and check.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rto_sentinel.configuration import load_generator_config
from rto_sentinel.contracts.monitoring import EnvironmentSpec, ShiftResult, ShiftStudy
from rto_sentinel.eval.shift import (
    MATERIAL_LIFT_DROP,
    OverrideError,
    apply_overrides,
    default_environments,
    generate_environment,
    summarise,
)
from rto_sentinel.settings import get_settings


@pytest.fixture(scope="module")
def generator_config() -> object:
    return load_generator_config(get_settings())


def result(
    name: str,
    *,
    pr_auc: float = 0.4,
    lift: float = 2.5,
    lift_delta: float | None = None,
    ece: float = 0.02,
    ece_delta: float | None = None,
    net: float = 3000.0,
    rto: float = 0.16,
) -> ShiftResult:
    return ShiftResult(
        environment=name,
        description=f"{name} description",
        n_orders=8000,
        observed_rto_rate=rto,
        pr_auc=pr_auc,
        pr_auc_lift=lift,
        pr_auc_lift_delta=lift_delta,
        roc_auc=0.78,
        brier_score=0.12,
        expected_calibration_error=ece,
        ece_delta=ece_delta,
        threshold=0.348,
        flag_rate=0.18,
        precision=0.42,
        recall=0.40,
        net_inr_per_1000=net,
    )


class TestOverrides:
    def test_a_nested_key_is_replaced(self, generator_config: object) -> None:
        shifted = apply_overrides(generator_config, {"payment.cod_share": 0.80})  # type: ignore[arg-type]
        assert shifted.payment.cod_share == 0.80
        assert generator_config.payment.cod_share != 0.80  # type: ignore[attr-defined]

    def test_the_original_config_is_not_mutated(self, generator_config: object) -> None:
        before = generator_config.base_rates.rto_given_cod  # type: ignore[attr-defined]
        apply_overrides(generator_config, {"base_rates.rto_given_cod": 0.44})  # type: ignore[arg-type]
        assert generator_config.base_rates.rto_given_cod == before  # type: ignore[attr-defined]

    def test_a_list_entry_is_addressed_by_name_not_index(self, generator_config: object) -> None:
        """Index addressing is silently wrong the moment somebody reorders the list."""
        shifted = apply_overrides(
            generator_config,  # type: ignore[arg-type]
            {
                "catalogue.categories.fashion.share": 0.60,
                "catalogue.categories.electronics.share": 0.08,
                "catalogue.categories.beauty.share": 0.12,
                "catalogue.categories.home.share": 0.12,
                "catalogue.categories.accessories.share": 0.08,
            },
        )
        shares = {entry.name: entry.share for entry in shifted.catalogue.categories}
        assert shares["fashion"] == 0.60

    def test_a_typo_is_fatal_rather_than_ignored(self, generator_config: object) -> None:
        """A silently skipped override produces a study that shifted nothing."""
        with pytest.raises(OverrideError, match="does not exist"):
            apply_overrides(generator_config, {"payment.cod_shre": 0.9})  # type: ignore[arg-type]

    def test_an_unknown_list_entry_is_fatal(self, generator_config: object) -> None:
        with pytest.raises(OverrideError, match="not among"):
            apply_overrides(
                generator_config,  # type: ignore[arg-type]
                {"catalogue.categories.groceries.share": 0.5},
            )

    def test_an_override_producing_an_invalid_world_is_refused(
        self, generator_config: object
    ) -> None:
        """Shares that no longer sum to one must fail here, not in the generator."""
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            apply_overrides(
                generator_config,  # type: ignore[arg-type]
                {"catalogue.categories.fashion.share": 0.99},
            )

    def test_every_default_environment_applies_cleanly(self, generator_config: object) -> None:
        """Each shipped environment must produce a valid world."""
        for spec in default_environments(seed=1, n_orders=500):
            apply_overrides(generator_config, spec.overrides)  # type: ignore[arg-type]


class TestEnvironmentSpecs:
    def test_the_reference_environment_takes_no_overrides(self) -> None:
        """The control defines the unshifted world by construction."""
        with pytest.raises(ValueError, match="takes no overrides"):
            EnvironmentSpec(
                name="reference",
                description="not really a control",
                overrides={"payment.cod_share": 0.9},
                seed=1,
                n_orders=100,
            )

    def test_the_shipped_set_covers_every_axis_the_spec_names(self) -> None:
        names = {spec.name for spec in default_environments(seed=1, n_orders=100)}
        assert "reference" in names
        # COD rate, RTO base rate, category mix, customer mix.
        assert {"cod_surge", "cod_collapse"} <= names
        assert {"rto_base_rate_up", "rto_base_rate_down"} <= names
        assert "category_mix_fashion" in names
        assert "customer_mix_new" in names

    def test_seeds_differ_across_environments(self) -> None:
        """Sharing one seed would correlate sampling noise and flatter comparisons."""
        seeds = [spec.seed for spec in default_environments(seed=100, n_orders=100)]
        assert len(set(seeds)) == len(seeds)

    def test_environment_definitions_are_stable_for_a_given_seed(self) -> None:
        first = default_environments(seed=42, n_orders=1000)
        second = default_environments(seed=42, n_orders=1000)
        assert [spec.model_dump() for spec in first] == [spec.model_dump() for spec in second]


class TestReproducibility:
    """Same seed, same world. Twice."""

    def test_generating_an_environment_twice_gives_identical_data(
        self, generator_config: object
    ) -> None:
        from rto_sentinel.configuration import load_features_config, load_splits_config

        settings = get_settings()
        spec = EnvironmentSpec(
            name="cod_surge",
            description="COD share rises",
            overrides={"payment.cod_share": 0.80},
            seed=777,
            n_orders=600,
        )
        kwargs = {
            "generator_config": generator_config,
            "features_config": load_features_config(settings),
            "splits_config": load_splits_config(settings),
        }

        first = generate_environment(spec, **kwargs)  # type: ignore[arg-type]
        second = generate_environment(spec, **kwargs)  # type: ignore[arg-type]

        assert first.labels.tolist() == second.labels.tolist()
        assert first.features.shape == second.features.shape
        assert first.orders["order_id"].tolist() == second.orders["order_id"].tolist()

    def test_a_different_seed_gives_a_different_world(self, generator_config: object) -> None:
        """Otherwise the seed is not doing anything and the study is one draw."""
        from rto_sentinel.configuration import load_features_config, load_splits_config

        settings = get_settings()
        kwargs = {
            "generator_config": generator_config,
            "features_config": load_features_config(settings),
            "splits_config": load_splits_config(settings),
        }
        base = {"description": "COD share rises", "overrides": {"payment.cod_share": 0.80}}

        first = generate_environment(
            EnvironmentSpec(name="a", seed=1, n_orders=600, **base),
            **kwargs,  # type: ignore[arg-type]
        )
        second = generate_environment(
            EnvironmentSpec(name="b", seed=2, n_orders=600, **base),
            **kwargs,  # type: ignore[arg-type]
        )
        assert first.orders["order_id"].tolist() != second.orders["order_id"].tolist()

    def test_the_override_actually_reaches_the_generated_data(
        self, generator_config: object
    ) -> None:
        """The whole study is worthless if the shift does not land in the orders."""
        from rto_sentinel.configuration import load_features_config, load_splits_config

        settings = get_settings()
        kwargs = {
            "generator_config": generator_config,
            "features_config": load_features_config(settings),
            "splits_config": load_splits_config(settings),
        }

        reference = generate_environment(
            EnvironmentSpec(
                name="reference", description="control", overrides={}, seed=5, n_orders=1500
            ),
            **kwargs,  # type: ignore[arg-type]
        )
        surge = generate_environment(
            EnvironmentSpec(
                name="cod_surge",
                description="COD share rises",
                overrides={"payment.cod_share": 0.85},
                seed=5,
                n_orders=1500,
            ),
            **kwargs,  # type: ignore[arg-type]
        )

        reference_cod = (reference.orders["payment_method"] == "cod").mean()
        surge_cod = (surge.orders["payment_method"] == "cod").mean()
        assert surge_cod > reference_cod + 0.10

    def test_only_matured_orders_are_evaluated(self, generator_config: object) -> None:
        """An immature order counted as a non-RTO flatters every environment."""
        from rto_sentinel.configuration import load_features_config, load_splits_config

        settings = get_settings()
        data = generate_environment(
            EnvironmentSpec(
                name="reference", description="control", overrides={}, seed=9, n_orders=800
            ),
            generator_config=generator_config,  # type: ignore[arg-type]
            features_config=load_features_config(settings),
            splits_config=load_splits_config(settings),
        )
        assert data.orders["is_rto"].notna().all()
        assert len(data.labels) == len(data.features) == len(data.orders)


class TestStudyContract:
    def test_a_study_without_its_control_is_refused(self) -> None:
        """Degradation measured against nothing is not degradation."""
        with pytest.raises(ValueError, match="reference environment"):
            ShiftStudy(
                generated_at=datetime.now(UTC),
                model_version="v1",
                threshold=0.348,
                environments=(
                    EnvironmentSpec(
                        name="cod_surge", description="d", overrides={}, seed=1, n_orders=10
                    ),
                ),
                results=(result("cod_surge"),),
            )

    def test_a_study_with_its_control_validates(self) -> None:
        study = ShiftStudy(
            generated_at=datetime.now(UTC),
            model_version="v1",
            threshold=0.348,
            environments=(
                EnvironmentSpec(
                    name="reference", description="d", overrides={}, seed=1, n_orders=10
                ),
            ),
            results=(result("reference"), result("cod_surge", lift_delta=-0.05)),
        )
        assert len(study.results) == 2


class TestSummarise:
    def test_findings_are_computed_from_lift_not_raw_pr_auc(self) -> None:
        """This is the trap the whole lift column exists to avoid.

        Raw PR-AUC rises when the base rate rises, because a random ranker scores
        PR-AUC equal to the positive rate. An environment whose raw PR-AUC went
        *up* while its lift went *down* must be reported as degraded.
        """
        findings = summarise(
            (
                result("reference", pr_auc=0.43, lift=2.57, rto=0.167),
                result(
                    "rto_base_rate_up",
                    pr_auc=0.56,  # higher than the reference
                    lift=2.0,  # but worse per unit of base rate
                    lift_delta=-0.57,
                    rto=0.237,
                ),
            )
        )
        text = " ".join(findings)
        assert "rto_base_rate_up" in text
        assert "ranking lift fell" in text

    def test_the_lift_caveat_is_stated_in_the_findings(self) -> None:
        findings = summarise((result("reference"), result("x", lift_delta=-0.01)))
        assert any("not comparable across environments" in finding for finding in findings)

    def test_a_small_lift_movement_is_not_called_degradation(self) -> None:
        findings = summarise(
            (
                result("reference"),
                result("mild", lift_delta=-MATERIAL_LIFT_DROP / 2),
            )
        )
        text = " ".join(findings)
        assert "ranking lift fell" not in text
        assert "left ranking lift within" in text

    def test_a_calibration_rise_is_called_the_more_serious_failure(self) -> None:
        findings = summarise(
            (result("reference"), result("drifted", ece=0.09, ece_delta=0.07, lift_delta=0.0))
        )
        assert any("more serious failure mode" in finding for finding in findings)

    def test_non_positive_economics_are_named(self) -> None:
        findings = summarise((result("reference"), result("bad", net=-2130.0, lift_delta=0.0)))
        assert any("stops paying for itself" in finding for finding in findings)

    def test_findings_avoid_the_rupee_sign(self) -> None:
        """These strings are printed to a terminal whose code page cannot encode it."""
        findings = summarise((result("reference"), result("bad", net=-1.0, lift_delta=0.0)))
        assert "₹" not in " ".join(findings)

    def test_a_study_with_no_shifted_environments_says_so(self) -> None:
        findings = summarise((result("reference"),))
        assert "nothing was shifted" in findings[0]

    def test_a_clean_study_does_not_claim_general_robustness(self) -> None:
        findings = summarise((result("reference"), result("mild", lift_delta=0.0, ece_delta=0.0)))
        text = " ".join(findings)
        assert "not a general robustness claim" in text
