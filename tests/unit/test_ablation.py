"""The ablation's reading rules, which are the only judgement it makes.

The arithmetic is a mean over per-order rupees and a bootstrap - neither is
subtle. What matters is that an arm whose interval spans zero is never reported
as a contribution, and that a family which cannot be disabled fails loudly rather
than silently running the full model again.
"""

from __future__ import annotations

import numpy as np
import pytest

from rto_sentinel.configuration import load_features_config
from rto_sentinel.eval.ablation import (
    MATERIAL_DELTA_INR,
    AblationResult,
    _bootstrap_net_delta,
    _per_order_net,
    disable_family,
    summarise,
)
from rto_sentinel.settings import get_settings


def arm(
    name: str,
    *,
    delta: float,
    low: float,
    high: float,
    pr_auc: float = 0.48,
    delta_pr: float = 0.0,
) -> AblationResult:
    return AblationResult(
        family_removed=name,
        n_features=45,
        net_inr_per_1000=5169.0 + delta,
        delta_vs_full=delta,
        delta_ci_low=low,
        delta_ci_high=high,
        pr_auc=pr_auc,
        delta_pr_auc_vs_full=delta_pr,
        flag_rate=0.19,
        precision=0.48,
        threshold=0.3481,
        chosen_candidate="minimal",
        calibration_method="platt",
    )


FULL = AblationResult(
    family_removed="__full__",
    n_features=54,
    net_inr_per_1000=5169.0,
    delta_vs_full=0.0,
    delta_ci_low=0.0,
    delta_ci_high=0.0,
    pr_auc=0.484,
    delta_pr_auc_vs_full=0.0,
    flag_rate=0.191,
    precision=0.482,
    threshold=0.3481,
    chosen_candidate="minimal",
    calibration_method="platt",
)


class TestDisableFamily:
    def test_the_named_family_is_switched_off(self) -> None:
        config = load_features_config(get_settings())
        assert config.families["geography_route"].enabled is True

        ablated = disable_family(config, "geography_route")

        assert ablated.families["geography_route"].enabled is False

    def test_the_other_families_are_untouched(self) -> None:
        config = load_features_config(get_settings())
        ablated = disable_family(config, "geography_route")

        for name, family in config.families.items():
            if name != "geography_route":
                assert ablated.families[name].enabled == family.enabled

    def test_the_original_config_is_not_mutated(self) -> None:
        config = load_features_config(get_settings())
        disable_family(config, "geography_route")
        assert config.families["geography_route"].enabled is True

    def test_an_unknown_family_is_fatal(self) -> None:
        """A silently-skipped arm would read as "this family contributes nothing".

        That is the single most misleading result this module could produce, so a
        typo raises rather than returning the full config.
        """
        config = load_features_config(get_settings())
        with pytest.raises(KeyError, match="unknown feature family"):
            disable_family(config, "geograhpy_route")


class TestPerOrderNet:
    def test_a_flagged_true_positive_saves_the_true_positive_saving(self) -> None:
        net = _per_order_net(
            np.array([1]),
            np.array([0.9]),
            threshold=0.5,
            cost_false_positive_inr=70.0,
            saving_true_positive_inr=132.0,
        )
        assert net.tolist() == [132.0]

    def test_a_flagged_false_positive_costs(self) -> None:
        net = _per_order_net(
            np.array([0]),
            np.array([0.9]),
            threshold=0.5,
            cost_false_positive_inr=70.0,
            saving_true_positive_inr=132.0,
        )
        assert net.tolist() == [-70.0]

    def test_an_unflagged_order_contributes_nothing_either_way(self) -> None:
        """The false-negative loss is identical under both arms and cancels.

        Charging it here would make every arm look worse by the same constant and
        change no delta, while making the absolute figures wrong.
        """
        net = _per_order_net(
            np.array([1, 0]),
            np.array([0.1, 0.1]),
            threshold=0.5,
            cost_false_positive_inr=70.0,
            saving_true_positive_inr=132.0,
        )
        assert net.tolist() == [0.0, 0.0]

    def test_a_probability_exactly_at_the_threshold_is_flagged(self) -> None:
        net = _per_order_net(
            np.array([1]),
            np.array([0.5]),
            threshold=0.5,
            cost_false_positive_inr=70.0,
            saving_true_positive_inr=132.0,
        )
        assert net.tolist() == [132.0]


class TestBootstrap:
    def test_identical_arms_give_an_interval_containing_zero(self) -> None:
        per_order = np.random.default_rng(1).choice([132.0, -70.0, 0.0], size=800)
        low, high = _bootstrap_net_delta(per_order, per_order.copy(), seed=1)
        assert low == 0.0 and high == 0.0

    def test_a_uniformly_worse_arm_gives_a_negative_interval(self) -> None:
        full = np.full(800, 132.0)
        worse = np.full(800, -70.0)
        _, high = _bootstrap_net_delta(full, worse, seed=2)
        assert high < 0

    def test_the_interval_is_paired_not_independent(self) -> None:
        """Both arms scored the same orders; independent resampling invents variance.

        Two arms that differ by a constant have zero variance in their paired
        difference, so a correctly paired bootstrap returns a degenerate interval.
        An unpaired one would return a wide one.
        """
        rng = np.random.default_rng(3)
        full = rng.choice([132.0, -70.0, 0.0], size=1000)
        arm_scores = full - 10.0
        low, high = _bootstrap_net_delta(full, arm_scores, seed=3)
        assert low == pytest.approx(-10_000.0)
        assert high == pytest.approx(-10_000.0)

    def test_an_empty_split_does_not_divide_by_zero(self) -> None:
        assert _bootstrap_net_delta(np.array([]), np.array([]), seed=1) == (0.0, 0.0)


class TestVerdict:
    def test_an_interval_spanning_zero_is_not_established(self) -> None:
        assert arm("x", delta=-757, low=-1879, high=472).verdict == "not established"

    def test_a_clearly_negative_delta_earns_its_place(self) -> None:
        """Removing it costs money, so keeping it pays."""
        assert arm("x", delta=-4313, low=-6163, high=-2510).verdict == "earns its place"

    def test_a_clearly_positive_delta_costs_money(self) -> None:
        """Removing it *improves* the number, so the family is a liability."""
        assert arm("x", delta=+900, low=+300, high=+1500).verdict == "costs money"

    def test_a_tiny_delta_is_no_material_effect_even_with_a_tight_interval(self) -> None:
        """One order changing sides is a rounding, not a decision."""
        tiny = MATERIAL_DELTA_INR / 2
        assert arm("x", delta=-tiny, low=-tiny - 1, high=-tiny + 1).verdict == (
            "no material effect"
        )

    def test_the_full_model_is_the_reference(self) -> None:
        assert FULL.verdict == "reference"


class TestSummarise:
    def test_the_reference_is_stated_first(self) -> None:
        findings = summarise(FULL, (arm("a", delta=-100, low=-500, high=300),))
        assert findings[0].startswith("Full model:")

    def test_an_unestablished_arm_is_not_reported_as_a_contribution(self) -> None:
        findings = summarise(FULL, (arm("customer_history", delta=-757, low=-1879, high=472),))
        text = " ".join(findings)
        assert "earns its place" not in text
        assert "no established economic effect" in text

    def test_the_overlap_caveat_is_always_attached_to_unestablished_arms(self) -> None:
        """ "Not established" must never be readable as "worthless"."""
        findings = summarise(FULL, (arm("session_intent", delta=-234, low=-1063, high=575),))
        text = " ".join(findings)
        assert "not evidence they are worthless" in text
        assert "overlapping signal hides individual value" in text

    def test_geography_failing_to_pay_is_called_out_against_its_fairness_cost(self) -> None:
        findings = summarise(FULL, (arm("geography_route", delta=-40, low=-900, high=800),))
        text = " ".join(findings)
        assert "highest fairness risk" in text
        assert "has not been shown to" in text

    def test_a_marginal_geography_result_is_flagged_as_marginal(self) -> None:
        """Clearing zero by a hair is not the same as earning its place comfortably.

        This is the measured case: the interval's upper bound was -88.
        """
        findings = summarise(FULL, (arm("geography_route", delta=-1043, low=-1948, high=-88),))
        text = " ".join(findings)
        assert "does pay for itself" in text
        assert "marginal result rather than a comfortable one" in text

    def test_a_comfortable_geography_result_is_not_dressed_up_as_marginal(self) -> None:
        findings = summarise(FULL, (arm("geography_route", delta=-3000, low=-4000, high=-2000),))
        assert "marginal result" not in " ".join(findings)

    def test_customer_history_gets_its_overlap_explanation(self) -> None:
        findings = summarise(FULL, (arm("customer_history", delta=-757, low=-1879, high=472),))
        text = " ".join(findings)
        assert "strongest honest signal" in text
        assert "cannot separate them" in text

    def test_findings_are_deterministic(self) -> None:
        arms = (arm("a", delta=-4313, low=-6163, high=-2510),)
        assert summarise(FULL, arms) == summarise(FULL, arms)
