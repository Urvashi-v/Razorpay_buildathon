"""Portfolio economics, the threshold sweep, and the merchant simulator.

The book-level arithmetic is where a sign error hides best: every individual
number looks plausible and only the total is wrong. So the tests here compute the
expected answers by hand on tiny books where the arithmetic can be written out,
and only then check the behaviour on a realistic one.
"""

from __future__ import annotations

import numpy as np
import pytest

from rto_sentinel.configuration.schemas import PolicyConfig
from rto_sentinel.contracts.decision import CostInputs
from rto_sentinel.contracts.enums import InterventionAction, RiskBand
from rto_sentinel.decision.cost_model import band_outcome_economics
from rto_sentinel.decision.policy import band_economics
from rto_sentinel.decision.portfolio import PortfolioError, evaluate_portfolio
from rto_sentinel.decision.simulation import (
    SimulationError,
    compare_ladder_against_uniform,
    simulate,
)
from rto_sentinel.decision.threshold import derive_threshold
from rto_sentinel.decision.threshold_analysis import SweepError, sweep_thresholds

SPEC_INPUTS = CostInputs(
    rto_cost_inr=220.0,
    contribution_margin_inr=250.0,
    abandonment_on_friction=0.25,
    intervention_success_rate=0.60,
    friction_support_cost_inr=8.0,
)


@pytest.fixture
def book() -> tuple[np.ndarray, np.ndarray]:
    """A deterministic book whose scores are calibrated by construction."""
    rng = np.random.default_rng(20260901)
    scores = rng.uniform(0.0, 1.0, size=4000)
    labels = rng.uniform(size=4000) < scores
    return scores, labels


# ---------------------------------------------------------------------------
# the arithmetic, hand-checked on a book small enough to write out
# ---------------------------------------------------------------------------


def test_a_two_order_book_prices_exactly(policy_config: PolicyConfig) -> None:
    """One order below the threshold, one in ELEVATED. Computed by hand."""
    threshold = derive_threshold(SPEC_INPUTS).threshold
    low, elevated = 0.10, threshold + 0.01
    result = evaluate_portfolio(
        np.array([low, elevated]), cost_inputs=SPEC_INPUTS, policy=policy_config
    )

    economics = band_outcome_economics(
        SPEC_INPUTS, band_economics(RiskBand.ELEVATED, policy_config)
    )
    expected_saving = elevated * economics.true_positive_saving_inr
    expected_cost = (1.0 - elevated) * economics.false_positive_cost_inr

    assert result.n_orders == 2
    assert result.expected_orders_affected == 1
    assert result.expected_savings_inr == pytest.approx(expected_saving)
    assert result.expected_false_positive_cost_inr == pytest.approx(expected_cost)
    assert result.expected_net_inr == pytest.approx(expected_saving - expected_cost)
    assert result.expected_net_inr_per_1000_orders == pytest.approx(
        (expected_saving - expected_cost) / 2 * 1000
    )


def test_the_residual_loss_counts_unflagged_and_unsaved_orders(
    policy_config: PolicyConfig,
) -> None:
    """An unflagged RTO costs the full amount; a flagged one costs what is not saved."""
    threshold = derive_threshold(SPEC_INPUTS).threshold
    result = evaluate_portfolio(
        np.array([0.05, threshold + 0.01]), cost_inputs=SPEC_INPUTS, policy=policy_config
    )
    elevated = band_economics(RiskBand.ELEVATED, policy_config)
    success = SPEC_INPUTS.intervention_success_rate * elevated.intervention_success_multiplier

    expected = (
        0.05 * SPEC_INPUTS.rto_cost_inr
        + (threshold + 0.01) * (1.0 - success) * SPEC_INPUTS.rto_cost_inr
    )
    assert result.expected_false_negative_loss_inr == pytest.approx(expected)


def test_doing_nothing_earns_exactly_zero(policy_config: PolicyConfig) -> None:
    """A book entirely below the threshold changes nothing, so the net is zero."""
    result = evaluate_portfolio(
        np.array([0.01, 0.02, 0.03]), cost_inputs=SPEC_INPUTS, policy=policy_config
    )
    assert result.expected_orders_affected == 0
    assert result.flag_rate == 0.0
    assert result.expected_net_inr_per_1000_orders == pytest.approx(0.0)
    assert result.expected_savings_inr == pytest.approx(0.0)


def test_the_bands_partition_the_book(policy_config: PolicyConfig, book) -> None:
    """Every order lands in exactly one band. Asserted, not assumed."""
    scores, labels = book
    result = evaluate_portfolio(
        scores, cost_inputs=SPEC_INPUTS, policy=policy_config, labels=labels
    )
    assert sum(band.n_orders for band in result.bands) == len(scores)
    assert sum(band.share_of_book for band in result.bands) == pytest.approx(1.0)


def test_flag_rate_and_intervention_rate_agree_for_this_ladder(
    policy_config: PolicyConfig, book
) -> None:
    """LOW's ceiling is the threshold, so the two coincide. They are still computed
    separately: a policy whose lowest acting band started above the threshold would
    make them differ, and a single number would hide it."""
    scores, labels = book
    result = evaluate_portfolio(
        scores, cost_inputs=SPEC_INPUTS, policy=policy_config, labels=labels
    )
    assert result.flag_rate == pytest.approx(result.intervention_rate)
    assert result.expected_orders_affected == round(result.flag_rate * len(scores))


def test_expected_and_realized_agree_on_calibrated_scores(
    policy_config: PolicyConfig, book
) -> None:
    """The fixture's labels are drawn from the scores, so calibration is exact.

    The gap between expected and realized is the calibration check the report
    reports; on a perfectly calibrated book it must be small.
    """
    scores, labels = book
    result = evaluate_portfolio(
        scores, cost_inputs=SPEC_INPUTS, policy=policy_config, labels=labels
    )
    assert result.calibration_gap is not None
    assert abs(result.calibration_gap) < 0.1 * result.expected_true_positives
    assert result.realized_net_inr_per_1000_orders == pytest.approx(
        result.expected_net_inr_per_1000_orders, rel=0.15
    )


def test_labels_are_optional(policy_config: PolicyConfig, book) -> None:
    """The live case: a merchant pricing today's unlabelled orders."""
    scores, _ = book
    result = evaluate_portfolio(scores, cost_inputs=SPEC_INPUTS, policy=policy_config)

    assert result.realized_net_inr_per_1000_orders is None
    assert result.calibration_gap is None
    assert result.expected_net_inr_per_1000_orders != 0.0


def test_the_holdout_reduces_the_net_by_its_share(policy_config: PolicyConfig, book) -> None:
    scores, labels = book
    result = evaluate_portfolio(
        scores, cost_inputs=SPEC_INPUTS, policy=policy_config, labels=labels
    )
    expected = result.expected_net_inr_per_1000_orders * (
        1.0 - policy_config.holdout_control.fraction_of_flagged
    )
    assert result.net_inr_per_1000_after_holdout == pytest.approx(expected)


def test_every_headline_quantity_carries_its_provenance(policy_config: PolicyConfig, book) -> None:
    """A merchant input and a measured metric must not be indistinguishable."""
    from rto_sentinel.contracts.provenance import Provenance

    scores, _ = book
    result = evaluate_portfolio(scores, cost_inputs=SPEC_INPUTS, policy=policy_config)
    by_name = {quantity.name: quantity for quantity in result.quantities}

    assert by_name["contribution_margin_inr"].provenance is Provenance.MERCHANT_INPUT
    assert by_name["intervention_success_rate"].provenance is Provenance.ASSUMED_INTERVENTION
    assert by_name["intervention_success_rate"].is_assumption
    assert by_name["operating_threshold"].provenance is Provenance.DERIVED
    # A rupee total inherits the assumption it rests on.
    assert by_name["net_inr_saved_per_1000_orders"].is_assumption


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_an_empty_book_is_refused(policy_config: PolicyConfig) -> None:
    with pytest.raises(PortfolioError, match="empty book"):
        evaluate_portfolio(np.array([]), cost_inputs=SPEC_INPUTS, policy=policy_config)


def test_out_of_range_probabilities_are_refused(policy_config: PolicyConfig) -> None:
    with pytest.raises(PortfolioError, match=r"\[0, 1\]"):
        evaluate_portfolio(np.array([0.5, 1.4]), cost_inputs=SPEC_INPUTS, policy=policy_config)


def test_mismatched_labels_are_refused(policy_config: PolicyConfig) -> None:
    with pytest.raises(PortfolioError, match="disagree in length"):
        evaluate_portfolio(
            np.array([0.5, 0.6]),
            cost_inputs=SPEC_INPUTS,
            policy=policy_config,
            labels=np.array([1]),
        )


def test_non_finite_probabilities_are_refused(policy_config: PolicyConfig) -> None:
    with pytest.raises(PortfolioError, match="non-finite"):
        evaluate_portfolio(np.array([0.5, np.nan]), cost_inputs=SPEC_INPUTS, policy=policy_config)


# ---------------------------------------------------------------------------
# monotonicity at the book level
# ---------------------------------------------------------------------------


def test_a_higher_margin_flags_fewer_orders(policy_config: PolicyConfig, book) -> None:
    scores, labels = book
    flag_rates = [
        evaluate_portfolio(
            scores,
            cost_inputs=SPEC_INPUTS.model_copy(update={"contribution_margin_inr": margin}),
            policy=policy_config,
            labels=labels,
        ).flag_rate
        for margin in (50.0, 250.0, 600.0, 1200.0)
    ]
    assert flag_rates == sorted(flag_rates, reverse=True)


def test_a_higher_rto_cost_flags_more_orders(policy_config: PolicyConfig, book) -> None:
    scores, labels = book
    flag_rates = [
        evaluate_portfolio(
            scores,
            cost_inputs=SPEC_INPUTS.model_copy(update={"rto_cost_inr": cost}),
            policy=policy_config,
            labels=labels,
        ).flag_rate
        for cost in (80.0, 220.0, 600.0, 1500.0)
    ]
    assert flag_rates == sorted(flag_rates)


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


def test_the_sweep_contains_the_derived_operating_point(book) -> None:
    scores, labels = book
    sweep = sweep_thresholds(scores, labels, cost_inputs=SPEC_INPUTS)

    marked = [point for point in sweep.points if point.is_derived_operating_point]
    assert len(marked) == 1
    assert marked[0].threshold == pytest.approx(sweep.derived_threshold)


def test_the_sweep_refuses_the_sealed_split(book) -> None:
    """A curve over test labels can only be used to tune against them."""
    scores, labels = book
    with pytest.raises(SweepError, match="sealed test split"):
        sweep_thresholds(scores, labels, cost_inputs=SPEC_INPUTS, split="test")


def test_the_sweep_states_that_the_threshold_is_not_read_off_it(book) -> None:
    scores, labels = book
    sweep = sweep_thresholds(scores, labels, cost_inputs=SPEC_INPUTS)
    assert "derived" in sweep.selection_methodology.lower()
    assert "never read off this curve" in sweep.selection_methodology


def test_the_peak_is_reported_separately_from_the_operating_point(book) -> None:
    """The two are different fields precisely so they cannot be conflated."""
    scores, labels = book
    sweep = sweep_thresholds(scores, labels, cost_inputs=SPEC_INPUTS)
    assert sweep.derived_threshold == pytest.approx(derive_threshold(SPEC_INPUTS).threshold)
    assert 0.0 <= sweep.best_net_threshold <= 1.0


def test_flag_rate_falls_monotonically_across_the_sweep(book) -> None:
    scores, labels = book
    sweep = sweep_thresholds(scores, labels, cost_inputs=SPEC_INPUTS)
    flag_rates = [point.flag_rate for point in sweep.points]
    assert flag_rates == sorted(flag_rates, reverse=True)


def test_recall_falls_monotonically_across_the_sweep(book) -> None:
    """Raising the bar can only catch fewer of the positives."""
    scores, labels = book
    sweep = sweep_thresholds(scores, labels, cost_inputs=SPEC_INPUTS)
    recalls = [point.recall for point in sweep.points if point.recall is not None]
    assert recalls == sorted(recalls, reverse=True)


def test_the_sweep_works_without_labels(book) -> None:
    scores, _ = book
    sweep = sweep_thresholds(scores, None, cost_inputs=SPEC_INPUTS)
    assert all(point.precision is None for point in sweep.points)
    assert all(point.realized_net_inr_per_1000_orders is None for point in sweep.points)


# ---------------------------------------------------------------------------
# the merchant simulator
# ---------------------------------------------------------------------------


def test_the_simulator_recomputes_everything(policy_config: PolicyConfig, book) -> None:
    """Margin 250 -> 400: threshold, bands, assignment and rupees all move."""
    scores, labels = book
    base = SPEC_INPUTS
    changed = SPEC_INPUTS.model_copy(update={"contribution_margin_inr": 400.0})

    before = simulate(scores, cost_inputs=base, policy=policy_config, labels=labels)
    after = simulate(
        scores, cost_inputs=changed, policy=policy_config, labels=labels, baseline=base
    )

    assert after.threshold.threshold > before.threshold.threshold
    assert after.economics.flag_rate < before.economics.flag_rate
    assert after.threshold_delta == pytest.approx(
        after.threshold.threshold - before.threshold.threshold
    )
    assert after.net_delta_inr_per_1000_orders is not None
    # Band boundaries moved with the threshold, not just the flag count.
    assert after.ladder[1].lower_bound > before.ladder[1].lower_bound


def test_the_simulator_refuses_the_sealed_split(policy_config: PolicyConfig, book) -> None:
    scores, labels = book
    with pytest.raises(SimulationError, match="sealed test split"):
        simulate(scores, cost_inputs=SPEC_INPUTS, policy=policy_config, labels=labels, split="test")


def test_simulation_is_deterministic(policy_config: PolicyConfig, book) -> None:
    scores, labels = book
    first = simulate(scores, cost_inputs=SPEC_INPUTS, policy=policy_config, labels=labels)
    second = simulate(scores, cost_inputs=SPEC_INPUTS, policy=policy_config, labels=labels)
    assert first.model_dump_json() == second.model_dump_json()


def test_the_ladder_is_priced_against_uniform_alternatives(
    policy_config: PolicyConfig, book
) -> None:
    """Whether graduation pays is measured, not assumed."""
    scores, labels = book
    comparison = compare_ladder_against_uniform(
        scores, cost_inputs=SPEC_INPUTS, policy=policy_config, labels=labels
    )

    acting = {
        band.action for band in policy_config.bands if band.action != InterventionAction.NONE.value
    }
    assert set(comparison.uniform_net_inr_per_1000) == acting
    assert comparison.best_uniform_action in acting
    assert "ASSUMED" in comparison.note
    assert isinstance(comparison.graduated_wins, bool)


def test_a_uniform_policy_of_the_anchor_action_matches_a_flat_ladder(
    policy_config: PolicyConfig, book
) -> None:
    """Sanity check on the comparison's construction, not on which policy wins."""
    scores, _ = book
    comparison = compare_ladder_against_uniform(
        scores, cost_inputs=SPEC_INPUTS, policy=policy_config
    )
    flat = policy_config.model_copy(
        update={
            "bands": [
                band
                if band.action == "none"
                else band.model_copy(
                    update={
                        "economics": band_economics(RiskBand.HIGH, policy_config),
                        "action": "confirmation_required",
                    }
                )
                for band in policy_config.bands
            ]
        }
    )
    direct = evaluate_portfolio(
        scores, cost_inputs=SPEC_INPUTS, policy=flat
    ).expected_net_inr_per_1000_orders
    assert comparison.uniform_net_inr_per_1000["confirmation_required"] == pytest.approx(direct)
