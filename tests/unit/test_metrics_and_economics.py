"""Metrics and the cost model, checked against hand-computed values.

The economics half of this file matters more than it looks. An earlier version of
``eval.economics`` double-counted the false-negative term against the do-nothing
baseline and reported savings roughly three times too large. It looked plausible -
positive numbers, sensible ordering between rungs - and nothing in the pipeline
would have caught it. These are the tests that would have.
"""

from __future__ import annotations

import numpy as np
import pytest

from rto_sentinel.contracts.decision import CostInputs
from rto_sentinel.contracts.evaluation import PointEstimate
from rto_sentinel.decision.cost_model import expected_value_of_flagging, outcome_economics
from rto_sentinel.decision.threshold import derive_threshold, threshold_sensitivity
from rto_sentinel.eval import (
    bootstrap_metric,
    calibration_metrics,
    confusion_at_threshold,
    do_nothing_net_per_1000,
    economic_result,
    expected_calibration_error,
    pr_auc,
    precision_at_k,
    ranking_metrics,
    recall_at_precision,
    roc_auc,
)

#: The specification's worked example, section 06.
SPEC_INPUTS = CostInputs(
    rto_cost_inr=220.0,
    contribution_margin_inr=250.0,
    abandonment_on_friction=0.25,
    intervention_success_rate=0.60,
    friction_support_cost_inr=0.0,
)


# ---------------------------------------------------------------------------
# confusion matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_counts_and_rates() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    y_prob = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])

    matrix = confusion_at_threshold(y_true, y_prob, 0.5)

    assert (matrix.true_positives, matrix.false_positives) == (2, 1)
    assert (matrix.false_negatives, matrix.true_negatives) == (1, 4)
    assert matrix.flag_rate == pytest.approx(3 / 8)
    assert matrix.precision == pytest.approx(2 / 3)
    assert matrix.recall == pytest.approx(2 / 3)
    assert matrix.f1 == pytest.approx(2 / 3)


def test_the_threshold_comparison_is_inclusive() -> None:
    """``>=``, so a heuristic scoring exactly 1.0 is flagged at any threshold."""
    y_true = np.array([1, 0], dtype=bool)
    matrix = confusion_at_threshold(y_true, np.array([1.0, 1.0]), 1.0)
    assert matrix.n_flagged == 2


def test_precision_is_undefined_when_nothing_is_flagged() -> None:
    """Rung 0's situation. NaN, not zero - the distinction is the whole point."""
    matrix = confusion_at_threshold(np.array([1, 0, 0], dtype=bool), np.array([0.1, 0.1, 0.1]), 0.5)
    assert matrix.n_flagged == 0
    assert np.isnan(matrix.precision)
    assert np.isnan(matrix.f1)
    assert matrix.recall == 0.0  # defined: we caught none of the positives


# ---------------------------------------------------------------------------
# ranking metrics
# ---------------------------------------------------------------------------


def test_pr_auc_of_a_constant_predictor_is_the_base_rate() -> None:
    """The floor every other rung must clear to have shown any ranking ability."""
    y_true = np.array([1] * 20 + [0] * 80, dtype=bool)
    assert pr_auc(y_true, np.full(100, 0.3)) == pytest.approx(0.20, abs=1e-9)


def test_roc_auc_is_undefined_for_a_constant_predictor() -> None:
    """NaN, not 0.5. There is no ranking to score, which is not the same as chance."""
    y_true = np.array([1] * 20 + [0] * 80, dtype=bool)
    assert np.isnan(roc_auc(y_true, np.full(100, 0.3)))


def test_a_perfect_ranking_scores_one() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=bool)
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert pr_auc(y_true, y_prob) == pytest.approx(1.0)
    assert roc_auc(y_true, y_prob) == pytest.approx(1.0)


def test_recall_at_precision_returns_none_when_unreachable() -> None:
    """None, not zero. "Cannot reach 90% precision" is not "reached it with no recall"."""
    rng = np.random.default_rng(0)
    y_true = rng.random(500) < 0.2
    y_prob = rng.random(500)  # pure noise cannot hold 90% precision
    assert recall_at_precision(y_true, y_prob, 0.90) is None


def test_recall_at_precision_finds_a_reachable_target() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    y_prob = np.array([0.99, 0.98, 0.97, 0.10, 0.05, 0.04, 0.03, 0.02])
    recall = recall_at_precision(y_true, y_prob, 0.80)
    assert recall is not None and recall >= 0.75


def test_precision_at_k_reads_the_top_slice() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    assert precision_at_k(y_true, y_prob, 0.2) == pytest.approx(1.0)
    assert precision_at_k(y_true, y_prob, 0.5) == pytest.approx(0.4)


def test_ranking_metrics_carry_intervals() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.random(400) < 0.25
    y_prob = np.clip(y_true * 0.35 + rng.random(400) * 0.5, 0, 1)

    metrics = ranking_metrics(y_true, y_prob, bootstrap_iterations=80, seed=1)

    assert metrics.pr_auc.n_bootstrap > 0
    assert metrics.pr_auc.ci_low <= metrics.pr_auc.value <= metrics.pr_auc.ci_high
    assert metrics.pr_auc.ci_low < metrics.pr_auc.ci_high


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_a_perfectly_calibrated_predictor_has_near_zero_ece() -> None:
    rng = np.random.default_rng(7)
    probabilities = rng.uniform(0.05, 0.95, size=20000)
    outcomes = rng.random(20000) < probabilities

    ece, bins = expected_calibration_error(outcomes, probabilities, n_bins=10)
    assert ece < 0.02
    assert len(bins) == 10


def test_a_systematically_overconfident_predictor_has_large_ece() -> None:
    """The failure mode Phase 5 exists to fix: ranks fine, lies about magnitude."""
    rng = np.random.default_rng(7)
    true_probability = rng.uniform(0.05, 0.55, size=8000)
    outcomes = rng.random(8000) < true_probability
    inflated = np.clip(true_probability * 1.8, 0, 1)

    ece, _ = expected_calibration_error(outcomes, inflated, n_bins=10)
    assert ece > 0.15


def test_empty_bins_are_skipped_not_counted_as_perfect() -> None:
    """Otherwise a two-valued predictor reports a flattering ECE by leaving bins empty."""
    y_true = np.array([1, 0, 1, 0], dtype=bool)
    y_prob = np.array([0.05, 0.05, 0.95, 0.95])
    _, bins = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert len(bins) == 2


def test_calibration_metrics_returns_reliability_bins() -> None:
    rng = np.random.default_rng(3)
    probabilities = rng.uniform(0.1, 0.9, size=2000)
    outcomes = rng.random(2000) < probabilities

    metrics = calibration_metrics(outcomes, probabilities, n_bins=8)
    assert metrics.n_bins == 8
    assert 0.0 <= metrics.brier_score <= 1.0
    assert all(count > 0 for _, _, count in metrics.reliability_bins)


# ---------------------------------------------------------------------------
# the cost model
# ---------------------------------------------------------------------------


def test_the_worked_example_reproduces_the_specification() -> None:
    """SPEC section 06: C_fp = 62.5, S_tp = 132, threshold = 0.3214. Not 0.5."""
    economics = outcome_economics(SPEC_INPUTS)
    assert economics.false_positive_cost_inr == pytest.approx(62.5)
    assert economics.true_positive_saving_inr == pytest.approx(132.0)

    derivation = derive_threshold(SPEC_INPUTS)
    assert derivation.threshold == pytest.approx(0.3214, abs=1e-4)
    assert derivation.threshold != pytest.approx(0.5)


def test_the_friction_support_cost_raises_the_threshold() -> None:
    """A per-friction ops cost makes a false positive dearer, so flag less readily."""
    with_support = SPEC_INPUTS.model_copy(update={"friction_support_cost_inr": 8.0})
    assert derive_threshold(with_support).threshold > derive_threshold(SPEC_INPUTS).threshold


def test_a_higher_margin_merchant_flags_less_readily() -> None:
    """The demo's central claim, asserted rather than narrated.

    A larger contribution margin makes a false positive more expensive, so the
    threshold rises and fewer orders are flagged.
    """
    thin = SPEC_INPUTS.model_copy(update={"contribution_margin_inr": 90.0})
    rich = SPEC_INPUTS.model_copy(update={"contribution_margin_inr": 520.0})
    assert derive_threshold(rich).threshold > derive_threshold(thin).threshold


def test_a_useless_intervention_yields_a_threshold_of_one() -> None:
    """If frictioning saves nothing, the honest answer is to flag nothing."""
    useless = SPEC_INPUTS.model_copy(update={"intervention_success_rate": 0.0})
    assert derive_threshold(useless).threshold == pytest.approx(1.0)


def test_expected_value_crosses_zero_at_the_threshold() -> None:
    """The threshold is where flagging stops being worth it, by construction."""
    threshold = derive_threshold(SPEC_INPUTS).threshold
    assert expected_value_of_flagging(threshold, SPEC_INPUTS) == pytest.approx(0.0, abs=1e-9)
    assert expected_value_of_flagging(threshold + 0.05, SPEC_INPUTS) > 0
    assert expected_value_of_flagging(threshold - 0.05, SPEC_INPUTS) < 0


def test_threshold_sensitivity_moves_in_the_expected_direction() -> None:
    curve = threshold_sensitivity(
        SPEC_INPUTS, parameter="contribution_margin_inr", perturbations=[-0.3, 0.0, 0.3]
    )
    thresholds = [derivation.threshold for _, derivation in curve]
    assert thresholds[0] < thresholds[1] < thresholds[2]


def test_an_unknown_cost_parameter_is_refused() -> None:
    with pytest.raises(ValueError, match="not a cost input"):
        threshold_sensitivity(SPEC_INPUTS, parameter="wishful_thinking", perturbations=[0.0])


# ---------------------------------------------------------------------------
# economics
# ---------------------------------------------------------------------------


def test_net_savings_are_measured_against_doing_nothing() -> None:
    """Hand-computed, because this is where the double count hid.

    30 true positives and 71 false positives on 419 orders::

        delta   = 30 x 132.0 - 71 x 62.5 = 3960 - 4437.5 = -477.5
        per 1000 = -477.5 / 419 x 1000   = -1139.6

    A negative figure is a real result: at this operating point the intervention
    costs more than it saves.
    """
    economics = outcome_economics(SPEC_INPUTS)
    delta = economics.net_versus_doing_nothing(tp=30, fp=71)
    assert delta == pytest.approx(30 * 132.0 - 71 * 62.5)
    assert delta / 419 * 1000 == pytest.approx(-1139.6, abs=0.5)


def test_doing_nothing_scores_exactly_zero() -> None:
    """The reference point. Flagging nothing changes nothing, by definition."""
    y_true = np.array([1] * 20 + [0] * 80, dtype=bool)
    result = economic_result(y_true, np.full(100, 0.05), threshold=0.5, cost_inputs=SPEC_INPUTS)
    assert result.true_positives == 0 and result.false_positives == 0
    assert result.net_inr_saved_per_1000_orders.value == pytest.approx(0.0)
    assert result.flag_rate == 0.0


def test_the_baseline_loss_is_reported_separately() -> None:
    """Context, not the headline: the size of the problem being attacked."""
    y_true = np.array([1] * 20 + [0] * 80, dtype=bool)
    expected = -(0.20 * 220.0) * 1000
    assert do_nothing_net_per_1000(y_true, SPEC_INPUTS) == pytest.approx(expected)


def test_a_perfect_classifier_saves_the_full_intervention_value() -> None:
    """No false positives, so the saving is exactly TP x S_tp."""
    y_true = np.array([1] * 20 + [0] * 80, dtype=bool)
    y_prob = np.where(y_true, 0.99, 0.01)

    result = economic_result(y_true, y_prob, threshold=0.5, cost_inputs=SPEC_INPUTS)
    assert result.false_positives == 0
    assert result.net_inr_saved_per_1000_orders.value == pytest.approx(20 * 132.0 / 100 * 1000)


def test_false_positive_cost_is_reported_and_never_netted_away() -> None:
    """SPEC section 07: stated separately, always."""
    y_true = np.array([1] * 20 + [0] * 80, dtype=bool)
    y_prob = np.full(100, 0.9)  # flag everything

    result = economic_result(y_true, y_prob, threshold=0.5, cost_inputs=SPEC_INPUTS)
    assert result.false_positives == 80
    assert result.total_false_positive_cost_inr == pytest.approx(80 * 62.5)
    assert result.total_false_positive_cost_inr > 0
    assert result.flag_rate == 1.0


def test_flagging_everything_loses_money_at_these_economics() -> None:
    """The blanket-block pathology, in one assertion."""
    y_true = np.array([1] * 20 + [0] * 80, dtype=bool)
    result = economic_result(y_true, np.full(100, 0.9), threshold=0.5, cost_inputs=SPEC_INPUTS)
    assert result.net_inr_saved_per_1000_orders.value < 0


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_produces_a_containing_interval() -> None:
    rng = np.random.default_rng(2)
    y_true = rng.random(600) < 0.3
    y_prob = np.clip(y_true * 0.4 + rng.random(600) * 0.5, 0, 1)

    estimate = bootstrap_metric(y_true, y_prob, pr_auc, iterations=200, seed=2)
    assert estimate.ci_low <= estimate.value <= estimate.ci_high
    assert estimate.n_bootstrap > 0


def test_bootstrap_is_reproducible_under_a_fixed_seed() -> None:
    rng = np.random.default_rng(2)
    y_true = rng.random(300) < 0.3
    y_prob = rng.random(300)

    first = bootstrap_metric(y_true, y_prob, pr_auc, iterations=100, seed=9)
    second = bootstrap_metric(y_true, y_prob, pr_auc, iterations=100, seed=9)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_zero_iterations_is_marked_as_having_no_interval() -> None:
    """The fast path for unit tests. ``n_bootstrap=0`` is what makes that visible."""
    y_true = np.array([1, 0, 1, 0], dtype=bool)
    estimate = bootstrap_metric(y_true, np.array([0.9, 0.1, 0.8, 0.2]), pr_auc, iterations=0)
    assert estimate.n_bootstrap == 0
    assert estimate.ci_low == estimate.value == estimate.ci_high


def test_an_undefined_metric_stays_undefined_through_the_bootstrap() -> None:
    y_true = np.array([1] * 10 + [0] * 40, dtype=bool)
    estimate = bootstrap_metric(y_true, np.full(50, 0.3), roc_auc, iterations=50, seed=1)
    assert not estimate.is_defined


def test_a_partially_undefined_estimate_is_refused() -> None:
    """A value with no interval is a bug, not a statement."""
    nan = float("nan")
    with pytest.raises(ValueError, match="partially undefined"):
        PointEstimate(value=0.5, ci_low=nan, ci_high=nan)
