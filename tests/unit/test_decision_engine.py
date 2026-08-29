"""The decision engine: arithmetic, boundaries, monotonicity and determinism.

This is the module a payments company would audit first, so the tests are
arranged around the four properties an auditor would ask about:

* the arithmetic reproduces the specification's worked example by hand;
* the boundaries behave at exactly the values where floating point and
  half-open intervals are most likely to disagree;
* the direction of every response to an input is the direction the formula
  implies, including the one this repository documented backwards until now;
* the same inputs produce the same decision, byte for byte, forever.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rto_sentinel.configuration.schemas import CostModelConfig, PolicyConfig
from rto_sentinel.contracts.decision import CostInputs, Decision
from rto_sentinel.contracts.enums import InterventionAction, RiskBand
from rto_sentinel.contracts.risk import FeatureContribution, RiskScore
from rto_sentinel.decision.cost_model import (
    band_outcome_economics,
    expected_value_of_flagging,
    outcome_economics,
)
from rto_sentinel.decision.engine import (
    SCORE_ONLY_REASON,
    DecisionEngine,
    UncalibratedScoreError,
)
from rto_sentinel.decision.policy import (
    PolicyError,
    band_economics,
    band_for,
    resolve_boundaries,
)
from rto_sentinel.decision.reason_codes import code_for, derive_reason_codes
from rto_sentinel.decision.threshold import derive_threshold

#: The specification's worked example, with no support cost so it reproduces the
#: published number exactly.
SPEC_INPUTS = CostInputs(
    rto_cost_inr=220.0,
    contribution_margin_inr=250.0,
    abandonment_on_friction=0.25,
    intervention_success_rate=0.60,
    friction_support_cost_inr=0.0,
)

FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def engine(policy_config: PolicyConfig) -> DecisionEngine:
    return DecisionEngine(policy_config)


def _score(probability: float, *, calibrated: bool = True, **kwargs: object) -> RiskScore:
    return RiskScore(
        order_id=kwargs.pop("order_id", "ORD-00000001"),  # type: ignore[arg-type]
        probability=probability,
        model_name="lightgbm",
        model_version="v1",
        calibration_method="platt" if calibrated else None,
        scored_at=FIXED_TIME,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# the arithmetic, checked by hand
# ---------------------------------------------------------------------------


def test_the_specification_worked_example_reproduces_exactly() -> None:
    """C_fp = 62.5, S_tp = 132, threshold = 0.3214. Not 0.5."""
    derivation = derive_threshold(SPEC_INPUTS)

    assert derivation.cost_false_positive_inr == pytest.approx(62.5)
    assert derivation.saving_true_positive_inr == pytest.approx(132.0)
    assert derivation.threshold == pytest.approx(62.5 / 194.5, abs=1e-9)
    assert derivation.threshold == pytest.approx(0.3214, abs=1e-4)
    assert derivation.threshold != pytest.approx(0.5)


def test_the_support_cost_enters_the_false_positive_side() -> None:
    """It makes friction dearer, so the threshold rises."""
    with_support = SPEC_INPUTS.model_copy(update={"friction_support_cost_inr": 8.0})
    economics = outcome_economics(with_support)

    assert economics.false_positive_cost_inr == pytest.approx(0.25 * 250.0 + 8.0)
    assert derive_threshold(with_support).threshold > derive_threshold(SPEC_INPUTS).threshold


def test_expected_value_is_zero_exactly_at_the_threshold() -> None:
    """The threshold is defined as the indifference point, so this must hold."""
    threshold = derive_threshold(SPEC_INPUTS).threshold
    assert expected_value_of_flagging(threshold, SPEC_INPUTS) == pytest.approx(0.0, abs=1e-9)
    assert expected_value_of_flagging(threshold + 0.05, SPEC_INPUTS) > 0
    assert expected_value_of_flagging(threshold - 0.05, SPEC_INPUTS) < 0


def test_net_against_doing_nothing_has_no_false_negative_term() -> None:
    """Hand-computed: the FN term cancels, and this is the arithmetic proving it."""
    economics = outcome_economics(SPEC_INPUTS)
    # 10 caught, 4 wrongly frictioned.
    expected = 10 * 132.0 - 4 * 62.5
    assert economics.net_versus_doing_nothing(tp=10, fp=4) == pytest.approx(expected)
    assert economics.net_versus_doing_nothing(tp=0, fp=0) == pytest.approx(0.0)


def test_band_economics_scale_the_merchant_rates(policy_config: PolicyConfig) -> None:
    """A gentler rung saves less and costs less. Checked against the config."""
    high = band_economics(RiskBand.HIGH, policy_config)
    elevated = band_economics(RiskBand.ELEVATED, policy_config)

    high_economics = band_outcome_economics(SPEC_INPUTS, high)
    elevated_economics = band_outcome_economics(SPEC_INPUTS, elevated)

    assert high_economics.true_positive_saving_inr == pytest.approx(
        0.60 * high.intervention_success_multiplier * 220.0
    )
    assert elevated_economics.true_positive_saving_inr < high_economics.true_positive_saving_inr
    assert elevated_economics.false_positive_cost_inr < high_economics.false_positive_cost_inr


def test_band_multipliers_cannot_push_a_rate_above_one() -> None:
    """A multiplier is a shape, not a licence to exceed a probability."""
    from rto_sentinel.configuration.schemas import BandEconomics

    absurd = BandEconomics(
        intervention_success_multiplier=5.0,
        abandonment_multiplier=5.0,
        support_cost_inr=0.0,
        rationale="test",
    )
    economics = band_outcome_economics(SPEC_INPUTS, absurd)

    # Both rates clamp at 1.0, so the saving is the full RTO cost and the cost is
    # the full margin.
    assert economics.true_positive_saving_inr == pytest.approx(220.0)
    assert economics.false_positive_cost_inr == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# monotonicity: the direction of every response
# ---------------------------------------------------------------------------


def test_a_higher_margin_raises_the_threshold() -> None:
    """The direction this repository documented backwards until Phase 6.

    The margin is what a false positive costs. A merchant with more margin to
    lose demands more certainty before frictioning, so the threshold rises and
    the merchant flags LESS. "They can afford it" is not the criterion.
    """
    thresholds = [
        derive_threshold(
            SPEC_INPUTS.model_copy(update={"contribution_margin_inr": margin})
        ).threshold
        for margin in (50.0, 250.0, 500.0, 1000.0)
    ]
    assert thresholds == sorted(thresholds)
    assert thresholds[0] < thresholds[-1]


def test_a_higher_rto_cost_lowers_the_threshold() -> None:
    """Returns costing more makes catching them worth more, so flag more."""
    thresholds = [
        derive_threshold(SPEC_INPUTS.model_copy(update={"rto_cost_inr": cost})).threshold
        for cost in (100.0, 220.0, 500.0, 1000.0)
    ]
    assert thresholds == sorted(thresholds, reverse=True)


def test_a_higher_intervention_success_rate_lowers_the_threshold() -> None:
    thresholds = [
        derive_threshold(
            SPEC_INPUTS.model_copy(update={"intervention_success_rate": rate})
        ).threshold
        for rate in (0.1, 0.3, 0.6, 0.9)
    ]
    assert thresholds == sorted(thresholds, reverse=True)


def test_a_higher_abandonment_rate_raises_the_threshold() -> None:
    thresholds = [
        derive_threshold(SPEC_INPUTS.model_copy(update={"abandonment_on_friction": rate})).threshold
        for rate in (0.05, 0.25, 0.5, 0.9)
    ]
    assert thresholds == sorted(thresholds)


def test_the_band_never_falls_as_the_probability_rises(engine: DecisionEngine) -> None:
    """Monotone by construction, and the one property a risk ladder must have."""
    from rto_sentinel.contracts.enums import band_rank

    ranks = [
        band_rank(engine.decide(_score(p), SPEC_INPUTS, decided_at=FIXED_TIME).band)
        for p in [i / 200 for i in range(201)]
    ]
    assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# boundaries
# ---------------------------------------------------------------------------


def test_a_probability_exactly_at_the_threshold_is_flagged(engine: DecisionEngine) -> None:
    """Half-open bands mean `p >= threshold` flags, matching the eval harness.

    This is the single value where the served flag rate and the measured flag
    rate could silently disagree, so it is pinned rather than assumed.
    """
    threshold = derive_threshold(SPEC_INPUTS).threshold
    at = engine.decide(_score(threshold), SPEC_INPUTS, decided_at=FIXED_TIME)
    just_below = engine.decide(_score(threshold - 1e-9), SPEC_INPUTS, decided_at=FIXED_TIME)

    assert at.flagged
    assert at.band is not RiskBand.LOW
    assert not just_below.flagged
    assert just_below.band is RiskBand.LOW


def test_every_band_cut_point_belongs_to_the_higher_band(policy_config: PolicyConfig) -> None:
    threshold = derive_threshold(SPEC_INPUTS).threshold
    ladder = resolve_boundaries(threshold, policy_config)

    for boundary in ladder.boundaries:
        assert band_for(boundary.lower_bound, ladder) is boundary.band
        if boundary.upper_bound is not None and boundary.upper_bound < 1.0:
            assert band_for(boundary.upper_bound, ladder) is not boundary.band


def test_the_extremes_are_scored(engine: DecisionEngine) -> None:
    lowest = engine.decide(_score(0.0), SPEC_INPUTS, decided_at=FIXED_TIME)
    highest = engine.decide(_score(1.0), SPEC_INPUTS, decided_at=FIXED_TIME)

    assert lowest.band is RiskBand.LOW
    assert not lowest.flagged
    assert highest.band is RiskBand.SEVERE
    assert highest.flagged


def test_bands_are_contiguous_and_cover_the_unit_interval(policy_config: PolicyConfig) -> None:
    ladder = resolve_boundaries(0.30, policy_config)
    bounds = [(b.lower_bound, b.upper_bound) for b in ladder.boundaries]

    assert bounds[0][0] == 0.0
    assert bounds[-1][1] is None
    for (_, upper), (lower, _) in itertools.pairwise(bounds):
        assert upper == pytest.approx(lower)


def test_bands_that_cannot_fire_are_collapsed_and_reported(
    policy_config: PolicyConfig,
) -> None:
    """A high threshold leaves no room above it for three more rungs."""
    ladder = resolve_boundaries(0.95, policy_config)

    assert RiskBand.LOW in ladder.bands
    assert ladder.collapsed, "the rungs that cannot fire must be reported, not dropped"
    for band, reason in ladder.collapsed:
        assert band is not RiskBand.LOW
        assert reason


def test_a_threshold_of_one_leaves_only_the_low_band(policy_config: PolicyConfig) -> None:
    """Zero RTO cost derives a threshold of 1.0: flag nothing, and mean it."""
    free_returns = SPEC_INPUTS.model_copy(update={"rto_cost_inr": 0.0})
    assert derive_threshold(free_returns).threshold == pytest.approx(1.0)

    ladder = resolve_boundaries(1.0, policy_config)
    assert ladder.bands == (RiskBand.LOW,)
    assert band_for(1.0, ladder) is RiskBand.LOW


def test_an_out_of_range_probability_is_refused(policy_config: PolicyConfig) -> None:
    ladder = resolve_boundaries(0.35, policy_config)
    with pytest.raises(PolicyError, match="out of range"):
        band_for(1.5, ladder)


def test_an_out_of_range_threshold_is_refused(policy_config: PolicyConfig) -> None:
    with pytest.raises(PolicyError, match="not a probability"):
        resolve_boundaries(1.4, policy_config)


# ---------------------------------------------------------------------------
# edge cases in the economics
# ---------------------------------------------------------------------------


def test_zero_margin_is_accepted_and_makes_friction_nearly_free() -> None:
    """A break-even order has no margin to lose, so the bar drops."""
    zero = SPEC_INPUTS.model_copy(
        update={"contribution_margin_inr": 0.0, "friction_support_cost_inr": 8.0}
    )
    derivation = derive_threshold(zero)

    assert derivation.cost_false_positive_inr == pytest.approx(8.0)
    assert 0.0 < derivation.threshold < 0.1


def test_zero_rto_cost_derives_a_threshold_of_one() -> None:
    """Returns are free, so no probability justifies friction. Flag nothing."""
    free = SPEC_INPUTS.model_copy(update={"rto_cost_inr": 0.0})
    assert derive_threshold(free).threshold == pytest.approx(1.0)


@pytest.mark.parametrize("field", ["rto_cost_inr", "contribution_margin_inr"])
def test_negative_rupee_inputs_are_refused_with_a_reason(field: str) -> None:
    """A negative margin inverts the decision rule. Refused, not clamped."""
    with pytest.raises(ValidationError, match="inverts the decision rule"):
        SPEC_INPUTS.model_copy(update={field: -1.0}).model_validate(
            {**SPEC_INPUTS.model_dump(), field: -1.0}
        )


@pytest.mark.parametrize("rate", [-0.1, 1.4])
@pytest.mark.parametrize("field", ["abandonment_on_friction", "intervention_success_rate"])
def test_invalid_probabilities_are_refused(field: str, rate: float) -> None:
    with pytest.raises(ValidationError):
        CostInputs(**{**SPEC_INPUTS.model_dump(), field: rate})


def test_degenerate_economics_are_refused_rather_than_defaulted() -> None:
    """No cost and no benefit means no threshold. Returning 0.5 would be a lie."""
    with pytest.raises(ValidationError, match="degenerate"):
        CostInputs(
            rto_cost_inr=220.0,
            contribution_margin_inr=250.0,
            abandonment_on_friction=0.0,
            intervention_success_rate=0.0,
            friction_support_cost_inr=0.0,
        )


def test_missing_economics_are_refused() -> None:
    """A partial cost model cannot derive a threshold, and says so."""
    with pytest.raises(ValidationError):
        CostInputs(rto_cost_inr=220.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# what the engine refuses
# ---------------------------------------------------------------------------


def test_an_uncalibrated_score_cannot_reach_a_decision(engine: DecisionEngine) -> None:
    """The rule that stops uncalibrated probabilities becoming rupee figures."""
    with pytest.raises(UncalibratedScoreError, match="uncalibrated"):
        engine.decide(_score(0.9, calibrated=False), SPEC_INPUTS)


def test_no_decision_can_remove_the_appeal_path() -> None:
    with pytest.raises(ValidationError, match="appeal path"):
        Decision(
            order_id="ORD-1",
            probability=0.9,
            threshold=0.3,
            band=RiskBand.SEVERE,
            action=InterventionAction.PREPAID_ONLY,
            flagged=True,
            reason_codes=("X",),
            expected_value_inr=1.0,
            decided_at=FIXED_TIME,
            engine_version="1.0.0",
            appeal_available=False,
            human_review_required=True,
        )


def test_severe_decisions_route_to_a_human(engine: DecisionEngine) -> None:
    decision = engine.decide(_score(0.99), SPEC_INPUTS, decided_at=FIXED_TIME)
    assert decision.band is RiskBand.SEVERE
    assert decision.human_review_required
    assert decision.appeal_available


def test_a_flagged_decision_always_carries_a_reason(engine: DecisionEngine) -> None:
    """Even with no SHAP attributions, which heuristic rungs never have."""
    decision = engine.decide(_score(0.9), SPEC_INPUTS, decided_at=FIXED_TIME)
    assert decision.flagged
    assert decision.reason_codes == (SCORE_ONLY_REASON,)


def test_reason_codes_come_from_risk_increasing_contributions() -> None:
    contributions = [
        FeatureContribution(
            feature="addr_token_count", family="address_quality", value=3, contribution=0.9
        ),
        FeatureContribution(
            feature="cust_prior_rto_rate", family="customer_history", value=0.4, contribution=0.5
        ),
        FeatureContribution(
            feature="cust_prepaid_share", family="customer_history", value=0.8, contribution=-0.7
        ),
    ]
    codes = derive_reason_codes(contributions)

    assert [code.code for code in codes] == ["ADDRESS_INCOMPLETE", "HISTORY_PRIOR_RTO_RATE"]
    assert all(code.direction == "increases_risk" for code in codes)


def test_reason_code_ordering_is_deterministic_on_ties() -> None:
    """Identical SHAP values must not reshuffle between runs."""
    contributions = [
        FeatureContribution(feature="zeta", family="order_shape", value=1, contribution=0.5),
        FeatureContribution(feature="alpha", family="order_shape", value=1, contribution=0.5),
    ]
    first = derive_reason_codes(contributions)
    second = derive_reason_codes(list(reversed(contributions)))
    assert [c.feature for c in first] == [c.feature for c in second] == ["alpha", "zeta"]


def test_an_unmapped_feature_still_produces_a_code() -> None:
    """A new feature must appear in the logs as something, not vanish."""
    contribution = FeatureContribution(
        feature="cust_brand_new_signal", family="customer_history", value=1, contribution=0.4
    )
    assert code_for(contribution) == "HISTORY_BRAND_NEW_SIGNAL"


# ---------------------------------------------------------------------------
# the control holdout
# ---------------------------------------------------------------------------


def test_a_holdout_order_is_banded_but_not_frictioned(engine: DecisionEngine) -> None:
    """The slice that keeps precision measurable once the system acts."""
    decision = engine.decide(
        _score(0.99), SPEC_INPUTS, is_control_holdout=True, decided_at=FIXED_TIME
    )

    assert decision.band is RiskBand.SEVERE, "the band is still recorded"
    assert decision.action is InterventionAction.NONE
    assert not decision.flagged
    assert decision.expected_value_inr == 0.0
    assert decision.is_control_holdout


def test_a_holdout_order_is_not_routed_to_review(engine: DecisionEngine) -> None:
    """Routing it to a human would destroy the counterfactual it preserves."""
    decision = engine.decide(
        _score(0.99), SPEC_INPUTS, is_control_holdout=True, decided_at=FIXED_TIME
    )
    assert not decision.human_review_required


def test_a_holdout_decision_cannot_be_flagged() -> None:
    with pytest.raises(ValidationError, match="receives no friction"):
        Decision(
            order_id="ORD-1",
            probability=0.9,
            threshold=0.3,
            band=RiskBand.HIGH,
            action=InterventionAction.CONFIRMATION_REQUIRED,
            flagged=True,
            reason_codes=("X",),
            expected_value_inr=1.0,
            decided_at=FIXED_TIME,
            engine_version="1.0.0",
            is_control_holdout=True,
        )


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_the_same_inputs_produce_the_same_decision(engine: DecisionEngine) -> None:
    """Byte for byte, with the clock pinned. An engine that drifts is unauditable."""
    score = _score(0.62)
    first = engine.decide(score, SPEC_INPUTS, decided_at=FIXED_TIME)
    second = engine.decide(score, SPEC_INPUTS, decided_at=FIXED_TIME)

    assert first.model_dump_json() == second.model_dump_json()


def test_a_fresh_engine_decides_identically(policy_config: PolicyConfig) -> None:
    score = _score(0.62)
    first = DecisionEngine(policy_config).decide(score, SPEC_INPUTS, decided_at=FIXED_TIME)
    second = DecisionEngine(policy_config).decide(score, SPEC_INPUTS, decided_at=FIXED_TIME)
    assert first.model_dump_json() == second.model_dump_json()


def test_the_engine_stamps_its_version(engine: DecisionEngine) -> None:
    """So a logged decision can be replayed against the engine that made it."""
    decision = engine.decide(_score(0.62), SPEC_INPUTS, decided_at=FIXED_TIME)
    assert decision.engine_version == engine.engine_version


def test_the_shipped_profiles_order_as_the_formula_requires(
    cost_config: CostModelConfig,
) -> None:
    """Thin margin flags most, high margin least. Asserted on what ships."""

    def threshold(key: str) -> float:
        profile = cost_config.profiles[key]
        return derive_threshold(
            CostInputs(
                rto_cost_inr=profile.rto_cost_inr,
                contribution_margin_inr=profile.contribution_margin_inr,
                abandonment_on_friction=profile.abandonment_on_friction,
                intervention_success_rate=profile.intervention_success_rate,
                friction_support_cost_inr=profile.friction_support_cost_inr,
            )
        ).threshold

    assert (
        threshold("thin_margin_reseller")
        < threshold("mid_margin_d2c")
        < threshold("high_margin_beauty")
    )
