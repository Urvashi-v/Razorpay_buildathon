"""The deterministic decision engine.

READ THIS BEFORE CHANGING ANYTHING IN THIS PACKAGE
==================================================

This module is the authority on what happens to an order. Its inputs are exactly
two things:

1. a **calibrated** :class:`~rto_sentinel.contracts.risk.RiskScore` from a model;
2. a :class:`~rto_sentinel.contracts.decision.CostInputs` from the merchant.

Given those, its output is a pure function. The same score and the same cost
inputs produce the same :class:`~rto_sentinel.contracts.decision.Decision`, on
every machine, forever. That property is not a nice-to-have: a non-deterministic
risk engine cannot be audited, and an unauditable risk engine cannot be deployed
by a payments company.

WHAT MAY NOT INFLUENCE A DECISION
---------------------------------
* **An LLM.** Not the probability, not the threshold, not the band, not the
  action. The language layer runs strictly downstream and only ever describes a
  decision that already exists. ``tests/architecture/test_layering.py`` asserts
  that no module in this package imports ``rto_sentinel.agents`` or any LLM SDK.
* **Network state.** No calls out. Sub-100ms, no external dependencies.
* **Wall-clock behaviour.** ``decided_at`` is recorded, never branched on. It is
  injectable for exactly that reason: a test can pin it and compare two decisions
  byte for byte.
* **An API handler.** Routers marshal and delegate. If risk logic appears in a
  route handler, it is in the wrong file.

FAILURE POSTURE
---------------
If the model artefact is missing, this engine raises. It does not fall back to a
default probability, and it does not pass an uncalibrated score through to a
threshold comparison. A system that cannot score an order says so; it does not
guess and call the guess a risk estimate.

WHAT ``expected_value_inr`` MEANS, AND WHAT IT RESTS ON
-------------------------------------------------------
It is the expected rupee gain from applying *this band's* action to *this* order,
against doing nothing::

    E[gain] = p x (success_rate x rto_cost) - (1 - p) x (abandonment x margin)
              - support_cost

The success and abandonment rates come from the merchant's inputs scaled by the
band's multipliers, and **those multipliers are assumptions nobody has measured**
(see ``config/policy.yaml``). The number is real arithmetic over declared inputs,
not a measurement, and every surface that reports it says so.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rto_sentinel.contracts.decision import Decision
from rto_sentinel.contracts.enums import InterventionAction, RiskBand
from rto_sentinel.decision.cost_model import band_outcome_economics
from rto_sentinel.decision.policy import (
    ResolvedLadder,
    action_for,
    band_economics,
    band_for,
    requires_human_review,
    resolve_boundaries,
)
from rto_sentinel.decision.reason_codes import derive_reason_codes
from rto_sentinel.decision.threshold import derive_threshold

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import PolicyConfig
    from rto_sentinel.contracts.decision import CostInputs, ThresholdDerivation
    from rto_sentinel.contracts.risk import RiskScore

#: Bumped whenever decision arithmetic changes. Stamped onto every Decision so a
#: logged decision can be replayed against the engine that produced it.
ENGINE_VERSION = "1.0.0"

#: Emitted when a decision applies friction but the model supplied no
#: attributions - heuristic rungs have none, and SHAP can be unavailable.
#: `Decision` requires every friction decision to carry a reason, and the honest
#: reason in that case is the score itself rather than a fabricated feature.
SCORE_ONLY_REASON = "MODEL_SCORE_ABOVE_THRESHOLD"


class UncalibratedScoreError(ValueError):
    """Raised when a decision is requested from an uncalibrated probability.

    SPEC section 05: if the model says 0.30 and the true rate is 0.55, the
    expected-value threshold is wrong and the rupee numbers are fiction. So this
    is a hard error rather than a warning.
    """


class DecisionEngine:
    """Converts a calibrated probability into a graduated, appealable action."""

    def __init__(self, policy: PolicyConfig, engine_version: str = ENGINE_VERSION) -> None:
        self._policy = policy
        self._engine_version = engine_version

    @property
    def policy(self) -> PolicyConfig:
        return self._policy

    @property
    def engine_version(self) -> str:
        return self._engine_version

    def threshold(self, cost_inputs: CostInputs) -> ThresholdDerivation:
        """Expose the derived threshold, with its arithmetic, for the console."""
        return derive_threshold(cost_inputs)

    def ladder(self, cost_inputs: CostInputs) -> ResolvedLadder:
        """The resolved band boundaries at this merchant's economics."""
        return resolve_boundaries(self.threshold(cost_inputs).threshold, self._policy)

    def expected_value(self, probability: float, band: RiskBand, cost_inputs: CostInputs) -> float:
        """Expected rupee gain from this band's action on one order.

        Zero for LOW by construction: no action is taken, so there is nothing to
        gain and nothing to risk.
        """
        economics = band_outcome_economics(cost_inputs, band_economics(band, self._policy))
        if action_for(band, self._policy) is InterventionAction.NONE:
            return 0.0
        return (
            probability * economics.true_positive_saving_inr
            - (1.0 - probability) * economics.false_positive_cost_inr
        )

    def decide(
        self,
        score: RiskScore,
        cost_inputs: CostInputs,
        *,
        is_control_holdout: bool = False,
        decided_at: datetime | None = None,
    ) -> Decision:
        """Produce the decision for one scored order.

        ``is_control_holdout`` marks the randomised no-friction slice described
        in SPEC section 11. Those orders are scored and banded exactly as usual
        but receive no friction, so the true outcome stays observable and
        precision remains measurable after the system starts acting. The flag is
        recorded on the decision rather than hidden, because a holdout order that
        returns is a *correct prediction*, not a miss, and the evaluation must be
        able to tell the difference.

        A holdout order receives no human review either. Routing it to a queue
        would let an operator act on it, which destroys exactly the
        counterfactual the holdout exists to preserve - so the band is recorded,
        the action is not taken, and the reason is on the decision.
        """
        if not score.is_calibrated:
            msg = (
                f"order {score.order_id}: refusing to decide from an uncalibrated score "
                f"produced by {score.model_name} v{score.model_version}. An uncalibrated "
                "probability compared against an expected-value threshold produces rupee "
                "figures that are fiction."
            )
            raise UncalibratedScoreError(msg)

        derivation = self.threshold(cost_inputs)
        ladder = resolve_boundaries(derivation.threshold, self._policy)
        band = band_for(score.probability, ladder)
        configured_action = action_for(band, self._policy)

        # The holdout takes the band and drops the action. Everything else about
        # the decision is identical, which is what makes the slice comparable.
        action = InterventionAction.NONE if is_control_holdout else configured_action
        flagged = action is not InterventionAction.NONE

        reason_codes: tuple[str, ...] = ()
        if flagged:
            derived = derive_reason_codes(list(score.contributions))
            reason_codes = tuple(entry.code for entry in derived) or (SCORE_ONLY_REASON,)

        return Decision(
            order_id=score.order_id,
            probability=score.probability,
            threshold=derivation.threshold,
            band=band,
            action=action,
            flagged=flagged,
            reason_codes=reason_codes,
            expected_value_inr=(
                0.0
                if is_control_holdout
                else self.expected_value(score.probability, band, cost_inputs)
            ),
            decided_at=decided_at or datetime.now(UTC),
            engine_version=self._engine_version,
            appeal_available=True,
            human_review_required=(
                False if is_control_holdout else requires_human_review(band, self._policy)
            ),
            is_control_holdout=is_control_holdout,
        )
