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
* **Wall-clock behaviour.** ``decided_at`` is recorded, never branched on.
* **An API handler.** Routers marshal and delegate. If risk logic appears in a
  route handler, it is in the wrong file.

FAILURE POSTURE
---------------
If the model artefact is missing, this engine raises. It does not fall back to a
default probability, and it does not pass an uncalibrated score through to a
threshold comparison. A system that cannot score an order says so; it does not
guess and call the guess a risk estimate.

STATUS: Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rto_sentinel.contracts.decision import Decision, ThresholdDerivation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import PolicyConfig
    from rto_sentinel.contracts.decision import CostInputs
    from rto_sentinel.contracts.risk import RiskScore

#: Bumped whenever decision arithmetic changes. Stamped onto every Decision so a
#: logged decision can be replayed against the engine that produced it.
ENGINE_VERSION = "0.1.0"


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

    def decide(
        self,
        score: RiskScore,
        cost_inputs: CostInputs,
        *,
        is_control_holdout: bool = False,
    ) -> Decision:
        """Produce the decision for one scored order.

        ``is_control_holdout`` marks the randomised no-friction slice described
        in SPEC section 11. Those orders are scored and banded exactly as usual
        but receive no friction, so the true outcome stays observable and
        precision remains measurable after the system starts acting. The flag is
        recorded on the decision rather than hidden, because a holdout order that
        returns is a *correct prediction*, not a miss, and the evaluation must be
        able to tell the difference.
        """
        raise NotImplementedError("Decision assembly lands in Phase 2.")

    def threshold(self, cost_inputs: CostInputs) -> ThresholdDerivation:
        """Expose the derived threshold, with its arithmetic, for the console."""
        raise NotImplementedError("Threshold exposure lands in Phase 2.")
