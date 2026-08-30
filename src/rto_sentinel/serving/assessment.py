"""The whole chain, end to end: a database order becomes a structured decision.

::

    order_id
      -> ServingRepository      the order, its address, its outcome if matured
      -> OrderFeatureService    context frame -> training pipeline -> one row
      -> ModelRegistry          the frozen artefact, fingerprint-checked
      -> calibrator             inside the artefact
      -> RiskScore              calibrated probability + provenance
      -> DecisionEngine         cost-derived threshold, friction ladder
      -> OrderAssessment        everything the console and the API need

Every step executes on every call. There is no cache between the order and the
decision, no precomputed score table, and no branch that returns a plausible
number when a step is unavailable.

WHAT THE ASSESSMENT CARRIES, AND WHY EACH PART
==============================================
The response deliberately refuses to hand back a probability on its own. It
travels with the threshold that interpreted it, the band that threshold produced,
the action, and the economics that derived the threshold - because a bare score
invites comparison against 0.5, which is the error this system exists to correct.

It also carries what it does *not* know: which features were null for this order,
how many rows of history the aggregates were computed over, and whether the
outcome has matured. A score built mostly from nulls on a cold-start customer is
a different object from one built on fifty prior orders, and the API says so
rather than leaving the caller to assume they are the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rto_sentinel.decision.policy import band_economics
from rto_sentinel.decision.threshold import derive_threshold

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.decision import CostInputs, Decision, ThresholdDerivation
    from rto_sentinel.contracts.risk import RiskScore
    from rto_sentinel.db.models import Order, OrderOutcomeRecord
    from rto_sentinel.db.repositories import ServingRepository
    from rto_sentinel.decision.engine import DecisionEngine
    from rto_sentinel.serving.features import OrderFeatures
    from rto_sentinel.serving.model_registry import LoadedModel
    from rto_sentinel.serving.scoring import ScoringService


class OrderNotFoundError(LookupError):
    """Raised when the requested order is not in the database."""


@dataclass(frozen=True, slots=True)
class OrderAssessment:
    """One order, scored and decided, with everything needed to justify it."""

    order: Order
    outcome: OrderOutcomeRecord | None
    score: RiskScore
    decision: Decision
    threshold: ThresholdDerivation
    cost_inputs: CostInputs
    cost_profile: str
    features: OrderFeatures
    model: LoadedModel
    #: The ASSUMED effectiveness of the action this band recommends. Carried so a
    #: consumer can label the rupee figure rather than presenting it as measured.
    assumed_intervention_success: float
    assumed_abandonment: float

    @property
    def label_is_known(self) -> bool:
        """Whether this order's outcome has matured.

        A scored order with a known outcome is a *historical* order being
        re-scored; a scored order without one is a live decision. Conflating the
        two is how a demo accidentally shows the model predicting the past.
        """
        return self.outcome is not None and bool(self.outcome.is_mature)


class AssessmentService:
    """Composes repository, features, model and engine for one order."""

    def __init__(
        self,
        repository: ServingRepository,
        scoring: ScoringService,
        engine: DecisionEngine,
        *,
        default_cost_inputs: CostInputs,
        default_cost_profile: str,
    ) -> None:
        self._repository = repository
        self._scoring = scoring
        self._engine = engine
        self._default_cost_inputs = default_cost_inputs
        self._default_cost_profile = default_cost_profile

    def assess(
        self,
        order_id: str,
        *,
        dataset_run_id: str | None = None,
        cost_inputs: CostInputs | None = None,
        include_contributions: bool = True,
        is_control_holdout: bool = False,
    ) -> OrderAssessment:
        """Run the whole chain for one stored order.

        The model is checked FIRST, before the order lookup. On a server with no
        artefact both preconditions fail, and "no model is loaded" is the one an
        operator can act on - reporting "no such order" instead would send them
        looking for a data problem that is not there. It also costs nothing: the
        registry is cached after the first successful load.
        """
        self._scoring.registry.load()

        order = self._repository.get_order(order_id, dataset_run_id=dataset_run_id)
        if order is None:
            scope = f" in dataset run {dataset_run_id!r}" if dataset_run_id else ""
            msg = f"no order {order_id!r}{scope} in the database"
            raise OrderNotFoundError(msg)

        score, features, loaded = self._scoring.score(
            order, include_contributions=include_contributions
        )

        inputs = cost_inputs or self._default_cost_inputs
        profile = self._default_cost_profile if cost_inputs is None else "custom"
        decision = self._engine.decide(score, inputs, is_control_holdout=is_control_holdout)
        economics = band_economics(decision.band, self._engine.policy)

        return OrderAssessment(
            order=order,
            outcome=self._repository.get_outcome(order),
            score=score,
            decision=decision,
            threshold=derive_threshold(inputs),
            cost_inputs=inputs,
            cost_profile=profile,
            features=features,
            model=loaded,
            assumed_intervention_success=min(
                inputs.intervention_success_rate * economics.intervention_success_multiplier, 1.0
            ),
            assumed_abandonment=min(
                inputs.abandonment_on_friction * economics.abandonment_multiplier, 1.0
            ),
        )
