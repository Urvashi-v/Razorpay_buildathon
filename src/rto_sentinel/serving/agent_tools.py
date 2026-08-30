"""The concrete toolset the language layer is handed.

WHY IT LIVES HERE AND NOT IN ``agents``
=======================================
``tests/architecture/test_layering.py`` forbids the ``agents`` package from
importing ``decision``, ``models``, ``features``, ``data`` or ``eval`` - the
layers where a probability, a threshold or an action could actually be produced.
That rule is the mechanical form of "the LLM is downstream of the decision".

The agents package therefore declares the tool *contract* and this module
implements it, because ``serving`` already composes those layers for the API. An
agent receives a toolset as an argument and has no other route to data: it cannot
import one, and there is nothing to import if it tried.

EVERY METHOD HERE IS A READ
===========================
The repositories this uses are the read paths from Phase 7. There is no write
method reachable from any tool - not through the assessment service, not through
the decision log. Re-scoring is *not* a write, and this toolset does re-score on
demand, which deserves saying plainly: ``get_risk_prediction`` runs the model
over the order's features. It produces the same number the API would, changes
nothing, and stores nothing.

WHAT IS WITHHELD, AND WHY
=========================
No tool returns a customer name, a phone number, or address text. The address is
represented by its derived quality signals - token count, whether a house number
is present, whether the pincode matches the city - because those are what the
model actually used, and because handing raw address text to a language model
that drafts customer-facing copy is how a delivery address ends up quoted back at
somebody. Pincode is reduced to its tier for the same reason it is withheld from
the model: an explanation keyed on a specific locality is an explanation about a
place rather than a behaviour.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sqlalchemy import select

from rto_sentinel.agents.audit import ToolInvocation
from rto_sentinel.agents.tools import (
    TOOLS_BY_NAME,
    CustomerHistoryFacts,
    CustomerHistoryRef,
    DigestFigures,
    DigestRef,
    EconomicDecisionFacts,
    FeatureAttribution,
    ModelExplanationFacts,
    OrderEvent,
    OrderEventFacts,
    OrderFacts,
    OrderRef,
    PriorOrder,
    RiskPredictionFacts,
    ToolResult,
)
from rto_sentinel.db.models import DeliveryEvent, Order, OrderOutcomeRecord
from rto_sentinel.decision.reason_codes import derive_reason_codes
from rto_sentinel.serving.assessment import OrderNotFoundError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.orm import Session

    from rto_sentinel.db.repositories import ServingRepository
    from rto_sentinel.serving.assessment import AssessmentService, OrderAssessment

#: Address columns an agent may see. Derived signals only - never the text.
_ADDRESS_SIGNALS = (
    "addr_token_count",
    "addr_has_house_number",
    "addr_has_floor_number",
    "addr_has_landmark",
    "addr_pincode_city_consistent",
)


class ApplicationToolset:
    """Read-only application tools, backed by the Phase 7 serving path.

    Assessments are memoised per order for the life of one agent run. An
    investigation calls ``get_risk_prediction``, ``get_model_explanation`` and
    ``get_economic_decision`` for the same order, and each would otherwise
    rebuild the feature context from the database - about a second apiece. The
    cache is per-instance and per-run, so it cannot serve a stale score across
    requests.
    """

    def __init__(
        self,
        repository: ServingRepository,
        assessment_service: AssessmentService,
        session: Session,
    ) -> None:
        self._repository = repository
        self._assessments = assessment_service
        self._session = session
        self._cache: dict[tuple[str, str | None], OrderAssessment] = {}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _assess(self, ref: OrderRef) -> OrderAssessment:
        key = (ref.order_id, ref.dataset_run_id)
        if key not in self._cache:
            self._cache[key] = self._assessments.assess(
                ref.order_id, dataset_run_id=ref.dataset_run_id, include_contributions=True
            )
        return self._cache[key]

    def _order(self, ref: OrderRef) -> Order | None:
        return self._repository.get_order(ref.order_id, dataset_run_id=ref.dataset_run_id)

    @staticmethod
    def _missing(order_id: str) -> str:
        return (
            f"no order {order_id!r} exists in the database, so there is no evidence to "
            "report. Do not describe this order; say the evidence is unavailable."
        )

    # ------------------------------------------------------------------
    # the tools
    # ------------------------------------------------------------------

    def get_order(self, ref: OrderRef) -> OrderFacts:
        order = self._order(ref)
        if order is None:
            return OrderFacts(found=False, reason=self._missing(ref.order_id))

        address = self._repository.get_address(order.address_pk)
        outcome = self._repository.get_outcome(order)
        return OrderFacts(
            found=True,
            order_id=order.order_id,
            merchant_id=order.merchant_id,
            customer_hash=order.customer_hash,
            ordered_at=order.ordered_at,
            payment_method=order.payment_method,
            is_cod=order.is_cod,
            order_value_inr=order.order_value_inr,
            discount_inr=order.discount_inr,
            discount_depth=order.discount_depth,
            item_count=order.item_count,
            category=order.category,
            courier_partner=order.courier_partner,
            pincode_tier=getattr(address, "pincode_tier", None),
            address_completeness={name: getattr(address, name, None) for name in _ADDRESS_SIGNALS}
            if address is not None
            else {},
            split=order.split,
            outcome=getattr(outcome, "outcome", None),
            is_rto=getattr(outcome, "is_rto", None),
            resolved_at=getattr(outcome, "resolved_at", None),
        )

    def get_customer_history(self, ref: CustomerHistoryRef) -> CustomerHistoryFacts:
        """History strictly as-of the referenced order.

        Only orders that had already **resolved** before this order's timestamp
        are included - the same cutoff the model's features use. An agent
        explaining a score must not be able to cite history the model did not
        have: that produces a sentence which is true about the customer and wrong
        about the decision, which is the worst of both.
        """
        order = self._repository.get_order(ref.order_id, dataset_run_id=ref.dataset_run_id)
        if order is None:
            return CustomerHistoryFacts(found=False, reason=self._missing(ref.order_id))

        rows = self._session.execute(
            select(Order, OrderOutcomeRecord)
            .join(OrderOutcomeRecord, OrderOutcomeRecord.order_pk == Order.id)
            .where(
                Order.dataset_run_id == order.dataset_run_id,
                Order.customer_hash == order.customer_hash,
                Order.id != order.id,
                OrderOutcomeRecord.resolved_at.isnot(None),
                OrderOutcomeRecord.resolved_at < order.ordered_at,
            )
            .order_by(Order.ordered_at.desc())
        ).all()

        prior = [
            PriorOrder(
                order_id=row.Order.order_id,
                ordered_at=row.Order.ordered_at,
                order_value_inr=row.Order.order_value_inr,
                payment_method=row.Order.payment_method,
                outcome=row.OrderOutcomeRecord.outcome,
                is_rto=row.OrderOutcomeRecord.is_rto,
                resolved_at=row.OrderOutcomeRecord.resolved_at,
            )
            for row in rows
        ]
        returned = sum(1 for entry in prior if entry.is_rto)
        last_ordered = prior[0].ordered_at if prior else None

        if not prior:
            return CustomerHistoryFacts(
                found=True,
                reason=(
                    "this customer had no resolved orders before this one. They were a "
                    "first-time buyer as far as the model could see - state that, do not "
                    "infer a history."
                ),
                customer_hash=order.customer_hash,
                as_of=order.ordered_at,
                prior_order_count=0,
                prior_rto_count=0,
                prior_rto_rate=None,
                is_new_customer=True,
                recent_orders=[],
            )

        return CustomerHistoryFacts(
            found=True,
            customer_hash=order.customer_hash,
            as_of=order.ordered_at,
            prior_order_count=len(prior),
            prior_rto_count=returned,
            prior_rto_rate=returned / len(prior),
            days_since_last_order=(
                (order.ordered_at - last_ordered).total_seconds() / 86400.0
                if last_ordered
                else None
            ),
            is_new_customer=False,
            recent_orders=prior[: ref.limit],
        )

    def get_risk_prediction(self, ref: OrderRef) -> RiskPredictionFacts:
        try:
            assessment = self._assess(ref)
        except OrderNotFoundError:
            return RiskPredictionFacts(found=False, reason=self._missing(ref.order_id))

        return RiskPredictionFacts(
            found=True,
            order_id=assessment.order.order_id,
            probability=assessment.score.probability,
            raw_score=assessment.score.raw_score,
            model_name=assessment.model.card.model_name,
            model_version=assessment.model.card.model_version,
            calibration_method=assessment.model.card.calibration_method,
            feature_version=assessment.features.feature_version,
            scored_at=assessment.score.scored_at,
            null_features=list(assessment.features.null_features),
            context_rows=assessment.features.context_rows,
        )

    def get_model_explanation(self, ref: OrderRef) -> ModelExplanationFacts:
        try:
            assessment = self._assess(ref)
        except OrderNotFoundError:
            return ModelExplanationFacts(found=False, reason=self._missing(ref.order_id))

        contributions = list(assessment.score.contributions)
        if not contributions:
            return ModelExplanationFacts(
                found=True,
                order_id=assessment.order.order_id,
                note=(
                    "this model produced no per-feature attributions for this order. Say "
                    "the drivers are unavailable; do not guess which features mattered."
                ),
                reason_codes=list(assessment.decision.reason_codes),
                permitted_features=[],
            )

        return ModelExplanationFacts(
            found=True,
            order_id=assessment.order.order_id,
            attributions=[
                FeatureAttribution(
                    feature=entry.feature,
                    family=entry.family,
                    value=entry.value,
                    contribution=entry.contribution,
                    direction=entry.direction,
                )
                for entry in contributions
            ],
            reason_codes=[code.code for code in derive_reason_codes(contributions)],
            permitted_features=[entry.feature for entry in contributions],
        )

    def get_economic_decision(self, ref: OrderRef) -> EconomicDecisionFacts:
        try:
            assessment = self._assess(ref)
        except OrderNotFoundError:
            return EconomicDecisionFacts(found=False, reason=self._missing(ref.order_id))

        decision = assessment.decision
        inputs = assessment.cost_inputs
        return EconomicDecisionFacts(
            found=True,
            order_id=decision.order_id,
            band=decision.band.value,
            action=decision.action.value,
            flagged=decision.flagged,
            threshold=decision.threshold,
            threshold_source="derived from merchant economics: C_fp / (C_fp + S_tp)",
            expected_value_inr=decision.expected_value_inr,
            human_review_required=decision.human_review_required,
            appeal_available=decision.appeal_available,
            is_control_holdout=decision.is_control_holdout,
            engine_version=decision.engine_version,
            cost_profile=assessment.cost_profile,
            rto_cost_inr=inputs.rto_cost_inr,
            contribution_margin_inr=inputs.contribution_margin_inr,
            cost_false_positive_inr=assessment.threshold.cost_false_positive_inr,
            saving_true_positive_inr=assessment.threshold.saving_true_positive_inr,
            assumed_intervention_success_rate=assessment.assumed_intervention_success,
            assumed_abandonment_rate=assessment.assumed_abandonment,
            decided_at=decision.decided_at,
        )

    def get_relevant_order_events(self, ref: OrderRef) -> OrderEventFacts:
        order = self._order(ref)
        if order is None:
            return OrderEventFacts(found=False, reason=self._missing(ref.order_id))

        rows = (
            self._session.execute(
                select(DeliveryEvent)
                .where(DeliveryEvent.order_pk == order.id)
                .order_by(DeliveryEvent.sequence.asc())
            )
            .scalars()
            .all()
        )
        if not rows:
            return OrderEventFacts(
                found=True,
                reason=(
                    "this order has no recorded delivery events. It had not been "
                    "dispatched, or the events were never ingested. Do not describe a "
                    "delivery timeline for it."
                ),
                order_id=order.order_id,
                events=[],
                dispatched_at=order.dispatched_at,
                first_attempt_at=order.first_attempt_at,
            )

        return OrderEventFacts(
            found=True,
            order_id=order.order_id,
            events=[
                OrderEvent(
                    sequence=row.sequence,
                    event_type=row.event_type,
                    occurred_at=row.occurred_at,
                )
                for row in rows
            ],
            dispatched_at=order.dispatched_at,
            first_attempt_at=order.first_attempt_at,
        )

    def get_digest_figures(self, ref: DigestRef) -> DigestFigures:
        """Aggregate figures computed in SQL. The LLM computes nothing."""
        from sqlalchemy import func

        window = (
            Order.merchant_id == ref.merchant_id,
            Order.ordered_at >= ref.period_start,
            Order.ordered_at < ref.period_end,
        )
        total = int(
            self._session.execute(
                select(func.count()).select_from(Order).where(*window)
            ).scalar_one()
        )
        if total == 0:
            return DigestFigures(
                found=False,
                reason=(
                    f"merchant {ref.merchant_id!r} has no orders in this period, so there "
                    "are no figures to summarise. Say so; do not describe a quiet week."
                ),
                merchant_id=ref.merchant_id,
                period_start=ref.period_start,
                period_end=ref.period_end,
            )

        cod = int(
            self._session.execute(
                select(func.count()).select_from(Order).where(Order.is_cod.is_(True), *window)
            ).scalar_one()
        )
        matured = int(
            self._session.execute(
                select(func.count())
                .select_from(OrderOutcomeRecord)
                .join(Order, OrderOutcomeRecord.order_pk == Order.id)
                .where(OrderOutcomeRecord.is_mature.is_(True), *window)
            ).scalar_one()
        )
        returned = int(
            self._session.execute(
                select(func.count())
                .select_from(OrderOutcomeRecord)
                .join(Order, OrderOutcomeRecord.order_pk == Order.id)
                .where(
                    OrderOutcomeRecord.is_mature.is_(True),
                    OrderOutcomeRecord.is_rto.is_(True),
                    *window,
                )
            ).scalar_one()
        )

        figures = {
            "orders": float(total),
            "cod_orders": float(cod),
            "cod_share": cod / total,
            "matured_orders": float(matured),
            "returned_orders": float(returned),
        }
        # Computed over matured orders only. Dividing by every order would count
        # "not yet resolved" as "did not return" and understate the rate.
        if matured:
            figures["rto_rate_among_matured"] = returned / matured

        return DigestFigures(
            found=True,
            merchant_id=ref.merchant_id,
            period_start=ref.period_start,
            period_end=ref.period_end,
            figures=figures,
        )


def invoke(
    toolset: ApplicationToolset, name: str, arguments: dict[str, object]
) -> tuple[ToolResult | None, ToolInvocation]:
    """Run one named tool, returning its result and an audit entry.

    Unknown names and malformed arguments are returned as failed invocations
    rather than raised. A model that asks for a tool that does not exist should
    be told so and allowed to continue with what it has; killing the whole run
    for it turns a recoverable confusion into an outage.
    """
    started = time.perf_counter()
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        return None, ToolInvocation(
            tool=name,
            arguments=dict(arguments),
            found=False,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            error=f"no such tool: {name!r}. Available: {sorted(TOOLS_BY_NAME)}",
        )

    try:
        parsed = spec.input_model.model_validate(arguments)
    except Exception as error:
        return None, ToolInvocation(
            tool=name,
            arguments=dict(arguments),
            found=False,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            error=f"invalid arguments ({type(error).__name__})",
        )

    try:
        result = getattr(toolset, name)(parsed)
    except Exception as error:
        return None, ToolInvocation(
            tool=name,
            arguments=dict(arguments),
            found=False,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            error=f"{type(error).__name__}: {error}",
        )

    return result, ToolInvocation(
        tool=name,
        arguments=dict(arguments),
        found=bool(result.found),
        reason=result.reason,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )
