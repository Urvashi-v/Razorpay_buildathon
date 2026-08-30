"""Order listing, retrieval, and the full risk assessment for one order.

``GET /v1/orders``                     a page of stored orders
``GET /v1/orders/{order_id}``          one order, with its outcome if matured
``GET /v1/orders/{order_id}/risk``     the whole chain: features, model, decision

EVERY FIELD HERE COMES FROM THE DATABASE OR FROM A MODEL RUN
=============================================================
Nothing on these responses is synthesised. The order fields are columns; the
probability is the trained artefact's output over features rebuilt by the
training pipeline; the band and action are the deterministic engine's. When any
link in that chain is unavailable the endpoint fails - 503 for a missing model,
404 for a missing order - rather than returning a number that looks like a score.

ORDER IDS ARE SCOPED TO A DATASET RUN
=====================================
Each generator run numbers orders from ``ORD-00000001``, so a database holding
two benchmark runs holds two orders with the same id belonging to different
synthetic universes. ``dataset_run`` disambiguates; without it the most recent
run wins, which is defined rather than arbitrary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Path, Query, status
from pydantic import BaseModel, Field

from rto_sentinel.api.deps import AssessmentServiceDep, ServingRepositoryDep
from rto_sentinel.api.errors import ApiError, ErrorCode, ErrorResponse
from rto_sentinel.contracts.enums import InterventionAction, RiskBand
from rto_sentinel.contracts.risk import FeatureContribution
from rto_sentinel.serving.assessment import OrderAssessment, OrderNotFoundError

router = APIRouter(prefix="/v1/orders", tags=["orders"])

#: Order ids are `ORD-` plus digits in this benchmark. Constrained rather than
#: free text so a malformed id is a 422 with a clear message instead of a
#: database round trip that finds nothing.
ORDER_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$"

OrderIdPath = Annotated[
    str, Path(pattern=ORDER_ID_PATTERN, max_length=64, description="Order identifier")
]
DatasetRunQuery = Annotated[
    str | None,
    Query(max_length=64, description="Disambiguates ids shared across runs; newest by default"),
]


class OrderSummary(BaseModel):
    """One order as a list row. Columns, nothing computed."""

    order_id: str
    merchant_id: str
    customer_hash: str
    ordered_at: datetime
    payment_method: str
    is_cod: bool
    order_value_inr: float
    discount_inr: float
    item_count: int
    category: str
    courier_partner: str | None = None
    split: str
    dataset_run_id: str | None = None
    #: NULL until the order resolves. Never defaulted to False - an immature
    #: order has no outcome, and saying otherwise manufactures optimism.
    is_rto: bool | None = None
    outcome: str | None = None
    resolved_at: datetime | None = None


class OrderPageResponse(BaseModel):
    """A page, with the total so a client can paginate without guessing."""

    orders: list[OrderSummary]
    total: int = Field(ge=0, description="Matching orders before pagination")
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class ModelProvenance(BaseModel):
    """Which model produced a score, and what it was trained on."""

    model_name: str
    model_version: str
    calibration_method: str | None
    calibration_fitted_on: str | None
    feature_version: str
    feature_fingerprint: str
    dataset_run_id: str
    generator_version: str
    trained_at: datetime
    training_rows: int
    n_features: int
    selection_manifest_id: str


class EconomicAssumptions(BaseModel):
    """The merchant inputs and the assumptions that derived the threshold.

    ``intervention_success_rate`` and ``abandonment_on_friction`` are labelled
    assumptions in every surface that reports them, because neither has been
    measured on this or any data.
    """

    cost_profile: str
    rto_cost_inr: float
    contribution_margin_inr: float
    friction_support_cost_inr: float
    abandonment_on_friction: float = Field(description="ASSUMED. Never measured.")
    intervention_success_rate: float = Field(description="ASSUMED. Never measured.")
    cost_false_positive_inr: float = Field(description="C_fp")
    saving_true_positive_inr: float = Field(description="S_tp")
    threshold_formula: str
    band_intervention_success_rate: float = Field(
        description="ASSUMED effectiveness of THIS band's action"
    )
    band_abandonment_rate: float = Field(description="ASSUMED abandonment for THIS band's action")


class FeatureProvenance(BaseModel):
    """What the score was actually computed from."""

    feature_version: str
    feature_fingerprint: str
    n_features: int
    null_features: list[str] = Field(
        description="Features with no value for this order. Cold start, not an error."
    )
    context_rows: int = Field(
        description="Rows of merchant history the aggregates were computed over"
    )


class RiskAssessmentResponse(BaseModel):
    """The full chain's output for one order.

    Probability, threshold, band and action travel together by design. A bare
    score invites comparison against 0.5, which is the error this whole system
    exists to correct.
    """

    order: OrderSummary
    probability: float = Field(ge=0.0, le=1.0, description="Calibrated P(RTO) at order time")
    raw_score: float | None = Field(default=None, description="Pre-calibration model output")
    threshold: float = Field(ge=0.0, le=1.0, description="Cost-derived, never 0.5")
    band: RiskBand
    action: InterventionAction
    flagged: bool
    reason_codes: list[str]
    expected_value_inr: float = Field(
        description="Expected rupee gain from this action. Rests on ASSUMED rates."
    )
    appeal_available: bool
    human_review_required: bool
    is_control_holdout: bool
    contributions: list[FeatureContribution] = Field(default_factory=list)
    model: ModelProvenance
    features: FeatureProvenance
    economics: EconomicAssumptions
    engine_version: str
    scored_at: datetime
    latency_ms: float | None = None
    outcome_is_known: bool = Field(
        description="True when this order has already resolved - a re-score, not a live decision"
    )
    data_provenance: str = Field(
        default=(
            "Model trained on synthetic benchmark data. Labels are simulated, not "
            "real-world ground truth; see docs/model_card.md."
        )
    )


def _summary(row: object) -> OrderSummary:
    """Build a summary from an ORM row, tolerating a missing outcome.

    `Order.outcome` is the relationship to the outcome row; `outcome.outcome` is
    the terminal state string on it. The names collide unhelpfully, which is why
    both are unpacked here once rather than at each call site.
    """
    outcome = getattr(row, "outcome", None)
    return OrderSummary(
        order_id=row.order_id,  # type: ignore[attr-defined]
        merchant_id=row.merchant_id,  # type: ignore[attr-defined]
        customer_hash=row.customer_hash,  # type: ignore[attr-defined]
        ordered_at=row.ordered_at,  # type: ignore[attr-defined]
        payment_method=row.payment_method,  # type: ignore[attr-defined]
        is_cod=row.is_cod,  # type: ignore[attr-defined]
        order_value_inr=row.order_value_inr,  # type: ignore[attr-defined]
        discount_inr=row.discount_inr,  # type: ignore[attr-defined]
        item_count=row.item_count,  # type: ignore[attr-defined]
        category=row.category,  # type: ignore[attr-defined]
        courier_partner=row.courier_partner,  # type: ignore[attr-defined]
        split=row.split,  # type: ignore[attr-defined]
        dataset_run_id=row.dataset_run_id,  # type: ignore[attr-defined]
        is_rto=getattr(outcome, "is_rto", None),
        outcome=getattr(outcome, "outcome", None),
        resolved_at=getattr(outcome, "resolved_at", None),
    )


def assessment_response(assessment: OrderAssessment) -> RiskAssessmentResponse:
    """Assemble the response. Shared so scoring and orders cannot diverge."""
    decision = assessment.decision
    derivation = assessment.threshold
    inputs = assessment.cost_inputs
    summary = _summary(assessment.order)
    summary = summary.model_copy(
        update={
            "is_rto": getattr(assessment.outcome, "is_rto", None),
            "outcome": getattr(assessment.outcome, "outcome", None),
            "resolved_at": getattr(assessment.outcome, "resolved_at", None),
        }
    )

    return RiskAssessmentResponse(
        order=summary,
        probability=decision.probability,
        raw_score=assessment.score.raw_score,
        threshold=decision.threshold,
        band=decision.band,
        action=decision.action,
        flagged=decision.flagged,
        reason_codes=list(decision.reason_codes),
        expected_value_inr=decision.expected_value_inr,
        appeal_available=decision.appeal_available,
        human_review_required=decision.human_review_required,
        is_control_holdout=decision.is_control_holdout,
        contributions=list(assessment.score.contributions),
        model=ModelProvenance(
            model_name=assessment.model.card.model_name,
            model_version=assessment.model.card.model_version,
            calibration_method=assessment.model.card.calibration_method,
            calibration_fitted_on=assessment.model.card.calibration_fitted_on,
            feature_version=assessment.model.card.feature_version,
            feature_fingerprint=assessment.model.card.feature_fingerprint,
            dataset_run_id=assessment.model.card.dataset_run_id,
            generator_version=assessment.model.card.generator_version,
            trained_at=assessment.model.card.trained_at,
            training_rows=assessment.model.card.training_rows,
            n_features=len(assessment.model.card.feature_names),
            selection_manifest_id=assessment.model.manifest.manifest_id,
        ),
        features=FeatureProvenance(
            feature_version=assessment.features.feature_version,
            feature_fingerprint=assessment.features.feature_fingerprint,
            n_features=len(assessment.features.feature_names),
            null_features=list(assessment.features.null_features),
            context_rows=assessment.features.context_rows,
        ),
        economics=EconomicAssumptions(
            cost_profile=assessment.cost_profile,
            rto_cost_inr=inputs.rto_cost_inr,
            contribution_margin_inr=inputs.contribution_margin_inr,
            friction_support_cost_inr=inputs.friction_support_cost_inr,
            abandonment_on_friction=inputs.abandonment_on_friction,
            intervention_success_rate=inputs.intervention_success_rate,
            cost_false_positive_inr=derivation.cost_false_positive_inr,
            saving_true_positive_inr=derivation.saving_true_positive_inr,
            threshold_formula=derivation.formula,
            band_intervention_success_rate=assessment.assumed_intervention_success,
            band_abandonment_rate=assessment.assumed_abandonment,
        ),
        engine_version=decision.engine_version,
        scored_at=assessment.score.scored_at,
        latency_ms=assessment.score.latency_ms,
        outcome_is_known=assessment.label_is_known,
    )


@router.get("", response_model=OrderPageResponse, summary="List stored orders")
def list_orders(
    repository: ServingRepositoryDep,
    merchant_id: Annotated[str | None, Query(max_length=64)] = None,
    customer_hash: Annotated[str | None, Query(max_length=64)] = None,
    split: Annotated[str | None, Query(max_length=32)] = None,
    payment_method: Annotated[str | None, Query(pattern=r"^(cod|prepaid)$")] = None,
    dataset_run: DatasetRunQuery = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> OrderPageResponse:
    """A page of orders, newest first.

    ``limit`` is capped at 200. An uncapped page size is a denial-of-service
    surface and an invitation to load the whole table into a browser.
    """
    page = repository.list_orders(
        merchant_id=merchant_id,
        customer_hash=customer_hash,
        split=split,
        payment_method=payment_method,
        dataset_run_id=dataset_run,
        limit=limit,
        offset=offset,
    )
    return OrderPageResponse(
        orders=[_summary(order) for order in page.orders],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/{order_id}",
    response_model=OrderSummary,
    summary="Retrieve one order",
    responses={404: {"model": ErrorResponse, "description": "No such order"}},
)
def get_order(
    order_id: OrderIdPath, repository: ServingRepositoryDep, dataset_run: DatasetRunQuery = None
) -> OrderSummary:
    order = repository.get_order(order_id, dataset_run_id=dataset_run)
    if order is None:
        raise ApiError(
            ErrorCode.ORDER_NOT_FOUND,
            f"no order {order_id!r} in the database",
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"order_id": order_id, "dataset_run": dataset_run},
        )
    outcome = repository.get_outcome(order)
    summary = _summary(order)
    return summary.model_copy(
        update={
            "is_rto": getattr(outcome, "is_rto", None),
            "outcome": getattr(outcome, "outcome", None),
            "resolved_at": getattr(outcome, "resolved_at", None),
        }
    )


@router.get(
    "/{order_id}/risk",
    response_model=RiskAssessmentResponse,
    summary="Score one order and return its economic decision",
    responses={
        404: {"model": ErrorResponse, "description": "No such order"},
        503: {"model": ErrorResponse, "description": "No calibrated model artefact loaded"},
    },
)
def assess_order(
    order_id: OrderIdPath,
    service: AssessmentServiceDep,
    dataset_run: DatasetRunQuery = None,
    include_contributions: Annotated[bool, Query()] = True,
) -> RiskAssessmentResponse:
    """Run the full chain: database, features, model, calibration, decision.

    Every step executes on every call. When the model artefact is missing this
    returns 503 with the command that would produce one - it does not fall back
    to a default probability.
    """
    try:
        assessment = service.assess(
            order_id,
            dataset_run_id=dataset_run,
            include_contributions=include_contributions,
        )
    except OrderNotFoundError as error:
        raise ApiError(
            ErrorCode.ORDER_NOT_FOUND,
            str(error),
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"order_id": order_id, "dataset_run": dataset_run},
        ) from error
    return assessment_response(assessment)
