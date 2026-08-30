"""Decision log, ops review queue, and overrides.

``GET  /v1/decisions/{order_id}``  the decision that was made, with reason codes
``GET  /v1/decisions/queue``       SEVERE-band orders awaiting a human
``POST /v1/decisions/override``    an ops human changing the recommendation

THE OVERRIDE ENDPOINT IS NOT A CONVENIENCE
------------------------------------------
SPEC section 09 lists "ops override always available, and logged" as a human
safeguard. It is available on every decision, including SEVERE, and it cannot be
switched off by configuration. An override is appended to the log with the
operator's hashed identity, the direction, and an optional note - never applied
by mutating the original decision, because an audit trail that can be edited is
not an audit trail.

Overrides are also *signal*. An associate relaxing a SEVERE band is telling the
system something the model did not know, and SPEC section 02 treats those as
counterfactual evidence feeding the outcome loop.

The decision log is APPEND-ONLY. There is no update endpoint and no update
method on the repository. When an operator disagrees, an override is appended
alongside the original, which stays exactly as it was.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Path, Query, status
from pydantic import BaseModel, Field

from rto_sentinel.api.deps import (
    AssessmentServiceDep,
    DecisionLogDep,
    OverrideRepositoryDep,
    ServingRepositoryDep,
)
from rto_sentinel.api.errors import ApiError, ErrorCode, ErrorResponse
from rto_sentinel.contracts.enums import InterventionAction, OverrideDirection, RiskBand, band_rank
from rto_sentinel.db.repositories import split_reason_codes
from rto_sentinel.serving.assessment import OrderNotFoundError

router = APIRouter(prefix="/v1/decisions", tags=["decisions"])

OrderIdPath = Annotated[
    str,
    Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$", max_length=64),
]


class DecisionRecord(BaseModel):
    """One logged decision, as the console renders it in the order queue."""

    order_id: str
    probability: float
    threshold: float
    band: RiskBand
    action: InterventionAction
    flagged: bool
    reason_codes: list[str]
    expected_value_inr: float
    appeal_available: bool
    human_review_required: bool
    is_control_holdout: bool
    model_name: str
    model_version: str
    engine_version: str
    decided_at: datetime


class QueueResponse(BaseModel):
    """The human review queue.

    Ordered oldest-first rather than by risk score. A queue sorted by score
    leaves the least risky appeals waiting longest, and those are
    disproportionately the false positives - customers who did nothing wrong and
    are the people the appeal path exists for.
    """

    items: list[DecisionRecord]
    total_pending: int


class DecideRequest(BaseModel):
    """Score an order and persist the resulting decision."""

    order_id: str = Field(max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
    dataset_run_id: str | None = Field(default=None, max_length=64)
    is_control_holdout: bool = Field(
        default=False,
        description=(
            "Mark this order as part of the randomised no-friction slice. It is banded "
            "as usual and no action is taken, which is what keeps precision measurable."
        ),
    )


class OverrideRequest(BaseModel):
    """A human changing the engine's recommendation.

    The reason is mandatory and length-checked. An override with no stated reason
    is unusable as the counterfactual evidence it is supposed to be - "an
    operator disagreed" tells the outcome loop nothing about *why*.
    """

    order_id: str = Field(max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
    override_band: RiskBand
    operator_id: str = Field(
        min_length=1, max_length=64, description="Hashed operator identity, never a name"
    )
    reason: str = Field(
        min_length=10,
        max_length=1000,
        description="Why the operator disagreed. Mandatory: an unexplained override is noise.",
    )


class OverrideResponse(BaseModel):
    accepted: bool
    order_id: str
    original_band: RiskBand
    new_band: RiskBand
    direction: OverrideDirection
    logged_at: datetime
    note: str = Field(default="", description="Free-text reason recorded with the override")
    original_decision_unchanged: bool = Field(
        default=True,
        description="The original decision row is never mutated. Overrides are appended.",
    )


def _record(row: object) -> DecisionRecord:
    return DecisionRecord(
        order_id=row.order_id,  # type: ignore[attr-defined]
        probability=row.probability,  # type: ignore[attr-defined]
        threshold=row.threshold,  # type: ignore[attr-defined]
        band=RiskBand(row.band),  # type: ignore[attr-defined]
        action=InterventionAction(row.action),  # type: ignore[attr-defined]
        flagged=row.flagged,  # type: ignore[attr-defined]
        reason_codes=split_reason_codes(row.reason_codes),  # type: ignore[attr-defined]
        expected_value_inr=row.expected_value_inr,  # type: ignore[attr-defined]
        appeal_available=row.appeal_available,  # type: ignore[attr-defined]
        human_review_required=row.human_review_required,  # type: ignore[attr-defined]
        is_control_holdout=row.is_control_holdout,  # type: ignore[attr-defined]
        model_name=row.model_name,  # type: ignore[attr-defined]
        model_version=row.model_version,  # type: ignore[attr-defined]
        engine_version=row.engine_version,  # type: ignore[attr-defined]
        decided_at=row.decided_at,  # type: ignore[attr-defined]
    )


@router.post(
    "",
    response_model=DecisionRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Score an order, take the economic decision, and log it",
    responses={
        404: {"model": ErrorResponse, "description": "No such order"},
        503: {"model": ErrorResponse, "description": "No calibrated model artefact loaded"},
    },
)
def decide(
    request: DecideRequest,
    service: AssessmentServiceDep,
    repository: ServingRepositoryDep,
    log: DecisionLogDep,
) -> DecisionRecord:
    """Run the full chain and append the result to the decision log.

    The log is append-only by construction: there is no update path, and an
    override is a separate row. An audit trail that can be edited is not an audit
    trail.
    """
    try:
        assessment = service.assess(
            request.order_id,
            dataset_run_id=request.dataset_run_id,
            is_control_holdout=request.is_control_holdout,
        )
    except OrderNotFoundError as error:
        raise ApiError(
            ErrorCode.ORDER_NOT_FOUND,
            str(error),
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"order_id": request.order_id},
        ) from error

    row = log.append(
        assessment.decision,
        order_pk=assessment.order.id,
        model_name=assessment.model.card.model_name,
        model_version=assessment.model.card.model_version,
        config_fingerprint=assessment.model.card.config_fingerprint,
    )
    return _record(row)


@router.get(
    "/queue",
    response_model=QueueResponse,
    summary="Decisions awaiting human review, oldest first",
)
def review_queue(
    log: DecisionLogDep,
    merchant_id: Annotated[str, Query(max_length=64, description="Merchant whose queue to read")],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> QueueResponse:
    rows = log.list_review_queue(merchant_id, limit=limit)
    return QueueResponse(items=[_record(row) for row in rows], total_pending=len(rows))


@router.get(
    "/{order_id}",
    response_model=DecisionRecord,
    summary="Fetch the latest logged decision for one order",
    responses={404: {"model": ErrorResponse, "description": "No decision for that order"}},
)
def get_decision(order_id: OrderIdPath, log: DecisionLogDep) -> DecisionRecord:
    row = log.get_latest_decision(order_id)
    if row is None:
        raise ApiError(
            ErrorCode.DECISION_NOT_FOUND,
            f"no decision has been logged for order {order_id!r}",
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"order_id": order_id},
        )
    return _record(row)


@router.post(
    "/override",
    response_model=OverrideResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a human override of the engine's recommendation",
    responses={
        404: {"model": ErrorResponse, "description": "No decision to override"},
        422: {"model": ErrorResponse, "description": "Override does not change the band"},
    },
)
def override_decision(
    request: OverrideRequest,
    log: DecisionLogDep,
    overrides: OverrideRepositoryDep,
) -> OverrideResponse:
    """Append an override against the latest decision. Never mutates it.

    The direction is derived from the band change rather than accepted from the
    client: a caller who says "relaxed" while raising the band would corrupt
    every aggregate built on that field, and the two facts must agree by
    construction rather than by trust.
    """
    from rto_sentinel.contracts.decision import OpsOverride

    decision = log.get_latest_decision(request.order_id)
    if decision is None:
        raise ApiError(
            ErrorCode.DECISION_NOT_FOUND,
            f"no decision has been logged for order {request.order_id!r}, so there is "
            "nothing to override. Score it first via POST /v1/decisions.",
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"order_id": request.order_id},
        )

    original = RiskBand(decision.band)
    if request.override_band is original:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            f"the override band {request.override_band.value} is the current band; an "
            "override must change the decision.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"order_id": request.order_id, "current_band": original.value},
        )

    direction = (
        OverrideDirection.ESCALATED
        if band_rank(request.override_band) > band_rank(original)
        else OverrideDirection.RELAXED
    )
    created_at = datetime.now(UTC)
    override = OpsOverride(
        order_id=request.order_id,
        original_band=original,
        override_band=request.override_band,
        direction=direction,
        operator_id=request.operator_id,
        note=request.reason,
        created_at=created_at,
    )
    overrides.append(override, decision_pk=decision.id)

    return OverrideResponse(
        accepted=True,
        order_id=request.order_id,
        original_band=original,
        new_band=request.override_band,
        direction=direction,
        logged_at=created_at,
        note=request.reason,
    )
