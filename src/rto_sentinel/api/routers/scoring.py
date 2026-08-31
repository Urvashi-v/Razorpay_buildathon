"""The scoring endpoint. The documented contract of this system.

``POST /v1/score`` takes an order and returns a calibrated probability, a
threshold, a band, an action and reason codes.

THE HANDLER'S JOB IS MARSHALLING, AND NOTHING ELSE
--------------------------------------------------
Validate the payload, ask the feature pipeline for a matrix, ask the model for a
probability, ask the decision engine for a decision, serialise the result. There
is no risk logic in this file and there must never be. If a threshold comparison,
a band cut point, or a rupee figure ever appears in a route handler, the decision
layer has been bypassed and the audit trail is no longer complete.

FAILURE POSTURE
---------------
No model loaded means ``503 MODEL_UNAVAILABLE``. It does not mean a default
probability, a base-rate guess, or a cached score from a similar order. A system
that cannot score an order says so.

Implemented against the real artefact: the request loads the stored order, runs
the feature pipeline over the merchant's book as of that order's timestamp,
scores it with the calibrated model, and passes the probability to the
deterministic decision engine. Nothing here synthesises a number.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, status
from pydantic import BaseModel, Field

from rto_sentinel.api.deps import AssessmentServiceDep
from rto_sentinel.api.errors import ApiError, ErrorCode, ErrorResponse
from rto_sentinel.api.routers.orders import RiskAssessmentResponse, assessment_response
from rto_sentinel.contracts.decision import CostInputs
from rto_sentinel.serving.assessment import OrderNotFoundError

#: Orders per batch request. Each one rebuilds its feature context from the
#: database, so this is a real cost ceiling rather than a formality.
MAX_BATCH = 25

router = APIRouter(prefix="/v1", tags=["scoring"])


# `ScoreRequest` and `ScoreResponse` were the Phase 1 placeholders for this
# surface. They accepted a whole order payload and returned a flat score.
#
# Both were removed in Phase 7 rather than kept alongside the real thing. The
# payload form could not be implemented honestly - an order that is not in the
# database has no history and no geographic context, so scoring it would produce
# a confident number for a customer the model would treat as brand new. The flat
# response had no room for the model version, the feature provenance or the
# economic assumptions that a probability needs to travel with.
#
# `RiskAssessmentResponse`, defined in `orders.py` and shared with
# `GET /v1/orders/{order_id}/risk`, replaces both. One shape, one code path, no
# way for the two surfaces to disagree about what a score means.


class StoredOrderScoreRequest(BaseModel):
    """Score an order already in the database, optionally under custom economics.

    WHY THE ORDER IS REFERENCED RATHER THAN SUBMITTED
    -------------------------------------------------
    The features this model needs are not all on the order. Customer history and
    geography aggregates are computed from the merchant's book as of the order's
    own timestamp, so an order that has never been persisted cannot be scored
    correctly - it would arrive with no history and be treated as a first-time
    customer in an unknown pincode.

    Accepting a full :class:`OrderPayload` and scoring it anyway would produce a
    number for every request and quietly wrong numbers for most of them.
    Ingestion of new orders is a separate concern from scoring stored ones, and
    conflating them is how a serving path starts lying.
    """

    order_id: str = Field(max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
    dataset_run_id: str | None = Field(
        default=None, max_length=64, description="Disambiguates ids shared across runs"
    )
    cost_inputs: CostInputs | None = Field(
        default=None, description="Overrides the merchant's configured cost profile"
    )
    include_contributions: bool = Field(
        default=True, description="Return SHAP contributions for the console explanation panel"
    )


@router.post(
    "/score",
    response_model=RiskAssessmentResponse,
    summary="Score one stored order and return a graduated, appealable action",
    responses={
        404: {"model": ErrorResponse, "description": "No such order"},
        503: {"model": ErrorResponse, "description": "No calibrated model artefact loaded"},
    },
)
def score_order(
    request: StoredOrderScoreRequest, service: AssessmentServiceDep
) -> RiskAssessmentResponse:
    """Run the whole chain for one order. See the module docstring for the posture.

    Identical in effect to ``GET /v1/orders/{order_id}/risk``, and calls the same
    service. It exists as a POST because it accepts custom cost inputs, which do
    not belong in a query string.
    """
    try:
        assessment = service.assess(
            request.order_id,
            dataset_run_id=request.dataset_run_id,
            cost_inputs=request.cost_inputs,
            include_contributions=request.include_contributions,
        )
    except OrderNotFoundError as error:
        raise ApiError(
            ErrorCode.ORDER_NOT_FOUND,
            str(error),
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"order_id": request.order_id},
        ) from error
    return assessment_response(assessment)


@router.post(
    "/score/batch",
    response_model=list[RiskAssessmentResponse],
    summary="Score several stored orders",
    responses={
        404: {"model": ErrorResponse, "description": "No such order"},
        503: {"model": ErrorResponse, "description": "No calibrated model artefact loaded"},
    },
)
def score_batch(
    requests: Annotated[list[StoredOrderScoreRequest], Body(max_length=MAX_BATCH)],
    service: AssessmentServiceDep,
) -> list[RiskAssessmentResponse]:
    """Batch scoring for the console's order stream.

    Capped at ``MAX_BATCH``. Each order rebuilds its own feature context from the
    database, so a batch is genuinely N times the work of one order rather than a
    vectorised shortcut - an uncapped batch would be a denial-of-service surface
    dressed as a convenience.

    Fails on the first missing order rather than returning partial results. A
    caller receiving nine scores where it asked for ten, with no indication which
    is absent, is worse off than one receiving an error.
    """
    return [score_order(request, service) for request in requests]
