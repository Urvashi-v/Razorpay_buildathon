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

STATUS: Phase 3. The contract below is fixed and documented in OpenAPI now, so
the console can be built against it; the implementation returns an explicit 501
until the model lands rather than fabricating a plausible response.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from rto_sentinel.api.deps import AppConfigDep, DecisionEngineDep, SettingsDep
from rto_sentinel.api.errors import ErrorResponse, not_implemented
from rto_sentinel.contracts.decision import CostInputs
from rto_sentinel.contracts.enums import InterventionAction, RiskBand
from rto_sentinel.contracts.orders import OrderPayload
from rto_sentinel.contracts.risk import FeatureContribution

router = APIRouter(prefix="/v1", tags=["scoring"])


class ScoreRequest(BaseModel):
    """An order to score, with optional per-merchant cost overrides.

    ``cost_inputs`` is optional: when omitted the merchant's configured profile
    is used. It is exposed on the request so the console's threshold sliders can
    re-score a live order stream without a configuration write, which is the
    thirty seconds of the demo that shows the model is embedded in a business.
    """

    order: OrderPayload
    cost_inputs: CostInputs | None = Field(
        default=None, description="Overrides the merchant's configured cost profile"
    )
    include_contributions: bool = Field(
        default=True, description="Return SHAP contributions for the console explanation panel"
    )


class ScoreResponse(BaseModel):
    """The full scoring result.

    Note that probability, threshold, band and action all travel together. The
    console never receives a probability without the threshold that interpreted
    it - a bare score invites someone to compare it against 0.5, which is the
    error this whole system exists to correct.
    """

    order_id: str
    probability: float = Field(ge=0.0, le=1.0, description="Calibrated P(RTO)")
    threshold: float = Field(ge=0.0, le=1.0, description="Cost-derived operating point")
    band: RiskBand
    action: InterventionAction
    flagged: bool
    reason_codes: list[str]
    expected_value_inr: float
    appeal_available: bool = True
    human_review_required: bool = False
    contributions: list[FeatureContribution] = Field(default_factory=list)
    model_name: str
    model_version: str
    engine_version: str
    scored_at: datetime
    latency_ms: float | None = None
    data_provenance: str = Field(
        default="Model trained on synthetic data; see README for what that does and does not claim."
    )


@router.post(
    "/score",
    response_model=ScoreResponse,
    summary="Score one order and return a graduated, appealable action",
    responses={
        501: {"model": ErrorResponse, "description": "Scoring not yet implemented (Phase 3)"},
        503: {"model": ErrorResponse, "description": "No model artefact loaded"},
    },
)
def score_order(
    request: ScoreRequest,
    settings: SettingsDep,
    config: AppConfigDep,
    engine: DecisionEngineDep,
) -> ScoreResponse:
    """Score an order. See the module docstring for the failure posture."""
    raise not_implemented("Order scoring", "Phase 3 (model training and inference)")


@router.post(
    "/score/batch",
    summary="Score a batch of orders",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    responses={501: {"model": ErrorResponse}},
)
def score_batch(
    requests: list[ScoreRequest],
    settings: SettingsDep,
    config: AppConfigDep,
    engine: DecisionEngineDep,
) -> list[ScoreResponse]:
    """Batch scoring for the console's live order stream."""
    raise not_implemented("Batch scoring", "Phase 3 (model training and inference)")
