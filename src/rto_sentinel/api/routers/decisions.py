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

STATUS: Phase 4.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from rto_sentinel.api.deps import DbSession
from rto_sentinel.api.errors import ErrorResponse, not_implemented
from rto_sentinel.contracts.decision import OpsOverride
from rto_sentinel.contracts.enums import InterventionAction, RiskBand

router = APIRouter(prefix="/v1/decisions", tags=["decisions"])


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


@router.get(
    "/queue",
    response_model=QueueResponse,
    summary="SEVERE-band decisions awaiting human review, oldest first",
)
def review_queue(
    session: DbSession,
    merchant_id: str = Query(description="Merchant whose queue to read"),
    limit: int = Query(default=50, ge=1, le=200),
) -> QueueResponse:
    raise not_implemented("Review queue", "Phase 4 (database and console)")


@router.get(
    "/{order_id}",
    response_model=DecisionRecord,
    summary="Fetch the logged decision for one order",
    responses={404: {"model": ErrorResponse, "description": "No decision for that order"}},
)
def get_decision(order_id: str, session: DbSession) -> DecisionRecord:
    raise not_implemented("Decision lookup", "Phase 4 (database and console)")


class OverrideResponse(BaseModel):
    accepted: bool
    order_id: str
    new_band: RiskBand
    logged_at: datetime
    note: str = Field(default="", description="Free-text reason recorded with the override")


@router.post(
    "/override",
    response_model=OverrideResponse,
    summary="Record a human override of the engine's recommendation",
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse, "description": "Override does not change the band"},
    },
)
def override_decision(override: OpsOverride, session: DbSession) -> OverrideResponse:
    """Append an override. Never mutates the original decision row."""
    raise not_implemented("Ops override", "Phase 4 (database and console)")
