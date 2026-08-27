"""Cost-model and threshold endpoints - the sliders behind the demo.

``GET  /v1/economics/profiles``   the configured merchant cost profiles
``POST /v1/economics/threshold``  derive the operating threshold from four inputs
``POST /v1/economics/what-if``    re-score the current book at a derived threshold

This is the most important surface in the console. Dragging the contribution
margin slider and watching the threshold, the flag rate, the false-positive cost
and the net saving all move together is the clearest demonstration that the model
is embedded in a business rather than floating above one.

The arithmetic is exposed, not hidden: ``/threshold`` returns ``C_fp`` and
``S_tp`` alongside the result, so a reviewer can check the number by hand. A risk
threshold that arrives without its derivation is a magic constant, and magic
constants are how 0.5 became the industry default in the first place.

STATUS: Phase 2 for threshold derivation, Phase 4 for what-if re-scoring.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from rto_sentinel.api.deps import AppConfigDep
from rto_sentinel.api.errors import ErrorResponse, not_implemented
from rto_sentinel.contracts.decision import CostInputs, ThresholdDerivation

router = APIRouter(prefix="/v1/economics", tags=["economics"])


class CostProfileSummary(BaseModel):
    """One named cost profile from ``config/cost_model.yaml``."""

    key: str
    label: str
    inputs: CostInputs


class ProfilesResponse(BaseModel):
    default_profile: str
    profiles: list[CostProfileSummary]
    bounds: dict[str, dict[str, float]] = Field(
        description="Accepted range per input; values outside are rejected, not clamped"
    )


class WhatIfRequest(BaseModel):
    """Re-score the evaluated book under different merchant economics."""

    cost_inputs: CostInputs
    split: str = Field(default="validation", description="Never 'test' - the seal forbids it")


class WhatIfResponse(BaseModel):
    """The four numbers that move together when a slider moves."""

    threshold: float
    flag_rate: float
    total_false_positive_cost_inr: float
    net_inr_saved_per_1000_orders: float
    n_orders: int


@router.get("/profiles", response_model=ProfilesResponse, summary="List cost profiles")
def list_profiles(config: AppConfigDep) -> ProfilesResponse:
    """Return the configured profiles and the bounds the API enforces.

    Implemented in Phase 2 alongside the threshold derivation it feeds.
    """
    raise not_implemented("Cost profile listing", "Phase 2 (cost model)")


@router.post(
    "/threshold",
    response_model=ThresholdDerivation,
    summary="Derive the operating threshold from merchant economics",
    responses={400: {"model": ErrorResponse, "description": "Degenerate cost inputs"}},
)
def derive(cost_inputs: CostInputs, config: AppConfigDep) -> ThresholdDerivation:
    """Solve ``threshold = C_fp / (C_fp + S_tp)`` and return the working.

    Not 0.5, and it moves with the merchant's margin.
    """
    raise not_implemented("Threshold derivation", "Phase 2 (cost model)")


@router.post(
    "/what-if",
    response_model=WhatIfResponse,
    summary="Re-score the evaluated book under different cost inputs",
    responses={400: {"model": ErrorResponse}},
)
def what_if(request: WhatIfRequest, config: AppConfigDep) -> WhatIfResponse:
    """Recompute flag rate, false-positive cost and net saving at a new threshold.

    Restricted to the validation split. The sealed test set is scored exactly
    once, and a slider that re-scores it on every drag is precisely the way that
    seal would get broken by accident.
    """
    raise not_implemented("What-if re-scoring", "Phase 4 (evaluation harness)")
