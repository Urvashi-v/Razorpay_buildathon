"""Cost-model, threshold and merchant-simulation endpoints.

``GET  /v1/economics/profiles``   the configured merchant cost profiles
``POST /v1/economics/threshold``  derive the operating threshold from four inputs
``POST /v1/economics/simulate``   recompute the whole policy under new economics
``POST /v1/economics/what-if``    the four headline numbers, for a slider
``GET  /v1/economics/sweep``      the threshold sweep, as a diagnostic

This is the most important surface in the console. Dragging the contribution
margin slider and watching the threshold, the flag rate, the false-positive cost
and the net saving all move together is the clearest demonstration that the model
is embedded in a business rather than floating above one.

THE RECALCULATION IS REAL, AND IT HAPPENS HERE
==============================================
``/simulate`` re-derives the threshold, re-resolves every band boundary,
re-assigns every order in the scored book to a band, and re-prices the result.
There is one implementation of that arithmetic - ``decision.portfolio`` - and the
CLI report calls the same one. Nothing is scaled, interpolated or cached, and no
economic arithmetic happens in the browser.

The arithmetic is also exposed, not hidden: ``/threshold`` returns ``C_fp`` and
``S_tp`` alongside the result, so a reviewer can check the number by hand. A risk
threshold that arrives without its derivation is a magic constant, and magic
constants are how 0.5 became the industry default in the first place.

WHAT THESE ENDPOINTS REFUSE
===========================
The sealed test split. Every one of them scores the validation book, and the
service layer raises if asked for ``test``. A slider is dragged dozens of times
in a demo; wiring one to the sealed set would consume it silently.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from rto_sentinel.api.deps import AppConfigDep, ScoredBookDep
from rto_sentinel.api.errors import ApiError, ErrorCode, ErrorResponse
from rto_sentinel.configuration.schemas import CostProfile
from rto_sentinel.contracts.decision import CostInputs, ThresholdDerivation
from rto_sentinel.contracts.economics import ThresholdSweep
from rto_sentinel.decision.portfolio import PortfolioError
from rto_sentinel.decision.simulation import SimulationError, SimulationResult, simulate
from rto_sentinel.decision.threshold import derive_threshold
from rto_sentinel.decision.threshold_analysis import SweepError, sweep_thresholds

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
    assumption_warning: str = Field(
        default=(
            "intervention_success_rate and abandonment_on_friction are ASSUMPTIONS. "
            "Neither has been measured on this or any data, and every rupee figure "
            "derived from them inherits that uncertainty."
        )
    )


class SimulationRequest(BaseModel):
    """Recompute the decision policy under different merchant economics."""

    cost_inputs: CostInputs
    split: str = Field(default="validation", description="Never 'test' - the seal forbids it")
    compare_to_profile: str | None = Field(
        default=None, description="Named profile to report deltas against"
    )


class WhatIfResponse(BaseModel):
    """The four numbers that move together when a slider moves."""

    threshold: float
    flag_rate: float
    total_false_positive_cost_inr: float
    net_inr_saved_per_1000_orders: float
    n_orders: int


def _cost_inputs_from(profile: CostProfile) -> CostInputs:
    """The five merchant inputs, lifted out of a configured profile."""
    return CostInputs(
        rto_cost_inr=profile.rto_cost_inr,
        contribution_margin_inr=profile.contribution_margin_inr,
        abandonment_on_friction=profile.abandonment_on_friction,
        intervention_success_rate=profile.intervention_success_rate,
        friction_support_cost_inr=profile.friction_support_cost_inr,
    )


def _run(request: SimulationRequest, config: AppConfigDep, book: ScoredBookDep) -> SimulationResult:
    """Shared path for every simulation endpoint. One implementation, one truth."""
    baseline = None
    if request.compare_to_profile is not None:
        profile = config.cost_model.profiles.get(request.compare_to_profile)
        if profile is None:
            raise ApiError(
                ErrorCode.VALIDATION_FAILED,
                f"unknown cost profile {request.compare_to_profile!r}",
                detail={"available": sorted(config.cost_model.profiles)},
            )
        baseline = _cost_inputs_from(profile)

    try:
        return simulate(
            book.probabilities,
            cost_inputs=request.cost_inputs,
            policy=config.policy,
            labels=book.labels,
            split=request.split,
            cost_profile=request.compare_to_profile or "custom",
            baseline=baseline,
        )
    except (SimulationError, PortfolioError) as error:
        raise ApiError(
            ErrorCode.INVALID_COST_INPUTS,
            str(error),
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from error


@router.get("/profiles", response_model=ProfilesResponse, summary="List cost profiles")
def list_profiles(config: AppConfigDep) -> ProfilesResponse:
    """Return the configured profiles and the bounds the API enforces."""
    cost_model = config.cost_model
    return ProfilesResponse(
        default_profile=cost_model.default_profile,
        profiles=[
            CostProfileSummary(key=key, label=profile.label, inputs=_cost_inputs_from(profile))
            for key, profile in cost_model.profiles.items()
        ],
        bounds={
            name: {"min": bound.min, "max": bound.max} for name, bound in cost_model.bounds.items()
        },
    )


@router.post(
    "/threshold",
    response_model=ThresholdDerivation,
    summary="Derive the operating threshold from merchant economics",
    responses={400: {"model": ErrorResponse, "description": "Degenerate cost inputs"}},
)
def derive(cost_inputs: CostInputs) -> ThresholdDerivation:
    """Solve ``threshold = C_fp / (C_fp + S_tp)`` and return the working.

    Not 0.5, and it moves with the merchant's margin. No book is needed: the
    derivation is a function of economics alone and never sees a label, which is
    exactly why it can be published before a sealed evaluation.
    """
    return derive_threshold(cost_inputs)


@router.post(
    "/simulate",
    response_model=SimulationResult,
    summary="Recompute threshold, bands, interventions and economics",
    responses={400: {"model": ErrorResponse}},
)
def simulate_policy(
    request: SimulationRequest, config: AppConfigDep, book: ScoredBookDep
) -> SimulationResult:
    """The full recomputation, server-side.

    Everything downstream of the cost inputs is rebuilt: the threshold, every
    band boundary, the assignment of each order to a band, and the rupee picture.
    """
    return _run(request, config, book)


@router.post(
    "/what-if",
    response_model=WhatIfResponse,
    summary="Re-score the evaluated book under different cost inputs",
    responses={400: {"model": ErrorResponse}},
)
def what_if(
    request: SimulationRequest, config: AppConfigDep, book: ScoredBookDep
) -> WhatIfResponse:
    """The compact form of ``/simulate``, for a slider that redraws four numbers.

    Restricted to the validation split. The sealed test set is scored exactly
    once, and a slider that re-scores it on every drag is precisely the way that
    seal would get broken by accident.
    """
    result = _run(request, config, book)
    return WhatIfResponse(
        threshold=result.threshold.threshold,
        flag_rate=result.economics.flag_rate,
        total_false_positive_cost_inr=result.economics.expected_false_positive_cost_inr,
        net_inr_saved_per_1000_orders=result.economics.expected_net_inr_per_1000_orders,
        n_orders=result.economics.n_orders,
    )


@router.get(
    "/sweep",
    response_model=ThresholdSweep,
    summary="Threshold sweep - a diagnostic, never the way the threshold is chosen",
    responses={400: {"model": ErrorResponse}},
)
def sweep(
    config: AppConfigDep,
    book: ScoredBookDep,
    profile: Annotated[str | None, Query(description="Cost profile; default when omitted")] = None,
) -> ThresholdSweep:
    """Precision, recall, flag rate, expected cost and net rupees across thresholds.

    The response carries ``selection_methodology``, which states that the
    operating point is derived from economics rather than read off this curve.
    That field is required by the contract, so the caveat cannot be dropped in
    transit.
    """
    cost_model = config.cost_model
    key = profile or cost_model.default_profile
    if key not in cost_model.profiles:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            f"unknown cost profile {key!r}",
            detail={"available": sorted(cost_model.profiles)},
        )
    try:
        return sweep_thresholds(
            book.probabilities,
            book.labels,
            cost_inputs=_cost_inputs_from(cost_model.profiles[key]),
            split=book.split,
            cost_profile=key,
        )
    except SweepError as error:
        raise ApiError(
            ErrorCode.VALIDATION_FAILED, str(error), status_code=status.HTTP_400_BAD_REQUEST
        ) from error
