"""Evaluation endpoints - how the console gets its numbers.

``GET /v1/evaluation/ladder``       every rung, scored identically
``GET /v1/evaluation/reliability``  the reliability diagram bins
``GET /v1/evaluation/fairness``     flag rate and precision by tier and value band

THE DASHBOARD READS THESE. IT DOES NOT HOLD NUMBERS OF ITS OWN.
---------------------------------------------------------------
No metric is hardcoded in the frontend, and no metric is computed there. Every
figure the console shows arrives from an evaluation report produced by the
harness and stored as an artefact. A chart that can invent its own numbers is a
picture, not a report - and the whole argument of this project is that the
numbers are checkable.

The fairness endpoint returns its slices whether or not the disparity trigger
fired. Reporting only the flattering runs would defeat the point of having an
audit at all.

STATUS: Phase 4.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from rto_sentinel.api.deps import SettingsDep
from rto_sentinel.api.errors import ErrorResponse, not_implemented
from rto_sentinel.contracts.evaluation import (
    CalibrationMetrics,
    EvaluationReport,
    FairnessAudit,
)

router = APIRouter(prefix="/v1/evaluation", tags=["evaluation"])


@router.get(
    "/ladder",
    response_model=list[EvaluationReport],
    summary="Every rung of the baseline ladder, scored on identical footing",
    responses={404: {"model": ErrorResponse, "description": "No evaluation artefacts found"}},
)
def ladder_results(
    settings: SettingsDep,
    split: str = Query(default="validation", description="validation | test"),
) -> list[EvaluationReport]:
    """Rungs 0-5 with PR-AUC, ECE, flag rate and net rupees.

    Including the rungs that beat the model, if any of them do.
    """
    raise not_implemented("Ladder results", "Phase 4 (evaluation harness)")


@router.get(
    "/reliability",
    response_model=CalibrationMetrics,
    summary="Reliability diagram bins and calibration error",
)
def reliability(
    settings: SettingsDep,
    model_name: str = Query(default="lightgbm_isotonic"),
    split: str = Query(default="validation"),
) -> CalibrationMetrics:
    """The bins behind the reliability diagram, plus ECE and Brier score.

    Returns the bins rather than an image so the console can draw it and a
    reviewer can recompute it.
    """
    raise not_implemented("Reliability diagram", "Phase 4 (evaluation harness)")


@router.get(
    "/fairness",
    response_model=FairnessAudit,
    summary="Flag rate and precision by pincode tier and order-value band",
)
def fairness(
    settings: SettingsDep,
    model_name: str = Query(default="lightgbm_isotonic"),
    split: str = Query(default="validation"),
) -> FairnessAudit:
    """The disparate-impact review, reported whether or not it tripped."""
    raise not_implemented("Fairness audit", "Phase 4 (evaluation harness)")
