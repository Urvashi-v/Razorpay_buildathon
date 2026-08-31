"""Operational statistics: what is loaded, what is stored, what has been decided.

``GET /v1/monitoring/model``     the loaded artefact and its provenance
``GET /v1/monitoring/data``      what the database actually contains
``GET /v1/monitoring/decisions`` decisions taken, by band, and human overrides
``GET /v1/monitoring/drift``     baseline period versus current period

WHAT THIS IS NOT
================
It is not a metrics endpoint. Model quality - PR-AUC, calibration error, net
rupees - lives under ``/v1/evaluation`` and is read from the frozen evaluation
artefacts, because a metric is a measurement against held-out labels and cannot
be recomputed from live traffic that has not resolved yet.

What lives here is operational: is a model loaded, which one, how much data is
stored, how many decisions have been taken and in which bands, and how often a
human disagreed. Every number is a count from the database or a field from the
loaded model card. None of it is estimated, and when something is unavailable the
response says so rather than reporting a zero that reads like a measurement.

WHY THE MODEL ENDPOINT DOES NOT RAISE
=====================================
"No model is loaded" is exactly what an operator queries monitoring to find out.
Returning 503 here would mean the endpoint that answers the question fails
whenever the answer is interesting.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from rto_sentinel.api.deps import (
    DbSession,
    DecisionLogDep,
    ModelRegistryDep,
    OverrideRepositoryDep,
    SettingsDep,
)
from rto_sentinel.api.errors import ApiError, ErrorCode, ErrorResponse
from rto_sentinel.contracts.monitoring import DriftReport
from rto_sentinel.db.models import DatasetRun, Order, OrderOutcomeRecord
from rto_sentinel.eval.responsible_report import RESPONSIBLE_DIR, read_contract_payload

router = APIRouter(prefix="/v1/monitoring", tags=["monitoring"])


class ModelStatusResponse(BaseModel):
    """What the server would score with, right now."""

    available: bool
    reason: str | None = Field(default=None, description="Why no model is loaded, when none is")
    model_name: str | None = None
    model_version: str | None = None
    calibration_method: str | None = None
    calibration_fitted_on: str | None = None
    feature_version: str | None = None
    feature_fingerprint: str | None = None
    dataset_run_id: str | None = None
    generator_version: str | None = None
    trained_at: datetime | None = None
    training_rows: int | None = None
    n_features: int | None = None
    selection_manifest_id: str | None = None
    artifact_path: str | None = None


class DatasetRunSummary(BaseModel):
    run_id: str
    generator_version: str
    seed: int
    n_orders: int
    created_at: datetime


class DataStatusResponse(BaseModel):
    """What the database holds. Counts, not estimates."""

    dataset_runs: list[DatasetRunSummary]
    total_orders: int
    orders_by_split: dict[str, int]
    orders_by_payment_method: dict[str, int]
    matured_orders: int = Field(
        description="Orders whose outcome has resolved and is therefore usable as a label"
    )
    immature_orders: int = Field(
        description="Outcome not yet known. NULL label, never defaulted to 'delivered'."
    )
    observed_rto_rate: float | None = Field(
        default=None,
        description="Among MATURED orders only. Null when nothing has matured yet.",
    )


class DecisionStatusResponse(BaseModel):
    """Decisions taken and human disagreement with them."""

    total_decisions: int
    decisions_by_band: dict[str, int]
    awaiting_human_review: int
    total_overrides: int
    overrides_by_direction: dict[str, int]
    override_rate: float | None = Field(
        default=None, description="Overrides per decision. Null when nothing has been decided."
    )


@router.get("/model", response_model=ModelStatusResponse, summary="The loaded model artefact")
def model_status(registry: ModelRegistryDep) -> ModelStatusResponse:
    """Never raises. "No model" is the answer, not a failure to answer."""
    return ModelStatusResponse(**registry.status())


@router.get("/data", response_model=DataStatusResponse, summary="What the database contains")
def data_status(
    session: DbSession,
    dataset_run: Annotated[str | None, Query(max_length=64)] = None,
) -> DataStatusResponse:
    filters = [Order.dataset_run_id == dataset_run] if dataset_run else []

    runs = [
        DatasetRunSummary(
            run_id=row.run_id,
            generator_version=row.generator_version,
            seed=row.seed,
            n_orders=row.n_orders,
            created_at=row.created_at,
        )
        for row in session.execute(select(DatasetRun).order_by(DatasetRun.created_at.desc()))
        .scalars()
        .all()
    ]

    total = int(
        session.execute(select(func.count()).select_from(Order).where(*filters)).scalar_one()
    )
    by_split = {
        str(split): int(count)
        for split, count in session.execute(
            select(Order.split, func.count()).where(*filters).group_by(Order.split)
        ).all()
    }
    by_payment = {
        str(method): int(count)
        for method, count in session.execute(
            select(Order.payment_method, func.count())
            .where(*filters)
            .group_by(Order.payment_method)
        ).all()
    }

    matured = int(
        session.execute(
            select(func.count())
            .select_from(OrderOutcomeRecord)
            .join(Order, OrderOutcomeRecord.order_pk == Order.id)
            .where(OrderOutcomeRecord.is_mature.is_(True), *filters)
        ).scalar_one()
    )
    returned = int(
        session.execute(
            select(func.count())
            .select_from(OrderOutcomeRecord)
            .join(Order, OrderOutcomeRecord.order_pk == Order.id)
            .where(
                OrderOutcomeRecord.is_mature.is_(True),
                OrderOutcomeRecord.is_rto.is_(True),
                *filters,
            )
        ).scalar_one()
    )

    return DataStatusResponse(
        dataset_runs=runs,
        total_orders=total,
        orders_by_split=by_split,
        orders_by_payment_method=by_payment,
        matured_orders=matured,
        immature_orders=max(total - matured, 0),
        # Computed over matured orders only. Dividing by the full order count
        # would quietly count "not yet resolved" as "did not return" and report a
        # rate lower than reality.
        observed_rto_rate=(returned / matured) if matured else None,
    )


@router.get(
    "/decisions",
    response_model=DecisionStatusResponse,
    summary="Decisions taken and human overrides",
)
def decision_status(
    decisions: DecisionLogDep,
    overrides: OverrideRepositoryDep,
    session: DbSession,
    merchant_id: Annotated[str | None, Query(max_length=64)] = None,
) -> DecisionStatusResponse:
    from rto_sentinel.db.models import Decision as DecisionRow
    from rto_sentinel.db.models import OpsOverrideRecord

    by_band = decisions.band_counts(merchant_id=merchant_id)
    total = sum(by_band.values())
    awaiting = int(
        session.execute(
            select(func.count())
            .select_from(DecisionRow)
            .where(DecisionRow.human_review_required.is_(True))
        ).scalar_one()
    )
    by_direction = overrides.direction_counts()
    total_overrides = int(
        session.execute(select(func.count()).select_from(OpsOverrideRecord)).scalar_one()
    )

    return DecisionStatusResponse(
        total_decisions=total,
        decisions_by_band=by_band,
        awaiting_human_review=awaiting,
        total_overrides=total_overrides,
        overrides_by_direction=by_direction,
        override_rate=(total_overrides / total) if total else None,
    )


@router.get(
    "/drift",
    response_model=DriftReport,
    summary="Baseline period versus current period",
    responses={501: {"model": ErrorResponse, "description": "No drift run exists"}},
)
def drift(settings: SettingsDep) -> DriftReport:
    """The drift comparison, read from the artefact `rto-sentinel monitor` wrote.

    **Drift is not failure, and this endpoint will not imply that it is.** A moved
    input distribution is a fact about the world; whether quality degraded is a
    separate question that needs labels. The two arrive in separate fields -
    ``signals`` carries distances with no verdict attached, ``performance``
    carries labelled comparisons - and ``labels_available`` says whether the
    second question could be answered at all.

    A consumer that renders only ``signals`` and shows green will be wrong in
    exactly the situation that matters: a recent window whose orders have not
    matured, where nothing can be measured and nothing is therefore alarming.
    """
    path = settings.artifact_path / RESPONSIBLE_DIR / "drift_report.json"
    if not path.is_file():
        raise ApiError(
            ErrorCode.NOT_IMPLEMENTED,
            "No drift comparison has been run. Run `rto-sentinel monitor` to produce "
            "one. This endpoint returns 501 rather than an all-clear: a monitoring "
            "page that reports stability without having measured anything is worse "
            "than one that reports nothing.",
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"status": "not_run"},
        )
    # `read_contract_payload` strips the annotations written for readers who open
    # the file directly; they are not part of the contract the console consumes.
    return DriftReport.model_validate(read_contract_payload(path))
