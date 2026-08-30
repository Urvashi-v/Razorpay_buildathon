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

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from rto_sentinel.api.deps import SettingsDep
from rto_sentinel.api.errors import ApiError, ErrorCode, ErrorResponse
from rto_sentinel.contracts.evaluation import CalibrationMetrics
from rto_sentinel.contracts.experiment import LadderResults
from rto_sentinel.contracts.final import FinalEvaluation, SelectionManifest
from rto_sentinel.models.final import FINAL_DIR, load_evaluation, load_manifest

router = APIRouter(prefix="/v1/evaluation", tags=["evaluation"])

SPLITS = Annotated[str, Query(pattern=r"^(validation|test)$", description="validation | test")]


class RungResult(BaseModel):
    """One ladder rung as it was measured. Every field read from the artefact."""

    rung_id: int
    model_name: str
    model_version: str
    evaluated_split: str
    is_calibrated: bool
    pr_auc: float
    pr_auc_ci_low: float
    pr_auc_ci_high: float
    roc_auc: float | None = Field(
        default=None, description="Null when undefined - a constant predictor has no ranking"
    )
    recall_at_precision_80: float | None = None
    expected_calibration_error: float
    brier_score: float
    train_pr_auc: float | None = None
    overfit_gap: float | None = Field(
        default=None, description="Train minus validation PR-AUC. Large means memorised."
    )
    flag_rate: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    net_inr_per_1000_orders: float | None = None
    false_positive_cost_inr: float | None = None


class LadderResponse(BaseModel):
    """The whole ladder, plus the provenance that makes it checkable."""

    dataset_run_id: str
    evaluated_split: str
    seed: int
    cost_profile: str
    threshold: float
    threshold_source: str
    config_fingerprint: str
    feature_fingerprint: str
    created_at: datetime
    rungs: list[RungResult]
    data_provenance: str


class FinalModelResponse(BaseModel):
    """The Phase 5 final model's measured evaluation on one split."""

    manifest_id: str
    model_name: str
    model_version: str
    evaluated_split: str
    calibration_method: str
    is_calibrated: bool
    n_rows: int
    positive_rate: float
    pr_auc: float
    pr_auc_ci_low: float
    pr_auc_ci_high: float
    pr_auc_uncalibrated: float
    roc_auc: float | None
    recall_at_precision_80: float | None
    recall_at_precision_90: float | None
    brier_score: float
    brier_score_uncalibrated: float
    expected_calibration_error: float
    expected_calibration_error_uncalibrated: float
    threshold: float
    flag_rate: float
    precision: float | None
    recall: float | None
    f1: float | None
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    net_inr_per_1000_orders: float
    net_ci_low: float
    net_ci_high: float
    false_positive_cost_inr: float
    do_nothing_loss_per_1000_orders: float
    unseal_reason: str | None = Field(
        default=None, description="Present for the sealed split: why it was opened"
    )
    evaluated_at: datetime
    data_provenance: str


class SelectionResponse(BaseModel):
    """The frozen record of how the shipped model was chosen."""

    manifest_id: str
    frozen_at: datetime
    base_rung: str
    chosen_candidate: str
    chosen_params: dict[str, object]
    calibration_method: str
    calibration_fitted_on: str
    calibration_folds: int
    threshold: float
    threshold_source: str
    cost_profile: str
    model_version: str
    seed: int
    candidates: list[dict[str, object]]
    calibration_candidates: list[dict[str, object]]
    notes: str


def _latest_ladder(settings: SettingsDep, split: str) -> LadderResults:
    root = settings.artifact_path / "experiments"
    candidates = sorted(root.rglob(f"ladder__{split}__*.json")) if root.is_dir() else []
    if not candidates:
        raise ApiError(
            ErrorCode.MODEL_UNAVAILABLE,
            f"no ladder results for the {split!r} split under {root}. Run "
            "`rto-sentinel train` to produce them.",
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"split": split},
        )
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return LadderResults.model_validate_json(newest.read_text(encoding="utf-8"))


def _latest_run_id(settings: SettingsDep) -> str:
    root = settings.artifact_path / FINAL_DIR
    runs = sorted(root.glob("*/selection_manifest.json")) if root.is_dir() else []
    if not runs:
        raise ApiError(
            ErrorCode.MODEL_UNAVAILABLE,
            f"no frozen selection manifest under {root}. Run `rto-sentinel final` first.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return max(runs, key=lambda path: path.stat().st_mtime).parent.name


@router.get(
    "/ladder",
    response_model=LadderResponse,
    summary="Every rung of the baseline ladder, scored on identical footing",
    responses={404: {"model": ErrorResponse, "description": "No evaluation artefacts found"}},
)
def ladder_results(settings: SettingsDep, split: SPLITS = "validation") -> LadderResponse:
    """Rungs 0-4 with PR-AUC, ECE, flag rate and net rupees.

    Read from the machine-readable artefacts `rto-sentinel train` wrote, not
    recomputed. Including the rungs that beat the model - on this benchmark
    logistic regression beats LightGBM, and the endpoint says so.
    """
    results = _latest_ladder(settings, split)
    rungs = []
    for record in results.ordered:
        head = record.headline()
        rungs.append(
            RungResult(
                rung_id=record.rung_id,
                model_name=record.model_name,
                model_version=record.model_version,
                evaluated_split=record.evaluated_split,
                is_calibrated=record.is_calibrated,
                pr_auc=record.ranking.pr_auc.value,
                pr_auc_ci_low=record.ranking.pr_auc.ci_low,
                pr_auc_ci_high=record.ranking.pr_auc.ci_high,
                roc_auc=(
                    record.ranking.roc_auc.value if record.ranking.roc_auc.is_defined else None
                ),
                recall_at_precision_80=record.ranking.recall_at_precision_80,
                expected_calibration_error=record.calibration.expected_calibration_error,
                brier_score=record.calibration.brier_score,
                train_pr_auc=record.train_pr_auc,
                overfit_gap=record.overfit_gap,
                flag_rate=head["flag_rate"],
                precision=head["precision"],
                recall=head["recall"],
                f1=head["f1"],
                net_inr_per_1000_orders=head["net_inr_per_1000"],
                false_positive_cost_inr=head["fp_cost_inr"],
            )
        )

    first = results.records[0]
    return LadderResponse(
        dataset_run_id=results.dataset_run_id,
        evaluated_split=results.evaluated_split,
        seed=results.seed,
        cost_profile=results.cost_profile,
        threshold=results.threshold,
        threshold_source=results.threshold_source,
        config_fingerprint=results.config_fingerprint,
        feature_fingerprint=results.feature_fingerprint,
        created_at=results.created_at,
        rungs=rungs,
        data_provenance=first.data_provenance,
    )


@router.get(
    "/final",
    response_model=FinalModelResponse,
    summary="The shipped model's measured evaluation",
    responses={404: {"model": ErrorResponse, "description": "No evaluation artefacts found"}},
)
def final_results(settings: SettingsDep, split: SPLITS = "test") -> FinalModelResponse:
    """The Phase 5 numbers, read from the artefacts that recorded them.

    The ``test`` split is the honest measurement: the model was selected and
    calibrated on validation, so those figures are optimistic and the artefact
    says so.
    """
    run_id = _latest_run_id(settings)
    try:
        evaluation: FinalEvaluation = load_evaluation(settings.artifact_path, run_id, split)
    except FileNotFoundError as error:
        raise ApiError(
            ErrorCode.MODEL_UNAVAILABLE,
            f"no {split!r} evaluation for dataset {run_id}. "
            + (
                "Run `rto-sentinel final-test` to score the sealed split - once."
                if split == "test"
                else "Run `rto-sentinel final` first."
            ),
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"split": split, "dataset_run_id": run_id},
        ) from error

    point = evaluation.operating_point
    net = evaluation.economics.net_inr_saved_per_1000_orders
    return FinalModelResponse(
        manifest_id=evaluation.manifest_id,
        model_name=evaluation.model_name,
        model_version=evaluation.model_version,
        evaluated_split=evaluation.evaluated_split,
        calibration_method=evaluation.calibration_method,
        is_calibrated=evaluation.is_calibrated,
        n_rows=evaluation.evaluation_summary.n_rows,
        positive_rate=evaluation.evaluation_summary.positive_rate,
        pr_auc=evaluation.ranking.pr_auc.value,
        pr_auc_ci_low=evaluation.ranking.pr_auc.ci_low,
        pr_auc_ci_high=evaluation.ranking.pr_auc.ci_high,
        pr_auc_uncalibrated=evaluation.uncalibrated_pr_auc,
        roc_auc=evaluation.ranking.roc_auc.value if evaluation.ranking.roc_auc.is_defined else None,
        recall_at_precision_80=evaluation.ranking.recall_at_precision_80,
        recall_at_precision_90=evaluation.ranking.recall_at_precision_90,
        brier_score=evaluation.calibration.brier_score,
        brier_score_uncalibrated=evaluation.uncalibrated_calibration.brier_score,
        expected_calibration_error=evaluation.calibration.expected_calibration_error,
        expected_calibration_error_uncalibrated=(
            evaluation.uncalibrated_calibration.expected_calibration_error
        ),
        threshold=point.threshold,
        flag_rate=point.flag_rate,
        precision=point.precision,
        recall=point.recall,
        f1=point.f1,
        true_positives=point.true_positives,
        false_positives=point.false_positives,
        false_negatives=point.false_negatives,
        true_negatives=point.true_negatives,
        net_inr_per_1000_orders=net.value,
        net_ci_low=net.ci_low,
        net_ci_high=net.ci_high,
        false_positive_cost_inr=evaluation.economics.total_false_positive_cost_inr,
        do_nothing_loss_per_1000_orders=evaluation.economics.baseline_net_inr_per_1000_orders,
        unseal_reason=evaluation.unseal_reason,
        evaluated_at=evaluation.evaluated_at,
        data_provenance=evaluation.data_provenance,
    )


@router.get(
    "/selection",
    response_model=SelectionResponse,
    summary="The frozen record of how the shipped model was chosen",
    responses={404: {"model": ErrorResponse}},
)
def selection(settings: SettingsDep) -> SelectionResponse:
    """Every candidate that was tried, not only the one that won.

    A selection record showing only the winner is an advertisement. This returns
    the field, so a reader can see whether the winner won by a margin worth
    having.
    """
    run_id = _latest_run_id(settings)
    manifest: SelectionManifest = load_manifest(settings.artifact_path, run_id)
    return SelectionResponse(
        manifest_id=manifest.manifest_id,
        frozen_at=manifest.frozen_at,
        base_rung=manifest.base_rung,
        chosen_candidate=manifest.chosen_candidate,
        chosen_params=manifest.chosen_params,
        calibration_method=manifest.calibration_method,
        calibration_fitted_on=manifest.calibration_fitted_on,
        calibration_folds=manifest.calibration_folds,
        threshold=manifest.threshold,
        threshold_source=manifest.threshold_source,
        cost_profile=manifest.cost_profile,
        model_version=manifest.model_version,
        seed=manifest.seed,
        candidates=[candidate.model_dump(mode="json") for candidate in manifest.candidates],
        calibration_candidates=[
            candidate.model_dump(mode="json") for candidate in manifest.calibration_candidates
        ],
        notes=manifest.notes,
    )


@router.get(
    "/reliability",
    response_model=CalibrationMetrics,
    summary="Reliability diagram bins and calibration error",
    responses={404: {"model": ErrorResponse}},
)
def reliability(settings: SettingsDep, split: SPLITS = "validation") -> CalibrationMetrics:
    """The bins behind the reliability diagram, plus ECE and Brier score.

    Returns the bins rather than an image so the console can draw it and a
    reviewer can recompute it.
    """
    run_id = _latest_run_id(settings)
    try:
        evaluation = load_evaluation(settings.artifact_path, run_id, split)
    except FileNotFoundError as error:
        raise ApiError(
            ErrorCode.MODEL_UNAVAILABLE,
            f"no {split!r} evaluation for dataset {run_id}.",
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"split": split},
        ) from error
    return evaluation.calibration


@router.get(
    "/fairness",
    summary="Flag rate and precision by pincode tier and order-value band",
    responses={501: {"model": ErrorResponse, "description": "The audit has not been run"}},
)
def fairness(settings: SettingsDep) -> dict[str, object]:
    """The disparate-impact review.

    NOT IMPLEMENTED, AND THAT IS THE HONEST ANSWER
    ----------------------------------------------
    The cohort audit is defined in ``config/evaluation.yaml`` and has never been
    run. Returning a plausible-looking breakdown here would be the single most
    damaging fake in this API: a fairness report nobody computed, presented as
    evidence that the model was checked.

    The 501 carries the reason. No fairness claim about this model should be made
    until the audit exists.
    """
    raise ApiError(
        ErrorCode.NOT_IMPLEMENTED,
        "The cohort and fairness audit has not been run. It is defined in "
        "config/evaluation.yaml and no results exist. This endpoint returns 501 rather "
        "than a fabricated breakdown: a fairness report nobody computed is worse than "
        "none. No fairness claim about this model should be made until it is run.",
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={"config": "config/evaluation.yaml", "status": "not_run"},
    )
