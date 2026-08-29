"""Selecting, calibrating and evaluating the final model. Phase 5.

THE ORDER OF OPERATIONS IS THE POINT
====================================
1. **Hyperparameters** are chosen on validation, by PR-AUC. Not by net rupees:
   rupees need a threshold, a threshold needs calibrated probabilities, and
   calibration has not happened yet. Ranking quality is the part calibration
   cannot repair, so it is what this step selects on.
2. **Calibration** is chosen on validation, cross-validated inside it, by
   expected calibration error. ``none`` is a candidate and wins ties, because
   correcting a score that did not need correcting adds a fitted component and
   therefore a way to be wrong.
3. **The threshold** is derived from merchant economics. It never sees a label,
   which is why it can be fixed before any evaluation without contaminating one.
4. **The manifest is frozen.** Every decision above is written to disk with a
   hash over it.
5. **Only then** may the test split be opened, once, by a separate command.

Steps 1-4 read train and validation. Step 5 reads test. Nothing reads test
before step 5, and :func:`evaluate_final_model` is the only function here that
can touch it - it requires a frozen manifest and an explicit unseal reason.

WHAT THIS MODULE DOES NOT DO
============================
It does not pick features - the feature set is frozen by Phase 3's contract and
its fingerprint is recorded. It does not tune the threshold. It does not decide
anything: a probability and an operating point are not a decision, and the
policy layer that turns them into one is a later phase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rto_sentinel.contracts.final import (
    CalibrationCandidate,
    CandidateResult,
    FinalEvaluation,
    SelectionManifest,
)
from rto_sentinel.contracts.risk import ModelCard
from rto_sentinel.decision.threshold import derive_threshold
from rto_sentinel.eval.economics import economic_result
from rto_sentinel.eval.metrics import calibration_metrics, pr_auc, ranking_metrics
from rto_sentinel.models.artifacts import artifact_dir
from rto_sentinel.models.calibrated import CalibratedModel
from rto_sentinel.models.calibration import build_calibrator, compare_calibrators
from rto_sentinel.models.experiment import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    SWEEP_THRESHOLDS,
    _model_version,
    cost_inputs_from_profile,
    split_summary,
    threshold_metrics_at,
)
from rto_sentinel.models.registry import resolve_rung

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import CostModelConfig, FinalModelConfig
    from rto_sentinel.contracts.decision import CostInputs
    from rto_sentinel.features.dataset import ModelingDataset, SplitView
    from rto_sentinel.models.base import RiskModel


class GuardrailViolation(RuntimeError):
    """Raised when the selected model fails a rule that makes it unshippable.

    Not a warning. A model that flags a quarter of all orders is an operational
    problem whatever its precision, and the honest response is to refuse to
    freeze it rather than to record the number and ship anyway.
    """


@dataclass(frozen=True, slots=True)
class FinalModel:
    """The calibrated model, its frozen manifest, and where it was written."""

    model: CalibratedModel
    manifest: SelectionManifest
    card: ModelCard
    artifact_path: Path | None
    #: Calibrated validation scores, kept so plots and the report work from the
    #: same numbers the metrics were computed on.
    validation_scores: np.ndarray
    validation_scores_raw: np.ndarray


# ---------------------------------------------------------------------------
# step 1: hyperparameter selection, on validation
# ---------------------------------------------------------------------------


def _fit_candidate(
    rung_name: str, params: dict[str, Any], train: SplitView
) -> tuple[RiskModel, float]:
    model = resolve_rung(rung_name)(params)
    started = time.perf_counter()
    model.fit(train.x, train.y, train.context)
    return model, time.perf_counter() - started


def search_hyperparameters(
    dataset: ModelingDataset,
    *,
    final_config: FinalModelConfig,
    seed: int,
    bootstrap_iterations: int | None = None,
) -> tuple[list[CandidateResult], RiskModel]:
    """Fit every candidate on train, score on validation, return the field.

    Returns the results *and* the winning fitted model, so the winner is not
    refitted afterwards - a refit is another chance for the shipped model to
    differ from the measured one.

    ``bootstrap_iterations`` comes from the config because the tie rule is
    expressed in units of that interval; the interval on the winner is reported
    later from the evaluation, where it is a result rather than a search
    statistic.
    """
    train, validation = dataset.train, dataset.validation
    y_validation = validation.y.to_numpy(dtype=bool)
    y_train = train.y.to_numpy(dtype=bool)
    iterations = (
        final_config.search.bootstrap_iterations
        if bootstrap_iterations is None
        else bootstrap_iterations
    )

    scored: list[tuple[CandidateResult, RiskModel]] = []
    for candidate in final_config.search.candidates:
        params = {**dict(candidate.params), "random_state": seed}
        model, duration = _fit_candidate(final_config.base_rung, params, train)

        validation_scores = model.predict_proba(validation.x, validation.context)
        train_scores = model.predict_proba(train.x, train.context)
        validation_pr = pr_auc(y_validation, validation_scores)
        train_pr = pr_auc(y_train, train_scores)

        interval: tuple[float, float] | None = None
        if iterations:
            estimate = ranking_metrics(
                y_validation, validation_scores, bootstrap_iterations=iterations, seed=seed
            ).pr_auc
            interval = (estimate.ci_low, estimate.ci_high)

        scored.append(
            (
                CandidateResult(
                    name=candidate.name,
                    note=candidate.note,
                    params=params,
                    train_pr_auc=float(train_pr) if np.isfinite(train_pr) else None,
                    validation_pr_auc=float(validation_pr),
                    validation_pr_auc_ci=interval,
                    train_duration_seconds=duration,
                ),
                model,
            )
        )

    measured = [result for result, _ in scored]
    best_index = _apply_tie_rule(measured, rule=final_config.search.tie_rule)
    results = [
        result.model_copy(update={"selected": index == best_index})
        for index, result in enumerate(measured)
    ]
    return results, scored[best_index][1]


def _capacity(params: dict[str, Any]) -> int:
    """Model size as trees x leaves. The axis the tie rule breaks ties along."""
    return int(params.get("n_estimators", 0)) * int(params.get("num_leaves", 1))


def _apply_tie_rule(candidates: list[CandidateResult], *, rule: str) -> int:
    """Index of the winner.

    ``one_standard_error_then_smallest`` implements the 1-SE rule: every
    candidate within one standard error of the best point estimate is treated as
    tied with it, and the smallest model among those wins.

    The standard error is recovered from the bootstrap interval - a 95%
    percentile interval spans about 2 x 1.96 standard errors - so it needs no
    second resampling pass. Without an interval the rule cannot be applied, and
    the function falls back to the best point estimate rather than inventing a
    tolerance.
    """
    best = max(range(len(candidates)), key=lambda i: candidates[i].validation_pr_auc)
    if rule != "one_standard_error_then_smallest":
        return best

    interval = candidates[best].validation_pr_auc_ci
    if interval is None:
        return best

    standard_error = (interval[1] - interval[0]) / (2 * 1.96)
    floor = candidates[best].validation_pr_auc - standard_error
    tied = [
        index for index, candidate in enumerate(candidates) if candidate.validation_pr_auc >= floor
    ]
    return min(
        tied, key=lambda i: (_capacity(candidates[i].params), -candidates[i].validation_pr_auc)
    )


# ---------------------------------------------------------------------------
# step 2: calibration selection, cross-validated inside validation
# ---------------------------------------------------------------------------


def select_calibration(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    final_config: FinalModelConfig,
    seed: int,
) -> list[CalibrationCandidate]:
    """Compare calibration methods out-of-fold and mark the winner.

    Two gates, and a candidate must pass both:

    1. It beats leaving the scores alone by at least ``minimum_improvement`` in
       expected calibration error.
    2. It does not make the **Brier score** worse by more than
       ``max_brier_degradation``.

    The second gate exists because ECE is computed over ten equal-width bins and
    is noisy: on a couple of thousand rows a perfectly calibrated model still
    scores an ECE around 0.015 from sampling alone, and a flexible calibrator can
    reduce that by fitting the noise in those bins rather than by producing
    better probabilities. Brier is a proper scoring rule and cannot be improved
    that way, so it is used as a veto.

    Anything failing either gate loses to ``none``: a calibrator that does not
    measurably help is a fitted component carrying risk for no return, and it
    would still have to be maintained, versioned and explained.
    """
    config = final_config.calibration
    measured = compare_calibrators(
        scores,
        y_true,
        methods=config.candidates,
        n_folds=config.n_folds,
        seed=seed,
    )
    baseline_ece = measured["none"].expected_calibration_error

    candidates = [
        CalibrationCandidate(
            method=method,
            expected_calibration_error=metrics.expected_calibration_error,
            brier_score=metrics.brier_score,
            improvement_over_none=baseline_ece - metrics.expected_calibration_error,
            n_folds=config.n_folds,
            fitted_on=config.fitted_on,
            reliability_bins=metrics.reliability_bins,
        )
        for method, metrics in measured.items()
    ]

    def rank(candidate: CalibrationCandidate) -> tuple[float, float]:
        return (candidate.expected_calibration_error, candidate.brier_score)

    baseline_brier = measured["none"].brier_score
    contenders = [
        candidate
        for candidate in candidates
        if candidate.method != "none"
        and candidate.improvement_over_none >= config.minimum_improvement
        and candidate.brier_score <= baseline_brier + config.max_brier_degradation
    ]
    winner = min(contenders, key=rank).method if contenders else "none"

    return [
        candidate.model_copy(update={"selected": candidate.method == winner})
        for candidate in candidates
    ]


# ---------------------------------------------------------------------------
# steps 3-4: derive the threshold, check the guardrails, freeze the manifest
# ---------------------------------------------------------------------------


def _check_guardrails(
    final_config: FinalModelConfig,
    *,
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> None:
    guardrails = final_config.guardrails
    flag_rate = float(np.mean(scores >= threshold))
    if flag_rate > guardrails.max_acceptable_flag_rate:
        msg = (
            f"the selected model flags {flag_rate:.1%} of validation orders at the "
            f"cost-derived threshold {threshold:.4f}, above the "
            f"{guardrails.max_acceptable_flag_rate:.0%} ceiling in config/models/final.yaml. "
            "Refusing to freeze it: a model that frictions this share of orders is an "
            "operational problem regardless of its precision."
        )
        raise GuardrailViolation(msg)

    base_rate = float(np.mean(y_true))
    achieved = pr_auc(y_true, scores)
    if achieved < base_rate + guardrails.min_pr_auc_over_base_rate:
        msg = (
            f"the selected model scores PR-AUC {achieved:.4f} against a base rate of "
            f"{base_rate:.4f}, short of the required margin of "
            f"{guardrails.min_pr_auc_over_base_rate:.4f}. Refusing to freeze a model that "
            "barely outranks flagging at random."
        )
        raise GuardrailViolation(msg)


def build_final_model(
    dataset: ModelingDataset,
    *,
    final_config: FinalModelConfig,
    cost_config: CostModelConfig,
    seed: int,
    cost_profile: str | None = None,
    artifact_root: Path | None = None,
    bootstrap_iterations: int | None = None,
) -> FinalModel:
    """Run selection, calibration and freezing. Reads train and validation only.

    The test split is not touched anywhere in this function, and cannot be: it is
    reached through ``dataset.test``, which raises until someone unseals it with
    a written reason.
    """
    train, validation = dataset.train, dataset.validation
    y_validation = validation.y.to_numpy(dtype=bool)

    # --- 1. hyperparameters --------------------------------------------------
    candidates, base_model = search_hyperparameters(
        dataset,
        final_config=final_config,
        seed=seed,
        bootstrap_iterations=bootstrap_iterations,
    )
    winner = next(candidate for candidate in candidates if candidate.selected)
    raw_scores = base_model.predict_proba(validation.x, validation.context)

    # --- 2. calibration ------------------------------------------------------
    calibration_candidates = select_calibration(
        raw_scores, y_validation, final_config=final_config, seed=seed
    )
    chosen_method = next(
        candidate.method for candidate in calibration_candidates if candidate.selected
    )

    # Refit the chosen method on the WHOLE validation split for the artefact.
    # The cross-validated number above stays the reported one: it was measured
    # out-of-fold, and this refit has seen every row it would be scored on.
    calibrator = build_calibrator(chosen_method)
    calibrator.fit(raw_scores, y_validation)
    model = CalibratedModel.wrap(base_model, calibrator)
    calibrated_scores = model.predict_proba(validation.x, validation.context)

    # --- 3. threshold, from economics alone ---------------------------------
    profile_name = cost_profile or cost_config.default_profile
    derivation = derive_threshold(cost_inputs_from_profile(cost_config.profiles[profile_name]))

    # --- 4. guardrails, then freeze -----------------------------------------
    _check_guardrails(
        final_config,
        scores=calibrated_scores,
        y_true=y_validation,
        threshold=derivation.threshold,
    )

    metadata = dataset.metadata
    model_version = _model_version(model.name, metadata.dataset_run_id, seed, winner.params)
    manifest = SelectionManifest(
        base_rung=final_config.base_rung,
        chosen_candidate=winner.name,
        chosen_params=winner.params,
        calibration_method=chosen_method,
        calibration_fitted_on=final_config.calibration.fitted_on,
        calibration_folds=final_config.calibration.n_folds,
        threshold=derivation.threshold,
        threshold_source=(
            f"derived from cost inputs: C_fp={derivation.cost_false_positive_inr:.2f}, "
            f"S_tp={derivation.saving_true_positive_inr:.2f}"
        ),
        cost_profile=profile_name,
        model_version=model_version,
        generator_version=metadata.generator_version,
        dataset_run_id=metadata.dataset_run_id,
        feature_version=metadata.feature_version,
        feature_fingerprint=metadata.feature_fingerprint,
        config_fingerprint=metadata.config_fingerprint,
        seed=seed,
        candidates=candidates,
        calibration_candidates=calibration_candidates,
        feature_names=tuple(train.x.columns),
        families_used=metadata.families_used,
        train_summary=split_summary(train),
        validation_summary=split_summary(validation),
        guardrails={
            "max_acceptable_flag_rate": final_config.guardrails.max_acceptable_flag_rate,
            "min_pr_auc_over_base_rate": final_config.guardrails.min_pr_auc_over_base_rate,
        },
        notes=(
            "Selected on validation. The test split was not read during selection, "
            "calibration or threshold derivation."
        ),
    )

    card = ModelCard(
        model_name=model.name,
        model_version=model_version,
        rung_id=model.rung_id,
        trained_at=datetime.now(UTC),
        training_rows=train.n_rows,
        feature_names=tuple(train.x.columns),
        enabled_families=metadata.families_used,
        random_seed=seed,
        config_fingerprint=metadata.config_fingerprint,
        feature_fingerprint=metadata.feature_fingerprint,
        feature_version=metadata.feature_version,
        dataset_run_id=metadata.dataset_run_id,
        generator_version=metadata.generator_version,
        calibration_method=chosen_method,
        calibration_fitted_on=final_config.calibration.fitted_on,
        hyperparameters=winner.params,
        notes=(
            f"Phase 5 final model. Base rung {final_config.base_rung} candidate "
            f"{winner.name!r}, calibrated with {chosen_method!r} fitted on "
            f"{final_config.calibration.fitted_on}. Manifest {manifest.manifest_id}."
        ),
    )

    artifact_path: Path | None = None
    if artifact_root is not None:
        artifact_path = model.save(artifact_dir(artifact_root, model.name, model_version), card)

    return FinalModel(
        model=model,
        manifest=manifest,
        card=card,
        artifact_path=artifact_path,
        validation_scores=calibrated_scores,
        validation_scores_raw=raw_scores,
    )


# ---------------------------------------------------------------------------
# step 5: evaluation, including the one test-set read
# ---------------------------------------------------------------------------


def evaluate_final_model(
    model: CalibratedModel,
    view: SplitView,
    *,
    manifest: SelectionManifest,
    cost_inputs: CostInputs,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    unseal_reason: str | None = None,
) -> tuple[FinalEvaluation, np.ndarray, np.ndarray]:
    """Score one split and compute every reported metric.

    Returns the evaluation plus the calibrated and raw score arrays, so plots are
    drawn from exactly the numbers the metrics were computed from.

    The threshold comes from the manifest, not from this split's labels. That is
    what makes a test-set evaluation a measurement rather than a fitting step.
    """
    if view.name == "test" and not unseal_reason:
        msg = "a test-set evaluation requires the written reason the seal was broken"
        raise ValueError(msg)

    y_true = view.y.to_numpy(dtype=bool)
    calibrated = model.predict_proba(view.x, view.context)
    raw = model.predict_raw(view.x, view.context)
    seed = manifest.seed

    ranking = ranking_metrics(
        y_true, calibrated, bootstrap_iterations=bootstrap_iterations, seed=seed
    )
    thresholds = [threshold_metrics_at(y_true, calibrated, manifest.threshold, "cost-derived")]
    thresholds += [
        threshold_metrics_at(y_true, calibrated, value, "sweep") for value in SWEEP_THRESHOLDS
    ]

    evaluation = FinalEvaluation(
        manifest_id=manifest.manifest_id,
        model_name=model.name,
        model_version=manifest.model_version,
        evaluated_split=view.name,
        is_calibrated=model.calibration_method != "none",
        calibration_method=model.calibration_method,
        generator_version=manifest.generator_version,
        dataset_run_id=manifest.dataset_run_id,
        feature_version=manifest.feature_version,
        feature_fingerprint=manifest.feature_fingerprint,
        config_fingerprint=manifest.config_fingerprint,
        seed=seed,
        ranking=ranking,
        calibration=calibration_metrics(y_true, calibrated),
        uncalibrated_calibration=calibration_metrics(y_true, raw),
        uncalibrated_pr_auc=float(pr_auc(y_true, raw)),
        threshold_metrics=thresholds,
        economics=economic_result(
            y_true,
            calibrated,
            threshold=manifest.threshold,
            cost_inputs=cost_inputs,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        ),
        evaluation_summary=split_summary(view),
        unseal_reason=unseal_reason,
    )
    return evaluation, calibrated, raw


def scores_frame(view: SplitView, calibrated: np.ndarray, raw: np.ndarray) -> pd.DataFrame:
    """Per-order calibrated and raw scores, for plotting and later analysis."""
    return pd.DataFrame(
        {
            "order_id": view.order_ids.to_numpy(),
            "ordered_at": view.ordered_at.to_numpy(),
            "label": view.y.to_numpy(dtype=int),
            "score_calibrated": calibrated,
            "score_raw": raw,
        }
    )


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

#: Where the frozen manifest and the evaluations live, under the artefact root.
FINAL_DIR = "final"


def final_dir(artifact_root: Path, dataset_run_id: str) -> Path:
    return artifact_root / FINAL_DIR / dataset_run_id


def save_manifest(manifest: SelectionManifest, artifact_root: Path) -> Path:
    """Freeze the decisions. The test-set command refuses to run without this."""
    directory = final_dir(artifact_root, manifest.dataset_run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "selection_manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_manifest(artifact_root: Path, dataset_run_id: str) -> SelectionManifest:
    """Read the frozen manifest, or say plainly that nothing was frozen."""
    path = final_dir(artifact_root, dataset_run_id) / "selection_manifest.json"
    if not path.is_file():
        msg = (
            f"no frozen selection manifest at {path}. The test split is opened only "
            "after model selection, calibration and threshold methodology are frozen. "
            "Run `rto-sentinel final` first."
        )
        raise FileNotFoundError(msg)
    return SelectionManifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_evaluation(evaluation: FinalEvaluation, artifact_root: Path) -> Path:
    """Write one split's metrics as JSON."""
    directory = final_dir(artifact_root, evaluation.dataset_run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"metrics__{evaluation.evaluated_split}.json"
    path.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_evaluation(artifact_root: Path, dataset_run_id: str, split: str) -> FinalEvaluation:
    path = final_dir(artifact_root, dataset_run_id) / f"metrics__{split}.json"
    if not path.is_file():
        msg = f"no {split} evaluation at {path}"
        raise FileNotFoundError(msg)
    return FinalEvaluation.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class ScoredBook:
    """A book of calibrated scores, ready to be priced by the decision layer.

    Loaded from the per-order scores the Phase 5 evaluation wrote. Carrying the
    split name rather than assuming it is what lets the decision layer refuse to
    simulate against the sealed set: the refusal is a property of the data, not
    of whoever remembered to pass the right flag.
    """

    probabilities: np.ndarray
    labels: np.ndarray | None
    split: str
    dataset_run_id: str
    model_version: str
    n_orders: int


def load_scored_book(
    artifact_root: Path, *, split: str = "validation", dataset_run_id: str | None = None
) -> ScoredBook:
    """Read the calibrated scores the final-model evaluation wrote.

    Raises ``FileNotFoundError`` when no such book exists. The API turns that
    into an explicit "no model loaded" response rather than inventing scores -
    a system that cannot score says so.
    """
    root = artifact_root / FINAL_DIR
    if dataset_run_id is None:
        runs = sorted(root.glob("*/selection_manifest.json")) if root.is_dir() else []
        if not runs:
            msg = (
                f"no final-model run under {root}. Run `rto-sentinel final` to produce a "
                "calibrated model and its scored book."
            )
            raise FileNotFoundError(msg)
        dataset_run_id = max(runs, key=lambda path: path.stat().st_mtime).parent.name

    path = final_dir(artifact_root, dataset_run_id) / f"scores__{split}.parquet"
    if not path.is_file():
        msg = f"no scored book for split {split!r} at {path}"
        raise FileNotFoundError(msg)

    frame = pd.read_parquet(path)
    manifest = load_manifest(artifact_root, dataset_run_id)
    labels = frame["label"].to_numpy(dtype=bool) if "label" in frame.columns else None
    return ScoredBook(
        probabilities=frame["score_calibrated"].to_numpy(dtype="float64"),
        labels=labels,
        split=split,
        dataset_run_id=dataset_run_id,
        model_version=manifest.model_version,
        n_orders=len(frame),
    )
