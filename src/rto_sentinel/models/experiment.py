"""Training and evaluating the ladder, with everything recorded.

THE RULE THIS MODULE EXISTS TO ENFORCE
======================================
Every rung is trained on the same training split, scored on the same evaluation
split, at the same threshold, under the same cost inputs. SPEC section 05: "Every
rung is evaluated identically on the same sealed test set. If a simpler rung
wins, it ships."

That only means something if the harness cannot tell the rungs apart, so it
cannot. It calls ``fit`` and ``predict_proba`` through the same interface for a
constant predictor and a gradient-boosted ensemble, and computes the same metrics
from the returned arrays.

WHERE THE THRESHOLD COMES FROM
==============================
Not 0.5, and not tuned on the evaluation labels. It is **derived** from the
merchant's cost inputs by ``decision.threshold.derive_threshold`` - a function of
economics alone that never sees a label. That is why the same operating point can
be published before the sealed test run without contaminating it.

The heuristic rungs score 1.0/0.0, so any threshold in ``(0, 1]`` produces the
same flags for them. That is a property of a binary policy, and it means the
comparison is fair: the derived threshold neither helps nor hurts them.

WHAT IS NOT HERE
================
No calibration. No decisions. No model selection. Phase 4 measures the ladder and
writes down what it found; choosing a winner on the sealed test set is Phase 6,
and doing it here would be tuning on validation and calling it a result.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rto_sentinel.contracts.experiment import (
    ExperimentRecord,
    LadderResults,
    SplitSummary,
    ThresholdMetrics,
)
from rto_sentinel.contracts.risk import ModelCard
from rto_sentinel.decision.threshold import derive_threshold
from rto_sentinel.eval.economics import economic_result
from rto_sentinel.eval.metrics import (
    calibration_metrics,
    confusion_at_threshold,
    pr_auc,
    ranking_metrics,
)
from rto_sentinel.models.artifacts import artifact_dir, write_json
from rto_sentinel.models.registry import resolve_rung

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import CostModelConfig, LadderConfig
    from rto_sentinel.contracts.decision import CostInputs
    from rto_sentinel.features.dataset import ModelingDataset, SplitView
    from rto_sentinel.models.base import RiskModel

#: Bumped when the training or evaluation procedure changes in a way that makes
#: old and new records incomparable. Distinct from a model's own version, which
#: identifies one artefact.
EXPERIMENT_VERSION = "1.0.0"

#: Extra operating points reported alongside the cost-derived one, so a reader
#: can see how each rung behaves across the range rather than at a single point.
SWEEP_THRESHOLDS = (0.1, 0.2, 0.3, 0.5)

DEFAULT_BOOTSTRAP_ITERATIONS = 500


@dataclass(frozen=True, slots=True)
class TrainedRung:
    """A fitted model with its card and the scores it produced."""

    model: RiskModel
    card: ModelCard
    record: ExperimentRecord
    scores: np.ndarray
    artifact_path: Path | None


def split_summary(view: SplitView) -> SplitSummary:
    first, last = view.date_range
    first_day, last_day = view.day_range
    return SplitSummary(
        name=view.name,
        n_rows=view.n_rows,
        n_positives=int(view.y.sum()),
        positive_rate=view.positive_rate,
        n_customers=int(view.customer_hashes.nunique()),
        first_day=first_day,
        last_day=last_day,
        first_ordered_at=first.to_pydatetime(),
        last_ordered_at=last.to_pydatetime(),
    )


def _nullable(value: float) -> float | None:
    """NaN becomes None, so the record serialises to valid JSON.

    A NaN precision means "nothing was flagged, so precision is undefined". JSON
    has no NaN, and writing 0.0 would turn "undefined" into "measured and
    terrible" - which is a different and wrong claim about rung 0.
    """
    return None if not np.isfinite(value) else float(value)


def threshold_metrics_at(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, source: str
) -> ThresholdMetrics:
    matrix = confusion_at_threshold(y_true, y_prob, threshold)
    return ThresholdMetrics(
        threshold=float(threshold),
        threshold_source=source,
        true_positives=matrix.true_positives,
        false_positives=matrix.false_positives,
        false_negatives=matrix.false_negatives,
        true_negatives=matrix.true_negatives,
        flag_rate=matrix.flag_rate,
        precision=_nullable(matrix.precision),
        recall=_nullable(matrix.recall),
        f1=_nullable(matrix.f1),
    )


def _experiment_id(model_name: str, dataset_run_id: str, seed: int, split: str) -> str:
    payload = f"{EXPERIMENT_VERSION}|{model_name}|{dataset_run_id}|{seed}|{split}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _model_version(
    model_name: str, dataset_run_id: str, seed: int, hyperparameters: dict[str, Any]
) -> str:
    """Deterministic version for one (model, data, seed, hyperparameter) combination.

    Deterministic on purpose: retraining the same thing overwrites its artefact
    rather than accumulating a second near-identical copy that a later comparison
    might silently pick between.
    """
    payload = f"{model_name}|{dataset_run_id}|{seed}|{sorted(hyperparameters.items())!s}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def train_rung(
    rung_name: str,
    dataset: ModelingDataset,
    *,
    ladder_config: LadderConfig,
    cost_inputs: CostInputs,
    threshold: float,
    seed: int,
    evaluation_split: str = "validation",
    artifact_root: Path | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> TrainedRung:
    """Train one rung and evaluate it. Nothing here is rung-specific."""
    rung = next((r for r in ladder_config.rungs if r.name == rung_name), None)
    if rung is None:
        msg = f"{rung_name!r} is not in config/models/ladder.yaml"
        raise KeyError(msg)

    train = dataset.train
    if evaluation_split == "validation":
        evaluation = dataset.validation
    elif evaluation_split == "test":
        # Deliberately explicit. Reaching the test split requires the caller to
        # have unsealed it with a written reason, which is recorded on the
        # dataset and appears in any traceback.
        evaluation = dataset.test
    else:
        msg = f"unknown evaluation split {evaluation_split!r}"
        raise ValueError(msg)

    params = {**dict(rung.params), "random_state": seed}
    model_class = resolve_rung(rung_name)
    model = model_class(params)

    started = time.perf_counter()
    model.fit(train.x, train.y, train.context)
    duration = time.perf_counter() - started

    scores = model.predict_proba(evaluation.x, evaluation.context)
    y_true = evaluation.y.to_numpy(dtype=bool)

    # Scored on the training split too. Not to report as a result - it is not one
    # - but so the train/evaluation gap travels with every record and a memorising
    # model is visible in the artefact rather than only to whoever thought to look.
    train_scores = model.predict_proba(train.x, train.context)
    train_pr = pr_auc(train.y.to_numpy(dtype=bool), train_scores)

    ranking = ranking_metrics(y_true, scores, bootstrap_iterations=bootstrap_iterations, seed=seed)
    calibration = calibration_metrics(y_true, scores)
    economics = economic_result(
        y_true,
        scores,
        threshold=threshold,
        cost_inputs=cost_inputs,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )

    thresholds = [threshold_metrics_at(y_true, scores, threshold, "cost-derived")]
    thresholds += [
        threshold_metrics_at(y_true, scores, value, "sweep") for value in SWEEP_THRESHOLDS
    ]

    metadata = dataset.metadata
    model_version = _model_version(rung_name, metadata.dataset_run_id, seed, params)
    trained_at = datetime.now(UTC)

    card = ModelCard(
        model_name=rung_name,
        model_version=model_version,
        rung_id=rung.id,
        trained_at=trained_at,
        training_rows=train.n_rows,
        feature_names=tuple(train.x.columns),
        enabled_families=metadata.families_used,
        random_seed=seed,
        config_fingerprint=metadata.config_fingerprint,
        feature_fingerprint=metadata.feature_fingerprint,
        feature_version=metadata.feature_version,
        dataset_run_id=metadata.dataset_run_id,
        generator_version=metadata.generator_version,
        # None through Phase 4. This is what stops the decision engine accepting
        # these scores; see the note in rung4_lightgbm.
        calibration_method=None,
        calibration_fitted_on=None,
        hyperparameters=params,
        notes=f"Phase 4 baseline ladder, evaluated on {evaluation_split}. Uncalibrated.",
    )

    record = ExperimentRecord(
        experiment_id=_experiment_id(rung_name, metadata.dataset_run_id, seed, evaluation_split),
        model_name=rung_name,
        model_version=model_version,
        rung_id=rung.id,
        rung_kind=rung.kind,
        generator_version=metadata.generator_version,
        dataset_run_id=metadata.dataset_run_id,
        feature_version=metadata.feature_version,
        feature_fingerprint=metadata.feature_fingerprint,
        config_fingerprint=metadata.config_fingerprint,
        seed=seed,
        hyperparameters=params,
        families_used=metadata.families_used,
        n_features=len(train.x.columns),
        trained_at=trained_at,
        train_duration_seconds=duration,
        evaluated_split=evaluation_split,
        split_strategy=metadata.split_strategy,
        split_pool_shares=metadata.split_pool_shares,
        train_summary=split_summary(train),
        evaluation_summary=split_summary(evaluation),
        ranking=ranking,
        calibration=calibration,
        threshold_metrics=thresholds,
        economics=economics,
        train_pr_auc=float(train_pr) if np.isfinite(train_pr) else None,
        is_calibrated=False,
        notes=rung.description.strip(),
    )

    artifact_path: Path | None = None
    if artifact_root is not None:
        artifact_path = model.save(artifact_dir(artifact_root, rung_name, model_version), card)

    return TrainedRung(
        model=model, card=card, record=record, scores=scores, artifact_path=artifact_path
    )


def run_ladder(
    dataset: ModelingDataset,
    *,
    ladder_config: LadderConfig,
    cost_config: CostModelConfig,
    seed: int,
    evaluation_split: str = "validation",
    artifact_root: Path | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    cost_profile: str | None = None,
) -> tuple[LadderResults, dict[str, TrainedRung]]:
    """Train and evaluate every enabled rung under identical conditions."""
    profile_name = cost_profile or cost_config.default_profile
    profile = cost_config.profiles[profile_name]
    cost_inputs = cost_inputs_from_profile(profile)

    derivation = derive_threshold(cost_inputs)
    threshold = derivation.threshold

    trained: dict[str, TrainedRung] = {}
    for rung in ladder_config.rungs:
        if not rung.enabled:
            continue
        trained[rung.name] = train_rung(
            rung.name,
            dataset,
            ladder_config=ladder_config,
            cost_inputs=cost_inputs,
            threshold=threshold,
            seed=seed,
            evaluation_split=evaluation_split,
            artifact_root=artifact_root,
            bootstrap_iterations=bootstrap_iterations,
        )

    metadata = dataset.metadata
    results = LadderResults(
        evaluated_split=evaluation_split,
        dataset_run_id=metadata.dataset_run_id,
        config_fingerprint=metadata.config_fingerprint,
        feature_fingerprint=metadata.feature_fingerprint,
        seed=seed,
        cost_profile=profile_name,
        threshold=threshold,
        threshold_source=(
            f"derived from cost inputs: C_fp={derivation.cost_false_positive_inr:.2f}, "
            f"S_tp={derivation.saving_true_positive_inr:.2f}"
        ),
        records=[run.record for run in trained.values()],
    )
    return results, trained


def cost_inputs_from_profile(profile: Any) -> CostInputs:
    from rto_sentinel.contracts.decision import CostInputs

    return CostInputs(
        rto_cost_inr=profile.rto_cost_inr,
        contribution_margin_inr=profile.contribution_margin_inr,
        abandonment_on_friction=profile.abandonment_on_friction,
        intervention_success_rate=profile.intervention_success_rate,
        friction_support_cost_inr=profile.friction_support_cost_inr,
    )


def save_results(results: LadderResults, artifact_root: Path) -> Path:
    """Write the machine-readable ladder results.

    One JSON file per run, plus a per-experiment file so a single rung can be
    inspected without parsing the whole ladder. Nothing in this project reads a
    metric from source; this file is where they live.
    """
    directory = artifact_root / "experiments" / results.dataset_run_id
    directory.mkdir(parents=True, exist_ok=True)

    target = directory / f"ladder__{results.evaluated_split}__seed{results.seed}.json"
    target.write_text(results.model_dump_json(indent=2), encoding="utf-8")

    for record in results.records:
        write_json(
            directory / f"{record.model_name}__{record.evaluated_split}.json",
            record.model_dump(mode="json"),
        )
    return target


def scores_frame(
    dataset: ModelingDataset, trained: dict[str, TrainedRung], split: str
) -> pd.DataFrame:
    """Per-order scores from every rung, for plotting and further analysis.

    Written out so the plots and any later analysis work from the same numbers the
    metrics were computed on, rather than recomputing and risking drift.
    """
    view = dataset.validation if split == "validation" else dataset.test
    frame = pd.DataFrame(
        {
            "order_id": view.order_ids.to_numpy(),
            "label": view.y.to_numpy(dtype=int),
        }
    )
    for name, run in trained.items():
        frame[f"score__{name}"] = run.scores
    return frame
