"""The final model's selection record and its evaluations.

WHY A "MANIFEST" AND NOT JUST A RESULT
======================================
The sealed test set is only meaningful if the thing being tested was decided
before it was opened. A claim to that effect is worth nothing after the fact -
anyone can say the choices were frozen first. So the choices are written to disk
as a :class:`SelectionManifest`, with a content hash over the decisions
themselves, and the test-set command refuses to run without one. The hash in the
test evaluation is what ties the numbers to a specific frozen decision rather
than to a story about one.

What the manifest freezes:

* which base rung, and which of its hyperparameter candidates won;
* which calibration method won, over how many folds, fitted on which split;
* the threshold *methodology* - a function of merchant economics, not of labels;
* the six versions that identify the data and the code that produced it.

The manifest also carries the losing candidates. A selection record showing only
the winner is a advertisement; showing the field is what lets someone check that
the winner won by a margin worth having.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rto_sentinel.contracts.evaluation import (
    CalibrationMetrics,
    EconomicResult,
    RankingMetrics,
)
from rto_sentinel.contracts.experiment import SplitSummary, ThresholdMetrics

#: Bumped when the selection procedure changes in a way that makes an old
#: manifest incomparable to a new one.
SELECTION_VERSION = "1.0.0"

_PROVENANCE = (
    "Synthetic benchmark data. Labels are simulated outcomes of the documented "
    "process in docs/simulator.md, not real-world ground truth. Absolute metric "
    "values are not a claim about production performance."
)


class CandidateResult(BaseModel):
    """One hyperparameter candidate, measured on validation.

    ``overfit_gap`` is carried because the Phase 4 finding that motivated this
    search was a 0.52 train-versus-validation gap, and a search that reported
    only validation scores could re-select the same failure without showing it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    note: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    train_pr_auc: float | None = None
    validation_pr_auc: float
    validation_pr_auc_ci: tuple[float, float] | None = None
    train_duration_seconds: float = Field(ge=0)
    selected: bool = False

    @property
    def overfit_gap(self) -> float | None:
        if self.train_pr_auc is None:
            return None
        return self.train_pr_auc - self.validation_pr_auc


class CalibrationCandidate(BaseModel):
    """One calibration method, measured out-of-fold on the validation split.

    ``improvement_over_none`` is the reduction in expected calibration error
    relative to leaving the scores alone. Negative means the method made
    calibration worse, which is a real outcome and is reported as such.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    expected_calibration_error: float
    brier_score: float
    improvement_over_none: float
    n_folds: int = Field(ge=2)
    fitted_on: str
    selected: bool = False
    reliability_bins: tuple[tuple[float, float, int], ...] = ()


class SelectionManifest(BaseModel):
    """The frozen record of every choice made before the test set was opened."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selection_version: str = SELECTION_VERSION
    frozen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # --- what was chosen --------------------------------------------------
    base_rung: str
    chosen_candidate: str
    chosen_params: dict[str, Any]
    calibration_method: str
    calibration_fitted_on: str
    calibration_folds: int = Field(ge=2)

    # --- how the operating point is derived -------------------------------
    threshold: float = Field(ge=0.0, le=1.0)
    threshold_source: str
    cost_profile: str

    # --- provenance: the six versions -------------------------------------
    model_version: str
    generator_version: str
    dataset_run_id: str
    feature_version: str
    feature_fingerprint: str
    config_fingerprint: str
    seed: int

    # --- the field, not just the winner -----------------------------------
    candidates: list[CandidateResult]
    calibration_candidates: list[CalibrationCandidate]

    # --- what it was fitted on --------------------------------------------
    feature_names: tuple[str, ...]
    families_used: tuple[str, ...]
    train_summary: SplitSummary
    validation_summary: SplitSummary

    guardrails: dict[str, float] = Field(default_factory=dict)
    notes: str = ""
    data_provenance: str = _PROVENANCE

    @property
    def manifest_id(self) -> str:
        """A hash over the decisions, so an edited manifest is a different one.

        Deliberately excludes ``frozen_at`` and the measured candidate metrics:
        the identity of a decision is *what was decided*, so re-running the
        selection and reaching the same choices reproduces the same id.
        """
        payload = json.dumps(
            {
                "selection_version": self.selection_version,
                "base_rung": self.base_rung,
                "chosen_candidate": self.chosen_candidate,
                "chosen_params": {k: str(v) for k, v in sorted(self.chosen_params.items())},
                "calibration_method": self.calibration_method,
                "calibration_fitted_on": self.calibration_fitted_on,
                "calibration_folds": self.calibration_folds,
                "threshold": round(self.threshold, 10),
                "dataset_run_id": self.dataset_run_id,
                "feature_fingerprint": self.feature_fingerprint,
                "config_fingerprint": self.config_fingerprint,
                "seed": self.seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    @property
    def winner(self) -> CandidateResult:
        for candidate in self.candidates:
            if candidate.name == self.chosen_candidate:
                return candidate
        msg = f"manifest names {self.chosen_candidate!r} but does not contain it"
        raise KeyError(msg)


class FinalEvaluation(BaseModel):
    """The final model measured on one split. Written once per split.

    ``uncalibrated_calibration`` is the same diagnostic computed on the raw
    scores, kept beside the calibrated one so the reader can see what the
    calibration step actually bought rather than being told.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_id: str
    model_name: str
    model_version: str
    evaluated_split: str
    is_calibrated: bool
    calibration_method: str
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # --- provenance -------------------------------------------------------
    generator_version: str
    dataset_run_id: str
    feature_version: str
    feature_fingerprint: str
    config_fingerprint: str
    seed: int

    # --- measured ---------------------------------------------------------
    ranking: RankingMetrics
    calibration: CalibrationMetrics
    uncalibrated_calibration: CalibrationMetrics
    uncalibrated_pr_auc: float
    threshold_metrics: list[ThresholdMetrics]
    economics: EconomicResult
    evaluation_summary: SplitSummary

    #: Present only for the sealed split, and required there: opening it is a
    #: decision someone has to justify in writing.
    unseal_reason: str | None = None

    notes: str = ""
    data_provenance: str = _PROVENANCE

    @property
    def primary_metric(self) -> float:
        return self.ranking.pr_auc.value

    @property
    def operating_point(self) -> ThresholdMetrics:
        return self.threshold_metrics[0]

    @property
    def calibration_improvement(self) -> float:
        """ECE reduction against the same model's raw scores. Negative is worse."""
        return (
            self.uncalibrated_calibration.expected_calibration_error
            - self.calibration.expected_calibration_error
        )
