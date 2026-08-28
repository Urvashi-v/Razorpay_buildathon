"""Experiment records - the machine-readable account of one training run.

WHY THIS IS A TYPED CONTRACT AND NOT A DICT
===========================================
"What produced this number?" has to be answerable months later by someone who was
not there. A loose dictionary answers it only as long as everyone remembers to
populate the same keys, which is until the second person touches the code.

So an :class:`ExperimentRecord` names every input that could change a result, and
Pydantic refuses to build one with a field missing. Six versions travel together
because a result is reproducible only if all six are pinned:

======================  ====================================================
``generator_version``   which generative process made the raw data
``dataset_run_id``      which specific dataset, deterministically identified
``feature_version``     which feature-engineering code
``feature_fingerprint`` SHA-256 over the feature declarations themselves
``config_fingerprint``  SHA-256 over the YAML configuration bundle
``model_version``       which trained artefact
======================  ====================================================

Plus the seed, the hyperparameters as actually passed to the estimator, the
training timestamp, and the split configuration the model was evaluated against.

NOTHING HERE IS WRITTEN BY HAND
===============================
Every metric on this record is computed from real predictions against held-out
data by ``rto_sentinel.eval``. There is no code path that sets a metric from a
literal, which is the point: a number in a report is either measured or it does
not exist.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rto_sentinel.contracts.evaluation import (
    CalibrationMetrics,
    EconomicResult,
    RankingMetrics,
)


class SplitSummary(BaseModel):
    """Shape of one split, recorded so a result can be sanity-checked later."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    n_rows: int = Field(ge=0)
    n_positives: int = Field(ge=0)
    positive_rate: float
    n_customers: int = Field(ge=0)
    first_day: int
    last_day: int
    first_ordered_at: datetime
    last_ordered_at: datetime


class ThresholdMetrics(BaseModel):
    """Confusion-matrix metrics at one operating threshold.

    ``precision`` and ``f1`` are nullable because they are genuinely undefined
    when nothing is flagged - which is rung 0's situation at any threshold above
    the base rate. NaN would serialise to invalid JSON; null says "not defined",
    which is the truth and is different from zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(ge=0.0, le=1.0)
    threshold_source: str = Field(
        description="How this threshold was chosen: cost-derived, fixed, or a sweep point"
    )

    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)

    flag_rate: float = Field(ge=0.0, le=1.0)
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class ExperimentRecord(BaseModel):
    """One trained rung, evaluated on one split. The unit of the results table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- identity ---------------------------------------------------------
    experiment_id: str = Field(description="Deterministic id for this run")
    model_name: str
    model_version: str
    rung_id: int = Field(ge=0)
    rung_kind: str = Field(description="baseline, heuristic, or model")

    # --- provenance: the six versions ------------------------------------
    generator_version: str
    dataset_run_id: str
    feature_version: str
    feature_fingerprint: str
    config_fingerprint: str
    seed: int

    # --- how it was run ---------------------------------------------------
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    families_used: tuple[str, ...] = ()
    n_features: int = Field(ge=0)
    trained_at: datetime
    train_duration_seconds: float = Field(ge=0)

    # --- what it was evaluated against ------------------------------------
    evaluated_split: str = Field(description="Which split these metrics come from")
    split_strategy: str
    split_pool_shares: dict[str, float]
    train_summary: SplitSummary
    evaluation_summary: SplitSummary

    # --- results, all measured -------------------------------------------
    ranking: RankingMetrics
    calibration: CalibrationMetrics
    threshold_metrics: list[ThresholdMetrics] = Field(default_factory=list)
    economics: EconomicResult | None = None

    #: PR-AUC on the TRAINING split. Recorded so overfitting is visible in the
    #: artefact rather than only to whoever happened to check.
    #:
    #: This exists because rung 4 at its configured 600 trees scores 0.96 on
    #: train and 0.44 on validation - a gap of 0.52 - and a record showing only
    #: the validation number would present a badly overfitting model as simply a
    #: mediocre one. Those call for different responses.
    train_pr_auc: float | None = None

    # --- honesty ----------------------------------------------------------
    is_calibrated: bool = Field(
        default=False,
        description="False through Phase 4. Uncalibrated scores must not reach the engine.",
    )
    notes: str = ""
    data_provenance: str = Field(
        default=(
            "Synthetic benchmark data. Labels are simulated outcomes of the documented "
            "process in docs/simulator.md, not real-world ground truth. Absolute metric "
            "values are not a claim about production performance."
        )
    )
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def primary_metric(self) -> float:
        """PR-AUC. The ranking metric this project leads with."""
        return self.ranking.pr_auc.value

    @property
    def overfit_gap(self) -> float | None:
        """Train PR-AUC minus evaluation PR-AUC. Large means memorised.

        A negative gap is normal and not a problem - it usually means the
        evaluation split happens to be slightly easier than the training one.
        """
        if self.train_pr_auc is None or not self.ranking.pr_auc.is_defined:
            return None
        return self.train_pr_auc - self.ranking.pr_auc.value

    def headline(self) -> dict[str, Any]:
        """The row this run contributes to the comparison table."""
        at_operating = self.threshold_metrics[0] if self.threshold_metrics else None
        return {
            "rung": self.rung_id,
            "model": self.model_name,
            "pr_auc": self.ranking.pr_auc.value,
            "pr_auc_ci": (self.ranking.pr_auc.ci_low, self.ranking.pr_auc.ci_high),
            "roc_auc": self.ranking.roc_auc.value,
            "recall_at_p80": self.ranking.recall_at_precision_80,
            "ece": self.calibration.expected_calibration_error,
            "brier": self.calibration.brier_score,
            "flag_rate": at_operating.flag_rate if at_operating else None,
            "precision": at_operating.precision if at_operating else None,
            "recall": at_operating.recall if at_operating else None,
            "f1": at_operating.f1 if at_operating else None,
            "net_inr_per_1000": (
                self.economics.net_inr_saved_per_1000_orders.value if self.economics else None
            ),
            "fp_cost_inr": (
                self.economics.total_false_positive_cost_inr if self.economics else None
            ),
            "is_calibrated": self.is_calibrated,
            "train_pr_auc": self.train_pr_auc,
            "overfit_gap": self.overfit_gap,
        }


class LadderResults(BaseModel):
    """Every rung, evaluated on the same split under the same conditions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluated_split: str
    dataset_run_id: str
    config_fingerprint: str
    feature_fingerprint: str
    seed: int
    cost_profile: str
    threshold: float
    threshold_source: str
    records: list[ExperimentRecord]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def by_name(self, name: str) -> ExperimentRecord:
        for record in self.records:
            if record.model_name == name:
                return record
        msg = f"no record for {name!r}"
        raise KeyError(msg)

    @property
    def ordered(self) -> list[ExperimentRecord]:
        return sorted(self.records, key=lambda record: record.rung_id)
