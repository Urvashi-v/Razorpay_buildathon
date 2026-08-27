"""Evaluation contracts - the shape of an honest result.

These types make the reporting rules from SPEC section 07 structural. Note that
:class:`EconomicResult` keeps ``total_false_positive_cost_inr`` as a required,
separate field: it can be displayed alongside the net figure but it can never be
quietly netted away inside it, because the type has nowhere to hide it.

Likewise :class:`PointEstimate` always carries an interval. A point estimate on
5,000 rows is not a result.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rto_sentinel.contracts.enums import DatasetSplit


class PointEstimate(BaseModel):
    """A statistic with its bootstrap interval attached, always."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float
    ci_low: float
    ci_high: float
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    n_bootstrap: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _interval_contains_value(self) -> PointEstimate:
        if not (self.ci_low <= self.value <= self.ci_high):
            msg = f"value {self.value} lies outside its interval [{self.ci_low}, {self.ci_high}]"
            raise ValueError(msg)
        return self


class RankingMetrics(BaseModel):
    """Ranking quality. PR-AUC leads; ROC-AUC is reported but not led with."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pr_auc: PointEstimate
    roc_auc: PointEstimate
    recall_at_precision_80: float | None = Field(default=None, ge=0.0, le=1.0)
    recall_at_precision_90: float | None = Field(default=None, ge=0.0, le=1.0)
    precision_at_k: dict[str, float] = Field(default_factory=dict)


class CalibrationMetrics(BaseModel):
    """Calibration is a headline metric here, not a footnote.

    ``reliability_bins`` holds (mean predicted, observed frequency, count) per
    bin so the console can draw the diagram without recomputing anything.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_calibration_error: float = Field(ge=0.0)
    brier_score: float = Field(ge=0.0)
    n_bins: int = Field(gt=1)
    reliability_bins: tuple[tuple[float, float, int], ...] = ()


class EconomicResult(BaseModel):
    """The headline. Costs are reported separately and never netted away."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(ge=0.0, le=1.0)
    flag_rate: float = Field(ge=0.0, le=1.0, description="Reported always")

    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)

    gross_saving_inr: float
    total_false_positive_cost_inr: float = Field(
        ge=0, description="Stated separately, never folded into the net figure"
    )
    residual_false_negative_loss_inr: float = Field(ge=0)

    net_inr_saved_per_1000_orders: PointEstimate
    baseline_net_inr_per_1000_orders: float = Field(
        description="Rung 0 (do nothing): the loss the merchant currently absorbs"
    )

    @property
    def n_orders(self) -> int:
        return (
            self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        )


class CohortResult(BaseModel):
    """One slice of the cohort or fairness breakdown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort: str = Field(max_length=64, description="e.g. pincode_tier")
    group: str = Field(max_length=64, description="e.g. tier_3")
    n_orders: int = Field(ge=0)
    flag_rate: float = Field(ge=0.0, le=1.0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    net_inr_per_1000: float | None = None


class FairnessAudit(BaseModel):
    """Disparate-impact review across pincode tier and order-value band.

    ``triggered`` is True when a group is flagged materially more often *and*
    with materially worse precision. That is the condition under which the
    geography features get pulled back - reported either way.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    slices: tuple[CohortResult, ...]
    max_flag_rate_ratio: float = Field(ge=0.0)
    worst_precision_drop: float
    triggered: bool
    narrative: str = Field(default="", max_length=4000)


class EvaluationReport(BaseModel):
    """The complete result for one rung on one split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    model_version: str
    rung_id: int
    split: DatasetSplit
    n_orders: int = Field(ge=0)
    evaluated_at: datetime
    config_fingerprint: str

    ranking: RankingMetrics
    calibration: CalibrationMetrics
    economics: EconomicResult
    cohorts: tuple[CohortResult, ...] = ()
    fairness: FairnessAudit | None = None

    # Stated on every report, in the report itself, not only in the README.
    data_provenance: str = Field(
        default=(
            "Synthetic data from config/generator.yaml. Absolute metric values are not a "
            "claim about production performance."
        )
    )
