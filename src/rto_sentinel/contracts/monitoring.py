"""Drift and distribution-shift contracts.

THE RULE THESE TYPES EXIST TO HOLD
==================================
**Drift is not failure.** A shifted input distribution is a fact about the world;
whether the model got worse is a separate question that needs labels to answer.
The two are kept structurally apart here: :class:`DriftSignal` describes a
distribution that moved, and it has no field in which to record a verdict.
:class:`PerformanceDelta` records a measured change in model quality, and it can
only be constructed where mature labels exist.

A monitoring system that conflates them cries wolf every festive season, when COD
share and order values move for entirely ordinary reasons. The console shows both
and says which is which.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: How a drift signal should be read. Deliberately not "pass" or "fail".
DriftSeverity = Literal["stable", "watch", "investigate"]

#: What kind of quantity moved. Feature and prediction drift need no labels;
#: outcome and calibration drift do, and are only computed on mature rows.
DriftKind = Literal["feature", "prediction", "outcome_rate", "flag_rate", "calibration"]


class WindowSummary(BaseModel):
    """One observation period, described well enough to be checked."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(max_length=64, description="e.g. baseline, current")
    n_orders: int = Field(ge=0)
    start: datetime | None = None
    end: datetime | None = None

    #: Rows whose outcome is known. Everything label-dependent is computed over
    #: these only, and the gap between this and ``n_orders`` is the reason a
    #: recent window can look artificially clean.
    n_matured: int = Field(default=0, ge=0)

    @property
    def maturity_rate(self) -> float | None:
        if self.n_orders == 0:
            return None
        return self.n_matured / self.n_orders


class DriftSignal(BaseModel):
    """One quantity, measured in two windows, with the distance between them.

    ``severity`` is a reading of the distance against configured bands. It says
    how much the distribution moved and nothing about whether that is bad.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(max_length=128, description="Feature name, or the quantity name")
    kind: DriftKind
    statistic: str = Field(max_length=32, description="psi, ks, or absolute_difference")
    distance: float = Field(ge=0.0)
    severity: DriftSeverity

    baseline_value: float | None = Field(default=None, description="Mean or rate in baseline")
    current_value: float | None = Field(default=None, description="Mean or rate in current")
    baseline_n: int = Field(default=0, ge=0)
    current_n: int = Field(default=0, ge=0)

    #: False when the windows are too thin for the distance to mean anything. A
    #: PSI computed on eleven rows is noise with a number attached.
    sufficient: bool = True
    note: str = Field(default="", max_length=512)


class PerformanceDelta(BaseModel):
    """A measured change in model quality between two windows.

    Separate from :class:`DriftSignal` on purpose. This requires labels, so it
    cannot be produced for a window whose orders have not matured - and where it
    cannot be produced, the monitoring report says the question is unanswered
    rather than reporting drift as though it were degradation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(max_length=64)
    baseline: float
    current: float
    delta: float
    n_baseline_matured: int = Field(ge=0)
    n_current_matured: int = Field(ge=0)
    sufficient: bool = True
    note: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def _delta_is_consistent(self) -> PerformanceDelta:
        if abs((self.current - self.baseline) - self.delta) > 1e-9:
            msg = (
                f"delta {self.delta} does not equal current {self.current} minus "
                f"baseline {self.baseline}"
            )
            raise ValueError(msg)
        return self


class DriftReport(BaseModel):
    """Baseline versus current, with warnings a human can act on.

    ``warnings`` are sentences, not codes. A monitoring page that prints
    ``PSI=0.27`` has told an operations manager nothing; one that prints "COD
    share rose from 61% to 74%, which moves the input distribution but does not
    by itself mean the model is worse" has told them what to do next.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime
    baseline: WindowSummary
    current: WindowSummary

    signals: tuple[DriftSignal, ...] = ()
    performance: tuple[PerformanceDelta, ...] = ()
    warnings: tuple[str, ...] = ()

    model_version: str = Field(default="", max_length=64)
    feature_version: str = Field(default="", max_length=32)

    #: True when at least one label-dependent comparison could be made. The
    #: console shows the negative case prominently: an all-green drift page
    #: computed with no labels is the most misleading thing this system could
    #: display.
    labels_available: bool = True

    data_provenance: str = Field(
        default=(
            "Synthetic benchmark data. Drift measured between two windows of the same "
            "simulated book; not evidence about production drift behaviour."
        )
    )

    @property
    def worst_severity(self) -> DriftSeverity:
        order: dict[str, int] = {"stable": 0, "watch": 1, "investigate": 2}
        if not self.signals:
            return "stable"
        worst = max(self.signals, key=lambda signal: order[signal.severity])
        return worst.severity


class EnvironmentSpec(BaseModel):
    """A named generator configuration for a distribution-shift experiment.

    ``overrides`` are dotted paths into ``config/generator.yaml``. Storing the
    overrides rather than a whole copied config is what makes an environment
    reviewable: the diff from the reference world is the experiment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(max_length=64)
    description: str = Field(max_length=512)
    overrides: dict[str, float] = Field(default_factory=dict)
    seed: int
    n_orders: int = Field(gt=0)

    @model_validator(mode="after")
    def _reference_has_no_overrides(self) -> EnvironmentSpec:
        if self.name == "reference" and self.overrides:
            msg = "the reference environment defines the unshifted world and takes no overrides"
            raise ValueError(msg)
        return self


class ShiftResult(BaseModel):
    """How the frozen model performed in one shifted environment.

    Everything here is measured with the *same* model artefact and the *same*
    threshold as the reference environment. Retraining per environment would
    measure something else entirely - how well the pipeline adapts - and would
    say nothing about what happens to a deployed model when the world moves.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment: str = Field(max_length=64)
    description: str = Field(default="", max_length=512)
    n_orders: int = Field(ge=0)

    observed_rto_rate: float = Field(ge=0.0, le=1.0)
    pr_auc: float
    roc_auc: float
    brier_score: float = Field(ge=0.0)
    expected_calibration_error: float = Field(ge=0.0)

    threshold: float = Field(ge=0.0, le=1.0)
    flag_rate: float = Field(ge=0.0, le=1.0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    net_inr_per_1000: float

    #: Difference from the reference environment. None on the reference itself.
    pr_auc_delta: float | None = None
    net_delta: float | None = None
    ece_delta: float | None = None


class ShiftStudy(BaseModel):
    """A complete controlled robustness experiment.

    The reference environment is not an IID resample presented as robustness. It
    is the control against which each deliberately perturbed world is compared,
    and every perturbation is a named parameter change recorded in
    :class:`EnvironmentSpec`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime
    model_version: str = Field(max_length=64)
    feature_version: str = Field(default="", max_length=32)
    generator_version: str = Field(default="", max_length=32)
    threshold: float = Field(ge=0.0, le=1.0)

    environments: tuple[EnvironmentSpec, ...]
    results: tuple[ShiftResult, ...]
    findings: tuple[str, ...] = ()

    data_provenance: str = Field(
        default=(
            "Controlled benchmark experiment. Each environment is a documented parameter "
            "change to the synthetic generator, not an observation of real distribution "
            "shift. Degradation measured here describes this simulator."
        )
    )

    @model_validator(mode="after")
    def _reference_is_present(self) -> ShiftStudy:
        names = {result.environment for result in self.results}
        if self.results and "reference" not in names:
            msg = (
                "a shift study without its reference environment has no control; "
                "degradation would be measured against nothing"
            )
            raise ValueError(msg)
        return self
