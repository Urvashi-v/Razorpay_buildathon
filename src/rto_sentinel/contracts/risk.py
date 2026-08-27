"""Risk-model outputs.

THE ARCHITECTURAL RULE THIS FILE ENCODES: a :class:`RiskScore` is produced by a
trained model and by nothing else. It is a calibrated probability, and it carries
no action, no band and no rupee figure - converting it into a decision is the
sole job of ``rto_sentinel.decision``.

An LLM never constructs one of these. ``tests/architecture/test_layering.py``
asserts that no module under ``agents/`` imports this type for construction.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FeatureContribution(BaseModel):
    """One feature's SHAP contribution to a single prediction.

    ``value`` is the raw feature value; ``contribution`` is its signed push on
    the model's log-odds output. Both are facts computed from the model - the
    language layer is allowed to *phrase* them and nothing else.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str = Field(max_length=128)
    family: str = Field(max_length=64, description="Feature family from config/features.yaml")
    value: float | str | bool | None
    contribution: float = Field(description="Signed SHAP value, in log-odds space")

    @property
    def direction(self) -> str:
        return "increases_risk" if self.contribution > 0 else "decreases_risk"


class RiskScore(BaseModel):
    """A calibrated probability that this order becomes an RTO.

    ``probability`` is post-calibration. ``raw_score`` is retained for debugging
    and for the reliability diagram; it is never used for a decision, because an
    uncalibrated boosting output is not an honest probability and the entire
    expected-value layer depends on honesty here (SPEC section 05).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    order_id: str = Field(max_length=64)
    probability: float = Field(ge=0.0, le=1.0, description="Calibrated P(RTO)")
    raw_score: float | None = Field(default=None, description="Pre-calibration model output")

    model_name: str = Field(max_length=128, description="Ladder rung that produced this")
    model_version: str = Field(max_length=64)
    calibration_method: str | None = Field(default=None, max_length=32)

    scored_at: datetime
    latency_ms: float | None = Field(default=None, ge=0)

    contributions: list[FeatureContribution] = Field(
        default_factory=list,
        description="Top SHAP contributions, ordered by absolute magnitude",
    )

    @property
    def is_calibrated(self) -> bool:
        """A decision may only be derived from a calibrated probability."""
        return self.calibration_method is not None


class ModelCard(BaseModel):
    """Provenance for a trained artefact. Written at training time, never edited.

    ``config_fingerprint`` ties the artefact to the exact configuration bundle
    that produced it, so a result can always be traced back.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    model_version: str
    rung_id: int
    trained_at: datetime
    training_rows: int
    feature_names: tuple[str, ...]
    enabled_families: tuple[str, ...]
    calibration_method: str | None
    calibration_fitted_on: str | None
    random_seed: int
    config_fingerprint: str
    notes: str = ""
