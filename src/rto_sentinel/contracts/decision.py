"""Decision-layer contracts: cost inputs, derived thresholds, and actions.

This is the boundary the whole submission turns on. A :class:`Decision` is
produced by the deterministic engine in ``rto_sentinel.decision`` from exactly
two things: a calibrated :class:`~rto_sentinel.contracts.risk.RiskScore` and a
:class:`CostInputs` supplied by the merchant. Nothing else may influence it - not
an LLM, not a heuristic bolted onto an API handler, not a hardcoded 0.5.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rto_sentinel.contracts.enums import InterventionAction, OverrideDirection, RiskBand


class CostInputs(BaseModel):
    """The four merchant-specific rupee inputs, plus the friction support cost.

    These are inputs, not truths. The console exposes them as sliders precisely
    because a high-margin brand should flag more aggressively than a thin-margin
    reseller, and the demo makes that visible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rto_cost_inr: float = Field(gt=0, description="Full cost of one RTO")
    contribution_margin_inr: float = Field(gt=0, description="Margin lost if a good order lapses")
    abandonment_on_friction: float = Field(
        ge=0.0, le=1.0, description="P(good customer abandons | frictioned)"
    )
    intervention_success_rate: float = Field(
        ge=0.0, le=1.0, description="P(risky order saved | frictioned)"
    )
    friction_support_cost_inr: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _threshold_is_derivable(self) -> CostInputs:
        """Reject inputs from which no finite threshold can be derived.

        If frictioning a bad order saves nothing *and* frictioning a good one
        costs nothing, the expected-value rule is undefined. Refusing here is
        better than silently returning 0.5, which is the exact failure mode this
        project exists to avoid.
        """
        cost_fp = self.abandonment_on_friction * self.contribution_margin_inr
        saving_tp = self.intervention_success_rate * self.rto_cost_inr
        if cost_fp + saving_tp + self.friction_support_cost_inr <= 0:
            msg = (
                "cost inputs are degenerate: frictioning has neither a cost nor a benefit, "
                "so no expected-value threshold exists"
            )
            raise ValueError(msg)
        return self


class ThresholdDerivation(BaseModel):
    """The derived operating threshold, with its arithmetic shown.

    The intermediate terms are part of the contract on purpose: the console
    displays them, and a reviewer can check the number by hand.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(ge=0.0, le=1.0)
    cost_false_positive_inr: float = Field(ge=0, description="C_fp")
    saving_true_positive_inr: float = Field(ge=0, description="S_tp")
    inputs: CostInputs
    formula: str = Field(default="threshold = C_fp / (C_fp + S_tp)")


class BandBoundary(BaseModel):
    """Resolved probability cut points for one rung of the ladder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    band: RiskBand
    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float | None = Field(default=None, ge=0.0, le=1.0)
    action: InterventionAction


class Decision(BaseModel):
    """The engine's output for one order: what to do, and why.

    ``reason_codes`` are machine-generated identifiers derived from SHAP
    contributions. The plain-language sentence an ops associate reads is produced
    separately by the agent layer and lives in
    :class:`~rto_sentinel.contracts.explanation.Explanation` - so that a failed
    LLM call degrades the wording, never the decision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    order_id: str = Field(max_length=64)
    probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    band: RiskBand
    action: InterventionAction

    flagged: bool = Field(description="True when any friction is applied")
    reason_codes: tuple[str, ...] = Field(default=())

    expected_value_inr: float = Field(
        description="Expected rupee benefit of this action versus doing nothing"
    )
    decided_at: datetime
    engine_version: str = Field(max_length=32)

    # Safeguards travelling with the decision, so the API cannot omit them.
    appeal_available: bool = Field(default=True)
    human_review_required: bool = Field(default=False)
    is_control_holdout: bool = Field(
        default=False,
        description="Randomised no-friction slice retained to keep precision measurable",
    )

    @model_validator(mode="after")
    def _safeguards_hold(self) -> Decision:
        """Invariants that must be true of every decision this system emits."""
        if self.action is InterventionAction.NONE and self.flagged:
            msg = "a decision with no action cannot be flagged"
            raise ValueError(msg)
        if self.action is not InterventionAction.NONE and not self.flagged:
            msg = "a decision that applies friction must be marked flagged"
            raise ValueError(msg)
        if self.flagged and not self.reason_codes:
            msg = "every friction decision must carry at least one reason code"
            raise ValueError(msg)
        if not self.appeal_available:
            msg = "no decision may remove the appeal path"
            raise ValueError(msg)
        if self.band is RiskBand.SEVERE and not self.human_review_required:
            msg = "SEVERE decisions must route to a human review queue"
            raise ValueError(msg)
        return self


class OpsOverride(BaseModel):
    """A human changing the engine's recommendation.

    Always available, always logged. Overrides are counterfactual evidence and
    feed the outcome loop (SPEC section 02, step 5).
    """

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(max_length=64)
    original_band: RiskBand
    override_band: RiskBand
    direction: OverrideDirection
    operator_id: str = Field(max_length=64, description="Hashed operator identity")
    note: str = Field(default="", max_length=1000)
    created_at: datetime

    @model_validator(mode="after")
    def _direction_matches(self) -> OpsOverride:
        from rto_sentinel.contracts.enums import band_rank

        moved_up = band_rank(self.override_band) > band_rank(self.original_band)
        expected = OverrideDirection.ESCALATED if moved_up else OverrideDirection.RELAXED
        if self.override_band == self.original_band:
            msg = "an override must change the band"
            raise ValueError(msg)
        if self.direction is not expected:
            msg = f"direction {self.direction} contradicts the band change"
            raise ValueError(msg)
        return self
