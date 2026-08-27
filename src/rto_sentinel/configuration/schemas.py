"""Typed schemas for the YAML files under ``config/``.

Every knob in this system lives in version-controlled YAML rather than in code,
and every one of those files is parsed into a validated model here. A malformed
or out-of-range configuration fails loudly at load time instead of quietly
producing a wrong threshold three layers down.

These are *configuration* schemas. Wire formats for the API live in
``rto_sentinel.contracts``; database tables live in ``rto_sentinel.db.models``.
Keeping the three separate is deliberate - see ARCHITECTURE.md.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    """Configuration is immutable once loaded. Nothing mutates it at runtime."""

    model_config = ConfigDict(frozen=True, extra="allow")


# ---------------------------------------------------------------------------
# config/cost_model.yaml
# ---------------------------------------------------------------------------


class CostProfile(_Frozen):
    """The four merchant-specific inputs that derive the operating threshold.

    SPEC section 06. ``friction_support_cost_inr`` is the small per-friction ops
    cost that rides along with every intervention.
    """

    label: str
    rto_cost_inr: float = Field(gt=0, description="Forward + reverse freight, repack, QC, support")
    contribution_margin_inr: float = Field(gt=0, description="Margin lost if a good order lapses")
    abandonment_on_friction: float = Field(ge=0.0, le=1.0)
    intervention_success_rate: float = Field(ge=0.0, le=1.0)
    friction_support_cost_inr: float = Field(default=0.0, ge=0.0)


class ParameterBound(_Frozen):
    min: float
    max: float
    source: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> ParameterBound:
        if self.min > self.max:
            msg = f"bound min ({self.min}) exceeds max ({self.max})"
            raise ValueError(msg)
        return self


class SensitivityConfig(_Frozen):
    perturbations: list[float]
    parameters: list[str]


class CostModelConfig(_Frozen):
    version: int
    default_profile: str
    profiles: dict[str, CostProfile]
    bounds: dict[str, ParameterBound]
    sensitivity: SensitivityConfig

    @model_validator(mode="after")
    def _default_profile_exists(self) -> CostModelConfig:
        if self.default_profile not in self.profiles:
            msg = f"default_profile {self.default_profile!r} is not defined in profiles"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# config/policy.yaml
# ---------------------------------------------------------------------------

BandName = Literal["LOW", "ELEVATED", "HIGH", "SEVERE"]
ActionName = Literal["none", "prepaid_nudge", "confirmation_required", "prepaid_only"]


class PolicyBand(_Frozen):
    """One rung of the friction ladder.

    ``upper_bound_multiplier`` is a multiple of the *cost-derived* threshold, not
    an absolute probability. That is what makes the whole ladder move when the
    merchant's economics move.
    """

    name: BandName
    upper_bound_multiplier: float | None = Field(default=None, gt=0)
    action: ActionName
    customer_visible: bool
    reversible: bool
    requires_reason_code: bool = False
    requires_appeal_path: bool = False
    requires_human_review_queue: bool = False
    channels: list[str] = Field(default_factory=list)
    description: str


class PolicySafeguards(_Frozen):
    hard_block_allowed: bool
    ops_override_enabled: bool
    ops_override_logged: bool
    decision_log_retention_days: int = Field(gt=0)
    neutral_customer_framing_required: bool


class HoldoutControlConfig(_Frozen):
    enabled: bool
    fraction_of_flagged: float = Field(ge=0.0, le=1.0)
    applies_to_bands: list[BandName]
    rationale: str


class PolicyConfig(_Frozen):
    version: int
    bands: list[PolicyBand]
    safeguards: PolicySafeguards
    holdout_control: HoldoutControlConfig

    @model_validator(mode="after")
    def _ladder_is_well_formed(self) -> PolicyConfig:
        """Enforce the ladder invariants declared at the top of policy.yaml."""
        expected: tuple[BandName, ...] = ("LOW", "ELEVATED", "HIGH", "SEVERE")
        actual = tuple(band.name for band in self.bands)
        if actual != expected:
            msg = f"bands must be exactly {expected} in order, got {actual}"
            raise ValueError(msg)

        # Multipliers must strictly increase, with only the top band open-ended.
        multipliers = [band.upper_bound_multiplier for band in self.bands]
        if multipliers[-1] is not None:
            msg = "the top band (SEVERE) must be open-ended (upper_bound_multiplier: null)"
            raise ValueError(msg)
        finite = [m for m in multipliers[:-1] if m is not None]
        if len(finite) != len(multipliers) - 1:
            msg = "only the top band may have a null upper_bound_multiplier"
            raise ValueError(msg)
        if any(a >= b for a, b in pairwise(finite)):
            msg = f"band multipliers must strictly increase, got {finite}"
            raise ValueError(msg)

        # Safety invariants: SPEC section 09.
        if self.safeguards.hard_block_allowed:
            msg = "hard_block_allowed must be false: no silent hard block, ever"
            raise ValueError(msg)
        for band in self.bands:
            if band.action != "none" and not band.requires_reason_code:
                msg = f"band {band.name} applies friction but carries no reason code"
                raise ValueError(msg)
            if not band.reversible:
                msg = f"band {band.name} is not reversible; every action must be appealable"
                raise ValueError(msg)
        severe = self.bands[-1]
        if not (severe.requires_appeal_path and severe.requires_human_review_queue):
            msg = "SEVERE must carry both an appeal path and a human review queue"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# config/splits.yaml
# ---------------------------------------------------------------------------


class DayRange(_Frozen):
    """Inclusive day range within the generated horizon."""

    start: int
    end: int

    @model_validator(mode="after")
    def _ordered(self) -> DayRange:
        if self.start > self.end:
            msg = f"day range start ({self.start}) exceeds end ({self.end})"
            raise ValueError(msg)
        return self


class TemporalSplitConfig(_Frozen):
    train_days: tuple[int, int]
    validation_days: tuple[int, int]
    test_days: tuple[int, int]

    @model_validator(mode="after")
    def _strictly_forward(self) -> TemporalSplitConfig:
        """Train must end before validation begins, and validation before test.

        This is the temporal-split rule from SPEC section 03, checked at load
        time so a mis-edited config cannot silently leak the future into the past.
        """
        windows = [self.train_days, self.validation_days, self.test_days]
        for (_, prev_end), (next_start, _) in pairwise(windows):
            if next_start <= prev_end:
                msg = f"splits overlap or are out of order: {windows}"
                raise ValueError(msg)
        for start, end in windows:
            if start > end:
                msg = f"split window ({start}, {end}) is inverted"
                raise ValueError(msg)
        return self


class GroupSplitConfig(_Frozen):
    key: str
    disjoint_across_splits: bool


class SealingConfig(_Frozen):
    test_set_sealed: bool
    max_test_evaluations: int = Field(ge=1)
    threshold_fitted_on: Literal["validation"]
    seal_receipt_path: str


class AsOfJoinConfig(_Frozen):
    enforced: bool
    order_timestamp_column: str
    resolution_timestamp_column: str
    rule: str


class LabelMaturityConfig(_Frozen):
    max_resolution_days: int = Field(gt=0)
    exclude_immature_tail: bool


class DriftWindowConfig(_Frozen):
    final_days: int = Field(gt=0)


class SplitsConfig(_Frozen):
    version: int
    strategy: Literal["temporal_grouped"]
    temporal: TemporalSplitConfig
    group: GroupSplitConfig
    sealing: SealingConfig
    as_of_join: AsOfJoinConfig
    label_maturity: LabelMaturityConfig
    drift_window: DriftWindowConfig

    @model_validator(mode="after")
    def _protocol_is_not_weakened(self) -> SplitsConfig:
        """The five split rules are not negotiable once results exist."""
        if not self.group.disjoint_across_splits:
            msg = "customer groups must stay disjoint across splits"
            raise ValueError(msg)
        if not self.sealing.test_set_sealed:
            msg = "the test set must remain sealed"
            raise ValueError(msg)
        if not self.as_of_join.enforced:
            msg = "as-of joins must stay enforced; otherwise aggregates leak the future"
            raise ValueError(msg)
        if not self.label_maturity.exclude_immature_tail:
            msg = "immature-label tail must be excluded, not optimistically labelled"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# config/features.yaml
# ---------------------------------------------------------------------------


class FeatureFamilyConfig(_Frozen):
    enabled: bool
    signals: list[str]
    risk_note: str
    as_of: bool = False
    shrinkage: Literal["bayesian"] | None = None
    min_support: int | None = None


class RefusedFeatureGroup(_Frozen):
    """A family of features this project refuses to use, and why.

    SPEC section 04. The patterns are matched against candidate column names by
    the feature pipeline, so the refusal is mechanical rather than a promise.
    """

    id: str
    patterns: list[str]
    reason: str


class FeaturesConfig(_Frozen):
    version: int
    families: dict[str, FeatureFamilyConfig]
    refused: list[RefusedFeatureGroup]

    @property
    def refused_patterns(self) -> frozenset[str]:
        return frozenset(p.lower() for group in self.refused for p in group.patterns)

    @property
    def enabled_families(self) -> tuple[str, ...]:
        return tuple(name for name, cfg in self.families.items() if cfg.enabled)


# ---------------------------------------------------------------------------
# config/models/ladder.yaml
# ---------------------------------------------------------------------------


class LadderRung(_Frozen):
    id: int = Field(ge=0)
    name: str
    kind: Literal["baseline", "heuristic", "model"]
    enabled: bool
    description: str
    params: dict[str, object] = Field(default_factory=dict)
    calibration: dict[str, object] | None = None
    promotion_rule: str | None = None


class ResamplingConfig(_Frozen):
    smote: bool
    reason: str


class SelectionConfig(_Frozen):
    primary_metric: str
    tiebreakers: list[str]
    max_acceptable_flag_rate: float = Field(gt=0.0, le=1.0)


class LadderConfig(_Frozen):
    version: int
    rungs: list[LadderRung]
    resampling: ResamplingConfig
    selection: SelectionConfig

    @model_validator(mode="after")
    def _ladder_is_ordered_and_honest(self) -> LadderConfig:
        ids = [rung.id for rung in self.rungs]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            msg = f"ladder rung ids must be unique and ascending, got {ids}"
            raise ValueError(msg)
        if ids and ids[0] != 0:
            msg = "the ladder must start at rung 0 (do-nothing baseline)"
            raise ValueError(msg)
        if self.resampling.smote:
            msg = "SMOTE is refused on tabular risk data; see the reason in ladder.yaml"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# config/evaluation.yaml
# ---------------------------------------------------------------------------


class BootstrapConfig(_Frozen):
    enabled: bool
    iterations: int = Field(gt=0)
    confidence: float = Field(gt=0.0, lt=1.0)
    note: str | None = None


class CalibrationMetricConfig(_Frozen):
    bins: int = Field(gt=1)
    strategy: Literal["uniform", "quantile"]
    note: str | None = None


class FairnessConfig(_Frozen):
    group_by: list[str]
    report: list[str]
    disparity_review_trigger: dict[str, float]


class EvaluationConfig(_Frozen):
    version: int
    primary_metric: str
    ranking_metrics: dict[str, object]
    calibration_metrics: dict[str, object]
    economics: dict[str, list[str]]
    uncertainty: dict[str, BootstrapConfig]
    cohorts: list[str]
    fairness: FairnessConfig
    ablation: dict[str, object]
    forbidden: list[str]

    @model_validator(mode="after")
    def _headline_is_rupees(self) -> EvaluationConfig:
        """The headline metric is net rupees, not a ranking statistic."""
        if self.primary_metric != "net_inr_saved_per_1000_orders":
            msg = "primary_metric must remain net_inr_saved_per_1000_orders"
            raise ValueError(msg)
        required = {"tune_threshold_on_test_set", "lead_with_roc_auc"}
        missing = required - set(self.forbidden)
        if missing:
            msg = f"evaluation config dropped required prohibitions: {sorted(missing)}"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# config/generator.yaml
# ---------------------------------------------------------------------------


class GeneratorHorizon(_Frozen):
    n_orders: int = Field(gt=0)
    start_date: str
    days: int = Field(gt=0)


class GeneratorBaseRates(_Frozen):
    rto_given_cod: float = Field(ge=0.0, le=1.0)
    rto_given_prepaid: float = Field(ge=0.0, le=1.0)
    marginal_tolerance: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _cod_is_the_problem(self) -> GeneratorBaseRates:
        """COD RTO must exceed prepaid RTO; the entire premise depends on it."""
        if self.rto_given_cod <= self.rto_given_prepaid:
            msg = "rto_given_cod must exceed rto_given_prepaid"
            raise ValueError(msg)
        return self


class GeneratorConfig(_Frozen):
    """Parameters for the synthetic order generator.

    SPEC section 09 names this as the one component deserving scrutiny. It is a
    labelled tabular sampler: order metadata plus a probabilistic RTO label drawn
    from published aggregate base rates. Nothing it emits is usable outside this
    repository's evaluation harness.
    """

    version: int
    seed: int
    horizon: GeneratorHorizon
    payment: dict[str, float]
    base_rates: GeneratorBaseRates
    causal_drivers: dict[str, dict[str, object]]
    customers: dict[str, object]
    geography: dict[str, object]
    address_quality: dict[str, object]
    catalogue: dict[str, object]
    couriers: list[dict[str, object]]
    label_maturity: dict[str, object]
    realism_anchors: dict[str, object]

    @model_validator(mode="after")
    def _labels_are_not_borrowed(self) -> GeneratorConfig:
        """Public datasets shape marginals only; they never supply an RTO label."""
        if self.realism_anchors.get("used_for_labels") is not False:
            msg = "realism_anchors.used_for_labels must be false: labels come from this generator"
            raise ValueError(msg)
        return self


class AppConfig(_Frozen):
    """The complete, validated configuration bundle for one run."""

    generator: GeneratorConfig
    splits: SplitsConfig
    features: FeaturesConfig
    cost_model: CostModelConfig
    policy: PolicyConfig
    ladder: LadderConfig
    evaluation: EvaluationConfig
