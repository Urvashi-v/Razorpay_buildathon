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


class BandEconomics(_Frozen):
    """How effective, and how costly, one rung's action is ASSUMED to be.

    Multipliers on the merchant's own rates rather than absolutes, so the ladder
    keeps its shape when a merchant changes their economics and there is one
    place - :class:`CostInputs` - where the scale lives.

    Every field here is an assumption. See the header of ``config/policy.yaml``
    and :class:`~rto_sentinel.contracts.provenance.Provenance` for why they are
    tagged ``assumed_intervention`` wherever they reach a report.
    """

    intervention_success_multiplier: float = Field(ge=0.0, le=5.0)
    abandonment_multiplier: float = Field(ge=0.0, le=5.0)
    support_cost_inr: float = Field(ge=0.0)
    rationale: str


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
    economics: BandEconomics


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
    """How customer disjointness is enforced across the temporal splits."""

    key: str
    disjoint_across_splits: bool
    assignment: Literal["customer_pool"]
    pool_shares: dict[str, float]
    pool_salt: str

    @model_validator(mode="after")
    def _pools_are_well_formed(self) -> GroupSplitConfig:
        expected = {"train", "validation", "test"}
        if set(self.pool_shares) != expected:
            msg = f"pool_shares must name exactly {sorted(expected)}"
            raise ValueError(msg)
        total = sum(self.pool_shares.values())
        if abs(total - 1.0) > 1e-6:
            msg = f"pool_shares must sum to 1.0, got {total}"
            raise ValueError(msg)
        if any(share <= 0 for share in self.pool_shares.values()):
            msg = "every pool must receive a positive share of customers"
            raise ValueError(msg)
        return self


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


class AllowedException(_Frozen):
    """A feature name that contains a refused token but is not the refused thing.

    Each needs a written justification. The list is deliberately tiny - if it
    grows, the refused patterns are wrong and should be fixed rather than
    exempted one name at a time.
    """

    feature: str
    contains_token: str
    reason: str

    @model_validator(mode="after")
    def _reason_is_present(self) -> AllowedException:
        if not self.reason.strip():
            msg = f"allowed exception for {self.feature!r} has no justification"
            raise ValueError(msg)
        return self


class FeaturesConfig(_Frozen):
    version: int
    families: dict[str, FeatureFamilyConfig]
    refused: list[RefusedFeatureGroup]
    allowed_exceptions: list[AllowedException] = Field(default_factory=list)

    @property
    def refused_patterns(self) -> frozenset[str]:
        return frozenset(p.lower() for group in self.refused for p in group.patterns)

    @property
    def exempt_features(self) -> frozenset[str]:
        return frozenset(item.feature.lower() for item in self.allowed_exceptions)

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
# config/models/final.yaml
# ---------------------------------------------------------------------------


class SearchCandidate(_Frozen):
    """One hyperparameter configuration, named so results can refer to it."""

    name: str
    note: str = ""
    params: dict[str, object] = Field(default_factory=dict)


class SearchConfig(_Frozen):
    metric: str
    candidates: list[SearchCandidate]
    bootstrap_iterations: int = Field(default=200, ge=0)
    tie_rule: Literal["best_point_estimate", "one_standard_error_then_smallest"] = (
        "one_standard_error_then_smallest"
    )

    @model_validator(mode="after")
    def _candidates_are_distinct(self) -> SearchConfig:
        names = [candidate.name for candidate in self.candidates]
        if len(set(names)) != len(names):
            msg = f"search candidate names must be unique, got {names}"
            raise ValueError(msg)
        if not names:
            msg = "the search needs at least one candidate"
            raise ValueError(msg)
        return self


class CalibrationSelectionConfig(_Frozen):
    candidates: list[str]
    n_folds: int = Field(ge=2, le=20)
    fitted_on: str
    metric: str
    tiebreaker: str
    minimum_improvement: float = Field(ge=0.0)
    max_brier_degradation: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _never_fitted_on_test(self) -> CalibrationSelectionConfig:
        """The one rule in this file that is a safety property rather than a choice."""
        if self.fitted_on != "validation":
            msg = (
                f"calibration must be fitted on validation, not {self.fitted_on!r}. "
                "Train calibrates the model to its own overfitting; test destroys the seal."
            )
            raise ValueError(msg)
        if "none" not in self.candidates:
            msg = (
                "'none' must remain a calibration candidate. Without it there is no "
                "measurement of whether calibrating helped at all."
            )
            raise ValueError(msg)
        return self


class FinalGuardrails(_Frozen):
    max_acceptable_flag_rate: float = Field(gt=0.0, le=1.0)
    min_pr_auc_over_base_rate: float = Field(ge=0.0)


class FinalModelConfig(_Frozen):
    version: int
    base_rung: str
    search: SearchConfig
    calibration: CalibrationSelectionConfig
    guardrails: FinalGuardrails


# ---------------------------------------------------------------------------
# config/models/model_card.yaml
# ---------------------------------------------------------------------------


class TrainingDataProse(_Frozen):
    description: str
    synthetic_disclaimer: str
    what_is_not_in_it: list[str]


class FeatureProse(_Frozen):
    description: str
    excluded_deliberately: list[str]


class MaintenanceProse(_Frozen):
    retraining_trigger: str
    monitoring: str


class ModelCardConfig(_Frozen):
    """The judgement half of the model card. Never the measured half.

    Every field here is prose someone has to stand behind. The metrics are read
    from the evaluation artefacts at render time, so this file cannot make a
    quantitative claim at all - which is the property that keeps a card from
    drifting away from the run that produced it.
    """

    version: int
    model_name: str
    owner: str
    summary: str
    intended_use: list[str]
    non_intended_use: list[str]
    training_data: TrainingDataProse
    features: FeatureProse
    known_limitations: list[str]
    evaluation_methodology: str
    calibration_methodology: str
    fairness_limitations: list[str]
    distribution_shift_limitations: list[str]
    maintenance: MaintenanceProse

    @model_validator(mode="after")
    def _the_disclaimer_cannot_be_softened(self) -> ModelCardConfig:
        """The synthetic-data disclaimer is not an editable pleasantry."""
        required = ("simulated", "not")
        lowered = self.training_data.synthetic_disclaimer.lower()
        if not all(word in lowered for word in required):
            msg = (
                "the synthetic-data disclaimer must state plainly that the labels are "
                "simulated and are not real-world ground truth"
            )
            raise ValueError(msg)
        if not self.non_intended_use:
            msg = "a model card without a non-intended-use section is an advertisement"
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
    """Which cohorts the audit examines, and what counts as evidence.

    ``group_by`` may only name operational cohorts. That is enforced in
    ``eval/fairness.py`` rather than here, because the check is a substring match
    against a token list that belongs next to the audit it protects - but the
    consequence is worth stating at the config surface: adding a sensitive
    attribute to this list makes the audit fail, not quietly widen.
    """

    group_by: list[str]
    report: list[str]
    disparity_review_trigger: dict[str, float]

    #: Orders a group needs before its rates count as evidence. Below this the
    #: Wilson interval on a proportion is wider than the disparities the audit is
    #: looking for, so any comparison is decided by noise.
    min_support_orders: int = Field(default=100, ge=1)

    #: Flagged orders a group needs before its precision counts as evidence.
    min_flagged_orders: int = Field(default=30, ge=1)


class AblationConfig(_Frozen):
    """Which feature families the leave-one-family-out study removes.

    Typed rather than left as `dict[str, object]`. The families list is iterated
    to build one training arm each, and an untyped dict pushed that check to
    runtime - where a typo would have produced an ablation that silently ran
    fewer arms than the config named.
    """

    mode: Literal["leave_one_family_out"]
    families: list[str]


class EvaluationConfig(_Frozen):
    version: int
    primary_metric: str
    ranking_metrics: dict[str, object]
    calibration_metrics: dict[str, object]
    economics: dict[str, list[str]]
    uncertainty: dict[str, BootstrapConfig]
    cohorts: list[str]
    fairness: FairnessConfig
    ablation: AblationConfig
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


class BetaParams(_Frozen):
    """Shape parameters for a Beta-distributed latent trait."""

    a: float = Field(gt=0)
    b: float = Field(gt=0)


class IntRange(_Frozen):
    min: int
    max: int

    @model_validator(mode="after")
    def _ordered(self) -> IntRange:
        if self.min > self.max:
            msg = f"range min ({self.min}) exceeds max ({self.max})"
            raise ValueError(msg)
        return self


class GeneratorCustomers(_Frozen):
    n_customers: int = Field(gt=0)
    orders_per_customer_alpha: float = Field(gt=0)
    activity_clip_quantile: float = Field(gt=0.0, le=1.0)
    reliability_beta: BetaParams
    address_quality_beta: BetaParams
    prepaid_affinity_beta: BetaParams


class CausalDriver(_Frozen):
    """One term in the simulator's latent logit.

    ``observable`` records whether a model could ever see this driver:
    ``true`` (measured directly), ``partial`` (a noisy proxy is available), or
    ``false`` (latent, and therefore a source of irreducible error). It is
    documentation with teeth - ``docs/simulator.md`` is generated against it and
    the generator asserts at least one driver is unobservable, because a
    simulation with no hidden state is a deterministic rule in disguise.
    """

    weight: float
    observable: Literal["true", "partial", "false"] | bool
    note: str

    @property
    def is_latent(self) -> bool:
        return self.observable is False or self.observable == "false"


class GeneratorNoise(_Frozen):
    logit_sigma: float = Field(ge=0.0)
    pincode_effect_sigma: float = Field(ge=0.0)
    label_flip_rate: float = Field(ge=0.0, le=0.2)


class GeneratorGeography(_Frozen):
    n_pincodes: int = Field(gt=0)
    tier_shares: dict[str, float]
    tier_risk_offset: dict[str, float]
    shrinkage_prior_strength: float = Field(gt=0)
    min_support_for_feature: int = Field(gt=0)

    @model_validator(mode="after")
    def _shares_sum_to_one(self) -> GeneratorGeography:
        total = sum(self.tier_shares.values())
        if abs(total - 1.0) > 1e-6:
            msg = f"tier_shares must sum to 1.0, got {total}"
            raise ValueError(msg)
        missing = set(self.tier_shares) - set(self.tier_risk_offset)
        if missing:
            msg = f"tier_risk_offset is missing tiers: {sorted(missing)}"
            raise ValueError(msg)
        return self


class GeneratorAddressQuality(_Frozen):
    tier_degradation: dict[str, float]
    pincode_city_inconsistency: dict[str, float]
    pincode_city_inconsistency_penalty: float = Field(ge=0.0, le=1.0)
    alternate_address_rate: float = Field(ge=0.0, le=1.0)


class CatalogueCategory(_Frozen):
    name: str
    share: float = Field(gt=0.0, le=1.0)
    rto_logit_offset: float
    margin_rate: float = Field(ge=0.0, le=1.0)


class OrderValueDistribution(_Frozen):
    distribution: Literal["lognormal"]
    mu: float
    sigma: float = Field(gt=0)
    min: float = Field(gt=0)
    max: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> OrderValueDistribution:
        if self.min >= self.max:
            msg = "order value min must be below max"
            raise ValueError(msg)
        return self


class GeneratorCatalogue(_Frozen):
    categories: list[CatalogueCategory]
    order_value: OrderValueDistribution
    items_per_order_lambda: float = Field(gt=0)

    @model_validator(mode="after")
    def _shares_sum_to_one(self) -> GeneratorCatalogue:
        total = sum(category.share for category in self.categories)
        if abs(total - 1.0) > 1e-6:
            msg = f"category shares must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self


class CourierConfig(_Frozen):
    name: str
    share: float = Field(gt=0.0, le=1.0)
    lane_quality: float = Field(ge=0.0, le=1.0)


class GeneratorTiming(_Frozen):
    hour_weights: list[float]
    late_night_hours: list[int]
    weekend_uplift: float = Field(gt=0)
    sale_days: list[int]
    sale_day_volume_multiplier: float = Field(gt=0)
    sale_day_discount_uplift: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _hours_are_well_formed(self) -> GeneratorTiming:
        if len(self.hour_weights) != 24:
            msg = f"hour_weights must have 24 entries, got {len(self.hour_weights)}"
            raise ValueError(msg)
        if any(weight < 0 for weight in self.hour_weights):
            msg = "hour_weights must be non-negative"
            raise ValueError(msg)
        if any(not 0 <= hour <= 23 for hour in self.late_night_hours):
            msg = "late_night_hours must be in 0-23"
            raise ValueError(msg)
        return self


class GeneratorFulfilment(_Frozen):
    dispatch_lag_hours: IntRange
    transit_days: IntRange
    rto_extra_days: IntRange
    cancellation_rate: float = Field(ge=0.0, le=1.0)


class GeneratorLabelMaturity(_Frozen):
    max_resolution_days: int = Field(gt=0)
    exclude_unresolved_tail: bool


class GeneratorRealismAnchors(_Frozen):
    basket_structure_reference: str
    used_for: list[str]
    used_for_labels: bool


class GeneratorPayment(_Frozen):
    cod_share: float = Field(gt=0.0, lt=1.0)
    prepaid_failure_to_cod: float = Field(ge=0.0, le=1.0)


class GeneratorConfig(_Frozen):
    """Parameters for the synthetic order generator.

    SPEC section 09 names this as the one component deserving scrutiny. It is a
    controlled benchmark generator: order metadata plus a probabilistic RTO label
    drawn from the documented causal process in ``docs/simulator.md``, calibrated
    to published aggregate base rates. Nothing it emits is usable outside this
    repository's own evaluation harness, and nothing it emits is ground truth
    about the real world.
    """

    version: int
    generator_version: str
    seed: int
    horizon: GeneratorHorizon
    payment: GeneratorPayment
    base_rates: GeneratorBaseRates
    customers: GeneratorCustomers
    causal_drivers: dict[str, CausalDriver]
    noise: GeneratorNoise
    geography: GeneratorGeography
    address_quality: GeneratorAddressQuality
    catalogue: GeneratorCatalogue
    couriers: list[CourierConfig]
    timing: GeneratorTiming
    fulfilment: GeneratorFulfilment
    label_maturity: GeneratorLabelMaturity
    realism_anchors: GeneratorRealismAnchors

    @model_validator(mode="after")
    def _labels_are_not_borrowed(self) -> GeneratorConfig:
        """Public datasets shape marginals only; they never supply an RTO label."""
        if self.realism_anchors.used_for_labels:
            msg = "realism_anchors.used_for_labels must be false: labels come from this generator"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _simulation_has_hidden_state(self) -> GeneratorConfig:
        """At least one driver must be unobservable.

        A simulation whose every driver is visible to the model is a deterministic
        rule waiting to be reverse-engineered, and a model trained on it reports a
        score that means nothing. The latent drivers are what create a real
        Bayes-optimal ceiling.
        """
        if not any(driver.is_latent for driver in self.causal_drivers.values()):
            msg = (
                "at least one causal driver must be unobservable; otherwise the "
                "learning task is recovering a deterministic rule"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _couriers_are_well_formed(self) -> GeneratorConfig:
        total = sum(courier.share for courier in self.couriers)
        if abs(total - 1.0) > 1e-6:
            msg = f"courier shares must sum to 1.0, got {total}"
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
