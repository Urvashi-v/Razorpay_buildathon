"""Leave-one-family-out ablation.

SPEC section 07: "how much does each feature family actually contribute?"

Answered by retraining with one family disabled at a time and reporting the
change in NET RUPEES, not in AUC. A family that adds ranking quality but no money
has not earned its place - and the geography family in particular has to justify
its fairness cost with a real economic contribution, not merely a lift.

THREE DECISIONS THAT SHAPE WHAT THIS MEASURES
=============================================

**1. Validation only. The sealed test set is never touched.**
An ablation is a feature-selection activity, and `config/evaluation.yaml`
forbids selecting anything on the test split. Running six arms against the
sealed set would burn it six times over and every subsequent number in this
project would be contaminated. `ModelingDataset.test` raises unless someone
unseals it with a written reason, so this is enforced rather than remembered.

**2. Hyperparameters are re-selected per arm, not held fixed.**
Removing a family can change which capacity is optimal - a smaller feature set
may want fewer leaves. Holding the shipped configuration fixed would measure
"what this family contributes to a model tuned with it present", which
systematically flatters every family. Each arm gets the same search and the same
one-standard-error rule the shipped model got.

The cost of that fairness is selection variance: two arms can differ because the
search landed differently, not because the family mattered. Which is why -

**3. Every delta carries a bootstrap interval, and the report leads with it.**
A ₹200 difference on ~2,000 validation orders is noise. Reporting it as a
finding would be exactly the "point estimate presented as precise" that
`config/evaluation.yaml` lists as forbidden. An arm whose interval spans zero has
not been shown to matter, and this module says so in those words.

WHAT AN ABLATION CANNOT TELL YOU
================================
Families overlap. Dropping `customer_history` may barely move the number because
`order_shape` carries some of the same signal, and concluding "customer history
is worthless" from that would be wrong - it is redundant *given the others*,
which is a different claim. Leave-one-out measures marginal contribution at the
full feature set, nothing more.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from rto_sentinel.configuration.schemas import (
        CostModelConfig,
        FeaturesConfig,
        FinalModelConfig,
        GeneratorConfig,
        SplitsConfig,
    )
    from rto_sentinel.data.generator import GenerationResult

#: Bootstrap resamples for the interval on each arm's net rupees.
#:
#: 500 rather than 2,000: an ablation runs one of these per arm on top of a full
#: retrain, and the interval is used to answer "does this overlap zero", which
#: does not need three decimal places of tail accuracy.
BOOTSTRAP_ITERATIONS = 500

#: Below this many rupees per 1,000 orders, a difference is not called a finding
#: regardless of where the interval falls.
#:
#: Anchored to the arithmetic rather than chosen: one avoided RTO is worth
#: `S_tp` (about 132 INR at the default profile), so on a 2,000-order validation
#: split a single order changing sides moves the per-1,000 figure by about 66.
#: A delta smaller than that is not a decision, it is a rounding.
MATERIAL_DELTA_INR = 66.0


@dataclass(frozen=True, slots=True)
class AblationResult:
    """One leave-one-out run, expressed as a delta against the full model."""

    family_removed: str
    n_features: int
    net_inr_per_1000: float
    delta_vs_full: float
    delta_ci_low: float
    delta_ci_high: float
    pr_auc: float
    delta_pr_auc_vs_full: float
    flag_rate: float
    precision: float | None
    threshold: float
    chosen_candidate: str
    calibration_method: str

    @property
    def interval_spans_zero(self) -> bool:
        """True when the data cannot say this family helped or hurt."""
        return self.delta_ci_low <= 0.0 <= self.delta_ci_high

    @property
    def verdict(self) -> str:
        """One phrase, chosen so a table can be read without a legend."""
        if self.family_removed == "__full__":
            return "reference"
        if abs(self.delta_vs_full) < MATERIAL_DELTA_INR:
            return "no material effect"
        if self.interval_spans_zero:
            return "not established"
        return "earns its place" if self.delta_vs_full < 0 else "costs money"


@dataclass(frozen=True, slots=True)
class AblationStudy:
    """Every arm, with the full model as the reference."""

    generated_at: datetime
    dataset_run_id: str
    split: str
    seed: int
    cost_profile: str
    full_model: AblationResult
    arms: tuple[AblationResult, ...]
    findings: tuple[str, ...]

    data_provenance: str = (
        "Synthetic benchmark data. Feature-family contributions measured here are "
        "properties of the documented simulator, not evidence about which signals "
        "matter in production."
    )


def disable_family(config: FeaturesConfig, family: str) -> FeaturesConfig:
    """Return a features config with one family switched off.

    Raises rather than silently returning the original when the family does not
    exist: an ablation arm that quietly ran the full model would appear as
    "this family contributes nothing", which is the most misleading result this
    module could produce.
    """
    if family not in config.families:
        known = ", ".join(sorted(config.families))
        msg = f"unknown feature family {family!r}; known families: {known}"
        raise KeyError(msg)

    payload = copy.deepcopy(config.model_dump())
    payload["families"][family]["enabled"] = False
    return type(config).model_validate(payload)


def _bootstrap_net_delta(
    full_per_order: np.ndarray,
    arm_per_order: np.ndarray,
    *,
    seed: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float, float]:
    """A 95% interval on the *paired* per-order difference in net rupees.

    Paired on purpose. The two arms scored the same validation orders, so
    resampling them independently would add variance that is not there and widen
    every interval until nothing is ever significant. Resampling order indices
    once and applying them to both arms is the comparison that was actually made.
    """
    if full_per_order.size == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    difference = arm_per_order - full_per_order
    n = difference.size
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        picks = rng.integers(0, n, size=n)
        draws[index] = float(difference[picks].mean()) * 1000.0
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def _per_order_net(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    cost_false_positive_inr: float,
    saving_true_positive_inr: float,
) -> np.ndarray:
    """Rupees gained or lost on each order, against doing nothing.

    A flagged true positive saves ``S_tp``; a flagged false positive costs
    ``C_fp``; an unflagged order contributes nothing either way, because the
    false-negative loss is identical under both arms and cancels in the
    comparison.
    """
    flagged = scores >= threshold
    positives = labels.astype(bool)
    return np.where(
        flagged & positives,
        saving_true_positive_inr,
        np.where(flagged & ~positives, -cost_false_positive_inr, 0.0),
    )


def run_ablation(
    generated: GenerationResult,
    *,
    features_config: FeaturesConfig,
    generator_config: GeneratorConfig,
    splits_config: SplitsConfig,
    final_config: FinalModelConfig,
    cost_config: CostModelConfig,
    seed: int,
    families: list[str],
    split_labels: object = None,
    artifact_root: Path | None = None,
    progress: object = None,
) -> AblationStudy:
    """Retrain with each family removed in turn and report the deltas.

    ``artifact_root`` is deliberately not forwarded to the per-arm trainer. An
    ablation arm is a diagnostic, not a shippable model, and writing six model
    artefacts into the store would leave the registry choosing between them by
    ``trained_at`` - so a diagnostic could become the served model.
    """
    from rto_sentinel.decision.cost_model import outcome_economics
    from rto_sentinel.eval.metrics import confusion_at_threshold, pr_auc
    from rto_sentinel.features import build_modeling_dataset
    from rto_sentinel.models import build_final_model
    from rto_sentinel.models.experiment import cost_inputs_from_profile

    profile_key = cost_config.default_profile
    economics = outcome_economics(cost_inputs_from_profile(cost_config.profiles[profile_key]))

    def one_arm(label: str, config: FeaturesConfig) -> tuple[AblationResult, np.ndarray]:
        if progress is not None:
            progress(label)  # type: ignore[operator]

        dataset = build_modeling_dataset(
            generated,
            features_config=config,
            generator_config=generator_config,
            splits_config=splits_config,
            split_labels=split_labels,  # type: ignore[arg-type]
        )
        final = build_final_model(
            dataset,
            final_config=final_config,
            cost_config=cost_config,
            seed=seed,
            artifact_root=None,
            bootstrap_iterations=0,
        )
        validation = dataset.validation
        scores = np.asarray(final.model.predict_proba(validation.x), dtype=float)
        labels = np.asarray(validation.y).astype(int)
        threshold = final.manifest.threshold

        per_order = _per_order_net(
            labels,
            scores,
            threshold=threshold,
            cost_false_positive_inr=economics.false_positive_cost_inr,
            saving_true_positive_inr=economics.true_positive_saving_inr,
        )
        matrix = confusion_at_threshold(labels, scores, threshold)

        return (
            AblationResult(
                family_removed=label,
                n_features=len(validation.x.columns),
                net_inr_per_1000=float(per_order.mean()) * 1000.0,
                delta_vs_full=0.0,
                delta_ci_low=0.0,
                delta_ci_high=0.0,
                pr_auc=pr_auc(labels, scores),
                delta_pr_auc_vs_full=0.0,
                flag_rate=matrix.flag_rate,
                precision=matrix.precision if matrix.n_flagged else None,
                threshold=threshold,
                chosen_candidate=final.manifest.chosen_candidate,
                calibration_method=final.manifest.calibration_method,
            ),
            per_order,
        )

    full, full_per_order = one_arm("__full__", features_config)

    arms: list[AblationResult] = []
    for index, family in enumerate(families):
        arm, arm_per_order = one_arm(family, disable_family(features_config, family))
        low, high = _bootstrap_net_delta(full_per_order, arm_per_order, seed=seed + index + 1)
        arms.append(
            AblationResult(
                family_removed=arm.family_removed,
                n_features=arm.n_features,
                net_inr_per_1000=arm.net_inr_per_1000,
                delta_vs_full=arm.net_inr_per_1000 - full.net_inr_per_1000,
                delta_ci_low=low,
                delta_ci_high=high,
                pr_auc=arm.pr_auc,
                delta_pr_auc_vs_full=arm.pr_auc - full.pr_auc,
                flag_rate=arm.flag_rate,
                precision=arm.precision,
                threshold=arm.threshold,
                chosen_candidate=arm.chosen_candidate,
                calibration_method=arm.calibration_method,
            )
        )

    return AblationStudy(
        generated_at=datetime.now(UTC),
        dataset_run_id=generated.metadata.run_id,
        split="validation",
        seed=seed,
        cost_profile=profile_key,
        full_model=full,
        arms=tuple(arms),
        findings=summarise(full, tuple(arms)),
    )


def summarise(full: AblationResult, arms: tuple[AblationResult, ...]) -> tuple[str, ...]:
    """Findings, stated as what was measured rather than as a recommendation."""
    findings: list[str] = [
        f"Full model: INR {full.net_inr_per_1000:,.0f} per 1,000 orders on validation "
        f"with {full.n_features} features, PR-AUC {full.pr_auc:.3f}. Every delta below "
        "is against this arm, measured on the same orders."
    ]

    established = [
        arm
        for arm in arms
        if not arm.interval_spans_zero and abs(arm.delta_vs_full) >= MATERIAL_DELTA_INR
    ]
    for arm in sorted(established, key=lambda a: a.delta_vs_full):
        direction = "COSTS money" if arm.delta_vs_full > 0 else "earns its place"
        findings.append(
            f"{arm.family_removed}: removing it moves net by INR "
            f"{arm.delta_vs_full:+,.0f} per 1,000 "
            f"[{arm.delta_ci_low:+,.0f}, {arm.delta_ci_high:+,.0f}] - the family "
            f"{direction}. PR-AUC moves {arm.delta_pr_auc_vs_full:+.3f}."
        )

    unestablished = [arm for arm in arms if arm not in established]
    if unestablished:
        names = ", ".join(arm.family_removed for arm in unestablished)
        findings.append(
            f"{len(unestablished)} of {len(arms)} families showed no established "
            f"economic effect: {names}. Either the interval spans zero or the delta "
            f"is under INR {MATERIAL_DELTA_INR:.0f} per 1,000, which is roughly one "
            "order changing sides on this split. That is not evidence they are "
            "worthless - leave-one-out measures marginal contribution given every "
            "other family, and overlapping signal hides individual value."
        )

    geography = next((a for a in arms if a.family_removed == "geography_route"), None)
    if geography is not None:
        if geography.interval_spans_zero or abs(geography.delta_vs_full) < MATERIAL_DELTA_INR:
            findings.append(
                "geography_route carries the highest fairness risk in the project and "
                "did NOT demonstrate an established economic contribution here. That "
                "is the combination the fairness note in docs/features.md warned "
                "about: a family that imposes friction on places should have to pay "
                "for itself, and on this benchmark it has not been shown to."
            )
        else:
            margin = min(abs(geography.delta_ci_low), abs(geography.delta_ci_high))
            note = (
                f" The interval clears zero by only INR {margin:,.0f}, so this is a "
                "marginal result rather than a comfortable one - and for the family "
                "that imposes friction on places rather than on behaviour, marginal "
                "is worth re-checking at the next retrain rather than treating as "
                "settled."
                if margin < MATERIAL_DELTA_INR * 3
                else ""
            )
            findings.append(
                f"geography_route does pay for itself here (INR "
                f"{-geography.delta_vs_full:+,.0f} per 1,000), which is the "
                "justification its fairness cost requires - on this benchmark, "
                f"against these cohorts, and no further.{note}"
            )

    history = next((a for a in arms if a.family_removed == "customer_history"), None)
    if history is not None and (
        history.interval_spans_zero or abs(history.delta_vs_full) < MATERIAL_DELTA_INR
    ):
        findings.append(
            "customer_history is described throughout this project as the strongest "
            "honest signal, and its marginal contribution is NOT established here. "
            "That is not a contradiction: prior RTO rate is highly informative on its "
            "own, and leave-one-out asks a different question - what it adds once "
            "every other family is present. The overlap with order_shape (which "
            "carries the COD flag) is the likely explanation, and this study cannot "
            "separate them."
        )

    return tuple(findings)
