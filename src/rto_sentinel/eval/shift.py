"""Controlled distribution-shift experiments.

WHAT MAKES THIS A ROBUSTNESS STUDY RATHER THAN A SECOND TEST SET
================================================================
Regenerating data with a fresh seed and evaluating on it measures sampling
variance. It is worth knowing, and it is not robustness: every draw comes from
the same distribution the model was trained on, so a model that has memorised
that distribution's quirks scores just as well on the resample as on the original.

Here, each environment is a **named, deliberate change to a generator parameter**
- COD share, RTO base rate, category mix, customer mix, geography mix - and the
model is not retrained. The same frozen artefact and the same frozen threshold
face a world that moved. That is the question a deployed model actually faces.

The reference environment exists so degradation is measured against something.
It is generated with the same seed discipline as the shifted worlds and differs
from them only in that it applies no overrides, which is why
:class:`ShiftStudy` refuses to validate without it.

WHY THE THRESHOLD IS HELD FIXED
===============================
Re-deriving the threshold per environment would quietly repair part of the damage
and understate what a real deployment suffers. In production the threshold is a
configuration value; it does not follow the distribution around on its own. The
study measures what happens to a fixed operating point, which is the thing that
is actually at risk.

WHAT THESE RESULTS ARE
======================
Statements about this simulator under documented perturbations. A 22% PR-AUC drop
when COD share rises is evidence that this model leans on payment method, which
is worth knowing and acting on. It is not a prediction of what would happen to a
production model during a festive season.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rto_sentinel.contracts.monitoring import EnvironmentSpec, ShiftResult, ShiftStudy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import (
        FeaturesConfig,
        GeneratorConfig,
        SplitsConfig,
    )
    from rto_sentinel.models.base import RiskModel


class OverrideError(KeyError):
    """Raised when an override names a path that does not exist in the config.

    Fatal rather than ignored. A typo in ``payment.cod_shre`` that is silently
    skipped produces an environment identical to the reference, and a study that
    confidently reports "the model is robust to COD-share shift" having never
    shifted anything.
    """


def apply_overrides(config: GeneratorConfig, overrides: dict[str, float]) -> GeneratorConfig:
    """Return a new generator config with dotted paths replaced.

    Two addressing forms are supported::

        payment.cod_share                 -> a plain nested key
        catalogue.categories.fashion.share -> a list of dicts, addressed by name

    The second exists because ``catalogue.categories[0].share`` is unreadable in a
    config file and silently wrong the moment somebody reorders the list. Naming
    the category means the override says what it means.

    The config is round-tripped through its own validator, so an override that
    produces an invalid world - a negative share, a probability above one - fails
    here rather than producing a quietly broken dataset.
    """
    payload: dict[str, Any] = copy.deepcopy(config.model_dump())

    for path, value in overrides.items():
        parts = path.split(".")
        cursor: Any = payload
        for part in parts[:-1]:
            cursor = _descend(cursor, part, path)
        leaf = parts[-1]
        if isinstance(cursor, dict) and leaf not in cursor:
            msg = (
                f"override path {path!r} names {leaf!r}, which does not exist in the "
                f"generator config. Available at that level: {sorted(cursor)[:12]}"
            )
            raise OverrideError(msg)
        cursor[leaf] = value

    return type(config).model_validate(payload)


def _descend(cursor: Any, part: str, path: str) -> Any:
    """One step down a dotted path, through dicts and name-keyed lists."""
    if isinstance(cursor, dict):
        if part not in cursor:
            msg = (
                f"override path {path!r} names {part!r}, which does not exist. "
                f"Available at that level: {sorted(cursor)[:12]}"
            )
            raise OverrideError(msg)
        return cursor[part]
    if isinstance(cursor, list):
        for entry in cursor:
            if isinstance(entry, dict) and entry.get("name") == part:
                return entry
        names = [entry.get("name") for entry in cursor if isinstance(entry, dict)]
        msg = f"override path {path!r} names {part!r}, which is not among {names}"
        raise OverrideError(msg)
    msg = f"override path {path!r} cannot descend into {type(cursor).__name__} at {part!r}"
    raise OverrideError(msg)


@dataclass(frozen=True, slots=True)
class EnvironmentData:
    """One generated world, ready to score. Features and labels, aligned."""

    spec: EnvironmentSpec
    features: pd.DataFrame
    labels: np.ndarray
    orders: pd.DataFrame


def generate_environment(
    spec: EnvironmentSpec,
    *,
    generator_config: GeneratorConfig,
    features_config: FeaturesConfig,
    splits_config: SplitsConfig,
    n_customers: int | None = None,
) -> EnvironmentData:
    """Generate one environment and build its feature matrix.

    The feature pipeline is the *same* one the model was trained with - same
    config, same version. Rebuilding features differently per environment would
    confound distribution shift with pipeline change, and the study would be
    unable to say which one caused the degradation.
    """
    from rto_sentinel.data.generator import ConfiguredOrderGenerator, GeneratorParams
    from rto_sentinel.data.splits import assign_splits
    from rto_sentinel.features.dataset import attach_customer_dimension
    from rto_sentinel.features.pipeline import FeaturePipeline

    shifted = apply_overrides(generator_config, spec.overrides)

    start = generator_config.horizon.start_date
    params = GeneratorParams(
        seed=spec.seed,
        generator_version=shifted.generator_version,
        # Scale the customer population with the order count so the orders-per-
        # customer ratio - and therefore how much history the model gets to see -
        # stays comparable across environments. Holding customers fixed while
        # shrinking orders would itself be a distribution shift, and an
        # undocumented one.
        n_customers=n_customers or max(int(spec.n_orders * 0.375), 100),
        n_orders=spec.n_orders,
        start_date=_as_datetime(start),
        end_date=_as_datetime(start) + pd.Timedelta(days=generator_config.horizon.days - 1),
    )

    result = ConfiguredOrderGenerator().generate(shifted, params)
    orders = result.orders.copy()
    orders["split"] = assign_splits(orders, splits_config).labels

    enriched = attach_customer_dimension(orders, result.customers)
    matrix = FeaturePipeline(features_config, generator_config).build(enriched)

    # Only matured rows can be evaluated. An immature order has no outcome, and
    # counting it as a non-RTO would flatter every environment equally and hide
    # the differences the study exists to measure.
    matured = orders["is_rto"].notna().to_numpy()
    features = matrix.matrix.loc[matured].reset_index(drop=True)
    labels = orders.loc[matured, "is_rto"].to_numpy().astype(int)

    return EnvironmentData(
        spec=spec,
        features=features,
        labels=labels,
        orders=orders.loc[matured].reset_index(drop=True),
    )


def _as_datetime(value: Any) -> datetime:
    """Horizon start, however the config spelled it.

    `horizon.start_date` round-trips through YAML as a string in some configs and
    as a date in others; normalising here keeps the environment generator
    indifferent to which.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value)
    else:
        parsed = datetime.combine(value, datetime.min.time())
    # The generator compares against timezone-aware horizon bounds, so a naive
    # datetime here fails deep inside the maturity check rather than at the edge.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def evaluate_environment(
    data: EnvironmentData,
    model: RiskModel,
    *,
    threshold: float,
    cost_false_positive_inr: float,
    saving_true_positive_inr: float,
    feature_names: tuple[str, ...] | None = None,
) -> ShiftResult:
    """Score one environment with the frozen model and price the outcome."""
    from rto_sentinel.eval.metrics import (
        confusion_at_threshold,
        expected_calibration_error,
        pr_auc,
        roc_auc,
    )

    features = data.features
    if feature_names:
        # Reindex rather than assume column order. A model handed its columns in
        # a different order returns confident nonsense, and the failure is silent.
        missing = [name for name in feature_names if name not in features.columns]
        if missing:
            msg = (
                f"environment {data.spec.name!r} is missing {len(missing)} feature(s) the "
                f"model expects, e.g. {missing[:5]}. Scoring anyway would silently "
                "misalign columns."
            )
            raise ValueError(msg)
        features = features[list(feature_names)]

    scores = np.asarray(model.predict_proba(features), dtype=float)
    labels = data.labels

    matrix = confusion_at_threshold(labels, scores, threshold)
    ece, _ = expected_calibration_error(labels, scores)
    brier = float(np.mean((scores - labels) ** 2))

    net = (
        (
            matrix.true_positives * saving_true_positive_inr
            - matrix.false_positives * cost_false_positive_inr
        )
        / max(len(labels), 1)
        * 1000.0
    )

    return ShiftResult(
        environment=data.spec.name,
        description=data.spec.description,
        n_orders=len(labels),
        observed_rto_rate=float(labels.mean()) if len(labels) else 0.0,
        pr_auc=pr_auc(labels, scores),
        roc_auc=roc_auc(labels, scores),
        brier_score=brier,
        expected_calibration_error=ece,
        threshold=threshold,
        flag_rate=matrix.flag_rate,
        precision=matrix.precision if matrix.n_flagged else None,
        recall=matrix.recall if labels.sum() else None,
        net_inr_per_1000=net,
    )


def run_shift_study(
    environments: tuple[EnvironmentSpec, ...],
    model: RiskModel,
    *,
    generator_config: GeneratorConfig,
    features_config: FeaturesConfig,
    splits_config: SplitsConfig,
    threshold: float,
    cost_false_positive_inr: float,
    saving_true_positive_inr: float,
    model_version: str,
    feature_version: str = "",
    feature_names: tuple[str, ...] | None = None,
    progress: Any = None,
) -> ShiftStudy:
    """Generate every environment, score them all with one frozen model, compare.

    Deltas are computed against the reference environment, which must be present.
    """
    if not any(spec.name == "reference" for spec in environments):
        msg = (
            "a shift study needs a reference environment to measure degradation "
            "against; without one the numbers are absolute values with nothing to "
            "compare them to"
        )
        raise ValueError(msg)

    raw: list[ShiftResult] = []
    for spec in environments:
        if progress is not None:
            progress(spec)
        data = generate_environment(
            spec,
            generator_config=generator_config,
            features_config=features_config,
            splits_config=splits_config,
        )
        raw.append(
            evaluate_environment(
                data,
                model,
                threshold=threshold,
                cost_false_positive_inr=cost_false_positive_inr,
                saving_true_positive_inr=saving_true_positive_inr,
                feature_names=feature_names,
            )
        )

    reference = next(result for result in raw if result.environment == "reference")
    results = tuple(
        result
        if result.environment == "reference"
        else result.model_copy(
            update={
                "pr_auc_delta": result.pr_auc - reference.pr_auc,
                "net_delta": result.net_inr_per_1000 - reference.net_inr_per_1000,
                "ece_delta": (
                    result.expected_calibration_error - reference.expected_calibration_error
                ),
            }
        )
        for result in raw
    )

    return ShiftStudy(
        generated_at=datetime.now(UTC),
        model_version=model_version,
        feature_version=feature_version,
        generator_version=generator_config.generator_version,
        threshold=threshold,
        environments=environments,
        results=results,
        findings=summarise(results),
    )


#: A PR-AUC drop larger than this is called out as material. Below it the change
#: is within the range two seeds of the same world routinely produce, so calling
#: it degradation would be reading noise.
MATERIAL_PR_AUC_DROP = 0.03

#: Likewise for calibration. ECE moving by more than this changes what the
#: threshold means, which is the failure with rupee consequences.
MATERIAL_ECE_RISE = 0.02


def summarise(results: tuple[ShiftResult, ...]) -> tuple[str, ...]:
    """Findings, stated as what was measured rather than as a verdict."""
    findings: list[str] = []
    shifted = [result for result in results if result.environment != "reference"]
    if not shifted:
        return ("Only the reference environment was run; nothing was shifted.",)

    ranking = [
        result
        for result in shifted
        if result.pr_auc_delta is not None and result.pr_auc_delta < -MATERIAL_PR_AUC_DROP
    ]
    for result in sorted(ranking, key=lambda entry: entry.pr_auc_delta or 0.0):
        findings.append(
            f"{result.environment}: PR-AUC fell by {abs(result.pr_auc_delta or 0):.3f} "
            f"to {result.pr_auc:.3f}. The model's ability to rank orders degraded when "
            f"{result.description.lower()}"
        )

    miscalibrated = [
        result
        for result in shifted
        if result.ece_delta is not None and result.ece_delta > MATERIAL_ECE_RISE
    ]
    for result in sorted(miscalibrated, key=lambda entry: -(entry.ece_delta or 0.0)):
        findings.append(
            f"{result.environment}: calibration error rose by {result.ece_delta:.3f} to "
            f"{result.expected_calibration_error:.3f}. This is the more serious failure "
            "mode: the threshold is compared against a probability, so a miscalibrated "
            "score makes every rupee figure downstream wrong even where ranking held up."
        )

    losing = [result for result in shifted if result.net_inr_per_1000 <= 0]
    for result in losing:
        findings.append(
            f"{result.environment}: net economics turned non-positive "
            f"(₹{result.net_inr_per_1000:,.0f} per 1,000 orders). Under this shift the "
            "system stops paying for itself at the frozen threshold."
        )

    if not findings:
        findings.append(
            "No environment produced a material drop in ranking quality, calibration or "
            "economics. That is a statement about these perturbations at these "
            "magnitudes, not a general robustness claim."
        )

    stable = [
        result
        for result in shifted
        if result.pr_auc_delta is not None and abs(result.pr_auc_delta) <= MATERIAL_PR_AUC_DROP
    ]
    if stable:
        findings.append(
            f"{len(stable)} of {len(shifted)} shifted environments left PR-AUC within "
            f"{MATERIAL_PR_AUC_DROP} of the reference: "
            + ", ".join(result.environment for result in stable)
            + "."
        )

    return tuple(findings)


def default_environments(*, seed: int, n_orders: int) -> tuple[EnvironmentSpec, ...]:
    """The perturbations named in the specification, as reviewable overrides.

    Every environment changes exactly one thing where possible, because a study
    that shifts three parameters at once cannot say which one the model was
    sensitive to. The two combined environments at the end are deliberate: real
    shifts arrive together, and the question of whether effects compound is worth
    a row of its own.

    Seeds are offset from a single base rather than shared. Reusing one seed
    across environments would correlate their sampling noise, which flatters
    comparisons; deriving them deterministically keeps the study reproducible.
    """
    return (
        EnvironmentSpec(
            name="reference",
            description="the unshifted world, generated with the configured parameters",
            overrides={},
            seed=seed,
            n_orders=n_orders,
        ),
        EnvironmentSpec(
            name="cod_surge",
            description="COD share rises from 62% to 80%, as in a festive-season shift",
            overrides={"payment.cod_share": 0.80},
            seed=seed + 1,
            n_orders=n_orders,
        ),
        EnvironmentSpec(
            name="cod_collapse",
            description="COD share falls to 35% as prepaid adoption accelerates",
            overrides={"payment.cod_share": 0.35},
            seed=seed + 2,
            n_orders=n_orders,
        ),
        EnvironmentSpec(
            name="rto_base_rate_up",
            description="the COD RTO base rate rises from 26% to 38%",
            overrides={"base_rates.rto_given_cod": 0.38},
            seed=seed + 3,
            n_orders=n_orders,
        ),
        EnvironmentSpec(
            name="rto_base_rate_down",
            description="the COD RTO base rate falls from 26% to 15%",
            overrides={"base_rates.rto_given_cod": 0.15},
            seed=seed + 4,
            n_orders=n_orders,
        ),
        EnvironmentSpec(
            name="category_mix_fashion",
            description="the catalogue swings towards fashion, the highest-return category",
            overrides={
                "catalogue.categories.fashion.share": 0.60,
                "catalogue.categories.electronics.share": 0.08,
                "catalogue.categories.beauty.share": 0.12,
                "catalogue.categories.home.share": 0.12,
                "catalogue.categories.accessories.share": 0.08,
            },
            seed=seed + 5,
            n_orders=n_orders,
        ),
        EnvironmentSpec(
            name="customer_mix_new",
            description=("the book fills with first-time customers, so history features go null"),
            overrides={"customers.orders_per_customer_alpha": 4.5},
            seed=seed + 6,
            n_orders=n_orders,
        ),
        EnvironmentSpec(
            name="geography_tier3",
            description="delivery mix shifts towards tier-3 pincodes",
            overrides={
                "geography.tier_shares.tier_1": 0.20,
                "geography.tier_shares.tier_2": 0.30,
                "geography.tier_shares.tier_3": 0.50,
            },
            seed=seed + 7,
            n_orders=n_orders,
        ),
        EnvironmentSpec(
            name="order_value_up",
            description="basket sizes rise; the median order value roughly doubles",
            overrides={"catalogue.order_value.mu": 7.55},
            seed=seed + 8,
            n_orders=n_orders,
        ),
        EnvironmentSpec(
            name="combined_festive",
            description=(
                "COD share, RTO base rate and fashion share all rise together, as they "
                "plausibly would during a festive peak"
            ),
            overrides={
                "payment.cod_share": 0.78,
                "base_rates.rto_given_cod": 0.34,
                "catalogue.categories.fashion.share": 0.48,
                "catalogue.categories.electronics.share": 0.10,
                "catalogue.categories.beauty.share": 0.16,
                "catalogue.categories.home.share": 0.16,
                "catalogue.categories.accessories.share": 0.10,
            },
            seed=seed + 9,
            n_orders=n_orders,
        ),
    )
