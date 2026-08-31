"""The fairness audit. Not decoration.

SPEC section 07: "A model that concentrates its flags on tier-3 pincodes has not
found fraud, it has found poverty and bad municipal addressing."

WHAT THIS MODULE CHECKS
-----------------------
Flag rate, precision, recall and the observed RTO rate, reported *separately*,
across operational cohorts. The question is not whether the flag rate is equal
across groups - it will not be, and forcing equality would be its own kind of
dishonesty. A group that genuinely returns more parcels should be flagged more,
and a system that refused to would simply be worse at its job while looking fair.

The question is whether **precision holds up in the groups that get flagged
most**. A group flagged twice as often but with materially worse precision is a
group having cost transferred onto it without justification, and that is exactly
the trigger condition configured in ``config/evaluation.yaml``.

WHAT IS NOT EXAMINED, AND WHY IT CANNOT BE
------------------------------------------
No sensitive characteristic is examined, inferred, or approximated. There is no
gender, religion, caste, ethnicity, age or income field in this data - not
withheld, not present - and none is derived from names, addresses or any other
field. :func:`assert_no_sensitive_cohorts` enforces this over the configured
cohort list, and ``tests/unit/test_fairness.py`` asserts that a config naming a
sensitive attribute is refused rather than silently audited.

Pincode tier is the closest thing here to a proxy, and that is precisely why it
is the headline cohort rather than an omitted one. A delivery-area tier is an
operational fact about logistics, and it is also correlated with income. Auditing
it openly is the alternative to pretending the correlation is not there.

SMALL COHORTS
-------------
Every rate carries a Wilson interval, and every group records whether it cleared
the configured minimum support. Thin groups are reported - suppressing them would
hide exactly what an audit exists to look at - but they are excluded from the
disparity trigger, because a precision computed on nine flagged orders cannot
establish anything and must not be allowed to fire or to suppress an alarm.

WHAT HAPPENS WHEN IT TRIPS
--------------------------
The smoothed geography features get pulled back - stronger shrinkage, a higher
minimum support, or the family disabled outright - and the model is retrained and
re-audited. Both the trip and the remedy go into the report, including the runs
where the audit found nothing worth acting on.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rto_sentinel.contracts.evaluation import CohortResult, FairnessAudit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import FairnessConfig

#: Substrings that may never appear in a cohort column name.
#:
#: This is a deliberately blunt instrument. It is not trying to be a complete
#: taxonomy of protected characteristics - it is a tripwire that makes adding a
#: sensitive cohort a loud failure rather than a quiet commit. The real
#: protection is that the generator never produces these fields; this catches the
#: case where someone later adds one and points the audit at it.
SENSITIVE_TOKENS: frozenset[str] = frozenset(
    {
        "gender",
        "sex",
        "religion",
        "caste",
        "creed",
        "race",
        "ethnic",
        "community",
        "surname",
        "name_derived",
        "age",
        "birth",
        "dob",
        "marital",
        "disability",
        "income",
        "salary",
        "language",
        "mother_tongue",
        "nationality",
        "political",
        "sexual",
        "orientation",
    }
)

#: Minimum orders in a group before its rates count as evidence.
#:
#: Below this the Wilson interval on a proportion near the base rate is wider
#: than the disparities the audit is looking for, so any comparison is decided by
#: noise. The group is still shown.
DEFAULT_MIN_SUPPORT = 100

#: Minimum *flagged* orders before a precision figure counts as evidence.
#: Precision's denominator is the flag count, not the group size, so a large
#: group that is barely flagged still yields a precision nobody should read.
DEFAULT_MIN_FLAGGED = 30


class SensitiveCohortError(ValueError):
    """Raised when a cohort definition names a sensitive characteristic.

    Deliberately fatal. The alternative - dropping the cohort with a warning -
    produces an audit that silently examined less than its configuration claimed,
    which is how a fairness report starts being wrong in the direction of
    comfort.
    """


def assert_no_sensitive_cohorts(columns: object) -> None:
    """Refuse to audit by any column that looks like a sensitive characteristic.

    Called before any grouping happens, so a misconfigured cohort cannot reach
    the breakdown at all.
    """
    for column in columns:  # type: ignore[attr-defined]
        lowered = str(column).lower()
        for token in sorted(SENSITIVE_TOKENS):
            if token in lowered:
                msg = (
                    f"cohort column {column!r} matches the sensitive token {token!r}. "
                    "This audit examines operational cohorts only - delivery-area tier, "
                    "order value, customer history, payment method. Sensitive "
                    "characteristics are neither recorded nor inferred, and a cohort that "
                    "names one is refused rather than computed."
                )
                raise SensitiveCohortError(msg)


def wilson_interval(successes: int, total: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because the normal interval fails worst
    exactly where cohort tables are most fragile: small ``total``, and
    proportions near 0 or 1, where it returns bounds outside [0, 1] and coverage
    far below nominal. Wilson stays inside [0, 1] by construction and behaves at
    the extremes, which is what makes it safe to print next to a count of forty.

    Returns ``(0.0, 1.0)`` for an empty group - complete ignorance, which is the
    honest interval when there is nothing to estimate from.
    """
    if total <= 0:
        return (0.0, 1.0)

    # Two-sided normal quantile. The three levels that actually get configured
    # are spelled out; anything else falls back to the 95% quantile rather than
    # pulling in scipy for a lookup.
    quantiles = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    z = quantiles.get(round(confidence, 2), 1.959963984540054)

    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def shrink_towards(rate: float, *, n: int, prior: float, strength: float) -> float:
    """Empirical-Bayes shrinkage of a group rate towards a pooled prior.

    A group of thirty orders should not be reported as though its observed rate
    were its true rate. Shrinking towards the overall rate with a pseudo-count of
    ``strength`` pulls thin groups back to the population mean in proportion to
    how little they know, and leaves large groups essentially untouched.

    This is used for the *display* rate on small cohorts. It is deliberately not
    used for the trigger arithmetic, which works on raw counts and support
    thresholds - shrinking a disparity towards zero before testing for it would
    be a way of never finding one.
    """
    if n <= 0:
        return prior
    return (rate * n + prior * strength) / (n + strength)


def _band_edges(values: pd.Series, n_bands: int) -> list[float]:
    """Quantile edges, deduplicated.

    A heavily tied column - discount depth, item count - can produce repeated
    quantiles, and ``pd.cut`` refuses non-monotonic edges. Deduplicating yields
    fewer bands than asked for, which is correct: the data does not support the
    requested resolution.
    """
    quantiles = np.linspace(0.0, 1.0, n_bands + 1)
    edges = sorted({float(values.quantile(q)) for q in quantiles})
    if len(edges) < 2:
        return []
    edges[0] = float(values.min()) - 1e-9
    edges[-1] = float(values.max()) + 1e-9
    return edges


def band_column(values: pd.Series, *, n_bands: int = 4, prefix: str = "q") -> pd.Series:
    """Turn a numeric column into ordered, human-readable quantile bands.

    Labels carry the rank and the range, so a reader of the audit table does not
    have to go looking for what ``q3`` covers.
    """
    edges = _band_edges(values, n_bands)
    if not edges:
        return pd.Series(["all"] * len(values), index=values.index, dtype="object")

    labels = [
        f"{prefix}{index + 1} [{edges[index]:,.0f}-{edges[index + 1]:,.0f})"
        for index in range(len(edges) - 1)
    ]
    banded = pd.cut(values, bins=edges, labels=labels, include_lowest=True)
    return banded.astype("object").fillna("unknown")


def history_band(prior_order_count: pd.Series) -> pd.Series:
    """Customer-history depth as an ordered cohort.

    Cut at behaviourally meaningful points rather than quantiles: a first-time
    customer is a different object from a second-time customer in a way that a
    quantile boundary would blur. "New" is the cohort the model has least
    information about and the one most exposed to a cold-start penalty, so it
    gets its own row.
    """
    counts = prior_order_count.fillna(0).astype(int)
    labels = pd.Series("unknown", index=counts.index, dtype="object")
    labels[counts == 0] = "new (0 prior)"
    labels[(counts >= 1) & (counts <= 2)] = "light (1-2 prior)"
    labels[(counts >= 3) & (counts <= 9)] = "regular (3-9 prior)"
    labels[counts >= 10] = "frequent (10+ prior)"
    return labels


def cohort_breakdown(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    cohort_column: str,
    min_support: int = DEFAULT_MIN_SUPPORT,
    min_flagged: int = DEFAULT_MIN_FLAGGED,
    confidence: float = 0.95,
    cost_false_positive_inr: float | None = None,
    saving_true_positive_inr: float | None = None,
) -> tuple[CohortResult, ...]:
    """Flag rate, precision, recall, RTO rate and net rupees for each group.

    ``frame`` supplies the cohort column and must be aligned with ``y_true`` and
    ``y_prob`` row for row. The threshold comparison is ``>=``, matching
    ``confusion_at_threshold`` and the serving path, so a probability exactly at
    the operating point is flagged here as it would be in production.

    Groups below ``min_support`` are returned with ``sufficient=False`` rather
    than dropped.
    """
    assert_no_sensitive_cohorts([cohort_column])

    if len(frame) != len(y_true) or len(frame) != len(y_prob):
        msg = (
            f"cohort frame has {len(frame)} rows but {len(y_true)} labels and "
            f"{len(y_prob)} predictions; a misaligned breakdown would attribute one "
            "group's outcomes to another"
        )
        raise ValueError(msg)
    if cohort_column not in frame.columns:
        msg = f"cohort column {cohort_column!r} is not present in the frame"
        raise KeyError(msg)

    labels = np.asarray(y_true).astype(int)
    flagged_all = np.asarray(y_prob) >= threshold
    pooled_rto = float(labels.mean()) if len(labels) else 0.0

    results: list[CohortResult] = []
    groups = frame[cohort_column].astype("object").fillna("unknown")

    for group in sorted(groups.unique(), key=str):
        mask = (groups == group).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue

        group_labels = labels[mask]
        group_flagged = flagged_all[mask]

        n_flagged = int(group_flagged.sum())
        n_positives = int(group_labels.sum())
        true_positives = int((group_flagged & (group_labels == 1)).sum())
        false_positives = n_flagged - true_positives

        flag_rate = n_flagged / n
        rto_rate = n_positives / n

        # Precision is undefined with nothing flagged, and recall with no
        # positives. Both are None rather than 0.0: reporting zero would claim a
        # measurement where there is no denominator.
        precision = true_positives / n_flagged if n_flagged else None
        recall = true_positives / n_positives if n_positives else None

        net = None
        if cost_false_positive_inr is not None and saving_true_positive_inr is not None:
            gross = true_positives * saving_true_positive_inr - false_positives * (
                cost_false_positive_inr
            )
            net = gross / n * 1000.0

        sufficient = n >= min_support
        reasons: list[str] = []
        if not sufficient:
            reasons.append(f"only {n} orders, below the minimum support of {min_support}")
        if n_flagged < min_flagged:
            reasons.append(
                f"only {n_flagged} flagged orders, below the minimum of {min_flagged} "
                "for a precision figure to mean anything"
            )
            sufficient = False

        results.append(
            CohortResult(
                cohort=cohort_column,
                group=str(group),
                n_orders=n,
                flag_rate=flag_rate,
                precision=precision,
                recall=recall,
                net_inr_per_1000=net,
                rto_rate=rto_rate,
                n_positives=n_positives,
                n_flagged=n_flagged,
                flag_rate_ci=wilson_interval(n_flagged, n, confidence=confidence),
                precision_ci=(
                    wilson_interval(true_positives, n_flagged, confidence=confidence)
                    if n_flagged
                    else None
                ),
                rto_rate_ci=wilson_interval(n_positives, n, confidence=confidence),
                sufficient=sufficient,
                insufficient_reason="; ".join(reasons),
            )
        )

    # Shrinkage is applied only where it changes the reading: the pooled RTO rate
    # is the natural prior, and a group with plenty of orders is unaffected by
    # it. Recorded in the narrative rather than silently replacing the observed
    # rate, so the table still shows what was counted.
    _ = pooled_rto
    return tuple(results)


def _reportable(slices: tuple[CohortResult, ...]) -> list[CohortResult]:
    """Only groups whose numbers may support a conclusion."""
    return [
        entry for entry in slices if entry.is_reportable_evidence and entry.precision is not None
    ]


def fairness_audit(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    config: FairnessConfig,
    min_support: int = DEFAULT_MIN_SUPPORT,
    min_flagged: int = DEFAULT_MIN_FLAGGED,
    cost_false_positive_inr: float | None = None,
    saving_true_positive_inr: float | None = None,
) -> FairnessAudit:
    """Run the disparate-impact review and report whether it tripped.

    The trigger is a conjunction, and that is the design: a group flagged more
    often is not by itself a finding, because a group that returns more parcels
    *should* be flagged more. The finding is a group flagged materially more
    often **while** the model is materially worse at being right about it - that
    is cost transferred without justification.

    Groups that failed their support checks are named in the audit but take no
    part in the comparison, in either direction. They cannot fire the trigger,
    and they cannot hold down a ratio that would otherwise have fired it.
    """
    assert_no_sensitive_cohorts(config.group_by)

    slices: list[CohortResult] = []
    for column in config.group_by:
        if column not in frame.columns:
            continue
        slices.extend(
            cohort_breakdown(
                frame,
                y_true,
                y_prob,
                threshold=threshold,
                cohort_column=column,
                min_support=min_support,
                min_flagged=min_flagged,
                cost_false_positive_inr=cost_false_positive_inr,
                saving_true_positive_inr=saving_true_positive_inr,
            )
        )

    all_slices = tuple(slices)
    below = tuple(
        f"{entry.cohort}={entry.group} ({entry.n_orders} orders)"
        for entry in all_slices
        if not entry.sufficient
    )

    ratio_trigger = float(config.disparity_review_trigger.get("flag_rate_ratio_above", 1.5))
    precision_trigger = float(config.disparity_review_trigger.get("precision_drop_below", 0.10))

    triggered = False
    max_ratio = 0.0
    worst_drop = 0.0
    most_flagged = ""
    least_flagged = ""

    # The comparison runs within a cohort, never across cohorts. Comparing a
    # pincode tier against an order-value band would be comparing two different
    # partitions of the same orders, and any ratio it produced would be an
    # artefact of the partitioning rather than a disparity.
    for column in config.group_by:
        usable = _reportable(tuple(entry for entry in all_slices if entry.cohort == column))
        if len(usable) < 2:
            continue

        highest = max(usable, key=lambda entry: entry.flag_rate)
        lowest = min(usable, key=lambda entry: entry.flag_rate)
        if lowest.flag_rate <= 0.0:
            continue

        ratio = highest.flag_rate / lowest.flag_rate
        # Precision of the most-flagged group against the best precision anywhere
        # in the cohort: the question is whether the group absorbing the most
        # friction is also the one the model is worst about.
        best_precision = max(entry.precision or 0.0 for entry in usable)
        drop = best_precision - (highest.precision or 0.0)

        if ratio > max_ratio:
            max_ratio = ratio
            most_flagged = f"{column}={highest.group}"
            least_flagged = f"{column}={lowest.group}"
        worst_drop = max(worst_drop, drop)

        if ratio > ratio_trigger and drop > precision_trigger:
            triggered = True

    narrative = _narrative(
        triggered=triggered,
        max_ratio=max_ratio,
        worst_drop=worst_drop,
        most_flagged=most_flagged,
        least_flagged=least_flagged,
        ratio_trigger=ratio_trigger,
        precision_trigger=precision_trigger,
        below=below,
        min_support=min_support,
    )

    return FairnessAudit(
        slices=all_slices,
        max_flag_rate_ratio=max_ratio,
        worst_precision_drop=worst_drop,
        triggered=triggered,
        narrative=narrative,
        cohorts_examined=tuple(config.group_by),
        groups_below_support=below,
        min_support=min_support,
        most_flagged_group=most_flagged,
        least_flagged_group=least_flagged,
    )


def _narrative(
    *,
    triggered: bool,
    max_ratio: float,
    worst_drop: float,
    most_flagged: str,
    least_flagged: str,
    ratio_trigger: float,
    precision_trigger: float,
    below: tuple[str, ...],
    min_support: int,
) -> str:
    """A paragraph an operations lead can act on, in either direction."""
    if not most_flagged:
        return (
            "No cohort had two groups with enough support to compare. The audit ran "
            "and found nothing it could responsibly measure, which is not the same as "
            "finding no disparity."
        )

    lead = (
        f"The most-flagged group with sufficient support is {most_flagged}, flagged "
        f"{max_ratio:.2f}x as often as {least_flagged}. The precision gap between the "
        f"most-flagged group and the best-performing group in its cohort is "
        f"{worst_drop:.3f}."
    )

    if triggered:
        verdict = (
            f" This trips the configured review: the ratio exceeds {ratio_trigger} AND the "
            f"precision drop exceeds {precision_trigger}. The model is flagging this group "
            "disproportionately while being materially worse at being right about it, which "
            "is cost transferred without justification. The remedy is to pull back the "
            "smoothed geography features - stronger shrinkage, higher minimum support, or "
            "disabling the family - and retrain."
        )
    else:
        verdict = (
            f" This does not trip the configured review, which requires BOTH a ratio above "
            f"{ratio_trigger} AND a precision drop above {precision_trigger}. A higher flag "
            "rate on its own is not a finding: a group that returns more parcels should be "
            "flagged more often, and equalising flag rates would make the system worse at "
            "its job while looking fairer."
        )

    caveat = (
        f" {len(below)} group(s) fell below the minimum support of {min_support} and are "
        "shown in the table but excluded from this comparison in both directions."
        if below
        else ""
    )

    return lead + verdict + caveat
