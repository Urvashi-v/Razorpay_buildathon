"""Drift measurement. Distances, not verdicts.

WHAT DRIFT IS AND IS NOT
========================
Drift means an input or output distribution moved between two windows. It does
**not** mean the model is broken, and this module is built so that it cannot
accidentally claim otherwise.

Every function here returns a distance and a severity band. None of them returns
a pass/fail, and :class:`DriftSignal` has no field in which to record one.
Whether quality actually degraded is answered by :func:`performance_delta`,
which needs labels and refuses to guess when it does not have them.

The distinction is not pedantic. Indian e-commerce moves hard during festive
season: COD share rises, order values rise, category mix swings towards fashion
and electronics. Every one of those shows up here as real drift, and none of them
means the risk model stopped working. A monitor that pages someone every Diwali
gets muted by March, and then it is worth nothing in the month it matters.

WHY PSI FOR DISTRIBUTIONS AND ABSOLUTE DIFFERENCE FOR RATES
===========================================================
PSI compares binned distributions and is the standard in credit risk, so its
bands (0.1 / 0.25) are numbers a risk reviewer already has intuitions about.
Applied to a single rate, though, PSI is hard to read - nobody knows what a PSI
of 0.04 on a flag rate means. For rates the honest statistic is the difference
itself: "the flag rate went from 15.9% to 22.4%" needs no interpretation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rto_sentinel.contracts.monitoring import (
    DriftKind,
    DriftSeverity,
    DriftSignal,
    PerformanceDelta,
)
from rto_sentinel.eval.metrics import expected_calibration_error

#: PSI bands, from the credit-risk convention.
#:
#: Below 0.10 the distribution is materially the same. 0.10-0.25 is worth an eye.
#: Above 0.25 something changed enough to go and look at. These are conventional
#: reading aids, not thresholds derived from this data - which is why they select
#: the words "watch" and "investigate" rather than "warn" and "fail".
PSI_WATCH = 0.10
PSI_INVESTIGATE = 0.25

#: Rate-difference bands, in absolute percentage points.
#:
#: Five points on a rate near 20% is a real move; ten is a large one. Chosen to
#: be legible rather than to hit a false-positive target that no labelled history
#: exists to calibrate against - and stated as such rather than dressed up.
RATE_WATCH = 0.05
RATE_INVESTIGATE = 0.10

#: Below this many rows in either window, a distance is not reported as evidence.
#: PSI on a handful of rows is dominated by which bin the rounding fell into.
MIN_WINDOW_ROWS = 50

#: Bins for the PSI histogram. Quantile edges come from the baseline, so the
#: baseline is by construction uniform across bins and the statistic measures how
#: far the current window departs from it.
PSI_BINS = 10

#: Laplace-style floor so an empty bin does not send PSI to infinity.
#:
#: An empty current bin is genuine information - a value range stopped occurring
#: - and it should push the distance up. It should not make it undefined, which
#: is what log(0) does. The floor is small enough not to mask a real emptiness.
EPSILON = 1e-6


def _severity(distance: float, watch: float, investigate: float) -> DriftSeverity:
    if distance >= investigate:
        return "investigate"
    if distance >= watch:
        return "watch"
    return "stable"


def population_stability_index(
    baseline: np.ndarray, current: np.ndarray, *, bins: int = PSI_BINS
) -> float:
    """PSI between two numeric samples, binned on baseline quantiles.

    Returns 0.0 when the baseline has no variation to bin - a constant feature
    cannot drift in shape, only in level, and the level is reported separately as
    the mean.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]

    if baseline.size == 0 or current.size == 0:
        return 0.0

    edges = np.unique(np.quantile(baseline, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return 0.0
    # Open the outer edges so current values outside the baseline range land in
    # the end bins rather than being dropped. A value the baseline never saw is
    # exactly the kind of drift this is supposed to catch.
    edges[0] = -np.inf
    edges[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    baseline_share = np.maximum(baseline_counts / baseline.size, EPSILON)
    current_share = np.maximum(current_counts / current.size, EPSILON)

    return float(np.sum((current_share - baseline_share) * np.log(current_share / baseline_share)))


def categorical_psi(baseline: pd.Series, current: pd.Series) -> float:
    """PSI over category shares, using the union of both windows' categories.

    Taking the union matters: a category that appears only in the current window
    is drift, and scoring it against the baseline's category list alone would
    silently discard it.
    """
    baseline_counts = baseline.astype("object").fillna("unknown").value_counts()
    current_counts = current.astype("object").fillna("unknown").value_counts()
    if baseline_counts.sum() == 0 or current_counts.sum() == 0:
        return 0.0

    categories = sorted(set(baseline_counts.index) | set(current_counts.index), key=str)
    total = 0.0
    for category in categories:
        baseline_share = max(
            float(baseline_counts.get(category, 0)) / float(baseline_counts.sum()), EPSILON
        )
        current_share = max(
            float(current_counts.get(category, 0)) / float(current_counts.sum()), EPSILON
        )
        total += (current_share - baseline_share) * math.log(current_share / baseline_share)
    return float(total)


def kolmogorov_smirnov(baseline: np.ndarray, current: np.ndarray) -> float:
    """Two-sample KS statistic: the largest gap between the two ECDFs.

    Reported alongside PSI for prediction drift because it is binning-free. PSI
    can be nudged by where the bin edges fell; KS cannot, so agreement between
    the two is a useful sign the move is real.
    """
    baseline = np.sort(np.asarray(baseline, dtype=float))
    current = np.sort(np.asarray(current, dtype=float))
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]
    if baseline.size == 0 or current.size == 0:
        return 0.0

    grid = np.concatenate([baseline, current])
    baseline_cdf = np.searchsorted(baseline, grid, side="right") / baseline.size
    current_cdf = np.searchsorted(current, grid, side="right") / current.size
    return float(np.max(np.abs(baseline_cdf - current_cdf)))


def feature_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    min_rows: int = MIN_WINDOW_ROWS,
) -> tuple[DriftSignal, ...]:
    """PSI per feature, numeric and categorical, over the shared columns.

    Columns present in only one window are skipped rather than treated as
    infinite drift: that is a schema change, which is a different problem with a
    different fix, and reporting it as drift would bury the actual distribution
    moves underneath it.
    """
    shared = [
        column
        for column in (columns or list(baseline.columns))
        if column in baseline.columns and column in current.columns
    ]

    signals: list[DriftSignal] = []
    sufficient = len(baseline) >= min_rows and len(current) >= min_rows

    for column in sorted(shared):
        baseline_column = baseline[column]
        current_column = current[column]

        if pd.api.types.is_numeric_dtype(baseline_column) and pd.api.types.is_numeric_dtype(
            current_column
        ):
            distance = population_stability_index(
                baseline_column.to_numpy(dtype=float), current_column.to_numpy(dtype=float)
            )
            baseline_value = float(baseline_column.mean()) if len(baseline_column) else None
            current_value = float(current_column.mean()) if len(current_column) else None
        else:
            distance = categorical_psi(baseline_column, current_column)
            baseline_value = None
            current_value = None

        signals.append(
            DriftSignal(
                name=column,
                kind="feature",
                statistic="psi",
                distance=distance,
                severity=_severity(distance, PSI_WATCH, PSI_INVESTIGATE)
                if sufficient
                else "stable",
                baseline_value=baseline_value,
                current_value=current_value,
                baseline_n=len(baseline_column),
                current_n=len(current_column),
                sufficient=sufficient,
                note=(
                    ""
                    if sufficient
                    else f"fewer than {min_rows} rows in one window; distance reported but not read"
                ),
            )
        )

    return tuple(signals)


def prediction_drift(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    min_rows: int = MIN_WINDOW_ROWS,
) -> tuple[DriftSignal, ...]:
    """How the score distribution itself moved.

    This is the signal that needs no labels and is available immediately, which
    makes it the first thing worth looking at when something is suspected. A
    score distribution that shifts while the input features have not is a sign
    the model is being fed something it is reading differently - often a feature
    that silently became null.
    """
    baseline_scores = np.asarray(baseline_scores, dtype=float)
    current_scores = np.asarray(current_scores, dtype=float)
    sufficient = baseline_scores.size >= min_rows and current_scores.size >= min_rows

    psi = population_stability_index(baseline_scores, current_scores)
    ks = kolmogorov_smirnov(baseline_scores, current_scores)

    baseline_mean = float(baseline_scores.mean()) if baseline_scores.size else None
    current_mean = float(current_scores.mean()) if current_scores.size else None

    return (
        DriftSignal(
            name="predicted_probability",
            kind="prediction",
            statistic="psi",
            distance=psi,
            severity=_severity(psi, PSI_WATCH, PSI_INVESTIGATE) if sufficient else "stable",
            baseline_value=baseline_mean,
            current_value=current_mean,
            baseline_n=int(baseline_scores.size),
            current_n=int(current_scores.size),
            sufficient=sufficient,
        ),
        DriftSignal(
            name="predicted_probability",
            kind="prediction",
            statistic="ks",
            # KS bands differ from PSI bands - it is a different statistic on a
            # different scale, and reusing PSI's numbers would be meaningless.
            distance=ks,
            severity=_severity(ks, 0.10, 0.20) if sufficient else "stable",
            baseline_value=baseline_mean,
            current_value=current_mean,
            baseline_n=int(baseline_scores.size),
            current_n=int(current_scores.size),
            sufficient=sufficient,
        ),
    )


def rate_drift(
    name: str,
    kind: DriftKind,
    baseline_successes: int,
    baseline_total: int,
    current_successes: int,
    current_total: int,
    *,
    min_rows: int = MIN_WINDOW_ROWS,
) -> DriftSignal:
    """Absolute movement in a rate - RTO rate, flag rate.

    The statistic is the plain difference in percentage points, because that is
    what a person reading a monitoring page can act on. "The RTO rate went from
    16.7% to 24.1%" is a sentence; "the RTO rate has a PSI of 0.08" is not.
    """
    baseline_rate = baseline_successes / baseline_total if baseline_total else 0.0
    current_rate = current_successes / current_total if current_total else 0.0
    distance = abs(current_rate - baseline_rate)
    sufficient = baseline_total >= min_rows and current_total >= min_rows

    return DriftSignal(
        name=name,
        kind=kind,
        statistic="absolute_difference",
        distance=distance,
        severity=_severity(distance, RATE_WATCH, RATE_INVESTIGATE) if sufficient else "stable",
        baseline_value=baseline_rate,
        current_value=current_rate,
        baseline_n=baseline_total,
        current_n=current_total,
        sufficient=sufficient,
        note=(
            ""
            if sufficient
            else (
                f"one window has fewer than {min_rows} rows with a known outcome; "
                "the difference is shown but is not evidence"
            )
        ),
    )


def calibration_drift(
    baseline_scores: np.ndarray,
    baseline_labels: np.ndarray,
    current_scores: np.ndarray,
    current_labels: np.ndarray,
    *,
    bins: int = 10,
    min_rows: int = MIN_WINDOW_ROWS,
) -> DriftSignal:
    """Movement in expected calibration error, measured only on mature rows.

    Calibration drift is the one that actually invalidates the economics. The
    threshold is derived as ``C_fp / (C_fp + S_tp)`` and compared against a
    probability; if that probability stops meaning what it says, the comparison
    is arithmetic on a number that no longer denotes anything, and every rupee
    figure downstream is wrong.

    This needs labels on both sides. Where the current window has not matured, it
    reports insufficiency rather than a reassuring number.
    """
    baseline_scores = np.asarray(baseline_scores, dtype=float)
    current_scores = np.asarray(current_scores, dtype=float)
    baseline_labels = np.asarray(baseline_labels).astype(int)
    current_labels = np.asarray(current_labels).astype(int)

    sufficient = baseline_scores.size >= min_rows and current_scores.size >= min_rows
    if not sufficient:
        return DriftSignal(
            name="expected_calibration_error",
            kind="calibration",
            statistic="absolute_difference",
            distance=0.0,
            severity="stable",
            baseline_n=int(baseline_scores.size),
            current_n=int(current_scores.size),
            sufficient=False,
            note=(
                f"fewer than {min_rows} matured orders in one window. Calibration cannot "
                "be measured without outcomes, and it is not assumed to be unchanged."
            ),
        )

    # `expected_calibration_error` returns (error, bins); only the error is
    # compared here, and the bins belong to the reliability diagram.
    baseline_ece, _ = expected_calibration_error(baseline_labels, baseline_scores, n_bins=bins)
    current_ece, _ = expected_calibration_error(current_labels, current_scores, n_bins=bins)
    distance = abs(current_ece - baseline_ece)

    return DriftSignal(
        name="expected_calibration_error",
        kind="calibration",
        statistic="absolute_difference",
        distance=distance,
        # Tighter bands than a raw rate: ECE is already a mean absolute gap, so a
        # two-point move in it is a larger event than a two-point move in a rate.
        severity=_severity(distance, 0.02, 0.05),
        baseline_value=baseline_ece,
        current_value=current_ece,
        baseline_n=int(baseline_scores.size),
        current_n=int(current_scores.size),
        sufficient=True,
    )


def performance_delta(
    metric: str,
    baseline: float,
    current: float,
    *,
    n_baseline_matured: int,
    n_current_matured: int,
    min_rows: int = MIN_WINDOW_ROWS,
) -> PerformanceDelta:
    """A measured change in model quality. Requires labels on both sides."""
    sufficient = n_baseline_matured >= min_rows and n_current_matured >= min_rows
    return PerformanceDelta(
        metric=metric,
        baseline=baseline,
        current=current,
        delta=current - baseline,
        n_baseline_matured=n_baseline_matured,
        n_current_matured=n_current_matured,
        sufficient=sufficient,
        note=(
            ""
            if sufficient
            else (
                f"fewer than {min_rows} matured orders in one window; this comparison "
                "cannot distinguish a real change from sampling noise"
            )
        ),
    )
