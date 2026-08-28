"""Bootstrap confidence intervals.

"A point estimate on 5,000 rows is not a result" - SPEC section 07. So every
headline number in this project ships with an interval, and
:class:`~rto_sentinel.contracts.evaluation.PointEstimate` has no constructor path
that omits one.

Resampling happens at the **order** level with a fixed seed, so intervals are
reproducible run to run. Where a metric is computed within customer groups, the
resample is at the customer level instead: resampling orders inside a correlated
group produces intervals that are too narrow and quietly overstate confidence.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from rto_sentinel.contracts.evaluation import PointEstimate

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: MetricFn,
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> PointEstimate:
    """Percentile bootstrap interval for any metric over ``(y_true, y_prob)``.

    Passing ``groups`` switches to a cluster bootstrap that resamples whole
    groups rather than individual rows.

    ``iterations=0`` returns the point estimate with a degenerate interval. That
    exists for fast unit tests; it is never used for a reported result, and the
    ``n_bootstrap=0`` on the returned object is what makes that visible.
    """
    point = float(metric_fn(y_true, y_prob))

    if iterations <= 0 or not np.isfinite(point):
        return PointEstimate(
            value=point, ci_low=point, ci_high=point, confidence=confidence, n_bootstrap=0
        )

    rng = np.random.default_rng(seed)
    samples: list[float] = []

    if groups is None:
        n = len(y_true)
        for _ in range(iterations):
            index = rng.integers(0, n, size=n)
            value = metric_fn(y_true[index], y_prob[index])
            if np.isfinite(value):
                samples.append(float(value))
    else:
        unique = np.unique(groups)
        positions = {key: np.flatnonzero(groups == key) for key in unique}
        for _ in range(iterations):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            index = np.concatenate([positions[key] for key in chosen])
            value = metric_fn(y_true[index], y_prob[index])
            if np.isfinite(value):
                samples.append(float(value))

    if not samples:
        # Every resample was degenerate - too few positives to compute the metric.
        # Reported as a zero-width interval with n_bootstrap=0 rather than an
        # invented range.
        return PointEstimate(
            value=point, ci_low=point, ci_high=point, confidence=confidence, n_bootstrap=0
        )

    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(samples, alpha))
    high = float(np.quantile(samples, 1.0 - alpha))

    # The percentile interval can exclude the point estimate on a skewed
    # distribution. Widening to contain it is the conservative choice and keeps
    # the PointEstimate contract satisfiable.
    return PointEstimate(
        value=point,
        ci_low=min(low, point),
        ci_high=max(high, point),
        confidence=confidence,
        n_bootstrap=len(samples),
    )
