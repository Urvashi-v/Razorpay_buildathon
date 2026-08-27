"""Bootstrap confidence intervals.

"A point estimate on 5,000 rows is not a result" - SPEC section 07. So every
headline number in this project ships with an interval, and
:class:`~rto_sentinel.contracts.evaluation.PointEstimate` has no constructor path
that omits one.

Resampling happens at the **order** level with a fixed seed, so intervals are
reproducible run to run. Where a metric is computed within customer groups, the
resample is at the customer level instead: resampling orders inside a correlated
group produces intervals that are too narrow and quietly overstate confidence.

STATUS: Phase 2.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from rto_sentinel.contracts.evaluation import PointEstimate


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> PointEstimate:
    """Percentile bootstrap interval for any metric over ``(y_true, y_prob)``.

    Passing ``groups`` switches to a cluster bootstrap that resamples whole
    groups rather than individual rows.
    """
    raise NotImplementedError("Bootstrap intervals land in Phase 2.")
