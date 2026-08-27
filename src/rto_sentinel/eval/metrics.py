"""Ranking and calibration metrics.

SPEC section 07 and appendix A1. What is here, and what is deliberately not:

* **PR-AUC leads.** It is not inflated by the large negative class.
* **ROC-AUC is computed and reported, but never led with.** It flatters
  imbalanced problems, and a submission that leads with it is telling you what
  it wants you to see.
* **Recall at fixed precision (80%, 90%).** The operationally meaningful form of
  "how good is it" - the question an ops team with a fixed review budget asks.
* **Expected Calibration Error and Brier score, with a reliability diagram.**
  First-class here, because the decision layer is only as honest as the
  probability feeding it.
* **No single accuracy figure.** There is no ``accuracy()`` function in this
  module, and that absence is deliberate. At a one-in-four positive rate,
  accuracy is a number that rewards doing nothing.

STATUS: Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from rto_sentinel.contracts.evaluation import CalibrationMetrics, RankingMetrics


def ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> RankingMetrics:
    """PR-AUC, ROC-AUC, recall at fixed precision, and precision at k."""
    raise NotImplementedError("Ranking metrics land in Phase 2.")


def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> CalibrationMetrics:
    """ECE, Brier score, and the reliability-diagram bins.

    Returns the bins rather than plotting them: this module computes, the report
    builder renders, and the console draws. Keeping plotting out of here means
    the metrics stay testable without a display.
    """
    raise NotImplementedError("Calibration metrics land in Phase 2.")


def recall_at_precision(y_true: np.ndarray, y_prob: np.ndarray, precision: float) -> float | None:
    """Highest recall achievable while holding precision at or above a target.

    Returns None when no threshold achieves that precision. That is a real and
    reportable outcome, not a zero: a model that simply cannot reach 90 percent
    precision should say so, rather than report 0.0 recall and leave a reader to
    assume it tried and failed narrowly.
    """
    raise NotImplementedError("Recall-at-precision lands in Phase 2.")
