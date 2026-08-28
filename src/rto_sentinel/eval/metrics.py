"""Ranking, threshold and calibration metrics.

SPEC section 07 and appendix A1. What is here, and what is deliberately not:

* **PR-AUC leads.** It is not inflated by the large negative class.
* **ROC-AUC is computed and reported, but never led with.** It flatters
  imbalanced problems, and a submission that leads with it is telling you what it
  wants you to see.
* **Recall at fixed precision (80%, 90%).** The operationally meaningful form of
  "how good is it" - the question an ops team with a fixed review budget asks.
* **Expected Calibration Error and Brier score.** Computed here as a *diagnostic*
  from Phase 4 onward. Fixing calibration is Phase 5 work; measuring it now is
  what tells us whether it needs fixing, and for raw boosting output it will.
* **No single accuracy figure.** There is no ``accuracy()`` function in this
  module, and that absence is deliberate. At a one-in-four positive rate,
  accuracy is a number that rewards doing nothing.

A NOTE ON CONSTANT PREDICTORS
=============================
Rung 0 predicts the same value for every row. Several metrics are undefined for
it - ROC-AUC has no meaningful ranking to score, recall-at-precision has no
threshold that achieves any given precision. Those return NaN or None rather than
0.0. A zero would read as "measured, and bad"; the truth is "not defined for this
predictor", and the ladder table shows the difference.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from rto_sentinel.contracts.evaluation import CalibrationMetrics, RankingMetrics
from rto_sentinel.eval.bootstrap import bootstrap_metric


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Counts at one operating threshold, plus the rates derived from them."""

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    threshold: float

    @property
    def n(self) -> int:
        return (
            self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        )

    @property
    def n_flagged(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def flag_rate(self) -> float:
        """Share of all orders receiving friction. Reported always.

        SPEC section 07: a model that flags 40% of orders is unusable regardless
        of its precision, so precision is never quoted without this.
        """
        return self.n_flagged / self.n if self.n else float("nan")

    @property
    def precision(self) -> float:
        """Of the orders we flagged, what share really were RTOs.

        NaN when nothing was flagged - which is rung 0's situation, and is a
        different statement from "precision zero".
        """
        return self.true_positives / self.n_flagged if self.n_flagged else float("nan")

    @property
    def recall(self) -> float:
        positives = self.true_positives + self.false_negatives
        return self.true_positives / positives if positives else float("nan")

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            return float("nan")
        return 2 * precision * recall / (precision + recall)

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "true_positives": float(self.true_positives),
            "false_positives": float(self.false_positives),
            "false_negatives": float(self.false_negatives),
            "true_negatives": float(self.true_negatives),
            "flag_rate": self.flag_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def confusion_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> ConfusionMatrix:
    """Confusion counts for ``y_prob >= threshold``.

    Inclusive ``>=`` matches the decision rule "flag when p is at least the
    threshold". The difference from ``>`` matters for the heuristic rungs, whose
    scores are exactly 0.0 and 1.0.
    """
    predicted = y_prob >= threshold
    actual = y_true.astype(bool)
    return ConfusionMatrix(
        true_positives=int(np.sum(predicted & actual)),
        false_positives=int(np.sum(predicted & ~actual)),
        false_negatives=int(np.sum(~predicted & actual)),
        true_negatives=int(np.sum(~predicted & ~actual)),
        threshold=float(threshold),
    )


def _is_constant(y_prob: np.ndarray) -> bool:
    return bool(np.allclose(y_prob, y_prob[0])) if y_prob.size else True


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the precision-recall curve. The primary ranking metric.

    For a constant predictor this equals the positive base rate, which is the
    correct and informative answer: it is the PR-AUC floor every other rung must
    beat to have demonstrated any ranking ability at all.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Reported, never led with. NaN for a constant predictor."""
    if len(np.unique(y_true)) < 2 or _is_constant(y_prob):
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def recall_at_precision(
    y_true: np.ndarray, y_prob: np.ndarray, target_precision: float
) -> float | None:
    """Highest recall achievable while holding precision at or above a target.

    Returns None when no threshold achieves that precision. That is a real and
    reportable outcome, not a zero: a model that simply cannot reach 90 percent
    precision should say so, rather than report 0.0 recall and leave a reader to
    assume it tried and failed narrowly.
    """
    if len(np.unique(y_true)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    achievable = precision >= target_precision
    if not achievable.any():
        return None
    best = float(np.max(recall[achievable]))
    # sklearn appends a (precision=1, recall=0) sentinel point that is not a real
    # operating point. A best recall of exactly zero means only that sentinel
    # qualified, so no usable threshold reaches the target.
    return best if best > 0.0 else None


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float) -> float:
    """Precision among the top ``k`` fraction of orders by score.

    The question an ops team with a fixed review budget actually asks: "if I can
    look at 5% of orders, how many of them are worth looking at?"
    """
    n = max(round(k * len(y_prob)), 1)
    top = np.argsort(-y_prob, kind="stable")[:n]
    return float(np.mean(y_true[top]))


def ranking_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    bootstrap_iterations: int = 0,
    confidence: float = 0.95,
    seed: int = 0,
) -> RankingMetrics:
    """PR-AUC, ROC-AUC, recall at fixed precision, and precision at k.

    ``bootstrap_iterations=0`` skips resampling and returns degenerate intervals.
    That is for fast unit tests only; every reported result uses a real interval,
    because a point estimate on a few thousand rows is not a result.
    """
    pr = bootstrap_metric(
        y_true, y_prob, pr_auc, iterations=bootstrap_iterations, confidence=confidence, seed=seed
    )
    roc = bootstrap_metric(
        y_true, y_prob, roc_auc, iterations=bootstrap_iterations, confidence=confidence, seed=seed
    )
    return RankingMetrics(
        pr_auc=pr,
        roc_auc=roc,
        recall_at_precision_80=recall_at_precision(y_true, y_prob, 0.80),
        recall_at_precision_90=recall_at_precision(y_true, y_prob, 0.90),
        precision_at_k={f"{k:.2f}": precision_at_k(y_true, y_prob, k) for k in (0.01, 0.05, 0.10)},
    )


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10
) -> tuple[float, tuple[tuple[float, float, int], ...]]:
    """Mean absolute gap between predicted probability and observed frequency.

    Equal-width bins over ``[0, 1]``, weighted by bin population. Returns the
    error and the bins, so the caller can draw a reliability diagram without
    recomputing anything.

    Empty bins are skipped rather than counted as perfectly calibrated. Counting
    them would let a model that predicts only two distinct values report a
    flattering ECE simply by leaving eight bins empty.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[tuple[float, float, int]] = []
    total_error = 0.0
    total_count = 0

    for lower, upper in itertools.pairwise(edges):
        # Right-closed on the final bin so p == 1.0 lands somewhere.
        in_bin = (y_prob >= lower) & (y_prob < upper if upper < 1.0 else y_prob <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        mean_predicted = float(y_prob[in_bin].mean())
        observed = float(y_true[in_bin].mean())
        bins.append((mean_predicted, observed, count))
        total_error += abs(mean_predicted - observed) * count
        total_count += count

    ece = total_error / total_count if total_count else float("nan")
    return ece, tuple(bins)


def calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10
) -> CalibrationMetrics:
    """ECE, Brier score, and the reliability-diagram bins.

    Returns the bins rather than plotting them: this module computes, the report
    builder renders, and the console draws. Keeping plotting out of here means
    the metrics stay testable without a display.

    **Phase 4 note.** These are diagnostics, not a claim that any rung is
    calibrated. Raw boosting output is systematically distorted, and the whole
    point of measuring it now is to show the size of the problem that Phase 5's
    isotonic regression has to solve.
    """
    ece, bins = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    return CalibrationMetrics(
        expected_calibration_error=ece,
        brier_score=float(brier_score_loss(y_true, y_prob)),
        n_bins=n_bins,
        reliability_bins=bins,
    )
