"""Sweeping the threshold, and being explicit that the sweep does not choose it.

WHAT THIS IS FOR
================
A merchant asks reasonable questions the operating point alone cannot answer:
what would happen if we frictioned twice as many orders? Where does precision
fall off? How wide is the plateau around our threshold - are we on a knife edge
or a flat stretch? The sweep answers those.

WHAT THIS IS NOT FOR
====================
**Choosing the threshold.** The operating point is derived from merchant
economics by ``derive_threshold``, and that derivation never sees a label. Reading
the peak off this curve would be fitting the operating point to the evaluation
data - and if the curve came from the sealed split, it would be fitting it to the
sealed split, which is the one thing the seal exists to prevent.

The distinction is not left to discipline:

* :class:`~rto_sentinel.contracts.economics.ThresholdSweep` has a required
  ``selection_methodology`` field whose validator refuses text that does not say
  the threshold is derived;
* the sweep reports ``best_net_threshold`` - where the curve peaks - as a
  *separate* field from ``derived_threshold``, so the gap between "what the data
  would have picked" and "what economics picked" is visible rather than
  collapsible;
* :func:`sweep_thresholds` refuses to run on the test split at all.

THE GAP BETWEEN THE TWO IS INFORMATION
======================================
If the derived threshold sits far from the curve's peak, that is worth knowing:
either the cost inputs are wrong, or the model's probabilities are miscalibrated
in that region. It is not, by itself, a reason to move the threshold. Moving it
to the peak would mean the merchant's economics no longer justify the operating
point, and the rupee figures would stop meaning what they claim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rto_sentinel.contracts.economics import ThresholdPoint, ThresholdSweep
from rto_sentinel.decision.cost_model import outcome_economics
from rto_sentinel.decision.threshold import derive_threshold

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.decision import CostInputs

PER_ORDERS = 1000.0

#: Stated on every sweep artefact. Written once here so the wording cannot drift
#: between the CLI, the report and the API.
SELECTION_METHODOLOGY = (
    "The operating threshold is DERIVED from merchant economics as "
    "C_fp / (C_fp + S_tp) and is never read off this curve. The curve is a "
    "diagnostic: it shows what other operating points would cost, how flat the "
    "region around the derived point is, and where the peak sits. Selecting the "
    "peak would fit the operating point to the evaluation labels, which is what "
    "the derivation exists to avoid. The sweep is computed on validation only; "
    "running it on the sealed test split is refused."
)


class SweepError(ValueError):
    """Raised when a sweep is requested in a way that would contaminate a result."""


def sweep_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray | None,
    *,
    cost_inputs: CostInputs,
    split: str = "validation",
    cost_profile: str = "custom",
    grid: np.ndarray | None = None,
) -> ThresholdSweep:
    """Precision, recall, flag rate, expected cost and net rupees across thresholds.

    ``labels`` are optional: without them the expected figures are computed from
    the probabilities alone and the realized columns stay empty. That is the live
    case, and it is the one the merchant simulator uses.
    """
    if split == "test":
        msg = (
            "refusing to sweep thresholds on the sealed test split. The operating point is "
            "derived from economics, so a sweep over test labels can only be used to tune "
            "against them - which is exactly what the seal prevents. Sweep on validation."
        )
        raise SweepError(msg)

    scores = np.asarray(probabilities, dtype="float64").ravel()
    if scores.size == 0:
        msg = "refusing to sweep an empty book"
        raise SweepError(msg)

    y: np.ndarray | None = None
    if labels is not None:
        y = np.asarray(labels).astype(bool).ravel()
        if y.shape != scores.shape:
            msg = f"labels and probabilities disagree in length: {y.shape} vs {scores.shape}"
            raise SweepError(msg)

    derivation = derive_threshold(cost_inputs)
    economics = outcome_economics(cost_inputs)
    n = int(scores.size)

    if grid is None:
        grid = np.round(np.linspace(0.01, 0.99, 99), 4)
    # The derived point is always on the grid, so the table always contains the
    # row a reader most wants to find.
    values = np.unique(np.append(np.asarray(grid, dtype="float64"), derivation.threshold))

    points: list[ThresholdPoint] = []
    for value in values:
        flagged = scores >= value
        n_flagged = int(flagged.sum())

        expected_tp = float(scores[flagged].sum())
        expected_fp = float((1.0 - scores[flagged]).sum())
        expected_fn = float(scores[~flagged].sum())

        expected_fp_cost = expected_fp * economics.false_positive_cost_inr
        expected_residual = (
            expected_tp * (1.0 - cost_inputs.intervention_success_rate) * cost_inputs.rto_cost_inr
            + expected_fn * cost_inputs.rto_cost_inr
        )
        expected_net = expected_tp * economics.true_positive_saving_inr - expected_fp_cost

        precision: float | None = None
        recall: float | None = None
        f1: float | None = None
        realized_net: float | None = None
        true_positives: int | None = None
        false_positives: int | None = None
        if y is not None:
            true_positives = int((flagged & y).sum())
            false_positives = int((flagged & ~y).sum())
            false_negatives = int((~flagged & y).sum())
            precision = true_positives / n_flagged if n_flagged else None
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives)
                else None
            )
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            realized_net = (
                economics.net_versus_doing_nothing(tp=true_positives, fp=false_positives) / n
            ) * PER_ORDERS

        points.append(
            ThresholdPoint(
                threshold=float(value),
                flag_rate=n_flagged / n,
                precision=precision,
                recall=recall,
                f1=f1,
                expected_cost_inr=expected_fp_cost + expected_residual,
                expected_net_inr_per_1000_orders=(expected_net / n) * PER_ORDERS,
                realized_net_inr_per_1000_orders=realized_net,
                true_positives=true_positives,
                false_positives=false_positives,
                is_derived_operating_point=bool(np.isclose(value, derivation.threshold, atol=1e-9)),
            )
        )

    # Where the curve peaks, on realized rupees if labels exist and on expected
    # rupees otherwise. Reported, never selected.
    def peak_key(point: ThresholdPoint) -> float:
        if point.realized_net_inr_per_1000_orders is not None:
            return point.realized_net_inr_per_1000_orders
        return point.expected_net_inr_per_1000_orders

    best = max(points, key=peak_key)

    return ThresholdSweep(
        split=split,
        cost_profile=cost_profile,
        derived_threshold=derivation.threshold,
        best_net_threshold=best.threshold,
        selection_methodology=SELECTION_METHODOLOGY,
        points=points,
    )


def sweep_to_rows(sweep: ThresholdSweep) -> list[dict[str, object]]:
    """The sweep as flat rows, for CSV and for the console table."""
    return [
        {
            "threshold": point.threshold,
            "flag_rate": point.flag_rate,
            "precision": point.precision,
            "recall": point.recall,
            "f1": point.f1,
            "expected_cost_inr": point.expected_cost_inr,
            "expected_net_inr_per_1000": point.expected_net_inr_per_1000_orders,
            "realized_net_inr_per_1000": point.realized_net_inr_per_1000_orders,
            "true_positives": point.true_positives,
            "false_positives": point.false_positives,
            "is_operating_point": point.is_derived_operating_point,
        }
        for point in sweep.points
    ]
