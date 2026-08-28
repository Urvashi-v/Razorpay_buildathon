"""The headline metric: net rupees saved per 1,000 orders.

SPEC section 07 and appendix A1::

    Net INR/1,000 = (TP savings - FP costs - FN losses) at the operating
                    threshold, minus the same quantity for the do-nothing
                    baseline, scaled to 1,000 orders.

Note the subtraction of the baseline. Reporting gross savings against zero would
credit the model with money the merchant was never going to lose. Rung 0 is the
reference point and every rung is scored against it.

TWO REPORTING RULES ENFORCED HERE
=================================
1. False-positive cost is returned as its own field and never netted away.
   :class:`~rto_sentinel.contracts.evaluation.EconomicResult` has a required slot
   for it, so it cannot be quietly omitted.
2. Flag rate is returned alongside precision, always. A model that flags 40
   percent of orders is unusable regardless of its precision, and precision
   quoted without the flag rate is a half-truth.

WHAT THE BASELINE IS, AND WHY THE FN TERM CANCELS
=================================================
Doing nothing flags no orders, so every RTO is a false negative and the merchant
absorbs the full loss::

    baseline_net_per_1000 = -(positive_rate x rto_cost) x 1000

Reported as context, because it is the size of the problem. But the *headline*
figure is the difference between running the model and doing nothing, and in that
difference the false-negative term cancels - see
``OutcomeEconomics.net_versus_doing_nothing`` for the derivation written out.

So::

    net_inr_saved_per_1000 = (TP x S_tp - FP x C_fp) / n x 1000

A rung scoring +400 per 1,000 saves the merchant ₹400 per thousand orders
compared with shipping everything. A negative figure means the intervention costs
more than it saves - which is what rung 1 is expected to show.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rto_sentinel.contracts.evaluation import EconomicResult
from rto_sentinel.decision.cost_model import outcome_economics
from rto_sentinel.eval.bootstrap import bootstrap_metric
from rto_sentinel.eval.metrics import confusion_at_threshold

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.decision import CostInputs

PER_ORDERS = 1000.0


def do_nothing_net_per_1000(y_true: np.ndarray, inputs: CostInputs) -> float:
    """The loss a merchant absorbs today, per 1,000 orders.

    Negative by construction: flagging nothing means every RTO lands in full.
    This is the reference point every other rung is measured against, and the
    reason a "saving" in this project is a difference rather than a total.
    """
    if len(y_true) == 0:
        return float("nan")
    total_loss = float(np.sum(y_true)) * inputs.rto_cost_inr
    return -(total_loss / len(y_true)) * PER_ORDERS


def economic_result(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    cost_inputs: CostInputs,
    bootstrap_iterations: int = 0,
    confidence: float = 0.95,
    seed: int = 0,
) -> EconomicResult:
    """Score one operating point in rupees, with a bootstrap interval.

    The baseline is computed from the same labels rather than passed in, so a
    caller cannot accidentally compare a rung against a different book than the
    one it was scored on.
    """
    economics = outcome_economics(cost_inputs)
    confusion = confusion_at_threshold(y_true, y_prob, threshold)
    baseline = do_nothing_net_per_1000(y_true, cost_inputs)

    gross_saving = confusion.true_positives * economics.true_positive_saving_inr
    false_positive_cost = confusion.false_positives * economics.false_positive_cost_inr
    residual_loss = confusion.false_negatives * economics.false_negative_loss_inr

    def net_per_1000(true_labels: np.ndarray, probabilities: np.ndarray) -> float:
        matrix = confusion_at_threshold(true_labels, probabilities, threshold)
        if matrix.n == 0:
            return float("nan")
        delta = economics.net_versus_doing_nothing(
            tp=matrix.true_positives, fp=matrix.false_positives
        )
        return (delta / matrix.n) * PER_ORDERS

    net = bootstrap_metric(
        y_true,
        y_prob,
        net_per_1000,
        iterations=bootstrap_iterations,
        confidence=confidence,
        seed=seed,
    )

    return EconomicResult(
        threshold=float(threshold),
        flag_rate=confusion.flag_rate,
        true_positives=confusion.true_positives,
        false_positives=confusion.false_positives,
        false_negatives=confusion.false_negatives,
        true_negatives=confusion.true_negatives,
        gross_saving_inr=float(gross_saving),
        total_false_positive_cost_inr=float(false_positive_cost),
        residual_false_negative_loss_inr=float(residual_loss),
        net_inr_saved_per_1000_orders=net,
        baseline_net_inr_per_1000_orders=baseline,
    )


def cost_sensitivity_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    cost_inputs: CostInputs,
    parameter: str,
    perturbations: list[float],
) -> list[tuple[float, float]]:
    """How net rupees move when one cost input is wrong by a given fraction.

    Answers the second half of the sensitivity question in SPEC section 07: not
    only where the threshold moves, but how much money being wrong actually
    costs. A system whose savings collapse under a 30 percent error in an assumed
    constant should say so before someone deploys it.

    The threshold is held fixed while the economics vary, which is the realistic
    failure mode: the merchant's true costs differ from what they told us, but
    the system is already running at the threshold those wrong numbers implied.
    """
    if not hasattr(cost_inputs, parameter):
        msg = f"{parameter!r} is not a cost input"
        raise ValueError(msg)

    baseline_value = getattr(cost_inputs, parameter)
    curve: list[tuple[float, float]] = []
    for shift in perturbations:
        value = baseline_value * (1.0 + shift)
        if parameter in {"abandonment_on_friction", "intervention_success_rate"}:
            value = min(max(value, 0.0), 1.0)
        else:
            value = max(value, 1e-6)
        perturbed = cost_inputs.model_copy(update={parameter: value})
        result = economic_result(
            y_true, y_prob, threshold=threshold, cost_inputs=perturbed, bootstrap_iterations=0
        )
        curve.append((shift, result.net_inr_saved_per_1000_orders.value))
    return curve
