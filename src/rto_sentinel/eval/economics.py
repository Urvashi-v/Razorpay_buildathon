"""The headline metric: net rupees saved per 1,000 orders.

SPEC section 07 and appendix A1::

    Net INR/1,000 = (TP savings - FP costs - FN losses) at the operating
                    threshold, minus the same quantity for the do-nothing
                    baseline, scaled to 1,000 orders.

Note the subtraction of the baseline. Reporting gross savings against zero would
credit the model with money the merchant was never going to lose. Rung 0 is the
reference point and every rung is scored against it.

TWO REPORTING RULES ENFORCED HERE
---------------------------------
1. False-positive cost is returned as its own field and never netted away. The
   return type has a required slot for it, so it cannot be quietly omitted.
2. Flag rate is returned alongside precision, always. A model that flags 40
   percent of orders is unusable regardless of its precision, and precision
   quoted without the flag rate is a half-truth.

STATUS: Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from rto_sentinel.contracts.decision import CostInputs
    from rto_sentinel.contracts.evaluation import EconomicResult


def economic_result(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
    cost_inputs: CostInputs,
    baseline_net_per_1000: float,
    bootstrap_iterations: int = 2000,
    confidence: float = 0.95,
) -> EconomicResult:
    """Score one operating point in rupees, with a bootstrap interval."""
    raise NotImplementedError("Economic scoring lands in Phase 2.")


def cost_sensitivity_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    cost_inputs: CostInputs,
    parameter: str,
    perturbations: list[float],
) -> list[tuple[float, float]]:
    """How net rupees move when one cost input is wrong by a given fraction.

    Answers the second half of the sensitivity question in SPEC section 07: not
    only where the threshold moves, but how much money being wrong actually
    costs. A system whose savings collapse under a 30 percent error in an assumed
    constant should say so before someone deploys it.
    """
    raise NotImplementedError("Cost sensitivity lands in Phase 2.")
