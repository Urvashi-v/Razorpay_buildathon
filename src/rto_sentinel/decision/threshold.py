"""Threshold derivation. Not 0.5, and not tuned on the test set.

SPEC section 06::

    Flag when   p > C_fp / (C_fp + S_tp)

    C_fp = expected cost of frictioning a good order
    S_tp = expected saving from frictioning a bad one

Worked example, which ``tests/unit/test_cost_model.py`` asserts: RTO cost 220,
contribution margin 250, abandonment-on-friction 25%, intervention success 60%.

TWO THINGS THIS MODULE MUST NEVER DO, BOTH ENFORCED BY TESTS
============================================================
**Never return a constant.** A hardcoded 0.5 is the failure this project exists
to correct. The threshold is a function of merchant economics and moves when they
move - a high-margin brand should flag more readily than a thin-margin reseller,
and the console demonstrates that live.

**Never see labels.** Threshold derivation takes cost inputs and nothing else. It
does not look at the validation set, let alone the test set. The operating point
is *derived*, not fitted, which is why it can be published before the sealed test
run without contaminating it.

A NOTE ON THE DEGENERATE CASE
=============================
If frictioning a bad order saves nothing and frictioning a good one costs
nothing, no finite threshold exists. :class:`CostInputs` refuses those inputs at
construction rather than letting this function invent 0.5 - which would look like
an answer and be a placeholder.
"""

from __future__ import annotations

from rto_sentinel.contracts.decision import CostInputs, ThresholdDerivation
from rto_sentinel.decision.cost_model import outcome_economics


def derive_threshold(inputs: CostInputs) -> ThresholdDerivation:
    """Solve the expected-value inequality for the flag/no-flag boundary.

    Returns the threshold *with its arithmetic attached* so the console can show
    the working and a reviewer can check it by hand.
    """
    economics = outcome_economics(inputs)
    cost_fp = economics.false_positive_cost_inr
    saving_tp = economics.true_positive_saving_inr

    # saving_tp <= 0 means frictioning a risky order costs more than it saves.
    # No probability is high enough to justify it, so the honest threshold is
    # 1.0 - flag nothing. Returned rather than clamped silently, because it says
    # the merchant's intervention is not worth running at all.
    threshold = 1.0 if saving_tp <= 0.0 else cost_fp / (cost_fp + saving_tp)

    return ThresholdDerivation(
        threshold=min(max(threshold, 0.0), 1.0),
        cost_false_positive_inr=cost_fp,
        saving_true_positive_inr=max(saving_tp, 0.0),
        inputs=inputs,
    )


def threshold_sensitivity(
    inputs: CostInputs,
    *,
    parameter: str,
    perturbations: list[float],
) -> list[tuple[float, ThresholdDerivation]]:
    """How far the threshold moves when one cost input is wrong.

    SPEC section 07 asks how fast the rupee savings degrade if the cost inputs
    are off by 30%. This is the first half of that answer; the second half lives
    in ``eval.economics``, which re-scores at each perturbed threshold.
    """
    if not hasattr(inputs, parameter):
        msg = f"{parameter!r} is not a cost input; expected one of {list(CostInputs.model_fields)}"
        raise ValueError(msg)

    results: list[tuple[float, ThresholdDerivation]] = []
    baseline = getattr(inputs, parameter)
    for shift in perturbations:
        value = baseline * (1.0 + shift)
        # Probabilities have to stay in [0, 1]; rupee amounts only have to stay
        # positive. Clamping rather than skipping keeps the sweep rectangular.
        if parameter in {"abandonment_on_friction", "intervention_success_rate"}:
            value = min(max(value, 0.0), 1.0)
        else:
            value = max(value, 1e-6)
        perturbed = inputs.model_copy(update={parameter: value})
        results.append((shift, derive_threshold(perturbed)))
    return results
