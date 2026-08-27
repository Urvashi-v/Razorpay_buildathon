"""Threshold derivation. Not 0.5, and not tuned on the test set.

SPEC section 06::

    Flag when   p > C_fp / (C_fp + S_tp)

    C_fp = expected cost of frictioning a good order
    S_tp = expected saving from frictioning a bad one

Worked example, which the Phase 2 unit test asserts to the fourth decimal:
RTO cost 220, contribution margin 250, abandonment-on-friction 25%, intervention
success 60%::

    C_fp      = 0.25 * 250 = 62.5
    S_tp      = 0.60 * 220 = 132.0
    threshold = 62.5 / (62.5 + 132.0) = 0.3214...

Two things this module must never do, both enforced by tests:

* **Never return a constant.** A hardcoded 0.5 is the failure this project
  exists to correct. The threshold is a function of merchant economics and
  moves when they move - a high-margin brand should flag more aggressively than
  a thin-margin reseller, and the console demonstrates that live.
* **Never see labels.** Threshold derivation takes cost inputs and nothing else.
  It does not look at the validation set, let alone the test set. The operating
  point is *derived*, not fitted, which is why it can be published before the
  sealed test run without contaminating it.

STATUS: Phase 2.
"""

from __future__ import annotations

from rto_sentinel.contracts.decision import CostInputs, ThresholdDerivation


def derive_threshold(inputs: CostInputs) -> ThresholdDerivation:
    """Solve the expected-value inequality for the flag/no-flag boundary.

    Returns the threshold *with its arithmetic attached* so the console can show
    the working and a reviewer can check it by hand.
    """
    raise NotImplementedError("Threshold derivation lands in Phase 2.")


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
    raise NotImplementedError("Threshold sensitivity lands in Phase 2.")
