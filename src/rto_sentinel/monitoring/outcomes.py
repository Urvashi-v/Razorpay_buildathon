"""The outcome feedback loop, and the counterfactual problem.

SPEC section 02 step 5, and section 11.

THE PROBLEM THIS MODULE EXISTS FOR
----------------------------------
Once friction is applied, the true outcome of that order is never observed. A
HIGH-band order that converts to prepaid and delivers cleanly does not tell you
whether it would have been an RTO. So the model stops seeing the very cases it
acts on, and measured precision slowly becomes fiction - it is computed only over
the orders the system chose not to touch.

THE ONLY CLEAN ANSWER
---------------------
A small randomised control slice of flagged orders that receives no friction.
Configured at 2 percent in ``config/policy.yaml``, marked on the decision as
``is_control_holdout``, and excluded from the intervention accounting. Those
orders are the only place true precision remains measurable after the system
starts acting.

The cost of the holdout is real and should be stated rather than buried: 2
percent of flagged orders ship unprotected, and some of them come back. That is
the price of continuing to know whether the model works, and it is cheaper than
the alternative, which is a metric that drifts into fantasy without anyone
noticing.

OVERRIDES AS EVIDENCE
---------------------
An ops associate relaxing a SEVERE band is asserting something the model did not
know. Aggregated, overrides are a signal about where the model is systematically
wrong - and where the override was mistaken, the outcome says so.

STATUS: NOT IMPLEMENTED.

The control-holdout flag is set on real decisions and the policy reserves the
slice, so the data needed for this would accumulate in production. The
measurement itself does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InterventionEffectiveness:
    """Measured effect of one friction rung, from the control comparison.

    SPEC section 11 is explicit that the 60 percent intervention success rate in
    the cost model comes from published studies rather than measurement here. The
    holdout is what eventually replaces that assumption with a number, and until
    it does, ``is_assumed`` stays True and the reports say so.
    """

    band: str
    n_treated: int
    n_control: int
    rto_rate_treated: float
    rto_rate_control: float
    measured_success_rate: float | None
    is_assumed: bool = True


def intervention_effectiveness(band: str) -> InterventionEffectiveness:
    """Compare treated versus control outcomes for one friction band."""
    raise NotImplementedError(
        "Intervention effectiveness has never been measured. The rate used by the "
        "decision engine is a stated ASSUMPTION - see contracts/provenance.py - and "
        "this function exists to mark where the measurement would go."
    )


def override_summary(merchant_id: str) -> dict[str, int]:
    """Counts of relaxed versus escalated overrides, by band.

    A band that ops relaxes most of the time is a band whose threshold is wrong,
    and this is how that becomes visible rather than remaining folklore in the
    operations team.
    """
    raise NotImplementedError(
        "Override analytics are not implemented. Overrides are logged, and "
        "GET /v1/monitoring/decisions reports their counts by direction; the "
        "analysis of what they imply is not built."
    )
