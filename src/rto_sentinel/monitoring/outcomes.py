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

STATUS: IMPLEMENTED, AND CURRENTLY RETURNS "NOT ENOUGH DATA".
-------------------------------------------------------------
The arithmetic below is real and runs against the decision log. What it reports
today is that no interventions have been applied - this system has never operated
on live traffic, so every flagged order in the benchmark carries a
"no friction was applied" outcome and there is nothing to compare.

That is the honest state, and it is deliberately a *working function returning
insufficient data* rather than a ``NotImplementedError``. The difference matters:
the day this runs in production the measurement starts accumulating, and
``is_assumed`` flips to False on its own once both arms clear their minimum. A
stub would have required someone to remember to come back.

**Until it flips, the 60 percent intervention success rate in the cost model
remains an assumption**, and every rupee figure in this project inherits that.

WHY THIS TAKES FRAMES RATHER THAN A SESSION
-------------------------------------------
``monitoring`` may not import ``db`` - the layering tests forbid it - and that
constraint is doing real work here. The arithmetic must not care whether the rows
came from PostgreSQL, a parquet file or a test fixture, because a measurement
that can only be exercised against a running database is a measurement nobody
runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Orders needed on BOTH arms before a success rate is reported as measured.
#:
#: The control holdout is 2 percent of flagged orders, so the control arm is the
#: binding constraint by a factor of fifty: reaching 200 controls means about
#: 10,000 frictioned orders. That is the price of measuring this honestly, and
#: quoting a rate off twenty controls would be worse than quoting the assumption,
#: because it would carry the authority of a measurement.
MIN_PER_ARM = 200


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
    rto_rate_treated: float | None
    rto_rate_control: float | None
    measured_success_rate: float | None
    is_assumed: bool = True
    note: str = ""


def intervention_effectiveness(
    decisions: pd.DataFrame,
    band: str,
    *,
    min_per_arm: int = MIN_PER_ARM,
) -> InterventionEffectiveness:
    """Compare treated versus control outcomes for one friction band.

    ``decisions`` needs three columns: ``band``, ``is_control_holdout`` and
    ``is_rto``. Rows whose outcome has not matured (``is_rto`` null) are dropped
    rather than counted as delivered - counting them would make every
    intervention look effective, since a frictioned order that has not resolved
    yet is exactly the shape of a success.

    The success rate is
    ``(control RTO rate - treated RTO rate) / control RTO rate``: the share of
    would-be returns that the friction actually prevented. It is clamped to
    [0, 1] because a negative value means the treated arm returned *more*, which
    is noise at these sample sizes rather than friction causing returns.
    """
    required = {"band", "is_control_holdout", "is_rto"}
    missing = required - set(decisions.columns)
    if missing:
        msg = f"decision frame is missing {sorted(missing)}; cannot measure effectiveness"
        raise KeyError(msg)

    in_band = decisions[decisions["band"] == band]
    matured = in_band[in_band["is_rto"].notna()]

    control = matured[matured["is_control_holdout"].astype(bool)]
    treated = matured[~matured["is_control_holdout"].astype(bool)]

    n_treated, n_control = len(treated), len(control)
    rate_treated = float(treated["is_rto"].astype(float).mean()) if n_treated else None
    rate_control = float(control["is_rto"].astype(float).mean()) if n_control else None

    if n_treated < min_per_arm or n_control < min_per_arm:
        return InterventionEffectiveness(
            band=band,
            n_treated=n_treated,
            n_control=n_control,
            rto_rate_treated=rate_treated,
            rto_rate_control=rate_control,
            measured_success_rate=None,
            is_assumed=True,
            note=(
                f"Not enough matured outcomes to measure: {n_treated} treated and "
                f"{n_control} control against a minimum of {min_per_arm} each. The "
                "intervention success rate remains the configured ASSUMPTION."
            ),
        )

    if not rate_control:
        return InterventionEffectiveness(
            band=band,
            n_treated=n_treated,
            n_control=n_control,
            rto_rate_treated=rate_treated,
            rto_rate_control=rate_control,
            measured_success_rate=None,
            is_assumed=True,
            note=(
                "The control arm saw no returns at all, so there is nothing for the "
                "friction to have prevented and no rate to compute. Not a success."
            ),
        )

    prevented = (rate_control - (rate_treated or 0.0)) / rate_control
    return InterventionEffectiveness(
        band=band,
        n_treated=n_treated,
        n_control=n_control,
        rto_rate_treated=rate_treated,
        rto_rate_control=rate_control,
        measured_success_rate=min(max(prevented, 0.0), 1.0),
        is_assumed=False,
        note=(
            f"Measured on {n_treated:,} treated and {n_control:,} control orders. "
            "This replaces the configured assumption for this band."
        ),
    )


@dataclass(frozen=True, slots=True)
class OverrideSummary:
    """What operations did with the decisions the engine made, by band."""

    band: str
    n_decisions: int
    n_relaxed: int
    n_escalated: int
    override_rate: float
    reading: str


def override_summary(overrides: pd.DataFrame, decisions: pd.DataFrame) -> list[OverrideSummary]:
    """Counts of relaxed versus escalated overrides, by band.

    A band that ops relaxes most of the time is a band whose threshold is wrong,
    and this is how that becomes visible rather than remaining folklore in the
    operations team.

    ``reading`` is prose rather than a flag because the two directions mean
    opposite things. Consistent relaxation says the model is flagging orders a
    human can see are fine - the threshold is too low, or a feature is
    misfiring. Consistent escalation says the model is missing something the
    human can see, which is a modelling gap rather than a threshold one.
    """
    # An EMPTY override log is the normal state, not a malformed one - a system
    # that has never been overridden is the common case and must not raise. Only
    # a non-empty frame is required to carry the columns.
    if overrides.empty:
        overrides = pd.DataFrame({"band": [], "direction": []})

    for frame, name, columns in (
        (overrides, "override", {"band", "direction"}),
        (decisions, "decision", {"band"}),
    ):
        missing = columns - set(frame.columns)
        if missing:
            msg = f"{name} frame is missing {sorted(missing)}"
            raise KeyError(msg)

    summaries: list[OverrideSummary] = []
    for band in sorted(set(decisions["band"].astype(str))):
        band_decisions = int((decisions["band"] == band).sum())
        band_overrides = overrides[overrides["band"] == band]
        relaxed = int((band_overrides["direction"] == "relaxed").sum())
        escalated = int((band_overrides["direction"] == "escalated").sum())
        total = relaxed + escalated
        rate = total / band_decisions if band_decisions else 0.0

        if total == 0:
            reading = "No overrides recorded in this band."
        elif relaxed > escalated * 2:
            reading = (
                "Operations relaxes this band far more often than it escalates. That "
                "points at the threshold being too low for this band, or a feature "
                "misfiring - not at the humans being wrong."
            )
        elif escalated > relaxed * 2:
            reading = (
                "Operations escalates this band far more often than it relaxes. The "
                "model is missing something a human can see, which is a modelling "
                "gap rather than a threshold one."
            )
        else:
            reading = "Overrides are roughly balanced between directions."

        summaries.append(
            OverrideSummary(
                band=band,
                n_decisions=band_decisions,
                n_relaxed=relaxed,
                n_escalated=escalated,
                override_rate=rate,
                reading=reading,
            )
        )
    return summaries
