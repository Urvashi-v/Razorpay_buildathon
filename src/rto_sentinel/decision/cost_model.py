"""The rupee cost model. The centre of this submission, not a footnote.

SPEC section 06. Four outcomes, four rupee values:

===============  ===========================================================
True positive    Risky order flagged, friction applied. Saves the RTO cost
                 times the intervention success rate.
False positive   A good customer is asked to prepay or confirm. Costs the
                 abandonment probability times contribution margin, plus a
                 small friction and support cost. **This is the number the
                 track's bar is asking for.**
False negative   Risky order ships unflagged and comes back. Costs the full
                 RTO amount.
True negative    Good order ships clean. Zero - the status quo working.
===============  ===========================================================

THE FORMULA IS THE SPECIFICATION'S, NOT AN IMPROVED ONE
=======================================================
::

    C_fp = abandonment_on_friction x contribution_margin + friction_support_cost
    S_tp = intervention_success_rate x rto_cost

With ``friction_support_cost = 0`` this reproduces the specification's worked
example exactly: ``0.25 x 250 = 62.5`` and ``0.60 x 220 = 132``, giving a
threshold of 0.3214. The default profile's ₹8 support cost moves it to 0.348.

A first implementation of this module also subtracted a "lapsed margin" term from
``S_tp`` - the contribution margin forgone on flagged orders that neither convert
nor would have returned anyway. That is arguably more careful, and it was
**removed**, because it requires an additional constant nobody has measured
(what share of non-converting flagged orders were genuinely lost). Smuggling an
unmeasured assumption into the headline metric is precisely the move this project
criticises elsewhere, and it moved the threshold by 14 points.

It is recorded here as a known simplification: the true-positive saving is
therefore an **upper bound**, and the net figures are correspondingly optimistic
in the model's favour. Stated rather than hidden.

TWO DESIGN DECISIONS
====================
This module is **pure arithmetic over declared inputs**. No model, no data
access, no I/O. That is what makes the rupee numbers checkable by hand, which is
the only reason anyone should believe them.

False-positive cost is computed and returned **separately**, never folded into a
net figure. :class:`OutcomeEconomics` keeps the values apart so the report has
somewhere honest to put each one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rto_sentinel.contracts.decision import CostInputs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import BandEconomics


@dataclass(frozen=True, slots=True)
class OutcomeEconomics:
    """Per-outcome rupee values under one set of cost inputs.

    All four are positive-signed magnitudes; the caller applies direction. A
    saving and a cost are both reported as "how many rupees", so the report can
    show false-positive cost on its own line rather than as a negative buried
    inside a net figure.
    """

    true_positive_saving_inr: float
    false_positive_cost_inr: float
    false_negative_loss_inr: float
    true_negative_value_inr: float = 0.0

    def net_versus_doing_nothing(self, *, tp: int, fp: int) -> float:
        """Rupees saved relative to flagging nothing.

        THE FALSE-NEGATIVE TERM CANCELS, AND THAT IS NOT AN OMISSION.

        SPEC appendix A1 defines the headline as "(TP savings - FP costs - FN
        losses) at the operating threshold, minus the same quantity for the
        do-nothing baseline". Writing both sides out::

            at threshold : TP x S_tp - FP x C_fp - FN x L_fn
            do nothing   :                       - (TP + FN) x L_fn

        Subtracting leaves ``TP x S_tp - FP x C_fp + TP x L_fn``. That last term
        is a double count: an order the model caught is credited both with the
        saving from catching it *and* with the loss it no longer incurs, which
        are the same money.

        The honest delta is ``TP x S_tp - FP x C_fp``. An earlier version of this
        code carried the double count and reported savings roughly three times
        too large, which is why the derivation is written out here rather than
        left as a one-line expression.
        """
        return tp * self.true_positive_saving_inr - fp * self.false_positive_cost_inr


def outcome_economics(inputs: CostInputs) -> OutcomeEconomics:
    """Expected rupee value of each confusion-matrix cell."""
    return OutcomeEconomics(
        true_positive_saving_inr=inputs.intervention_success_rate * inputs.rto_cost_inr,
        false_positive_cost_inr=(
            inputs.abandonment_on_friction * inputs.contribution_margin_inr
            + inputs.friction_support_cost_inr
        ),
        false_negative_loss_inr=inputs.rto_cost_inr,
        true_negative_value_inr=0.0,
    )


def expected_value_of_flagging(probability: float, inputs: CostInputs) -> float:
    """Expected rupee gain from applying friction to an order at ``probability``.

    Positive means flagging is worth it. This is the quantity
    :func:`~rto_sentinel.decision.threshold.derive_threshold` sets to zero and
    solves for ``p``::

        p x S_tp - (1 - p) x C_fp
    """
    economics = outcome_economics(inputs)
    return (
        probability * economics.true_positive_saving_inr
        - (1.0 - probability) * economics.false_positive_cost_inr
    )


def band_outcome_economics(inputs: CostInputs, band: BandEconomics) -> OutcomeEconomics:
    """Per-outcome rupee values for ONE RUNG of the friction ladder.

    THE MULTIPLIERS ARE ASSUMPTIONS AND THE ARITHMETIC CANNOT HIDE THAT
    ==================================================================
    A graduated ladder is only worth building if the rungs differ economically -
    a declinable nudge should not be credited with the same save rate as removing
    cash on delivery. So each band scales the merchant's own rates::

        S_tp(band) = (success_rate x success_multiplier) x rto_cost
        C_fp(band) = (abandonment x abandonment_multiplier) x margin
                     + band_support_cost

    But **nobody has measured any of those multipliers on this data**, and they
    cannot be measured without running the interventions and observing the
    counterfactual. They are declared in ``config/policy.yaml`` with a rationale
    each, tagged ``assumed_intervention`` wherever they reach a report, and they
    are the single largest source of uncertainty in every rupee figure this
    system produces.

    Rates are clamped to ``[0, 1]`` after scaling: a multiplier can push a
    success or abandonment rate above 1, and a probability greater than one is
    not a stronger intervention, it is a broken input.
    """
    success = min(
        max(inputs.intervention_success_rate * band.intervention_success_multiplier, 0.0), 1.0
    )
    abandonment = min(max(inputs.abandonment_on_friction * band.abandonment_multiplier, 0.0), 1.0)
    return OutcomeEconomics(
        true_positive_saving_inr=success * inputs.rto_cost_inr,
        false_positive_cost_inr=(
            abandonment * inputs.contribution_margin_inr + band.support_cost_inr
        ),
        false_negative_loss_inr=inputs.rto_cost_inr,
        true_negative_value_inr=0.0,
    )
