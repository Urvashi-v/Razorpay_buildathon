"""The rupee cost model. The centre of this submission, not a footnote.

SPEC section 06. Four outcomes, four rupee values:

===============  ===========================================================
True positive    Risky order flagged, friction applied. Saves the RTO cost
                 times the intervention success rate, minus the margin on
                 orders that lapse rather than convert.
False positive   A good customer is asked to prepay or confirm. Costs the
                 abandonment probability times contribution margin, plus a
                 small friction and support cost. **This is the number the
                 track's bar is asking for.**
False negative   Risky order ships unflagged and comes back. Costs the full
                 RTO amount.
True negative    Good order ships clean. Zero - the status quo working.
===============  ===========================================================

Two design decisions worth stating:

* This module is **pure arithmetic over declared inputs**. No model, no data
  access, no I/O. That is what makes the rupee numbers checkable by hand, which
  is the only reason anyone should believe them.
* False-positive cost is computed and returned **separately**, never folded into
  a net figure. ``EconomicResult`` has a required field for it precisely so it
  cannot be hidden.

STATUS: Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from rto_sentinel.contracts.decision import CostInputs


@dataclass(frozen=True, slots=True)
class OutcomeEconomics:
    """Per-outcome rupee values under one set of cost inputs.

    All four are positive-signed magnitudes; the aggregation decides direction.
    Keeping them separate is what lets the report state false-positive cost on
    its own line.
    """

    true_positive_saving_inr: float
    false_positive_cost_inr: float
    false_negative_loss_inr: float
    true_negative_value_inr: float = 0.0


def outcome_economics(inputs: CostInputs) -> OutcomeEconomics:
    """Expected rupee value of each confusion-matrix cell.

    Phase 2 implementation follows SPEC section 06 exactly::

        S_tp = intervention_success_rate * rto_cost
               - (1 - intervention_success_rate) * <margin on orders that lapse>
        C_fp = abandonment_on_friction * contribution_margin
               + friction_support_cost
        L_fn = rto_cost
        V_tn = 0
    """
    raise NotImplementedError("Outcome economics land in Phase 2.")


def expected_value_of_flagging(probability: float, inputs: CostInputs) -> float:
    """Expected rupee gain from applying friction to an order at ``probability``.

    Positive means flagging is worth it. This is the quantity the threshold
    derivation sets to zero and solves for ``p``.
    """
    raise NotImplementedError("Expected-value arithmetic lands in Phase 2.")
