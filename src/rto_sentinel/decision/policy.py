"""The friction ladder: turning a probability and a threshold into an action.

SPEC section 06 and section 09.

The four rungs, and why each exists:

* **LOW** - nothing. COD offered freely. The overwhelming majority of orders,
  and zero friction is the default state rather than a reward.
* **ELEVATED** - prepaid nudge with a small incentive. Positive-sum: cheaper
  than an RTO, and the customer gains something. Nobody is punished.
* **HIGH** - COD allowed after one-tap WhatsApp or OTP confirmation.
  Confirmation before dispatch is one of the highest-leverage documented
  interventions: low friction, high signal.
* **SEVERE** - prepaid-only, with a visible appeal path and a human review
  queue. A small tail. Never silent, never permanent, always reversible.

THE INVARIANT THAT MATTERS MOST
-------------------------------
There is no rung that hard-blocks an order without recourse. That is not a
default this module offers and a caller opts out of - it is not expressible.
``PolicyConfig`` rejects ``hard_block_allowed: true`` at load time, ``Decision``
rejects ``appeal_available=False`` at construction, and the band boundaries are
derived from the cost threshold rather than being independently settable. A
future contributor who wants a silent block has to defeat three separate checks,
in three separate files, on purpose.

Band cut points are *multipliers on the derived threshold*, so the whole ladder
slides when the merchant's economics change. Nothing here is an absolute
probability.

STATUS: Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rto_sentinel.contracts.decision import BandBoundary
from rto_sentinel.contracts.enums import InterventionAction, RiskBand

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.configuration.schemas import PolicyConfig


def resolve_boundaries(threshold: float, config: PolicyConfig) -> tuple[BandBoundary, ...]:
    """Turn the derived threshold into concrete probability cut points.

    LOW spans ``[0, threshold)``; each subsequent band spans up to
    ``multiplier * threshold``, clamped at 1.0; SEVERE is open-ended.

    Clamping matters: with a high threshold, ``2.4 * threshold`` can exceed 1.0,
    which would leave SEVERE unreachable. The Phase 2 implementation collapses
    unreachable bands rather than emitting boundaries that cannot fire, and
    reports that it did so.
    """
    raise NotImplementedError("Band resolution lands in Phase 2.")


def band_for(probability: float, boundaries: tuple[BandBoundary, ...]) -> RiskBand:
    """Select the band a probability falls into."""
    raise NotImplementedError("Band selection lands in Phase 2.")


def action_for(band: RiskBand, config: PolicyConfig) -> InterventionAction:
    """The configured action for a band."""
    raise NotImplementedError("Action lookup lands in Phase 2.")


def requires_human_review(band: RiskBand, config: PolicyConfig) -> bool:
    """Whether this band routes to the ops review queue. True for SEVERE."""
    raise NotImplementedError("Review routing lands in Phase 2.")
