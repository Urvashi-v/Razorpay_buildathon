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

BOUNDARIES ARE HALF-OPEN, AND THE SIDE MATTERS
----------------------------------------------
A band spans ``[lower, upper)``. A probability exactly equal to a cut point
therefore lands in the **higher** band, which makes the flag/no-flag rule
``p >= threshold`` - matching ``confusion_at_threshold`` in the evaluation
harness. Two components disagreeing about that one boundary would make the
measured flag rate differ from the served flag rate, silently, for exactly the
orders sitting on the line.

WHEN A BAND CANNOT EXIST
------------------------
``multiplier * threshold`` can exceed 1.0 - a thin-margin merchant with a high
threshold has no room above it for three more rungs. Emitting a band whose lower
bound is 1.0 would be emitting a rung that can never fire while telling the
console it exists. Such bands are collapsed away, and
:func:`resolve_boundaries` reports which, so a merchant can be told "at your
economics there is no SEVERE tier" rather than being shown an empty one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rto_sentinel.contracts.decision import BandBoundary
from rto_sentinel.contracts.enums import InterventionAction, RiskBand

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

    from rto_sentinel.configuration.schemas import BandEconomics, PolicyBand, PolicyConfig


class PolicyError(ValueError):
    """Raised when a policy cannot be resolved into a usable ladder."""


@dataclass(frozen=True, slots=True)
class ResolvedLadder:
    """The ladder at one threshold, plus what had to be dropped to build it."""

    boundaries: tuple[BandBoundary, ...]
    #: Bands that cannot fire at this threshold, with the reason. Reported rather
    #: than silently omitted: "your economics leave no SEVERE tier" is
    #: information a merchant needs.
    collapsed: tuple[tuple[RiskBand, str], ...] = ()

    def __iter__(self) -> Iterator[BandBoundary]:
        return iter(self.boundaries)

    def __len__(self) -> int:
        return len(self.boundaries)

    @property
    def bands(self) -> tuple[RiskBand, ...]:
        return tuple(boundary.band for boundary in self.boundaries)


def _band_config(band: RiskBand, config: PolicyConfig) -> PolicyBand:
    for entry in config.bands:
        if entry.name == band.value:
            return entry
    msg = f"{band} is not configured in policy.yaml"
    raise PolicyError(msg)


def resolve_boundaries(threshold: float, config: PolicyConfig) -> ResolvedLadder:
    """Turn the derived threshold into concrete probability cut points.

    LOW spans ``[0, threshold)``; each subsequent band spans up to
    ``multiplier * threshold``, clamped at 1.0; the top band is open-ended.

    A band is collapsed when its span is empty - that is, when the previous
    band's upper bound has already reached 1.0. The result always contains at
    least LOW, because ``[0, threshold)`` is non-empty for any threshold above
    zero, and at a threshold of zero LOW is empty and every order is flagged,
    which is itself a legitimate (if aggressive) policy.
    """
    if not 0.0 <= threshold <= 1.0:
        msg = f"threshold {threshold} is not a probability"
        raise PolicyError(msg)

    boundaries: list[BandBoundary] = []
    collapsed: list[tuple[RiskBand, str]] = []
    lower = 0.0

    for index, entry in enumerate(config.bands):
        band = RiskBand(entry.name)
        is_top = index == len(config.bands) - 1

        if is_top:
            upper: float | None = None
        else:
            multiplier = entry.upper_bound_multiplier
            if multiplier is None:  # pragma: no cover - refused by PolicyConfig
                msg = f"only the top band may be open-ended; {band} is not"
                raise PolicyError(msg)
            upper = min(multiplier * threshold, 1.0)

        # An empty span cannot fire. `lower >= 1.0` means the ladder ran out of
        # probability space; `upper <= lower` means this band's ceiling is at or
        # below its floor.
        if lower >= 1.0 or (upper is not None and upper <= lower):
            reason = (
                f"{entry.upper_bound_multiplier} x threshold {threshold:.4f} leaves no room "
                f"above {lower:.4f}"
                if upper is not None
                else f"no probability remains above {lower:.4f}"
            )
            collapsed.append((band, reason))
            continue

        boundaries.append(
            BandBoundary(
                band=band,
                lower_bound=lower,
                upper_bound=upper,
                action=InterventionAction(entry.action),
            )
        )
        if upper is None:
            break
        lower = upper

    if not boundaries:  # pragma: no cover - unreachable while LOW starts at 0.0
        msg = f"no band can fire at threshold {threshold}"
        raise PolicyError(msg)

    return ResolvedLadder(boundaries=tuple(boundaries), collapsed=tuple(collapsed))


def band_for(probability: float, ladder: ResolvedLadder) -> RiskBand:
    """Select the band a probability falls into.

    Half-open ``[lower, upper)``, so a probability exactly on a cut point lands
    in the higher band. See the module docstring for why that side was chosen.
    """
    if not 0.0 <= probability <= 1.0:
        msg = f"probability {probability} is out of range"
        raise PolicyError(msg)

    for boundary in ladder.boundaries:
        if boundary.upper_bound is None:
            if probability >= boundary.lower_bound:
                return boundary.band
        elif boundary.lower_bound <= probability < boundary.upper_bound:
            return boundary.band

    # Reachable when the top band was collapsed and p sits above every ceiling -
    # for instance a ladder whose highest band ends exactly at 1.0 scored with
    # p == 1.0. The highest surviving band is the honest answer.
    return ladder.boundaries[-1].band


def action_for(band: RiskBand, config: PolicyConfig) -> InterventionAction:
    """The configured action for a band."""
    return InterventionAction(_band_config(band, config).action)


def requires_human_review(band: RiskBand, config: PolicyConfig) -> bool:
    """Whether this band routes to the ops review queue. True for SEVERE."""
    return _band_config(band, config).requires_human_review_queue


def requires_reason_code(band: RiskBand, config: PolicyConfig) -> bool:
    """Whether a decision in this band must carry a reason code."""
    return _band_config(band, config).requires_reason_code


def band_economics(band: RiskBand, config: PolicyConfig) -> BandEconomics:
    """The ASSUMED effectiveness and cost multipliers for this band's action."""
    return _band_config(band, config).economics
