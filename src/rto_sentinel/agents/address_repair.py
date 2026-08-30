"""Address repair suggestions: DEFERRED, deliberately and with reasons.

STATUS: not implemented. This module raises, and that is the finished state of
it for now rather than a gap someone forgot to fill.

WHY IT IS NOT BUILT
===================
The Phase 8 brief says to implement this "only if it can be implemented safely
and with actual data/logic. Otherwise explicitly defer it rather than producing a
fake feature." It cannot, for three separate reasons, any one of which would be
enough.

**1. There is no address to repair.** The generator produces address *strings*
and the derived quality signals computed from them, but those strings are
synthetic - assembled from a small vocabulary of road names and localities. A
suggestion engine trained or prompted on them would be repairing text that has no
relationship to how Indian addresses are actually written, and its apparent
accuracy would be a measurement of the generator, not of anything useful.

**2. The agent layer is deliberately denied the address text.**
``serving.agent_tools`` returns ``addr_token_count``, whether a house number is
present, whether the pincode matches the city - never the line itself. That was a
privacy decision made on purpose: handing raw delivery addresses to a language
model that also drafts customer-facing copy is how an address ends up quoted back
at somebody. Implementing repair would mean reversing it, and the reason for it
has not changed.

**3. Correctness would need a source this project does not have.** A real address
suggester validates against a postal database - India Post's PIN directory, or a
commercial geocoder. Without one, the model can only guess at what a malformed
address *meant*, and a confidently wrong "corrected" address is worse than a
flagged incomplete one: it routes a parcel to a place nobody lives while looking
like a fix.

WHAT WOULD MAKE IT BUILDABLE
============================
Real merchant addresses (with the consent and handling that implies), a postal
reference dataset to validate against, and a measurement of suggestion accuracy
against accepted-versus-rejected outcomes. Until all three exist, the honest
product is the one already shipping: the address-quality features feed the model,
a low completeness score contributes to the risk score, and the customer is asked
to confirm their own address rather than having it rewritten for them.

That last part is not a lesser feature. Nothing in this system silently rewrites
a delivery address, which is why
:class:`~rto_sentinel.contracts.explanation.AddressRepairSuggestion` has no
"applied" field: acceptance would always be the customer's action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.explanation import AddressRepairSuggestion

#: Quoted by the API so a caller reads the reason rather than a bare 501.
DEFERRAL_REASON = (
    "Address repair suggestions are deliberately not implemented. Three reasons, each "
    "sufficient on its own: (1) the addresses in this benchmark are synthetic strings, so "
    "any suggester would be repairing text unrelated to real Indian addresses; (2) the "
    "agent layer is denied raw address text by design, because a model that drafts "
    "customer-facing copy should not hold delivery addresses; (3) correctness needs a "
    "postal reference dataset this project does not have, and a confidently wrong "
    "'corrected' address is worse than a flagged incomplete one. The shipped behaviour is "
    "to ask the customer to confirm their own address - nothing here rewrites one."
)


class AddressRepairDeferred(NotImplementedError):
    """Raised on any attempt to use the deferred address-repair feature."""


def suggest_address_repair(*args: object, **kwargs: object) -> NoReturn:
    """Refuse, with the reason. See the module docstring.

    Deliberately not a stub that returns a plausible suggestion. A fake repair
    would be indistinguishable from a real one to every caller, and the first
    time anyone relied on it would be the last time anyone trusted the rest.
    """
    raise AddressRepairDeferred(DEFERRAL_REASON)


__all__ = ["DEFERRAL_REASON", "AddressRepairDeferred", "suggest_address_repair"]

if TYPE_CHECKING:  # pragma: no cover - keeps the contract import meaningful
    _: type[AddressRepairSuggestion]
