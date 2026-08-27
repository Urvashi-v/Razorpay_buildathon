"""Language job 4 of 4: suggest a repair for a low-quality address.

SPEC section 08.

WHAT IT DOES
    Proposes a corrected address for low-quality inputs, shown to the customer
    for confirmation at checkout.

THE GUARDRAIL
    ALWAYS a suggestion the customer accepts or rejects. It never silently
    rewrites a delivery address. There is no code path in this module, or in the
    API, that applies a suggestion without an explicit customer action - which is
    why :class:`~rto_sentinel.contracts.explanation.AddressRepairSuggestion` has
    no "applied" field for this module to set.

WHY THIS IS THE MOST USEFUL OF THE FOUR
    Address quality is one of the strongest honest signals in the model and,
    unlike most risk features, it is *fixable at the point of sale*. A customer
    who adds a missing house number converts a HIGH-band order into a LOW-band
    one and gets their package. That is a better outcome than any friction rung,
    and it is the one place the language layer can reduce the number of orders
    needing an intervention at all rather than just explaining the ones that do.

STATUS: Phase 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.agents.provider import LLMProvider
    from rto_sentinel.contracts.explanation import AddressRepairSuggestion

SYSTEM_PROMPT = """You help a customer complete an incomplete Indian delivery \
address at checkout.

You will be given the address as entered, plus the city, state and PIN code. \
Suggest a clearer version that keeps every detail the customer provided and adds \
structure only where something is plainly missing, such as a house or flat \
number placeholder or a landmark slot. Never invent a house number, a street \
name, a locality, or any detail the customer did not supply. Never change the \
PIN code. If the address is already adequate, say so and suggest nothing."""


def suggest_repair(
    provider: LLMProvider,
    *,
    order_id: str,
    original_line: str,
    city: str,
    state: str,
    pincode: str,
) -> AddressRepairSuggestion:
    """Propose a clearer address for the customer to accept or reject.

    The Phase 5 implementation verifies that the suggestion contains no invented
    tokens - every substantive token in the suggestion must appear in the input
    or be a structural placeholder - before returning ``grounded=True``. A
    fabricated street name would be worse than no suggestion: it produces a
    confident-looking address that does not exist.
    """
    raise NotImplementedError("Address repair suggestion lands in Phase 5.")
