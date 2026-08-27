"""Language job 2 of 4: draft the confirmation message for a HIGH-band order.

SPEC section 08.

WHAT IT DOES
    Writes the WhatsApp or SMS confirmation for HIGH-band orders, in the
    merchant's tone and the customer's likely language.

THE GUARDRAIL
    Templated with variable slots. The message NEVER states or implies that the
    customer is suspected of anything. A human reviews the template once, before
    it goes live - not each message. Generated bodies pass
    ``agents.grounding.validate_neutral_framing`` before they can be sent.

WHY NEUTRAL FRAMING IS A HARD REQUIREMENT
    SPEC section 09 lists it under human safeguards: customers are never told
    they are flagged. A confirmation request that reads as an accusation converts
    a cheap, positive-sum intervention into a reputational cost - and it lands on
    a customer who, given the model's precision, may well have done nothing at
    all. The message has to read as ordinary delivery logistics because for most
    recipients that is exactly what it is.

STATUS: Phase 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.agents.provider import LLMProvider
    from rto_sentinel.contracts.explanation import ConfirmationMessage

SYSTEM_PROMPT = """You write short delivery-confirmation messages for an Indian \
e-commerce merchant.

The message asks the customer to confirm their order before dispatch. It must \
read as routine logistics. Never suggest the customer is suspected of anything. \
Never use the words fraud, risk, suspicious, flagged, verification, or security. \
Never explain why this order in particular was selected. Fill only the variable \
slots you are given. Keep it under 40 words, warm and matter-of-fact."""


def draft_confirmation(
    provider: LLMProvider,
    *,
    order_id: str,
    template_id: str,
    template_body: str,
    slots: dict[str, str],
    channel: str,
    language: str,
) -> ConfirmationMessage:
    """Fill a human-reviewed template and verify neutral framing.

    ``neutral_framing_verified`` is set only when the validator passed. The
    sending path in the merchant's own system is expected to refuse any message
    where it is False - drafting and sending are separate concerns, and this
    module only drafts.
    """
    raise NotImplementedError("Confirmation drafting lands in Phase 5.")
