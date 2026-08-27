"""Language-layer endpoints. Everything here degrades gracefully.

``POST /v1/explanations/{order_id}``  one plain sentence from the reason codes
``POST /v1/explanations/confirmation`` draft a customer confirmation message
``POST /v1/explanations/address-repair`` suggest an address correction
``GET  /v1/explanations/digest``      the weekly merchant digest

THE CONTRACT THESE ENDPOINTS MAKE
---------------------------------
Every response carries ``grounded``. When it is False the caller gets the
deterministic artefact - the reason codes, the figures table - and a reason why
the prose is missing. When the language layer is entirely unavailable the
response is ``503 AGENT_UNAVAILABLE``, which the console renders as a quiet
"explanation unavailable" next to a fully functional decision.

WHAT THESE ENDPOINTS CANNOT DO
------------------------------
Change a decision. There is no path from any handler in this file into the
decision engine or the decision log. They read a decision that was already made
and describe it. That is the entire capability surface, by construction: the
response models here come from ``contracts.explanation``, none of which has a
field for a probability, a threshold, a band or an action.

STATUS: Phase 5.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from rto_sentinel.api.deps import DbSession, LLMProviderDep
from rto_sentinel.api.errors import ErrorResponse, not_implemented
from rto_sentinel.contracts.explanation import (
    AddressRepairSuggestion,
    ConfirmationMessage,
    Explanation,
    MerchantDigest,
)

router = APIRouter(prefix="/v1/explanations", tags=["explanations"])

_AGENT_RESPONSES: dict[int | str, dict[str, object]] = {
    503: {
        "model": ErrorResponse,
        "description": "Language layer unavailable; the decision itself is unaffected.",
    },
    422: {
        "model": ErrorResponse,
        "description": "Generation rejected by the grounding validator; raw reason codes returned.",
    },
}


@router.post(
    "/{order_id}",
    response_model=Explanation,
    summary="Render an existing decision's reason codes into one sentence",
    responses=_AGENT_RESPONSES,
)
def explain_decision(
    order_id: str,
    provider: LLMProviderDep,
    session: DbSession,
) -> Explanation:
    """Phrase the reason codes. Never computes or alters them."""
    raise not_implemented("Reason-code phrasing", "Phase 5 (agent layer)")


@router.post(
    "/confirmation",
    response_model=ConfirmationMessage,
    summary="Draft a neutral confirmation message for a HIGH-band order",
    responses=_AGENT_RESPONSES,
)
def draft_confirmation_message(
    order_id: str,
    provider: LLMProviderDep,
    session: DbSession,
    channel: str = Query(default="whatsapp"),
    language: str = Query(default="en-IN"),
) -> ConfirmationMessage:
    """Fill a human-reviewed template. Drafts only - it does not send."""
    raise not_implemented("Confirmation drafting", "Phase 5 (agent layer)")


@router.post(
    "/address-repair",
    response_model=AddressRepairSuggestion,
    summary="Suggest a clearer delivery address for the customer to accept or reject",
    responses=_AGENT_RESPONSES,
)
def suggest_address_repair(
    order_id: str,
    provider: LLMProviderDep,
    session: DbSession,
) -> AddressRepairSuggestion:
    """Propose a correction. Never silently rewrites a delivery address."""
    raise not_implemented("Address repair suggestion", "Phase 5 (agent layer)")


@router.get(
    "/digest",
    response_model=MerchantDigest,
    summary="Weekly merchant digest: prose around SQL-computed figures",
    responses=_AGENT_RESPONSES,
)
def merchant_digest(
    provider: LLMProviderDep,
    session: DbSession,
    merchant_id: str = Query(description="Merchant to summarise"),
) -> MerchantDigest:
    """Figures come from SQL. The model writes prose and computes nothing."""
    raise not_implemented("Merchant digest", "Phase 5 (agent layer)")
