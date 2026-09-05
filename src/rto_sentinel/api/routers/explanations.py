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

The risk investigation agent runs a real tool loop: it is given the read-only
application tools and decides what to fetch. Nothing is pre-loaded into its
prompt, because a prompt stuffed with pre-fetched context is one where nobody can
tell which evidence the model actually used.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, status
from pydantic import BaseModel

from rto_sentinel.agents.address_repair import DEFERRAL_REASON
from rto_sentinel.agents.audit import AgentAuditRecord
from rto_sentinel.agents.confirmation_writer import ConfirmationWriter
from rto_sentinel.agents.digest_writer import DigestWriter
from rto_sentinel.agents.investigator import (
    InvestigationError,
    RiskInvestigation,
    RiskInvestigationAgent,
)
from rto_sentinel.agents.provider import AgentUnavailableError
from rto_sentinel.agents.reason_code_writer import write_explanation
from rto_sentinel.agents.tools import DigestRef
from rto_sentinel.api.deps import (
    AgentToolsetDep,
    AssessmentServiceDep,
    LLMProviderDep,
)
from rto_sentinel.api.errors import ApiError, ErrorCode, ErrorResponse
from rto_sentinel.api.security import enforce_rate_limit
from rto_sentinel.contracts.explanation import (
    ConfirmationMessage,
    Explanation,
    MerchantDigest,
    ReasonCode,
)
from rto_sentinel.decision.reason_codes import derive_reason_codes
from rto_sentinel.serving.assessment import OrderNotFoundError

# Authentication and rate limiting apply to every route below.
#
# Declared on the router rather than per handler so a new endpoint is
# protected by default. The alternative - remembering to add a dependency to
# each one - fails silently the first time somebody forgets, and the failure
# is an open endpoint.
router = APIRouter(
    prefix="/v1/explanations", dependencies=[Depends(enforce_rate_limit)], tags=["explanations"]
)

OrderIdPath = Annotated[str, Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$", max_length=64)]

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


class InvestigationResponse(BaseModel):
    """The agent's answer, plus the audit trail of how it got there."""

    investigation: RiskInvestigation
    audit: AgentAuditRecord


def _unavailable(reason: str) -> ApiError:
    """503 with the reason, never a substituted sentence."""
    return ApiError(
        ErrorCode.AGENT_UNAVAILABLE,
        f"{reason} The decision, its band and its reason codes are unaffected and "
        "remain available from /v1/orders/{order_id}/risk.",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


# ROUTE ORDER MATTERS HERE.
#
# `POST /{order_id}` is a catch-all: it matches any single path segment,
# including the literal ones. Registered first, it swallowed
# `POST /address-repair` as an order called "address-repair" and returned a 503
# about a missing model instead of the 501 that endpoint exists to give.
#
# Literal paths are therefore declared before the parameterised one. FastAPI
# matches in declaration order, so this is the fix and not a preference.


@router.post(
    "/address-repair",
    summary="DEFERRED: address repair is deliberately not implemented",
    responses={501: {"model": ErrorResponse, "description": "Deferred, with reasons"}},
)
def suggest_address_repair(
    order_id: Annotated[str, Query(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$", max_length=64)],
) -> dict[str, str]:
    """Refuse, with the reasoning.

    See ``agents.address_repair`` for the full argument. In short: the benchmark
    addresses are synthetic, the agent layer is denied raw address text by
    design, and correctness would need a postal reference dataset this project
    does not have. A confidently wrong "corrected" address is worse than a
    flagged incomplete one.
    """
    raise ApiError(
        ErrorCode.NOT_IMPLEMENTED,
        DEFERRAL_REASON,
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={"order_id": order_id, "status": "deferred"},
    )


@router.get(
    "/digest",
    response_model=MerchantDigest,
    summary="Weekly merchant digest: prose around SQL-computed figures",
    responses=_AGENT_RESPONSES,
)
def merchant_digest(
    provider: LLMProviderDep,
    toolset: AgentToolsetDep,
    merchant_id: Annotated[str, Query(max_length=64, description="Merchant to summarise")],
    days: Annotated[int, Query(ge=1, le=365)] = 7,
) -> MerchantDigest:
    """Figures come from SQL. The model writes prose and computes nothing.

    Returns 200 even with the language layer down: the figures are the substance
    and a digest without sentences is still a digest. `grounded=false` says why
    the prose is missing.
    """
    end = datetime.now(UTC)
    figures = toolset.get_digest_figures(
        DigestRef(merchant_id=merchant_id, period_start=end - timedelta(days=days), period_end=end)
    )
    return DigestWriter(provider).write(figures)


class ToolCatalogueEntry(BaseModel):
    """One tool, as the agent layer declares it."""

    name: str
    purpose: str
    permission: str
    input_schema: dict[str, object]


@router.get(
    "/tools",
    response_model=list[ToolCatalogueEntry],
    summary="The complete set of tools any agent may call",
)
def tool_catalogue() -> list[ToolCatalogueEntry]:
    """Every capability the language layer has, and its permission boundary.

    Published because "the agents are read-only" is a claim, and this is the list
    a reviewer checks it against.
    """
    from rto_sentinel.agents.tools import TOOL_SPECS

    return [
        ToolCatalogueEntry(
            name=spec.name,
            purpose=spec.purpose,
            permission=spec.permission,
            input_schema=spec.input_model.model_json_schema(),
        )
        for spec in TOOL_SPECS
    ]


@router.post(
    "/{order_id}/investigate",
    response_model=InvestigationResponse,
    summary="Ask why an order received its risk level; the agent retrieves its own evidence",
    responses=_AGENT_RESPONSES,
)
def investigate_order(
    order_id: OrderIdPath,
    provider: LLMProviderDep,
    toolset: AgentToolsetDep,
    question: Annotated[str, Query(min_length=5, max_length=500)] = (
        "Why did this order receive its risk level?"
    ),
    dataset_run: Annotated[str | None, Query(max_length=64)] = None,
) -> InvestigationResponse:
    """The risk investigation agent, running a real tool loop.

    The agent is given six read-only tools and no pre-loaded context. It decides
    what to fetch; the loop executes those calls against the live database and
    model. Nothing it can call writes, scores or overrides anything.

    Returns 503 when the language layer is unavailable. It does not substitute a
    scripted explanation - the reason codes are already available without one.
    """
    agent = RiskInvestigationAgent(provider, toolset)
    try:
        investigation = agent.investigate(order_id, question, dataset_run_id=dataset_run)
    except AgentUnavailableError as error:
        raise _unavailable(error.reason) from error
    except InvestigationError as error:
        raise ApiError(
            ErrorCode.GROUNDING_REJECTED,
            f"{error} The reason codes remain available.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        ) from error

    record = agent.audit_log.last()
    assert record is not None  # noqa: S101 - every run records exactly one entry
    return InvestigationResponse(investigation=investigation, audit=record)


@router.post(
    "/{order_id}",
    response_model=Explanation,
    summary="Render an existing decision's reason codes into one sentence",
    responses=_AGENT_RESPONSES,
)
def explain_decision(
    order_id: OrderIdPath,
    provider: LLMProviderDep,
    service: AssessmentServiceDep,
    dataset_run: Annotated[str | None, Query(max_length=64)] = None,
) -> Explanation:
    """Phrase the reason codes. Never computes or alters them.

    Unlike the investigation endpoint this returns 200 even when the language
    layer is down: the reason codes are the artefact, the sentence is a
    convenience, and an `Explanation` with `grounded=false` carries the codes and
    says why there is no sentence.
    """
    try:
        assessment = service.assess(
            order_id, dataset_run_id=dataset_run, include_contributions=True
        )
    except OrderNotFoundError as error:
        raise ApiError(
            ErrorCode.ORDER_NOT_FOUND,
            str(error),
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"order_id": order_id},
        ) from error

    codes: tuple[ReasonCode, ...] = derive_reason_codes(list(assessment.score.contributions))
    return write_explanation(provider, order_id=order_id, reason_codes=codes)


@router.post(
    "/{order_id}/confirmation",
    response_model=ConfirmationMessage,
    summary="Draft a neutral confirmation message for a frictioned order",
    responses=_AGENT_RESPONSES,
)
def draft_confirmation_message(
    order_id: OrderIdPath,
    provider: LLMProviderDep,
    service: AssessmentServiceDep,
    channel: Annotated[str, Query(pattern=r"^(whatsapp|sms)$")] = "whatsapp",
    language: Annotated[str, Query(max_length=16)] = "en-IN",
    dataset_run: Annotated[str | None, Query(max_length=64)] = None,
) -> ConfirmationMessage:
    """Fill a human-reviewed template. Drafts only - it does not send.

    The action comes from the decision engine, never from the agent. A LOW-band
    order has no confirmation to draft and the endpoint says so rather than
    writing one anyway.
    """
    try:
        assessment = service.assess(
            order_id, dataset_run_id=dataset_run, include_contributions=False
        )
    except OrderNotFoundError as error:
        raise ApiError(
            ErrorCode.ORDER_NOT_FOUND,
            str(error),
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"order_id": order_id},
        ) from error

    action = assessment.decision.action.value
    if action == "none":
        raise ApiError(
            ErrorCode.VALIDATION_FAILED,
            f"order {order_id} is in the {assessment.decision.band.value} band and receives "
            "no friction, so there is no confirmation to draft. Drafting one anyway would "
            "mean messaging a customer the policy chose not to contact.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"band": assessment.decision.band.value, "action": action},
        )

    return ConfirmationWriter(provider).draft(
        order_id=order_id, action=action, channel=channel, language=language
    )


class AgentStatusResponse(BaseModel):
    """Whether the language layer can run, and what it would use."""

    available: bool
    reason: str | None = None
    provider: str
    model: str
    required_environment_variable: str
    enable_switch: str
    tools: list[str]
    note: str


@router.get(
    "/status",
    response_model=AgentStatusResponse,
    summary="Whether the language layer is configured, and what it needs",
)
def agent_status(provider: LLMProviderDep) -> AgentStatusResponse:
    """Never raises. "No language layer" is the answer, not a failure to answer."""
    from rto_sentinel.agents.provider import API_KEY_VARIABLE, ENABLE_VARIABLE
    from rto_sentinel.agents.tools import TOOL_SPECS

    return AgentStatusResponse(
        available=provider.available,
        reason=getattr(provider, "reason", None),
        provider="anthropic",
        model=provider.model,
        required_environment_variable=API_KEY_VARIABLE,
        enable_switch=ENABLE_VARIABLE,
        tools=[spec.name for spec in TOOL_SPECS],
        note=(
            "The risk system does not depend on this layer. Scoring, calibration, the "
            "decision engine and every endpoint outside /v1/explanations work with it off."
        ),
    )
