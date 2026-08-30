"""Tools the agent layer may call. All read-only, all narrow, all logged.

This is the complete list of capabilities available to the language layer. It is
short on purpose, and everything absent from it is absent deliberately.

WHAT THE AGENTS CAN DO
----------------------
* Read an order, its customer's prior history, and its delivery events.
* Read a risk prediction and the model attributions behind it.
* Read a decision that has already been made, with its reason codes.
* Read aggregate figures computed by SQL, for the weekly digest.

WHAT NO TOOL HERE PERMITS
-------------------------
* Writing a risk score, a threshold, a band or an action.
* Overriding, re-running, or re-scoring a decision.
* Reading raw customer PII. Tools return hashed identities and derived
  attributes; nothing returns a name, a phone number, or a full address.
* Reaching any system outside this application's own database.
* Sending a message to a customer. The agent *drafts*; a human-reviewed
  template and the merchant's own messaging system send.

The asymmetry is the whole design: agents can read what happened and describe
it, and they cannot change what happens.

WHY THIS MODULE DEFINES AN INTERFACE AND NOT AN IMPLEMENTATION
==============================================================
``tests/architecture/test_layering.py`` forbids this package from importing
``decision``, ``models``, ``features``, ``data`` or ``eval`` - the layers where a
probability, a threshold or an action could actually be produced. That rule is
the mechanical form of "the LLM is downstream of the decision", and it would be
worth very little if this module simply imported the decision engine to read from
it.

So the package declares the *contract* - schemas, the protocol, the permission
boundary - and the concrete toolset is built in ``serving.agent_tools``, which is
already allowed to compose those layers. The agent receives it as an argument. An
agent cannot reach for a capability nobody handed it.

EVERY TOOL RETURNS A TYPED RESULT OR A TYPED ABSENCE
====================================================
No tool returns ``None`` to mean "not found" and no tool raises to mean "no
data". Each returns a result object carrying ``found`` and, when false, a
``reason``. That is what lets the grounding rules downstream distinguish "the
evidence says no" from "the evidence is missing" - a distinction an LLM will
otherwise paper over, and the one this whole layer exists to preserve.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# tool inputs
# ---------------------------------------------------------------------------


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrderRef(_Schema):
    """Identifies one order. The only handle most tools need."""

    order_id: str = Field(
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$",
        description="The order identifier, e.g. ORD-00008874",
    )
    dataset_run_id: str | None = Field(
        default=None,
        max_length=64,
        description="Disambiguates ids shared across benchmark runs; newest run by default",
    )


class CustomerHistoryRef(_Schema):
    """Identifies a customer's history, as of one order's timestamp."""

    order_id: str = Field(
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$",
        description="History is read AS OF this order - never including it or anything later",
    )
    dataset_run_id: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=10, ge=1, le=50, description="Most recent prior orders to return")


class DigestRef(_Schema):
    merchant_id: str = Field(max_length=64)
    period_start: datetime
    period_end: datetime


# ---------------------------------------------------------------------------
# tool outputs
# ---------------------------------------------------------------------------


class ToolResult(_Schema):
    """Base for every tool output. Absence is a value, not an exception."""

    found: bool = Field(description="False when the evidence does not exist")
    reason: str | None = Field(default=None, description="Why nothing was found, when nothing was")


class OrderFacts(ToolResult):
    """The order as stored. Columns only - nothing derived, nothing inferred."""

    order_id: str | None = None
    merchant_id: str | None = None
    customer_hash: str | None = Field(
        default=None, description="Hashed identity. There is no tool that returns a name."
    )
    ordered_at: datetime | None = None
    payment_method: str | None = None
    is_cod: bool | None = None
    order_value_inr: float | None = None
    discount_inr: float | None = None
    discount_depth: float | None = None
    item_count: int | None = None
    category: str | None = None
    courier_partner: str | None = None
    pincode_tier: str | None = Field(
        default=None, description="Tier only. The raw pincode is never returned to an agent."
    )
    address_completeness: dict[str, Any] = Field(
        default_factory=dict, description="Derived address-quality signals, never the address text"
    )
    split: str | None = None
    outcome: str | None = Field(default=None, description="Terminal state, or null if unresolved")
    is_rto: bool | None = Field(
        default=None, description="Null when the outcome has not matured. Never defaulted."
    )
    resolved_at: datetime | None = None


class PriorOrder(_Schema):
    """One earlier order by the same customer, resolved before the one in question."""

    order_id: str
    ordered_at: datetime
    order_value_inr: float
    payment_method: str
    outcome: str | None = None
    is_rto: bool | None = None
    resolved_at: datetime | None = None


class CustomerHistoryFacts(ToolResult):
    """What the merchant knew about this customer when the order was placed.

    AS-OF, STRICTLY. Only orders that had already *resolved* before this order's
    timestamp are included. That is the same cutoff the model's features use, so
    an agent explaining a score cannot cite history the model did not have -
    which would be a true statement and a wrong explanation.
    """

    customer_hash: str | None = None
    as_of: datetime | None = None
    prior_order_count: int = 0
    prior_rto_count: int = 0
    prior_rto_rate: float | None = None
    days_since_last_order: float | None = None
    is_new_customer: bool | None = None
    recent_orders: list[PriorOrder] = Field(default_factory=list)


class RiskPredictionFacts(ToolResult):
    """The model's output. Read, never produced here."""

    order_id: str | None = None
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_score: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    calibration_method: str | None = None
    feature_version: str | None = None
    scored_at: datetime | None = None
    null_features: list[str] = Field(
        default_factory=list,
        description="Features with no value for this order. Cold start, not an error.",
    )
    context_rows: int | None = Field(
        default=None, description="Rows of merchant history the aggregates were computed over"
    )


class FeatureAttribution(_Schema):
    """One feature's contribution, as the model computed it."""

    feature: str
    family: str
    value: Any = None
    contribution: float
    direction: str


class ModelExplanationFacts(ToolResult):
    """SHAP attributions and the reason codes derived from them.

    ``permitted_features`` is the allow-list the grounding validator enforces: an
    explanation naming anything outside it is rejected rather than repaired.
    """

    order_id: str | None = None
    attributions: list[FeatureAttribution] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    permitted_features: list[str] = Field(default_factory=list)
    note: str | None = Field(
        default=None,
        description="Set when the model produced no attributions - heuristic rungs have none",
    )


class EconomicDecisionFacts(ToolResult):
    """The decision, its threshold, and the economics that derived the threshold."""

    order_id: str | None = None
    band: str | None = None
    action: str | None = None
    flagged: bool | None = None
    threshold: float | None = None
    threshold_source: str | None = None
    expected_value_inr: float | None = None
    human_review_required: bool | None = None
    appeal_available: bool | None = None
    is_control_holdout: bool | None = None
    engine_version: str | None = None
    cost_profile: str | None = None
    rto_cost_inr: float | None = None
    contribution_margin_inr: float | None = None
    cost_false_positive_inr: float | None = None
    saving_true_positive_inr: float | None = None
    assumed_intervention_success_rate: float | None = Field(
        default=None, description="ASSUMED, never measured. Must be described as an assumption."
    )
    assumed_abandonment_rate: float | None = Field(
        default=None, description="ASSUMED, never measured. Must be described as an assumption."
    )
    decided_at: datetime | None = None


class OrderEvent(_Schema):
    """One delivery event, as recorded."""

    sequence: int
    event_type: str
    occurred_at: datetime
    detail: str | None = None


class OrderEventFacts(ToolResult):
    """The delivery timeline. Empty is a legitimate answer for an undispatched order."""

    order_id: str | None = None
    events: list[OrderEvent] = Field(default_factory=list)
    dispatched_at: datetime | None = None
    first_attempt_at: datetime | None = None


class DigestFigures(ToolResult):
    """Aggregate figures for one merchant over one period, computed in SQL.

    Handed to the digest writer as the complete set of numbers it may mention.
    The LLM does not compute, sum, or infer any figure - a wrong number in the
    digest is therefore a bug in a query, which is findable, rather than a
    hallucination, which is not.
    """

    merchant_id: str | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    figures: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# the protocol
# ---------------------------------------------------------------------------


class AgentToolset(Protocol):
    """Read-only accessors available to the language jobs.

    Implemented by ``serving.agent_tools.ApplicationToolset``. An agent is handed
    one of these; it has no other route to data.
    """

    def get_order(self, ref: OrderRef) -> OrderFacts: ...

    def get_customer_history(self, ref: CustomerHistoryRef) -> CustomerHistoryFacts: ...

    def get_risk_prediction(self, ref: OrderRef) -> RiskPredictionFacts: ...

    def get_model_explanation(self, ref: OrderRef) -> ModelExplanationFacts: ...

    def get_economic_decision(self, ref: OrderRef) -> EconomicDecisionFacts: ...

    def get_relevant_order_events(self, ref: OrderRef) -> OrderEventFacts: ...

    def get_digest_figures(self, ref: DigestRef) -> DigestFigures: ...


# ---------------------------------------------------------------------------
# the declared tool surface
# ---------------------------------------------------------------------------


class ToolSpec(BaseModel):
    """One tool, fully declared: what it does and what it may not do."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    purpose: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    permission: str = Field(description="The boundary this tool is bound by")

    def anthropic_definition(self) -> dict[str, Any]:
        """The tool as the Messages API expects to receive it.

        The permission boundary is appended to the description the model sees.
        Not as a security control - the control is that no write path exists -
        but because a model told what a tool is *for* uses it more sensibly than
        one left to infer it from a name.
        """
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": f"{self.purpose}\n\nPermission boundary: {self.permission}",
            "input_schema": schema,
        }


READ_ONLY = "Read-only. Cannot write, score, re-score, override or send anything."
NO_PII = (
    "Returns hashed identities and derived attributes only. Never names, phones or address text."
)

TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_order",
        purpose=(
            "Fetch one order as stored: value, payment method, category, courier, pincode "
            "tier, address-quality signals, and its outcome if it has resolved."
        ),
        input_model=OrderRef,
        output_model=OrderFacts,
        permission=f"{READ_ONLY} {NO_PII}",
    ),
    ToolSpec(
        name="get_customer_history",
        purpose=(
            "Fetch what was known about this customer when the order was placed: prior "
            "order count, prior RTO count and rate, and the most recent prior orders. "
            "Strictly as-of the order's timestamp - only orders that had already resolved."
        ),
        input_model=CustomerHistoryRef,
        output_model=CustomerHistoryFacts,
        permission=(
            f"{READ_ONLY} {NO_PII} As-of the referenced order; never includes the order "
            "itself or anything resolved later."
        ),
    ),
    ToolSpec(
        name="get_risk_prediction",
        purpose=(
            "Fetch the calibrated RTO probability the model produced for this order, with "
            "the model and feature versions that produced it."
        ),
        input_model=OrderRef,
        output_model=RiskPredictionFacts,
        permission=(
            f"{READ_ONLY} Returns the model's output. It cannot compute, adjust or "
            "re-score a probability."
        ),
    ),
    ToolSpec(
        name="get_model_explanation",
        purpose=(
            "Fetch the per-feature SHAP attributions behind this order's probability and "
            "the deterministic reason codes derived from them."
        ),
        input_model=OrderRef,
        output_model=ModelExplanationFacts,
        permission=(
            f"{READ_ONLY} The returned permitted_features list is the ONLY set of drivers "
            "an explanation may name. Anything else is rejected by the grounding validator."
        ),
    ),
    ToolSpec(
        name="get_economic_decision",
        purpose=(
            "Fetch the decision taken for this order: band, action, the cost-derived "
            "threshold and the merchant economics behind it."
        ),
        input_model=OrderRef,
        output_model=EconomicDecisionFacts,
        permission=(
            f"{READ_ONLY} Reads a decision the deterministic engine already made. It "
            "cannot choose a threshold, change a band, or approve or block an order."
        ),
    ),
    ToolSpec(
        name="get_relevant_order_events",
        purpose=(
            "Fetch the delivery timeline for this order: dispatch, attempts and their "
            "outcomes, in sequence."
        ),
        input_model=OrderRef,
        output_model=OrderEventFacts,
        permission=f"{READ_ONLY} {NO_PII}",
    ),
)

TOOLS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


def anthropic_tool_definitions() -> list[dict[str, Any]]:
    """Every tool, in the shape the Messages API expects."""
    return [spec.anthropic_definition() for spec in TOOL_SPECS]
