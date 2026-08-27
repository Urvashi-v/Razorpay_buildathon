"""Language-layer contracts - everything the LLM is allowed to touch.

SPEC section 08: the LLM is downstream of the decision, never inside it. These
types exist so that boundary is expressed in the type system rather than in a
comment. Note what is absent: no probability, no threshold, no band, no action.
An :class:`Explanation` can only ever *describe* a decision that already exists.

Every generated artefact carries :attr:`GroundedOutput.grounded` and the list of
features it was permitted to mention. The validator in
``rto_sentinel.agents.grounding`` rejects any output naming a feature that was
not in its input - which is what stops the model inventing a cause.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class GroundedOutput(BaseModel):
    """Base for anything an LLM produced. Carries its own provenance."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    llm_model: str = Field(max_length=64)
    grounded: bool = Field(description="False when the grounding validator rejected the generation")
    rejection_reason: str | None = Field(
        default=None, description="Why the validator rejected it, when it did"
    )


class ReasonCode(BaseModel):
    """A deterministic, machine-generated reason identifier.

    Produced by the decision layer from SHAP contributions - NOT by an LLM. The
    LLM only ever renders these into a sentence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(max_length=64, description="e.g. ADDRESS_INCOMPLETE")
    feature: str = Field(max_length=128)
    family: str = Field(max_length=64)
    contribution: float
    direction: str


class Explanation(GroundedOutput):
    """One plain sentence an ops associate can act on.

    ``permitted_features`` is the exact allow-list handed to the generator. If
    the sentence mentions anything outside it, ``grounded`` is False and the
    console shows the raw reason codes instead. Degrading to a bar chart is an
    acceptable failure; inventing a cause is not.
    """

    order_id: str = Field(max_length=64)
    sentence: str = Field(max_length=500)
    reason_codes: tuple[ReasonCode, ...]
    permitted_features: tuple[str, ...]


class ConfirmationMessage(GroundedOutput):
    """Customer-facing confirmation copy for a HIGH-band order.

    SPEC section 09: customers are never told they are "flagged". The message
    must read as a routine delivery confirmation. ``neutral_framing_verified``
    records that the validator checked for accusatory language.
    """

    order_id: str = Field(max_length=64)
    channel: str = Field(max_length=32, description="whatsapp | sms")
    language: str = Field(max_length=16, description="BCP-47 tag")
    body: str = Field(max_length=1000)
    template_id: str = Field(max_length=64, description="Human-reviewed template this filled")
    neutral_framing_verified: bool = Field(default=False)


class AddressRepairSuggestion(GroundedOutput):
    """A proposed correction to a low-quality address.

    ALWAYS a suggestion the customer accepts or rejects. Nothing in this system
    silently rewrites a delivery address, so there is no "applied" field here by
    design - acceptance is recorded against the order, by the customer's action.
    """

    order_id: str = Field(max_length=64)
    original_line: str = Field(max_length=512)
    suggested_line: str = Field(max_length=512)
    fields_changed: tuple[str, ...]
    confidence_note: str = Field(default="", max_length=300)


class DigestSection(BaseModel):
    """One section of the weekly merchant digest.

    ``figures`` come from SQL. The LLM writes ``prose`` around numbers it is
    handed and is not permitted to compute - so a wrong number here is a bug in
    the query, never a hallucination.
    """

    model_config = ConfigDict(extra="forbid")

    heading: str = Field(max_length=120)
    figures: dict[str, float]
    prose: str = Field(max_length=2000)


class MerchantDigest(GroundedOutput):
    """The weekly summary: where losses concentrated, what paid for itself."""

    merchant_id: str = Field(max_length=64)
    period_start: datetime
    period_end: datetime
    sections: tuple[DigestSection, ...]
    computed_figures: dict[str, float] = Field(
        default_factory=dict,
        description="Every figure the digest is allowed to mention, computed in SQL",
    )
