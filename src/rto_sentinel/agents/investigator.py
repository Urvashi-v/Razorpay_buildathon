"""The risk investigation agent: why did this order get this risk level?

WHAT IT ACTUALLY DOES
=====================
Runs a real tool-use loop against the Messages API. The model is given the six
read-only application tools and no context beyond the question; it decides what
to fetch, the loop executes those calls against the live database and model, and
the results go back as tool results. Nothing is pre-loaded into the prompt,
because a prompt stuffed with pre-fetched context is a prompt where nobody can
tell which evidence the model actually used.

WHAT IT CANNOT DO, STRUCTURALLY
===============================
It has no tool that writes, scores, re-bands or approves anything. The probability
and the band arrive through ``get_risk_prediction`` and ``get_economic_decision``
as facts already established by the deterministic pipeline. If the model writes a
different number in its answer, the number in the response is still the one from
the tool - :class:`RiskInvestigation` carries the retrieved values as separate
fields, not as prose the model composed.

That is the difference between "the agent explains the decision" and "the agent
makes one". The prose is a rendering; the fields are the record.

THE THREE OUTCOMES
==================
1. **Grounded explanation.** Evidence retrieved, structured output validated,
   nothing fabricated.
2. **Insufficient evidence.** The tools came back empty - an order that does not
   exist, a model that produced no attributions. The agent says so. This is a
   success, not a failure: an honest "I cannot tell you" is the correct output.
3. **Rejected.** The model produced something, and the grounding validator
   refused it. The reason codes are returned without prose.

There is no fourth outcome where a plausible sentence is produced anyway.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rto_sentinel.agents.audit import AuditBuilder, AuditLog
from rto_sentinel.agents.grounding import (
    validate_evidence_references,
    validate_feature_grounding,
)
from rto_sentinel.agents.provider import AgentUnavailableError
from rto_sentinel.agents.tools import anthropic_tool_definitions

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.agents.provider import LLMProvider

#: Turns the loop will run before giving up. Six tools exist; a model needing
#: more than this many rounds is looping, not investigating.
MAX_TURNS = 6

AGENT_TYPE = "risk_investigation"

SYSTEM_PROMPT = """\
You are a risk investigation assistant for RTO Sentinel, a return-to-origin risk \
system for Indian cash-on-delivery commerce. An operations associate is asking \
why a particular order received the risk level it did.

WHAT YOU ARE AND ARE NOT
You explain decisions that have already been made by a deterministic pipeline. \
You do not make them. You cannot compute a risk probability, choose a threshold, \
change a band, or approve or block an order, and you must never write as though \
you could.

HOW TO WORK
Call the tools to retrieve evidence. Do not answer from assumption. The tools are \
the only evidence you have; if a tool reports found=false, or returns a reason \
explaining that something is absent, that absence IS the finding and you must \
report it as such.

RULES YOU MUST FOLLOW
1. Only name risk drivers that appear in get_model_explanation's \
permitted_features. Naming any other driver - chargebacks, credit history, fraud \
scores, anything this system does not measure - is a fabrication, and the output \
will be rejected.
2. Quote the probability and threshold exactly as the tools return them. Do not \
round them into a different number and do not recompute anything.
3. The intervention success rate and abandonment rate are ASSUMPTIONS that have \
never been measured. If you mention the expected rupee value, say that it rests \
on assumptions.
4. If the evidence is insufficient to explain the decision, set \
sufficient_evidence to false and say what is missing. Never fill a gap with a \
plausible guess.
5. The labels in this system are simulated benchmark data, not real outcomes.

OUTPUT
Reply with a single JSON object and nothing else:
{
  "sufficient_evidence": true | false,
  "summary": "one or two sentences an ops associate can act on",
  "key_drivers": ["feature_name", ...],
  "evidence_used": ["tool_name", ...],
  "uncertainty": "what you could not establish, or an empty string",
  "caveats": ["any assumption the answer rests on", ...]
}
"""


class RiskInvestigation(BaseModel):
    """The agent's structured answer, with the retrieved facts kept separate.

    ``probability``, ``band`` and ``threshold`` are copied from the tool results,
    not parsed out of the model's prose. A model that writes the wrong number in
    its summary cannot change what this object reports.
    """

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(max_length=64)
    sufficient_evidence: bool
    summary: str = Field(max_length=1200)
    key_drivers: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)
    uncertainty: str = Field(default="", max_length=800)
    caveats: list[str] = Field(default_factory=list)

    # --- retrieved, never generated ---------------------------------------
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    band: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    model_version: str | None = None
    reason_codes: list[str] = Field(default_factory=list)

    # --- provenance --------------------------------------------------------
    generated_at: datetime
    llm_model: str = Field(max_length=64)
    grounded: bool = True
    rejection_reason: str | None = None


class ModelAnswer(BaseModel):
    """The JSON the model is asked to return. Anything else is rejected."""

    model_config = ConfigDict(extra="forbid")

    sufficient_evidence: bool
    summary: str = Field(max_length=1200)
    key_drivers: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)
    uncertainty: str = Field(default="", max_length=800)
    caveats: list[str] = Field(default_factory=list)


class InvestigationError(RuntimeError):
    """Raised when the agent cannot produce any answer at all."""


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a reply that may be wrapped in prose.

    Models sometimes fence the JSON or precede it with a sentence. Tolerating
    that is not the same as tolerating a malformed answer: what comes out is
    still validated against :class:`ModelAnswer`, and a reply with no object at
    all fails.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1] if "```" in stripped[3:] else stripped[3:]
        stripped = stripped.removeprefix("json").strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        msg = "the model did not return a JSON object"
        raise InvestigationError(msg)
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as error:
        msg = f"the model returned malformed JSON ({error.msg})"
        raise InvestigationError(msg) from error
    if not isinstance(parsed, dict):
        msg = "the model returned JSON that is not an object"
        raise InvestigationError(msg)
    return parsed


class RiskInvestigationAgent:
    """Runs the tool loop and validates what comes back."""

    def __init__(
        self,
        provider: LLMProvider,
        toolset: Any,
        *,
        audit_log: AuditLog | None = None,
        max_turns: int = MAX_TURNS,
    ) -> None:
        self._provider = provider
        self._toolset = toolset
        self._audit = audit_log or AuditLog()
        self._max_turns = max_turns

    @property
    def audit_log(self) -> AuditLog:
        return self._audit

    def investigate(
        self, order_id: str, question: str, *, dataset_run_id: str | None = None
    ) -> RiskInvestigation:
        """Answer one question about one order, from retrieved evidence only."""
        from rto_sentinel.serving.agent_tools import invoke

        builder = AuditBuilder(
            agent_type=AGENT_TYPE,
            request=question,
            provider="anthropic",
            model=self._provider.model,
            subject_id=order_id,
        )
        builder.dataset_run_id = dataset_run_id

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"{question}\n\nThe order in question is {order_id}."
                    + (f" Dataset run: {dataset_run_id}." if dataset_run_id else "")
                ),
            }
        ]
        retrieved: dict[str, Any] = {}
        permitted_features: list[str] = []

        try:
            answer = self._run_loop(builder, messages, retrieved, permitted_features, invoke)
        except AgentUnavailableError as error:
            self._audit.record(builder.finish(error=error.reason))
            raise
        except InvestigationError as error:
            self._audit.record(builder.finish(error=str(error)))
            raise

        return self._finalise(builder, order_id, answer, retrieved, tuple(permitted_features))

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def _run_loop(
        self,
        builder: AuditBuilder,
        messages: list[dict[str, Any]],
        retrieved: dict[str, Any],
        permitted_features: list[str],
        invoke: Any,
    ) -> ModelAnswer:
        tools = anthropic_tool_definitions()

        for _ in range(self._max_turns):
            completion = self._provider.converse(
                system=SYSTEM_PROMPT, messages=messages, tools=tools
            )
            builder.llm_turns += 1
            builder.add_usage(completion.input_tokens, completion.output_tokens)

            if not completion.wants_tools:
                return self._parse(completion.text)

            messages.append({"role": "assistant", "content": completion.raw_content})
            results: list[dict[str, Any]] = []

            for call in completion.tool_calls:
                result, invocation = invoke(self._toolset, call.name, call.arguments)
                builder.add_tool(invocation)

                if result is None:
                    payload = {"error": invocation.error}
                else:
                    retrieved[call.name] = result
                    payload = json.loads(result.model_dump_json())
                    if call.name == "get_model_explanation":
                        permitted_features.extend(result.permitted_features)

                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(payload, default=str),
                        "is_error": result is None,
                    }
                )

            messages.append({"role": "user", "content": results})

        msg = (
            f"the agent did not reach an answer within {self._max_turns} turns. No "
            "explanation is returned; the reason codes remain available."
        )
        raise InvestigationError(msg)

    def _parse(self, text: str) -> ModelAnswer:
        payload = _extract_json(text)
        try:
            return ModelAnswer.model_validate(payload)
        except ValidationError as error:
            count = len(error.errors())
            msg = (
                f"the model's answer failed schema validation ({count} problem(s)). "
                "Rejected rather than repaired."
            )
            raise InvestigationError(msg) from error

    # ------------------------------------------------------------------
    # validation and assembly
    # ------------------------------------------------------------------

    def _finalise(
        self,
        builder: AuditBuilder,
        order_id: str,
        answer: ModelAnswer,
        retrieved: dict[str, Any],
        permitted_features: tuple[str, ...],
    ) -> RiskInvestigation:
        prediction = retrieved.get("get_risk_prediction")
        decision = retrieved.get("get_economic_decision")
        explanation = retrieved.get("get_model_explanation")

        verdict = validate_feature_grounding(answer.summary, permitted_features)
        if verdict.grounded:
            verdict = validate_evidence_references(
                answer.summary,
                cited=tuple(answer.evidence_used),
                available=tuple(retrieved),
            )
        if verdict.grounded and answer.key_drivers and permitted_features:
            unknown = sorted(set(answer.key_drivers) - set(permitted_features))
            if unknown:
                verdict = type(verdict)(
                    grounded=False,
                    rejection_reason=(
                        f"the answer lists drivers that are not in the model's "
                        f"attributions: {unknown}. Permitted: {sorted(permitted_features)}."
                    ),
                    offending_terms=tuple(unknown),
                )

        investigation = RiskInvestigation(
            order_id=order_id,
            sufficient_evidence=answer.sufficient_evidence,
            summary=answer.summary,
            key_drivers=answer.key_drivers,
            evidence_used=answer.evidence_used,
            uncertainty=answer.uncertainty,
            caveats=answer.caveats,
            probability=getattr(prediction, "probability", None),
            band=getattr(decision, "band", None),
            threshold=getattr(decision, "threshold", None),
            model_version=getattr(prediction, "model_version", None),
            reason_codes=list(getattr(explanation, "reason_codes", []) or []),
            generated_at=datetime.now(UTC),
            llm_model=self._provider.model,
            grounded=verdict.grounded,
            rejection_reason=verdict.rejection_reason,
        )

        self._audit.record(
            builder.finish(
                output=json.loads(investigation.model_dump_json()),
                grounded=verdict.grounded,
                rejection_reason=verdict.rejection_reason,
            )
        )
        return investigation
