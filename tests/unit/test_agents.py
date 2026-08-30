"""The agent layer: what it retrieves, what it refuses, and what it cannot do.

ABOUT THE TEST DOUBLE
=====================
:class:`ScriptedProvider` replays a fixed list of completions. It lives here, in
``tests/``, and there is deliberately no equivalent in ``src/``: a double in a
test proves the orchestration works, while a double in the product hides that it
does not. ``test_no_scripted_responder_ships_in_the_product`` asserts that
distinction rather than trusting it.

Everything the double stands in for is one HTTP call. The tool loop, the
schemas, the grounding validators, the audit trail and the refusals are all real
code running against real data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rto_sentinel.agents.audit import AgentAuditRecord, AuditLog
from rto_sentinel.agents.confirmation_writer import ConfirmationWriter
from rto_sentinel.agents.digest_writer import DigestWriter
from rto_sentinel.agents.grounding import (
    validate_evidence_references,
    validate_feature_grounding,
    validate_figure_grounding,
    validate_neutral_framing,
)
from rto_sentinel.agents.investigator import (
    InvestigationError,
    RiskInvestigationAgent,
)
from rto_sentinel.agents.provider import (
    AgentUnavailableError,
    Completion,
    ToolCall,
    UnavailableProvider,
    get_provider,
)
from rto_sentinel.agents.tools import (
    TOOL_SPECS,
    CustomerHistoryFacts,
    DigestFigures,
    EconomicDecisionFacts,
    ModelExplanationFacts,
    OrderFacts,
    RiskPredictionFacts,
    anthropic_tool_definitions,
)
from rto_sentinel.settings import LLMSettings

# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Replays prepared completions. Records what it was asked."""

    def __init__(self, *completions: Completion, model: str = "scripted-model") -> None:
        self._queue = list(completions)
        self._model = model
        self.requests: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return True

    @property
    def model(self) -> str:
        return self._model

    def complete(self, *, system: str, prompt: str, max_tokens: int | None = None) -> str:
        return self.converse(
            system=system, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens
        ).text

    def converse(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        self.requests.append({"system": system, "messages": messages, "tools": tools})
        if not self._queue:
            msg = "the scripted provider ran out of completions"
            raise AssertionError(msg)
        return self._queue.pop(0)


class FailingProvider:
    """Stands in for an upstream that is reachable and broken."""

    available = True
    model = "failing-model"

    def complete(self, **_: Any) -> str:
        raise AgentUnavailableError("the Anthropic API call failed (APITimeoutError)")

    def converse(self, **_: Any) -> Completion:
        raise AgentUnavailableError("the Anthropic API call failed (APITimeoutError)")


class StubToolset:
    """A toolset returning prepared facts, for testing the loop rather than SQL."""

    def __init__(self, **results: Any) -> None:
        self._results = results
        self.calls: list[str] = []

    def _get(self, name: str, default: Any) -> Any:
        self.calls.append(name)
        return self._results.get(name, default)

    def get_order(self, ref: Any) -> OrderFacts:
        return self._get("get_order", OrderFacts(found=False, reason="absent"))

    def get_customer_history(self, ref: Any) -> CustomerHistoryFacts:
        return self._get("get_customer_history", CustomerHistoryFacts(found=False, reason="absent"))

    def get_risk_prediction(self, ref: Any) -> RiskPredictionFacts:
        return self._get("get_risk_prediction", RiskPredictionFacts(found=False, reason="absent"))

    def get_model_explanation(self, ref: Any) -> ModelExplanationFacts:
        return self._get(
            "get_model_explanation", ModelExplanationFacts(found=False, reason="absent")
        )

    def get_economic_decision(self, ref: Any) -> EconomicDecisionFacts:
        return self._get(
            "get_economic_decision", EconomicDecisionFacts(found=False, reason="absent")
        )

    def get_relevant_order_events(self, ref: Any) -> Any:
        from rto_sentinel.agents.tools import OrderEventFacts

        return self._get("get_relevant_order_events", OrderEventFacts(found=False, reason="absent"))

    def get_digest_figures(self, ref: Any) -> DigestFigures:
        return self._get("get_digest_figures", DigestFigures(found=False, reason="absent"))


def answer(**fields: Any) -> Completion:
    """A completion carrying the agent's JSON answer."""
    payload = {
        "sufficient_evidence": True,
        "summary": "",
        "key_drivers": [],
        "evidence_used": [],
        "uncertainty": "",
        "caveats": [],
        **fields,
    }
    return Completion(text=json.dumps(payload), stop_reason="end_turn")


def tool_turn(*names: str) -> Completion:
    """A completion asking for tools by name."""
    return Completion(
        text="",
        tool_calls=tuple(
            ToolCall(id=f"call_{index}", name=name, arguments={"order_id": "ORD-00000001"})
            for index, name in enumerate(names)
        ),
        stop_reason="tool_use",
        raw_content=[{"type": "tool_use", "id": "call_0", "name": names[0], "input": {}}],
    )


PREDICTION = RiskPredictionFacts(
    found=True,
    order_id="ORD-00000001",
    probability=0.5719,
    model_name="lightgbm_platt",
    model_version="a0d780424b79",
    calibration_method="platt",
)
DECISION = EconomicDecisionFacts(
    found=True,
    order_id="ORD-00000001",
    band="HIGH",
    action="confirmation_required",
    threshold=0.3481,
    flagged=True,
)
EXPLANATION = ModelExplanationFacts(
    found=True,
    order_id="ORD-00000001",
    reason_codes=["ORDER_IS_COD", "HISTORY_PRIOR_RTO_RATE"],
    permitted_features=["order_is_cod", "cust_prior_rto_rate"],
)


# ---------------------------------------------------------------------------
# the provider boundary
# ---------------------------------------------------------------------------


def test_no_key_yields_an_unavailable_provider_with_a_reason() -> None:
    provider = get_provider(LLMSettings(_env_file=None, agents_enabled=True, api_key=None))

    assert not provider.available
    assert isinstance(provider, UnavailableProvider)
    with pytest.raises(AgentUnavailableError, match="ANTHROPIC_API_KEY"):
        provider.complete(system="s", prompt="p")


def test_agents_off_yields_an_unavailable_provider() -> None:
    """A key alone is not enough. The operator has to switch the layer on."""
    from pydantic import SecretStr

    provider = get_provider(
        LLMSettings(_env_file=None, agents_enabled=False, api_key=SecretStr("placeholder"))
    )
    assert not provider.available
    with pytest.raises(AgentUnavailableError, match="RTO_AGENTS_ENABLED"):
        provider.converse(system="s", messages=[])


def test_an_unavailable_provider_never_returns_text() -> None:
    """The failure mode that matters: no substituted sentence, ever."""
    provider = UnavailableProvider("no key")
    for call in (
        lambda: provider.complete(system="s", prompt="p"),
        lambda: provider.converse(system="s", messages=[]),
    ):
        with pytest.raises(AgentUnavailableError):
            call()


def test_no_scripted_responder_ships_in_the_product() -> None:
    """A test double belongs in tests/. In src/ it would be a fake feature.

    Greps the shipped agent package for the shapes a canned responder takes. The
    check is crude on purpose: it is a tripwire against someone adding a
    "development mode" that returns plausible text when the API is down.
    """
    package = Path(__file__).resolve().parents[2] / "src" / "rto_sentinel" / "agents"
    banned = ("ScriptedProvider", "FakeProvider", "MockProvider", "StubProvider")

    offenders = [
        f"{path.name}: {name}"
        for path in package.glob("*.py")
        for name in banned
        if name in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "the agent package must contain no scripted responder. A double in a test "
        f"proves the orchestration works; one in the product hides that it does not: {offenders}"
    )


# ---------------------------------------------------------------------------
# the tool surface
# ---------------------------------------------------------------------------


def test_every_tool_declares_its_permission_boundary() -> None:
    for spec in TOOL_SPECS:
        assert spec.name
        assert spec.purpose
        assert spec.permission
        assert "Read-only" in spec.permission
        assert spec.input_model is not None
        assert spec.output_model is not None


def test_the_tool_definitions_are_valid_for_the_messages_api() -> None:
    for definition in anthropic_tool_definitions():
        assert set(definition) == {"name", "description", "input_schema"}
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "Permission boundary" in definition["description"]


def test_no_tool_can_write_score_or_override() -> None:
    """The capability list is the security boundary. This is what it contains."""
    names = {spec.name for spec in TOOL_SPECS}

    assert names == {
        "get_order",
        "get_customer_history",
        "get_risk_prediction",
        "get_model_explanation",
        "get_economic_decision",
        "get_relevant_order_events",
    }
    assert all(name.startswith("get_") for name in names)
    for verb in ("set", "update", "delete", "override", "approve", "block", "send", "write"):
        assert not any(verb in name for name in names)


def test_the_agents_package_cannot_import_the_decision_engine() -> None:
    """Asserted here as well as in the layering tests, because it is the rule."""
    import rto_sentinel.agents.tools as tools_module

    source = Path(tools_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "from rto_sentinel.decision",
        "from rto_sentinel.models",
        "from rto_sentinel.db",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# the investigation loop
# ---------------------------------------------------------------------------


def test_the_agent_retrieves_evidence_before_answering() -> None:
    """It calls tools; it does not answer from the prompt alone."""
    toolset = StubToolset(
        get_risk_prediction=PREDICTION,
        get_economic_decision=DECISION,
        get_model_explanation=EXPLANATION,
    )
    provider = ScriptedProvider(
        tool_turn("get_risk_prediction", "get_economic_decision", "get_model_explanation"),
        answer(
            summary="The order was placed with cash on delivery and the customer's prior "
            "return rate is elevated.",
            key_drivers=["order_is_cod", "cust_prior_rto_rate"],
            evidence_used=["get_risk_prediction", "get_model_explanation"],
        ),
    )
    agent = RiskInvestigationAgent(provider, toolset)

    result = agent.investigate("ORD-00000001", "Why was this HIGH?")

    assert toolset.calls == [
        "get_risk_prediction",
        "get_economic_decision",
        "get_model_explanation",
    ]
    assert result.grounded
    assert result.sufficient_evidence


def test_retrieved_values_are_reported_not_the_models_prose() -> None:
    """The model can write any number it likes; the fields come from the tools.

    This is the structural reason the agent cannot alter a risk decision. Even a
    model that insists the probability is 0.05 and the band is LOW cannot change
    what the response reports.
    """
    toolset = StubToolset(
        get_risk_prediction=PREDICTION,
        get_economic_decision=DECISION,
        get_model_explanation=EXPLANATION,
    )
    provider = ScriptedProvider(
        tool_turn("get_risk_prediction", "get_economic_decision", "get_model_explanation"),
        answer(
            summary="This order is actually low risk at 5 percent and should be LOW band.",
            evidence_used=["get_risk_prediction"],
        ),
    )

    result = RiskInvestigationAgent(provider, toolset).investigate("ORD-00000001", "why?")

    assert result.probability == pytest.approx(0.5719)
    assert result.band == "HIGH"
    assert result.threshold == pytest.approx(0.3481)
    assert result.model_version == "a0d780424b79"


def test_missing_evidence_produces_an_uncertainty_response() -> None:
    """The order does not exist. The agent must say so, not describe it."""
    toolset = StubToolset()  # every tool returns found=False
    provider = ScriptedProvider(
        tool_turn("get_order"),
        answer(
            sufficient_evidence=False,
            summary="No order with that identifier exists in the database, so there is no "
            "decision to explain.",
            uncertainty="The order could not be found; nothing about it can be established.",
            evidence_used=["get_order"],
        ),
    )

    result = RiskInvestigationAgent(provider, toolset).investigate("ORD-99999999", "why?")

    assert result.sufficient_evidence is False
    assert result.uncertainty
    assert result.probability is None
    assert result.band is None


def test_the_absence_reason_reaches_the_model() -> None:
    """A tool that found nothing tells the model what to do about it."""
    toolset = StubToolset()
    provider = ScriptedProvider(
        tool_turn("get_order"),
        answer(sufficient_evidence=False, summary="No evidence.", evidence_used=["get_order"]),
    )
    RiskInvestigationAgent(provider, toolset).investigate("ORD-99999999", "why?")

    tool_results = provider.requests[1]["messages"][-1]["content"]
    assert "absent" in json.dumps(tool_results)


def test_a_fabricated_driver_is_rejected() -> None:
    """The model cites chargebacks. This system has no chargeback feature."""
    toolset = StubToolset(
        get_model_explanation=EXPLANATION,
        get_risk_prediction=PREDICTION,
        get_economic_decision=DECISION,
    )
    provider = ScriptedProvider(
        tool_turn("get_model_explanation"),
        answer(
            summary="The customer has a history of chargebacks and a poor credit score.",
            evidence_used=["get_model_explanation"],
        ),
    )

    result = RiskInvestigationAgent(provider, toolset).investigate("ORD-00000001", "why?")

    assert result.grounded is False
    assert "chargeback" in (result.rejection_reason or "")


def test_a_driver_outside_the_attributions_is_rejected() -> None:
    """Naming a real-looking feature the model was not given also fails."""
    toolset = StubToolset(get_model_explanation=EXPLANATION)
    provider = ScriptedProvider(
        tool_turn("get_model_explanation"),
        answer(
            summary="Driven mainly by the delivery route.",
            key_drivers=["geo_courier_rto_rate_smoothed"],
            evidence_used=["get_model_explanation"],
        ),
    )

    result = RiskInvestigationAgent(provider, toolset).investigate("ORD-00000001", "why?")

    assert result.grounded is False
    assert "not in the model's attributions" in (result.rejection_reason or "")


def test_citing_evidence_that_was_never_retrieved_is_rejected() -> None:
    toolset = StubToolset(get_risk_prediction=PREDICTION)
    provider = ScriptedProvider(
        tool_turn("get_risk_prediction"),
        answer(
            summary="Two delivery attempts failed.",
            evidence_used=["get_risk_prediction", "get_relevant_order_events"],
        ),
    )

    result = RiskInvestigationAgent(provider, toolset).investigate("ORD-00000001", "why?")

    assert result.grounded is False
    assert "never retrieved" in (result.rejection_reason or "")


def test_invalid_structured_output_is_rejected_not_repaired() -> None:
    toolset = StubToolset(get_risk_prediction=PREDICTION)
    provider = ScriptedProvider(
        Completion(text='{"summary": "missing the required fields"}', stop_reason="end_turn")
    )

    with pytest.raises(InvestigationError, match="schema validation"):
        RiskInvestigationAgent(provider, toolset).investigate("ORD-00000001", "why?")


def test_a_non_json_reply_is_rejected() -> None:
    toolset = StubToolset()
    provider = ScriptedProvider(Completion(text="I think it was risky.", stop_reason="end_turn"))

    with pytest.raises(InvestigationError, match="did not return a JSON object"):
        RiskInvestigationAgent(provider, toolset).investigate("ORD-00000001", "why?")


def test_malformed_json_is_rejected() -> None:
    toolset = StubToolset()
    provider = ScriptedProvider(Completion(text='{"summary": "a" "b", }', stop_reason="end_turn"))

    with pytest.raises(InvestigationError, match="malformed JSON"):
        RiskInvestigationAgent(provider, toolset).investigate("ORD-00000001", "why?")


def test_a_fenced_json_reply_is_accepted() -> None:
    """Models fence their JSON. Tolerating that is not tolerating malformed output."""
    toolset = StubToolset(get_risk_prediction=PREDICTION)
    payload = json.dumps(
        {
            "sufficient_evidence": True,
            "summary": "A calibrated probability above the derived threshold.",
            "key_drivers": [],
            "evidence_used": [],
            "uncertainty": "",
            "caveats": [],
        }
    )
    provider = ScriptedProvider(Completion(text=f"```json\n{payload}\n```", stop_reason="end_turn"))

    result = RiskInvestigationAgent(provider, toolset).investigate("ORD-00000001", "why?")
    assert result.grounded


def test_an_api_failure_propagates_rather_than_being_papered_over() -> None:
    with pytest.raises(AgentUnavailableError, match="APITimeoutError"):
        RiskInvestigationAgent(FailingProvider(), StubToolset()).investigate("ORD-1", "why?")


def test_a_runaway_loop_is_stopped() -> None:
    """A model that only ever asks for tools does not run forever."""
    toolset = StubToolset(get_order=OrderFacts(found=True, order_id="ORD-00000001"))
    provider = ScriptedProvider(*[tool_turn("get_order") for _ in range(4)])

    with pytest.raises(InvestigationError, match="within 3 turns"):
        RiskInvestigationAgent(provider, toolset, max_turns=3).investigate("ORD-1", "why?")


def test_an_unknown_tool_does_not_kill_the_run() -> None:
    """The model asks for a tool that does not exist; it is told and continues."""
    toolset = StubToolset(get_risk_prediction=PREDICTION)
    provider = ScriptedProvider(
        Completion(
            text="",
            tool_calls=(ToolCall(id="c1", name="get_customer_bank_details", arguments={}),),
            stop_reason="tool_use",
            raw_content=[{"type": "tool_use", "id": "c1", "name": "x", "input": {}}],
        ),
        answer(sufficient_evidence=False, summary="Insufficient evidence."),
    )

    result = RiskInvestigationAgent(provider, toolset).investigate("ORD-00000001", "why?")
    assert result.sufficient_evidence is False


# ---------------------------------------------------------------------------
# the audit trail
# ---------------------------------------------------------------------------


def test_every_run_is_audited() -> None:
    toolset = StubToolset(
        get_risk_prediction=PREDICTION,
        get_model_explanation=EXPLANATION,
        get_economic_decision=DECISION,
    )
    provider = ScriptedProvider(
        tool_turn("get_risk_prediction", "get_model_explanation"),
        answer(summary="A calibrated probability above the threshold.", evidence_used=[]),
    )
    log = AuditLog()
    RiskInvestigationAgent(provider, toolset, audit_log=log).investigate(
        "ORD-00000001", "Why was this HIGH?"
    )

    record = log.last()
    assert isinstance(record, AgentAuditRecord)
    assert record.agent_type == "risk_investigation"
    assert record.request == "Why was this HIGH?"
    assert record.subject_id == "ORD-00000001"
    assert record.model == "scripted-model"
    assert record.tool_names == ("get_risk_prediction", "get_model_explanation")
    assert record.llm_turns == 2
    assert record.output is not None
    assert record.duration_ms >= 0


def test_a_failed_run_is_audited_with_its_error() -> None:
    log = AuditLog()
    agent = RiskInvestigationAgent(FailingProvider(), StubToolset(), audit_log=log)

    with pytest.raises(AgentUnavailableError):
        agent.investigate("ORD-1", "why?")

    record = log.last()
    assert record is not None
    assert record.error is not None
    assert record.output is None
    assert not record.succeeded


def test_the_audit_record_carries_no_secret() -> None:
    """The key never reaches this layer. Asserted rather than assumed."""
    toolset = StubToolset(get_risk_prediction=PREDICTION)
    provider = ScriptedProvider(answer(summary="Above the derived threshold."))
    log = AuditLog()
    RiskInvestigationAgent(provider, toolset, audit_log=log).investigate("ORD-1", "why?")

    serialised = log.last().model_dump_json().lower()  # type: ignore[union-attr]
    for secret in ("api_key", "sk-ant", "authorization", "bearer", "password"):
        assert secret not in serialised


# ---------------------------------------------------------------------------
# grounding validators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "The customer has a history of chargebacks.",
        "Their credit score is low.",
        "This matches a known fraud score pattern.",
        "The device fingerprint was reused.",
    ],
)
def test_fabricated_drivers_are_caught(sentence: str) -> None:
    verdict = validate_feature_grounding(sentence, ("order_is_cod", "cust_prior_rto_rate"))
    assert not verdict.grounded
    assert verdict.offending_terms


@pytest.mark.parametrize(
    "sentence",
    [
        "The order was placed with cash on delivery, which raises the estimate.",
        "This customer's prior return rate is above the merchant's average.",
        "The address is missing a house number, and the discount is unusually deep.",
    ],
)
def test_ordinary_language_passes(sentence: str) -> None:
    verdict = validate_feature_grounding(
        sentence,
        ("order_is_cod", "cust_prior_rto_rate", "addr_has_house_number", "order_discount_depth"),
    )
    assert verdict.grounded, verdict.rejection_reason


def test_a_figure_that_was_not_computed_is_caught() -> None:
    verdict = validate_figure_grounding(
        "RTO cost the merchant 60,000 rupees this week.", {"rto_cost": 41967.0}
    )
    assert not verdict.grounded
    assert "60000" in verdict.offending_terms


def test_a_rounded_figure_passes() -> None:
    verdict = validate_figure_grounding(
        "Roughly 23 percent of matured orders came back.", {"rto_rate": 0.2341}
    )
    assert verdict.grounded, verdict.rejection_reason


@pytest.mark.parametrize(
    "message",
    [
        "Your order was flagged as suspicious.",
        "We detected a fraud risk on this order.",
        "Your return rate is high, so we need prepayment.",
    ],
)
def test_accusatory_customer_copy_is_rejected(message: str) -> None:
    verdict = validate_neutral_framing(message)
    assert not verdict.grounded
    assert verdict.offending_terms


def test_neutral_customer_copy_passes() -> None:
    verdict = validate_neutral_framing(
        "Hi! We're getting your order ready to ship. Please confirm your delivery address "
        "and that someone will be available to receive it."
    )
    assert verdict.grounded, verdict.rejection_reason


def test_evidence_references_must_have_been_retrieved() -> None:
    assert validate_evidence_references(
        "x", cited=("get_order",), available=("get_order",)
    ).grounded
    assert not validate_evidence_references(
        "x", cited=("get_relevant_order_events",), available=("get_order",)
    ).grounded


# ---------------------------------------------------------------------------
# confirmation writer
# ---------------------------------------------------------------------------


def test_a_confirmation_falls_back_to_the_reviewed_template_when_the_llm_is_down() -> None:
    message = ConfirmationWriter(FailingProvider()).draft(
        order_id="ORD-00008874", action="confirmation_required"
    )

    assert message.body
    assert message.grounded is False, "a template is not a generation and must not claim to be"
    assert message.llm_model == "template"
    assert message.rejection_reason
    assert message.neutral_framing_verified


def test_an_accusatory_draft_is_replaced_by_the_template() -> None:
    provider = ScriptedProvider(
        Completion(text=json.dumps({"body": "Your order was flagged as suspicious."}))
    )
    message = ConfirmationWriter(provider).draft(
        order_id="ORD-00008874", action="confirmation_required"
    )

    assert message.grounded is False
    assert "suspicious" not in message.body
    assert message.llm_model == "template"


def test_a_neutral_draft_is_kept() -> None:
    body = "Hi! Please confirm your delivery address so we can ship order 008874."
    provider = ScriptedProvider(Completion(text=json.dumps({"body": body})))

    message = ConfirmationWriter(provider).draft(
        order_id="ORD-00008874", action="confirmation_required"
    )
    assert message.grounded
    assert message.body == body
    assert message.neutral_framing_verified


def test_no_template_exists_for_an_unknown_action() -> None:
    with pytest.raises(KeyError, match="no confirmation template"):
        ConfirmationWriter(FailingProvider()).draft(order_id="ORD-1", action="seize_goods")


# ---------------------------------------------------------------------------
# digest writer
# ---------------------------------------------------------------------------


FIGURES = DigestFigures(
    found=True,
    merchant_id="M-DEMO-001",
    period_start=datetime(2026, 2, 1, tzinfo=UTC),
    period_end=datetime(2026, 3, 1, tzinfo=UTC),
    figures={"orders": 6716.0, "rto_rate_among_matured": 0.1646},
)


def test_a_digest_with_an_invented_figure_is_rejected() -> None:
    provider = ScriptedProvider(
        Completion(
            text=json.dumps(
                {"sections": [{"heading": "Week", "prose": "You lost 95,000 rupees to returns."}]}
            )
        )
    )
    digest = DigestWriter(provider).write(FIGURES)

    assert digest.grounded is False
    assert digest.sections == ()
    assert digest.computed_figures == FIGURES.figures, "the figures survive a rejected generation"


def test_a_digest_quoting_only_given_figures_is_kept() -> None:
    provider = ScriptedProvider(
        Completion(
            text=json.dumps(
                {
                    "sections": [
                        {
                            "heading": "This period",
                            "prose": "You shipped 6716 orders and about 16 percent of the "
                            "matured ones came back.",
                        }
                    ]
                }
            )
        )
    )
    digest = DigestWriter(provider).write(FIGURES)

    assert digest.grounded, digest.rejection_reason
    assert len(digest.sections) == 1


def test_a_digest_with_no_figures_produces_no_prose() -> None:
    """A model asked to summarise an empty week will write something. Do not ask."""
    provider = ScriptedProvider()  # would raise if called
    digest = DigestWriter(provider).write(
        DigestFigures(
            found=False,
            reason="no orders in this period",
            merchant_id="M-1",
            period_start=datetime(2026, 2, 1, tzinfo=UTC),
            period_end=datetime(2026, 3, 1, tzinfo=UTC),
        )
    )

    assert digest.grounded is False
    assert digest.sections == ()
    assert provider.requests == []


def test_a_digest_survives_the_llm_being_down() -> None:
    digest = DigestWriter(FailingProvider()).write(FIGURES)

    assert digest.grounded is False
    assert digest.computed_figures == FIGURES.figures
    assert digest.rejection_reason


# ---------------------------------------------------------------------------
# address repair: deferred
# ---------------------------------------------------------------------------


def test_address_repair_refuses_rather_than_fabricating() -> None:
    from rto_sentinel.agents.address_repair import (
        AddressRepairDeferred,
        suggest_address_repair,
    )

    with pytest.raises(AddressRepairDeferred) as excinfo:
        suggest_address_repair(order_id="ORD-1")

    reason = str(excinfo.value)
    assert "synthetic" in reason
    assert "postal reference" in reason
