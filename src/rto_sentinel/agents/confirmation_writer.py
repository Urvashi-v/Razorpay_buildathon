"""Customer-facing confirmation copy for a frictioned order.

WHAT THIS AGENT IS FOR
======================
When the decision engine asks for a confirmation before dispatch, somebody has to
write the message the customer reads. That message must sound like routine
delivery logistics, because it is: the customer has done nothing wrong, and the
overwhelming majority of frictioned orders would have been delivered fine.

WHAT IT CANNOT DO
=================
It receives the decision; it does not make one. The band, the action and the
channel are inputs. There is no code path here that changes what happens to the
order - the agent only chooses words, and if it chooses them badly the message is
rejected and the human-reviewed template is sent instead.

THE GUARDRAIL THAT MATTERS
==========================
SPEC section 09: customers are never told they are "flagged". The generated body
is checked against :func:`~rto_sentinel.agents.grounding.validate_neutral_framing`
and **rejected**, not rewritten, if it uses risk-disclosing or accusatory
vocabulary. A message edited into neutrality by a regex is a message nobody
reviewed; a rejected one falls back to copy a human wrote.

The fallback is a real template, not a generated one. It is the reason this agent
can fail safely: the worst case is that the customer gets slightly less tailored
wording, never that they get told they are suspected of something.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rto_sentinel.agents.audit import AuditBuilder, AuditLog
from rto_sentinel.agents.grounding import validate_neutral_framing
from rto_sentinel.agents.provider import AgentUnavailableError
from rto_sentinel.contracts.explanation import ConfirmationMessage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.agents.provider import LLMProvider

AGENT_TYPE = "confirmation_message"

#: Human-reviewed fallbacks, used when the language layer is unavailable or the
#: generation is rejected. These are not "generated text pretending to be a
#: model's" - they are templates, they are labelled as such on the response, and
#: the caller can tell the difference from `grounded` and `llm_model`.
TEMPLATES: dict[str, str] = {
    "confirmation_required": (
        "Hi! We're getting your order {order_ref} ready to ship. Please confirm your "
        "delivery address and that someone will be available to receive it. Reply YES to "
        "confirm, or tap the link to update your address."
    ),
    "prepaid_nudge": (
        "Hi! Your order {order_ref} is ready to ship. Pay online now and get it on its way "
        "sooner - plus a small discount at checkout. Prefer cash on delivery? That's fine "
        "too, just confirm and we'll dispatch."
    ),
    "prepaid_only": (
        "Hi! For your order {order_ref}, we're currently able to ship on prepayment. Tap "
        "the link to complete payment and we'll dispatch right away. If that doesn't work "
        "for you, reply and our team will help."
    ),
}

SYSTEM_PROMPT = """\
You write short delivery-confirmation messages for an Indian e-commerce merchant.

TONE AND CONTENT
The message is routine logistics. The customer has done nothing wrong and must \
never be told, implied, or hinted that their order was flagged, scored, assessed \
for risk, or found suspicious. Do not mention risk, fraud, returns, verification \
because of a problem, or any assessment of the customer.

Write warmly and briefly, as a helpful merchant would. Ask for what the action \
requires and nothing more. Two or three sentences at most.

CONSTRAINTS
- Do not invent order details, delivery dates, prices, discounts or offers that \
you were not given.
- Do not promise anything about timing you were not told.
- Refer to the order only by the reference you are given.
- Write in the requested language.

OUTPUT
Reply with a single JSON object and nothing else:
{"body": "the message text"}
"""


class ConfirmationWriter:
    """Drafts the customer message for one frictioned order."""

    def __init__(self, provider: LLMProvider, *, audit_log: AuditLog | None = None) -> None:
        self._provider = provider
        self._audit = audit_log or AuditLog()

    @property
    def audit_log(self) -> AuditLog:
        return self._audit

    def draft(
        self,
        *,
        order_id: str,
        action: str,
        channel: str = "whatsapp",
        language: str = "en-IN",
    ) -> ConfirmationMessage:
        """Draft a message, or fall back to the reviewed template.

        Never raises for an unavailable provider. A confirmation message is
        operationally required - the order is waiting on it - so the template is
        the answer when the model cannot be reached, and the response says which
        it was.
        """
        template = TEMPLATES.get(action)
        if template is None:
            msg = (
                f"no confirmation template exists for action {action!r}. A message is not "
                "drafted for an action nobody has written copy for."
            )
            raise KeyError(msg)

        builder = AuditBuilder(
            agent_type=AGENT_TYPE,
            request=f"draft {channel} confirmation for {action}",
            provider="anthropic",
            model=self._provider.model,
            subject_id=order_id,
        )
        reference = order_id[-6:]
        fallback = template.format(order_ref=reference)

        try:
            completion = self._provider.complete(
                system=SYSTEM_PROMPT,
                prompt=(
                    f"Action required: {action}\n"
                    f"Channel: {channel}\n"
                    f"Language: {language}\n"
                    f"Order reference to use: {reference}\n"
                    f"Approved template for this action (match its intent):\n{fallback}"
                ),
            )
            builder.llm_turns = 1
        except AgentUnavailableError as error:
            return self._fallback(
                builder, order_id, channel, language, fallback, action, error.reason
            )

        try:
            body = str(json.loads(_only_object(completion))["body"]).strip()
        except (KeyError, TypeError, ValueError) as error:
            return self._fallback(
                builder,
                order_id,
                channel,
                language,
                fallback,
                action,
                f"the model's reply was not the expected JSON object ({type(error).__name__})",
            )

        verdict = validate_neutral_framing(body)
        if not verdict.grounded:
            return self._fallback(
                builder, order_id, channel, language, fallback, action, verdict.rejection_reason
            )

        message = ConfirmationMessage(
            generated_at=datetime.now(UTC),
            llm_model=self._provider.model,
            grounded=True,
            order_id=order_id,
            channel=channel,
            language=language,
            body=body,
            template_id=action,
            neutral_framing_verified=True,
        )
        self._audit.record(
            builder.finish(output=json.loads(message.model_dump_json()), grounded=True)
        )
        return message

    def _fallback(
        self,
        builder: AuditBuilder,
        order_id: str,
        channel: str,
        language: str,
        body: str,
        action: str,
        reason: str | None,
    ) -> ConfirmationMessage:
        """Send the reviewed template, and say plainly that is what happened."""
        message = ConfirmationMessage(
            generated_at=datetime.now(UTC),
            llm_model="template",
            grounded=False,
            rejection_reason=reason,
            order_id=order_id,
            channel=channel,
            language=language,
            body=body,
            template_id=action,
            # The template was written and reviewed by a human, so its framing is
            # verified by construction rather than by the validator.
            neutral_framing_verified=True,
        )
        self._audit.record(
            builder.finish(
                output=json.loads(message.model_dump_json()),
                grounded=False,
                rejection_reason=reason,
            )
        )
        return message


def _only_object(text: str) -> str:
    """The JSON object in a reply, tolerating a code fence around it."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        msg = "no JSON object in the reply"
        raise ValueError(msg)
    return stripped[start : end + 1]
