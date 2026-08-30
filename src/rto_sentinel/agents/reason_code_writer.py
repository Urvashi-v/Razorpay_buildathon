"""Language job 1 of 4: turn reason codes into one plain sentence.

SPEC section 08.

WHAT IT DOES
    Renders the top SHAP contributions into a single sentence an ops associate
    can act on, instead of a bar chart they have to decode.

THE GUARDRAIL
    Strictly grounded. The prompt receives only feature names and contribution
    values, and is instructed to describe them without adding causes. Any output
    naming a feature not in the input is rejected by
    ``agents.grounding.validate_feature_grounding``.

WHAT IT IS NOT
    It is not the reason code. The codes themselves are derived deterministically
    in ``decision.reason_codes`` from SHAP values, and they are what gets logged,
    counted and alerted on. This job only phrases them. If it fails, the console
    shows the codes and the contribution bars, and nothing about the decision or
    the audit trail changes.

STATUS: Phase 5.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rto_sentinel.agents.grounding import validate_feature_grounding
from rto_sentinel.agents.provider import AgentUnavailableError
from rto_sentinel.contracts.explanation import Explanation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.agents.provider import LLMProvider
    from rto_sentinel.contracts.explanation import ReasonCode

SYSTEM_PROMPT = """You describe, in one sentence, why an order was given extra \
delivery friction by a risk system.

You will be given a list of feature names with their contribution values. \
Describe only those features. Do not introduce any cause, motive, or customer \
characteristic that is not in the list. Do not speculate about intent. Do not \
mention fraud. Do not use the word "flagged". Write one sentence, under 30 \
words, in plain operational English."""


def write_explanation(
    provider: LLMProvider,
    *,
    order_id: str,
    reason_codes: tuple[ReasonCode, ...],
) -> Explanation:
    """Generate and validate a one-sentence explanation.

    Returns an :class:`Explanation` either way: ``grounded=True`` with the
    sentence, or ``grounded=False`` with the rejection reason and an empty
    sentence. It does not raise on rejection, because a missing sentence is a
    display concern, not an error the ops queue should see as a failure.
    """
    permitted = tuple(code.feature for code in reason_codes)
    if not reason_codes:
        return Explanation(
            generated_at=datetime.now(UTC),
            llm_model="none",
            grounded=False,
            rejection_reason=(
                "no reason codes were derived for this order, so there is nothing to "
                "phrase. The decision stands; it simply has no per-feature account."
            ),
            order_id=order_id,
            sentence="",
            reason_codes=(),
            permitted_features=(),
        )

    listing = "\n".join(
        f"  {code.feature} ({code.family}): contribution {code.contribution:+.3f}"
        for code in reason_codes
    )
    try:
        sentence = provider.complete(
            system=SYSTEM_PROMPT,
            prompt=f"Order {order_id}. Features that raised this order's risk:\n{listing}",
        ).strip()
    except AgentUnavailableError as error:
        return Explanation(
            generated_at=datetime.now(UTC),
            llm_model="none",
            grounded=False,
            rejection_reason=error.reason,
            order_id=order_id,
            sentence="",
            reason_codes=reason_codes,
            permitted_features=permitted,
        )

    verdict = validate_feature_grounding(sentence, permitted)
    return Explanation(
        generated_at=datetime.now(UTC),
        llm_model=provider.model,
        grounded=verdict.grounded,
        rejection_reason=verdict.rejection_reason,
        order_id=order_id,
        # A rejected sentence is not returned at all. Showing it beside a
        # "grounded: false" flag invites someone to read it anyway.
        sentence=sentence[:500] if verdict.grounded else "",
        reason_codes=reason_codes,
        permitted_features=permitted,
    )
