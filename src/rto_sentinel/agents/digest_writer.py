"""The weekly merchant digest: prose around figures the model did not compute.

THE DIVISION OF LABOUR
======================
SQL computes every number. The model writes sentences around them. It is not
asked to sum, average, compare or infer anything, and
:func:`~rto_sentinel.agents.grounding.validate_figure_grounding` rejects any
output containing a numeral that does not correspond to a figure it was handed.

That division is what makes a wrong number in a digest a *bug in a query* -
findable, reproducible, fixable - rather than a hallucination, which is none of
those things.

WHY THE FIGURES ARE ALSO RETURNED SEPARATELY
============================================
:class:`~rto_sentinel.contracts.explanation.MerchantDigest` carries
``computed_figures`` alongside the prose. If the prose is rejected the merchant
still gets the table; if it is accepted the table is still the record. The
sentences are a convenience laid over the numbers and never a substitute for
them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rto_sentinel.agents.audit import AuditBuilder, AuditLog
from rto_sentinel.agents.grounding import validate_figure_grounding
from rto_sentinel.agents.provider import AgentUnavailableError
from rto_sentinel.contracts.explanation import DigestSection, MerchantDigest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from datetime import datetime as DateTime

    from rto_sentinel.agents.provider import LLMProvider
    from rto_sentinel.agents.tools import DigestFigures

AGENT_TYPE = "merchant_digest"

SYSTEM_PROMPT = """\
You write a short weekly summary for an Indian e-commerce merchant about \
return-to-origin activity on their orders.

THE ONLY NUMBERS YOU MAY USE
You will be given a list of figures. Those are the only numbers you may write. \
Do not compute, sum, average, project, or estimate anything. Do not introduce a \
number that is not in the list, not even an obvious one. If a comparison would \
need a figure you were not given, describe it in words instead.

Rounding a given figure for readability is fine - 0.2341 may be written as \
"about 23 percent".

TONE
Plain and useful. This is an operations summary, not marketing. Do not \
congratulate the merchant and do not alarm them. If the figures are unremarkable, \
say so.

WHAT THIS DATA IS
The labels behind these figures are simulated benchmark data, not real customer \
outcomes. Do not present them as measured business results.

OUTPUT
Reply with a single JSON object and nothing else:
{"sections": [{"heading": "...", "prose": "..."}]}
Two or three sections at most.
"""


class DigestWriter:
    """Writes the weekly digest from figures computed in SQL."""

    def __init__(self, provider: LLMProvider, *, audit_log: AuditLog | None = None) -> None:
        self._provider = provider
        self._audit = audit_log or AuditLog()

    @property
    def audit_log(self) -> AuditLog:
        return self._audit

    def write(self, figures: DigestFigures) -> MerchantDigest:
        """Produce the digest. Prose is optional; the figures are not.

        When the figures tool found nothing, no prose is generated at all - there
        is nothing to write about, and a model asked to summarise an empty week
        will write something rather than nothing.
        """
        builder = AuditBuilder(
            agent_type=AGENT_TYPE,
            request=f"weekly digest for {figures.merchant_id}",
            provider="anthropic",
            model=self._provider.model,
            subject_id=figures.merchant_id,
        )

        if not figures.found:
            return self._without_prose(
                builder,
                figures,
                figures.reason or "no figures were computed for this period",
            )

        prompt = "Figures for this period:\n" + "\n".join(
            f"  {name}: {value:,.4g}" for name, value in sorted(figures.figures.items())
        )
        try:
            completion = self._provider.complete(system=SYSTEM_PROMPT, prompt=prompt)
            builder.llm_turns = 1
        except AgentUnavailableError as error:
            return self._without_prose(builder, figures, error.reason)

        try:
            payload = json.loads(_only_object(completion))
            raw_sections = payload["sections"]
            if not isinstance(raw_sections, list) or not raw_sections:
                msg = "sections must be a non-empty list"
                raise ValueError(msg)
        except (KeyError, TypeError, ValueError) as error:
            return self._without_prose(
                builder,
                figures,
                f"the model's reply was not the expected JSON object ({type(error).__name__})",
            )

        prose = " ".join(str(section.get("prose", "")) for section in raw_sections)
        verdict = validate_figure_grounding(prose, figures.figures)
        if not verdict.grounded:
            return self._without_prose(builder, figures, verdict.rejection_reason)

        sections = tuple(
            DigestSection(
                heading=str(section.get("heading", "Summary"))[:120],
                figures=figures.figures,
                prose=str(section.get("prose", ""))[:2000],
            )
            for section in raw_sections
        )
        digest = MerchantDigest(
            generated_at=datetime.now(UTC),
            llm_model=self._provider.model,
            grounded=True,
            merchant_id=figures.merchant_id or "",
            period_start=_required(figures.period_start),
            period_end=_required(figures.period_end),
            sections=sections,
            computed_figures=figures.figures,
        )
        self._audit.record(
            builder.finish(output=json.loads(digest.model_dump_json()), grounded=True)
        )
        return digest

    def _without_prose(
        self, builder: AuditBuilder, figures: DigestFigures, reason: str | None
    ) -> MerchantDigest:
        """The figures table, with no sentences. A complete, honest answer."""
        digest = MerchantDigest(
            generated_at=datetime.now(UTC),
            llm_model="none",
            grounded=False,
            rejection_reason=reason,
            merchant_id=figures.merchant_id or "",
            period_start=_required(figures.period_start),
            period_end=_required(figures.period_end),
            sections=(),
            computed_figures=figures.figures,
        )
        self._audit.record(
            builder.finish(
                output=json.loads(digest.model_dump_json()),
                grounded=False,
                rejection_reason=reason,
            )
        )
        return digest


def _required(value: DateTime | None) -> DateTime:
    if value is None:  # pragma: no cover - the caller always supplies a period
        msg = "a digest needs a period"
        raise ValueError(msg)
    return value


def _only_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        msg = "no JSON object in the reply"
        raise ValueError(msg)
    return stripped[start : end + 1]
