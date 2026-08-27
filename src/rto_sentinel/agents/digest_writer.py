"""Language job 3 of 4: the weekly merchant digest.

SPEC section 08.

WHAT IT DOES
    Summarises where losses concentrated, which interventions paid for
    themselves, and which cost more than they saved.

THE GUARDRAIL
    Numbers come from SQL, not from the model. The LLM writes prose around
    figures it is handed and is **not permitted to compute**. Any numeral in the
    output that does not correspond to a permitted figure fails
    ``agents.grounding.validate_figure_grounding`` and the digest falls back to
    the figures table with no prose.

WHY THIS SPLIT
    "Interventions that cost more than they saved" is precisely the kind of
    finding a merchant needs and a system is tempted to soften. Computing it in
    SQL and handing the model only the result means the unflattering number
    survives the writing step. The model can make a bad week readable; it cannot
    make it look better than it was.

STATUS: Phase 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.agents.provider import LLMProvider
    from rto_sentinel.agents.tools import DigestFigures
    from rto_sentinel.contracts.explanation import MerchantDigest

SYSTEM_PROMPT = """You write a short weekly operations digest for an e-commerce \
merchant about return-to-origin losses.

You will be given a set of figures that have already been computed. Use only \
those figures. Do not calculate, estimate, extrapolate, or infer any number that \
you were not given. Do not round in a way that changes a figure's meaning. If an \
intervention cost more than it saved, say so plainly. Write in short paragraphs, \
no marketing language, no reassurance the figures do not support."""


def write_digest(
    provider: LLMProvider,
    *,
    figures: DigestFigures,
) -> MerchantDigest:
    """Generate the digest prose around SQL-computed figures.

    The returned digest always carries ``computed_figures`` - the full set the
    prose was allowed to mention - so a reader can check every number in the text
    against the source without leaving the object.
    """
    raise NotImplementedError("Digest writing lands in Phase 5.")
