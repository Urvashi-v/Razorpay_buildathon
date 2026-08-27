"""The grounding validator - what stops a language model inventing a cause.

SPEC section 08. Each of the four language jobs has a guardrail, and this module
is where three of them are mechanised.

THE CORE RULE
-------------
Every generation is handed an explicit allow-list: the feature names and
contribution values it may talk about, or the figures it may quote. Any output
naming something outside that list is **rejected**, not edited, not softened. The
caller then falls back to the deterministic artefact - raw reason codes, or the
figures table without prose.

Rejection is a normal outcome here, not an exception path. It is recorded on
:class:`~rto_sentinel.contracts.explanation.GroundedOutput` as ``grounded=False``
with a reason, so the rate of rejections is measurable rather than invisible.

WHY REJECT RATHER THAN REPAIR
-----------------------------
A repaired hallucination is still a hallucination that got close enough to pass.
For a risk system whose explanations may be shown to an ops team making a call
about a real customer, "we could not explain this in words" is a safe answer and
"we explained it slightly wrong" is not.

STATUS: Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GroundingVerdict:
    """The result of validating one generation against its allow-list."""

    grounded: bool
    rejection_reason: str | None = None
    offending_terms: tuple[str, ...] = ()


def validate_feature_grounding(text: str, permitted_features: tuple[str, ...]) -> GroundingVerdict:
    """Reject a sentence that names a feature it was not given.

    The Phase 5 implementation matches against the human-readable aliases of the
    permitted features plus a curated vocabulary of risk-domain nouns, so that
    ordinary connective language passes while a fabricated driver ("the customer
    has a history of chargebacks", when no such feature exists) does not.
    """
    raise NotImplementedError("Feature grounding validation lands in Phase 5.")


def validate_figure_grounding(text: str, permitted_figures: dict[str, float]) -> GroundingVerdict:
    """Reject prose containing a number that was not handed to the generator.

    Used by the weekly merchant digest. The figures come from SQL; the model
    writes prose *around* numbers it is given and is not permitted to compute.
    Any numeral in the output that does not correspond to a permitted figure -
    at the tolerance a human would read it, so 23.4 percent may be written as
    "roughly 23 percent" - fails the check.
    """
    raise NotImplementedError("Figure grounding validation lands in Phase 5.")


def validate_neutral_framing(text: str) -> GroundingVerdict:
    """Reject customer-facing copy that implies suspicion.

    SPEC section 09: customers are never told they are "flagged", and friction is
    framed neutrally. A confirmation message must read as routine logistics. This
    check looks for accusatory and risk-disclosing vocabulary - suspicious,
    flagged, fraud, risk, verification-because-of, and similar - in the generated
    body, and rejects rather than rewrites.
    """
    raise NotImplementedError("Neutral-framing validation lands in Phase 5.")
