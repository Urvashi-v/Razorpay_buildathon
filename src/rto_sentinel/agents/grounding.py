"""The grounding validator - what stops a language model inventing a cause.

SPEC section 08. Each of the language jobs has a guardrail, and this module is
where they are mechanised.

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

WHAT THESE CHECKS ARE AND ARE NOT
---------------------------------
They are a **blocklist over a known vocabulary**, not a proof of truthfulness.
:func:`validate_feature_grounding` catches a sentence that cites a risk driver
this system does not have - chargebacks, credit scores, a fraud history - because
those are the fabrications that matter and they are nameable in advance. It
cannot catch a fluent sentence that misdescribes a driver it *was* given.

That limit is real and is the reason the reason codes, not the prose, are the
artefact of record. The sentence is a convenience laid over them; if it is
rejected the codes are still there, and if it passes it is still only prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Risk-domain nouns this system has no feature for. A sentence naming one is
#: describing evidence that does not exist anywhere in the pipeline, whatever the
#: model believed while writing it.
#:
#: Chosen by walking the feature catalogue and listing the plausible-sounding
#: drivers it does NOT contain - which is the set a model is most likely to reach
#: for, because they are what a human would expect a risk system to use.
FABRICATED_DRIVERS: frozenset[str] = frozenset(
    {
        "chargeback",
        "chargebacks",
        "credit score",
        "credit history",
        "credit rating",
        "fraud history",
        "fraud score",
        "blacklist",
        "blocklist",
        "watchlist",
        "criminal",
        "police",
        "identity theft",
        "stolen card",
        "card testing",
        "velocity check",
        "device fingerprint",
        "ip address",
        "vpn",
        "proxy",
        "social media",
        "phone verification",
        "kyc",
        "aadhaar",
        "pan card",
        "gst",
        "bank statement",
        "income",
        "salary",
        "employment",
        "age",
        "gender",
        "religion",
        "caste",
        "ethnicity",
        "nationality",
    }
)

#: Vocabulary a customer-facing message may never contain. SPEC section 09:
#: customers are never told they are "flagged", and friction is framed neutrally.
ACCUSATORY_TERMS: frozenset[str] = frozenset(
    {
        "suspicious",
        "suspect",
        "flagged",
        "flag",
        "fraud",
        "fraudulent",
        "risk",
        "risky",
        "high-risk",
        "unreliable",
        "untrustworthy",
        "blacklisted",
        "blocked",
        "denied",
        "rejected",
        "refused",
        "penalty",
        "penalise",
        "penalize",
        "warning",
        "violation",
        "abuse",
        "abusive",
        "return rate",
        "rto",
    }
)

#: Numbers a model may always write without them counting as an unsupported
#: figure: small integers used as ordinary language ("one of the two attempts").
_FREE_NUMERALS = frozenset({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"})

_NUMERAL = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[a-z][a-z'-]*")


@dataclass(frozen=True, slots=True)
class GroundingVerdict:
    """The result of validating one generation against its allow-list."""

    grounded: bool
    rejection_reason: str | None = None
    offending_terms: tuple[str, ...] = ()

    @classmethod
    def ok(cls) -> GroundingVerdict:
        return cls(grounded=True)


def _mentions(lowered: str, term: str) -> bool:
    """Whether a term appears as a WHOLE WORD (or whole phrase) in the text.

    Substring matching is the obvious implementation and it is wrong. "age" is a
    protected attribute this system must never key on - and it is also inside
    "average", "manage" and "package", every one of which is ordinary language in
    a risk explanation. A validator that rejects "above the merchant's average"
    gets switched off within a week, and then it protects nothing.

    So terms are matched at word boundaries. Multi-word terms match as phrases,
    which lets "credit score" be caught without "credit" alone tripping on an
    unrelated sentence.
    """
    return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", lowered) is not None


def _humanise(feature: str) -> set[str]:
    """The words a feature name licenses in prose.

    ``cust_prior_rto_rate`` licenses "prior", "rto" and "rate" - so a sentence
    about the customer's prior RTO rate passes while one about their chargebacks
    does not. The family prefix is dropped: nobody writes "cust".
    """
    parts = feature.lower().split("_")
    return {part for part in parts[1:] if len(part) > 2} or set(parts)


def validate_feature_grounding(text: str, permitted_features: tuple[str, ...]) -> GroundingVerdict:
    """Reject a sentence that names a risk driver it was not given.

    Two passes. The first looks for any term in :data:`FABRICATED_DRIVERS` that
    the permitted features do not license - a model writing about chargebacks
    when no chargeback feature exists. The second is not attempted: enumerating
    every legitimate English sentence is not possible, and a validator that
    rejects ordinary connective language would simply be switched off.
    """
    lowered = text.lower()
    licensed: set[str] = set()
    for feature in permitted_features:
        licensed |= _humanise(feature)

    offending = sorted(
        term
        for term in FABRICATED_DRIVERS
        if _mentions(lowered, term) and not any(word in licensed for word in term.split())
    )
    if offending:
        return GroundingVerdict(
            grounded=False,
            rejection_reason=(
                f"the explanation names {len(offending)} risk driver(s) this system has no "
                f"feature for: {offending}. Nothing in the model's inputs could support "
                "that claim, so the sentence is rejected rather than shown."
            ),
            offending_terms=tuple(offending),
        )
    return GroundingVerdict.ok()


def _numerals(text: str) -> list[str]:
    return [match.group().replace(",", "") for match in _NUMERAL.finditer(text)]


def validate_figure_grounding(
    text: str, permitted_figures: dict[str, float], *, tolerance: float = 0.05
) -> GroundingVerdict:
    """Reject prose containing a number that was not handed to the generator.

    Used by the weekly merchant digest. The figures come from SQL; the model
    writes prose *around* numbers it is given and is not permitted to compute.
    Any numeral in the output that does not correspond to a permitted figure -
    at the tolerance a human would read it, so 23.4 percent may be written as
    "roughly 23 percent" - fails the check.

    ``tolerance`` is relative, so a rounded ₹41,967 passes as "about ₹42,000"
    while an invented ₹60,000 does not.
    """
    permitted = list(permitted_figures.values())
    offending: list[str] = []

    for token in _numerals(text):
        if token in _FREE_NUMERALS:
            continue
        try:
            value = float(token)
        except ValueError:  # pragma: no cover - the regex only matches numerals
            continue
        if not any(_close_enough(value, allowed, tolerance) for allowed in permitted):
            offending.append(token)

    if offending:
        return GroundingVerdict(
            grounded=False,
            rejection_reason=(
                f"the digest quotes {len(offending)} figure(s) that were not computed and "
                f"handed to it: {offending}. Every number in a merchant digest must come "
                "from SQL, so this generation is rejected and the figures table is shown "
                "without prose."
            ),
            offending_terms=tuple(offending),
        )
    return GroundingVerdict.ok()


def _close_enough(value: float, allowed: float, tolerance: float) -> bool:
    """Whether a written number plausibly refers to a permitted figure.

    Rounding is expected: a model given 0.2341 may write "23 percent", and a
    model given 41967.4 may write "42,000". Both the percentage form and the
    rounded magnitude are accepted; an unrelated number is not.
    """
    candidates = (allowed, allowed * 100.0, round(allowed), round(allowed * 100.0))
    for candidate in candidates:
        if candidate == 0:
            if abs(value) < 1e-9:
                return True
            continue
        if abs(value - candidate) <= abs(candidate) * tolerance:
            return True
    return False


def validate_neutral_framing(text: str) -> GroundingVerdict:
    """Reject customer-facing copy that implies suspicion.

    SPEC section 09: customers are never told they are "flagged", and friction is
    framed neutrally. A confirmation message must read as routine logistics. This
    check looks for accusatory and risk-disclosing vocabulary and rejects rather
    than rewrites - a message edited into neutrality by a regex is a message
    nobody reviewed.
    """
    lowered = text.lower()

    offending = sorted(term for term in ACCUSATORY_TERMS if _mentions(lowered, term))
    if offending:
        return GroundingVerdict(
            grounded=False,
            rejection_reason=(
                f"the customer-facing message uses {len(offending)} term(s) that disclose "
                f"risk assessment or imply suspicion: {offending}. Friction is framed as "
                "routine logistics; a customer is never told they were flagged."
            ),
            offending_terms=tuple(offending),
        )
    return GroundingVerdict.ok()


def validate_evidence_references(
    text: str, *, cited: tuple[str, ...], available: tuple[str, ...]
) -> GroundingVerdict:
    """Reject an explanation citing evidence that was never retrieved.

    The agent reports which tools it drew on. If it claims support from evidence
    no tool returned - a delivery timeline for an order that has none, a customer
    history that came back empty - the claim has nothing behind it whatever the
    sentence says.
    """
    unavailable = sorted(set(cited) - set(available))
    if unavailable:
        return GroundingVerdict(
            grounded=False,
            rejection_reason=(
                f"the explanation cites evidence that was never retrieved: {unavailable}. "
                f"Available evidence: {sorted(available) or 'none'}."
            ),
            offending_terms=tuple(unavailable),
        )
    return GroundingVerdict.ok()
