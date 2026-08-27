"""Synthetic address rendering and observable address-quality signals.

TWO DIRECTIONS, DELIBERATELY SEPARATED
--------------------------------------
:func:`render_address` goes *latent quality → text*. The simulator draws a
latent quality level for a customer's address and this renders text consistent
with it: a high-quality address has a house number, a floor, a landmark and clean
tokens; a degraded one is missing some of those.

:func:`observable_signals` goes *text → measurable signals*. It reads only the
rendered string, exactly as a production system would read what a customer
actually typed. It has no access to the latent quality that produced the text.

Keeping the two directions apart is what makes the address features honest: the
model sees what a real system would see, and the noise between latent quality and
its observable proxy is real irreducible noise rather than a hidden shortcut.

WHAT THESE ADDRESSES ARE NOT
----------------------------
They are token patterns, not locations. Street names come from a small fixed
vocabulary, house numbers are integers, and the pincodes are synthetic six-digit
identifiers from a generated pool - not a map of real Indian postcodes. Nothing
here is deliverable, geocodable, or usable outside this repository's own
evaluation harness.

FAIRNESS NOTE
-------------
The signals below measure *structural completeness* - is there a house number, is
the pincode consistent with the city - and never fluency. An address written in
imperfect English is not a risky address, and a feature that conflated the two
would be a literacy proxy. The fairness audit in Phase 4 checks whether these
features concentrate on tier-3 pincodes without a matching precision.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# Small fixed vocabularies. Deliberately generic: these are shapes of addresses,
# not places.
STREET_STEMS: tuple[str, ...] = (
    "MG Road",
    "Station Road",
    "Park Street",
    "Gandhi Marg",
    "Ring Road",
    "Market Lane",
    "Church Street",
    "Nehru Road",
    "Lake View Road",
    "Industrial Area",
)

LOCALITY_STEMS: tuple[str, ...] = (
    "Sector 4",
    "Phase 2",
    "Block C",
    "Ward 11",
    "Colony",
    "Extension",
    "Layout",
    "Nagar",
)

LANDMARK_STEMS: tuple[str, ...] = (
    "near the water tank",
    "opposite the bus stop",
    "behind the school",
    "next to the clinic",
    "near the temple",
    "beside the petrol pump",
)

CITY_STEMS: tuple[str, ...] = (
    "Pune",
    "Indore",
    "Jaipur",
    "Kochi",
    "Lucknow",
    "Nagpur",
    "Surat",
    "Bhopal",
    "Patna",
    "Guwahati",
)

STATE_BY_CITY: dict[str, str] = {
    "Pune": "Maharashtra",
    "Indore": "Madhya Pradesh",
    "Jaipur": "Rajasthan",
    "Kochi": "Kerala",
    "Lucknow": "Uttar Pradesh",
    "Nagpur": "Maharashtra",
    "Surat": "Gujarat",
    "Bhopal": "Madhya Pradesh",
    "Patna": "Bihar",
    "Guwahati": "Assam",
}

# Consonant runs that a human would not type in a real address. Used to simulate
# the keyboard-mash and autocomplete-garbage that genuinely appears in address
# fields, and detected as a gibberish signal.
_GIBBERISH_TOKENS: tuple[str, ...] = ("qwe", "asdf", "xyz", "zxcv", "nnn", "hjk")

_HOUSE_NUMBER_RE = re.compile(r"\b\d+[a-zA-Z]?(?:/\d+)?\b")
_FLOOR_RE = re.compile(r"\b(?:\d+(?:st|nd|rd|th)\s+floor|floor\s+\d+|flat\s+\w+)\b", re.IGNORECASE)
_LANDMARK_RE = re.compile(r"\b(?:near|opposite|behind|next to|beside|opp\.?)\b", re.IGNORECASE)
_GIBBERISH_RE = re.compile("|".join(_GIBBERISH_TOKENS), re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AddressSignals:
    """Everything observable from an address string at order time.

    No field here depends on the delivery outcome, and none depends on the latent
    quality that generated the text. These are exactly the measurements a
    production system could take at checkout.
    """

    token_count: int
    has_house_number: bool
    has_floor_number: bool
    has_landmark: bool
    pincode_city_consistent: bool
    allcaps_ratio: float
    gibberish_ratio: float


@dataclass(frozen=True, slots=True)
class RenderedAddress:
    """A generated address plus its stable fingerprint."""

    line: str
    city: str
    state: str
    pincode: str
    fingerprint: str


def address_fingerprint(line: str, pincode: str) -> str:
    """Stable 16-hex-character identity for an address.

    A fingerprint rather than the raw text, so the same physical address used
    across orders can be joined without the joining key being the address itself.
    Deterministic across runs because it is a plain digest with no salt - this is
    synthetic data, and reproducibility matters more here than unlinkability.
    """
    normalised = " ".join(line.lower().split()) + "|" + pincode
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def render_address(
    rng: np.random.Generator,
    *,
    quality: float,
    city: str,
    pincode: str,
    consistent_pincode: bool,
) -> RenderedAddress:
    """Render an address whose text reflects ``quality`` in ``[0, 1]``.

    Higher quality means more of the components a courier actually needs. The
    mapping is probabilistic rather than a threshold, so two addresses of the same
    latent quality do not render identically - that stochasticity is what stops
    the observable signals from being a lossless encoding of the latent.

    ``consistent_pincode=False`` renders a city that does not match the pincode's
    real city, which is the single most common structural defect in Indian
    address data and a legitimate, fair predictor of delivery failure.
    """
    parts: list[str] = []

    # House number: the component whose absence most reliably fails a delivery.
    if rng.random() < 0.35 + 0.6 * quality:
        house = int(rng.integers(1, 400))
        if rng.random() < 0.3:
            parts.append(f"{house}/{int(rng.integers(1, 20))}")
        else:
            parts.append(str(house))

    # Floor or flat.
    if rng.random() < 0.15 + 0.5 * quality:
        parts.append(f"{int(rng.integers(1, 12))}{_ordinal_suffix(rng)} Floor")

    parts.append(str(STREET_STEMS[int(rng.integers(0, len(STREET_STEMS)))]))

    if rng.random() < 0.3 + 0.4 * quality:
        parts.append(str(LOCALITY_STEMS[int(rng.integers(0, len(LOCALITY_STEMS)))]))

    # Landmark: how deliveries actually get completed in much of India.
    if rng.random() < 0.20 + 0.55 * quality:
        parts.append(str(LANDMARK_STEMS[int(rng.integers(0, len(LANDMARK_STEMS)))]))

    # Gibberish tokens: keyboard mash and autocomplete garbage.
    if rng.random() < 0.28 * (1.0 - quality):
        parts.append(str(_GIBBERISH_TOKENS[int(rng.integers(0, len(_GIBBERISH_TOKENS)))]))

    line = ", ".join(parts)

    # All-caps entry: common on older mobile keyboards. Correlated with, but not
    # determined by, low quality - and it is NOT itself treated as a risk driver
    # by the simulator, only measured. See docs/simulator.md.
    if rng.random() < 0.10 + 0.25 * (1.0 - quality):
        line = line.upper()

    rendered_city = city
    if not consistent_pincode:
        # Pick a different city, so the pincode and city genuinely disagree.
        others = [c for c in CITY_STEMS if c != city]
        rendered_city = others[int(rng.integers(0, len(others)))]

    return RenderedAddress(
        line=line,
        city=rendered_city,
        state=STATE_BY_CITY[rendered_city],
        pincode=pincode,
        fingerprint=address_fingerprint(line, pincode),
    )


def _ordinal_suffix(rng: np.random.Generator) -> str:
    return ("st", "nd", "rd", "th")[int(rng.integers(0, 4))]


def observable_signals(line: str, *, city: str, expected_city: str) -> AddressSignals:
    """Measure an address string exactly as a production system would.

    Reads the text and nothing else. ``expected_city`` is the city the pincode
    belongs to, which a merchant knows from a pincode lookup table; comparing it
    to the city the customer typed is the consistency check.
    """
    tokens = [token for token in re.split(r"[,\s]+", line) if token]
    token_count = len(tokens)

    alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
    allcaps = sum(1 for token in alpha_tokens if token.isupper() and len(token) > 1)
    gibberish = sum(1 for token in tokens if _GIBBERISH_RE.fullmatch(token.strip(".,")))

    return AddressSignals(
        token_count=token_count,
        has_house_number=bool(_HOUSE_NUMBER_RE.search(line)),
        has_floor_number=bool(_FLOOR_RE.search(line)),
        has_landmark=bool(_LANDMARK_RE.search(line)),
        pincode_city_consistent=city.strip().lower() == expected_city.strip().lower(),
        allcaps_ratio=(allcaps / len(alpha_tokens)) if alpha_tokens else 0.0,
        gibberish_ratio=(gibberish / token_count) if token_count else 0.0,
    )
