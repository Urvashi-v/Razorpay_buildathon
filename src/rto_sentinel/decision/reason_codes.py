"""Deterministic reason codes derived from SHAP contributions.

These are produced HERE, in the decision layer, from model attributions - not by
an LLM. The language layer only ever renders an existing code into a sentence.

The distinction matters operationally: reason codes are stable identifiers that
can be counted, filtered, alerted on and audited. "ADDRESS_INCOMPLETE" means the
same thing in every decision log row, forever. A generated sentence does not, and
cannot be aggregated.

WHY THE CODE IS BUILT FROM THE FAMILY, NOT THE FEATURE NAME
===========================================================
``FAMILY_CODE_STEMS`` maps a feature *family* to a stem, and the suffix comes
from a small explicit table below. Deriving the code from the raw feature name
would mean that renaming ``addr_token_count`` to ``address_token_count`` - a
harmless refactor - silently changed an operational identifier that ops
dashboards and alerts are keyed on. The indirection is the point.

A feature with no entry in the suffix table still produces a code, built from its
family stem and a normalised form of its own name. That is a deliberate fallback
rather than a silent drop: a new feature appearing in the top contributions
should show up in the logs as something, and an unfamiliar code is a prompt to
add it to the table.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rto_sentinel.contracts.explanation import ReasonCode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.risk import FeatureContribution

#: Feature family -> reason code stem. Kept explicit rather than derived from
#: feature names so that renaming a feature cannot silently change an
#: operational code that ops dashboards and alerts are keyed on.
FAMILY_CODE_STEMS: dict[str, str] = {
    "address_quality": "ADDRESS",
    "customer_history": "HISTORY",
    "order_shape": "ORDER",
    "session_intent": "INTENT",
    "geography_route": "ROUTE",
    "temporal_velocity": "VELOCITY",
}

#: Feature -> suffix. Stable operational vocabulary, changed only deliberately.
FEATURE_SUFFIXES: dict[str, str] = {
    "addr_token_count": "INCOMPLETE",
    "addr_has_house_number": "NO_HOUSE_NUMBER",
    "addr_has_floor_number": "NO_FLOOR_NUMBER",
    "addr_has_landmark": "NO_LANDMARK",
    "addr_pincode_city_consistent": "PINCODE_CITY_MISMATCH",
    "addr_allcaps_ratio": "FORMATTING",
    "addr_gibberish_ratio": "UNPARSEABLE",
    "cust_prior_rto_rate": "PRIOR_RTO_RATE",
    "cust_prior_order_count": "THIN_HISTORY",
    "cust_days_since_last_order": "DORMANT",
    "cust_prepaid_share": "COD_ONLY",
    "cust_is_new_customer": "NEW_CUSTOMER",
    "order_value_inr": "ORDER_VALUE",
    "discount_depth": "DEEP_DISCOUNT",
    "item_count": "BASKET_SIZE",
    "is_cod": "COD_PAYMENT",
    "hour_of_day": "ORDER_HOUR",
    "is_weekend": "WEEKEND_ORDER",
}

#: Codes are `[A-Z0-9_]`, capped so they fit the 64-character contract field.
_UNSAFE = re.compile(r"[^A-Z0-9]+")
_MAX_CODE_LENGTH = 64


def _suffix_for(feature: str) -> str:
    """The stable suffix for a feature, or a normalised fallback."""
    known = FEATURE_SUFFIXES.get(feature)
    if known is not None:
        return known
    # Strip the family prefix if present, then normalise. `cust_foo_bar` becomes
    # `FOO_BAR` rather than `CUST_FOO_BAR`, so the stem is not repeated.
    stripped = feature.split("_", 1)[1] if "_" in feature else feature
    return _UNSAFE.sub("_", stripped.upper()).strip("_") or "UNKNOWN"


def code_for(contribution: FeatureContribution) -> str:
    """The stable operational identifier for one contribution."""
    stem = FAMILY_CODE_STEMS.get(contribution.family)
    suffix = _suffix_for(contribution.feature)
    code = f"{stem}_{suffix}" if stem else suffix
    return code[:_MAX_CODE_LENGTH]


def derive_reason_codes(
    contributions: list[FeatureContribution],
    *,
    top_k: int = 3,
) -> tuple[ReasonCode, ...]:
    """Turn the top risk-increasing contributions into stable reason codes.

    Only risk-*increasing* contributions become reason codes: an ops associate
    reading why an order was frictioned does not need the list of things that
    made it look safe. Those remain available in the full contribution list for
    the console explanation panel.

    Ordering is by contribution magnitude, then by feature name. The tiebreak is
    not decoration: two features with identical SHAP values would otherwise come
    out in whatever order the model happened to emit them, and a decision log
    whose reason codes reshuffle between identical runs is not auditable.
    """
    if top_k <= 0:
        msg = f"top_k must be positive, got {top_k}"
        raise ValueError(msg)

    increasing = [entry for entry in contributions if entry.contribution > 0]
    ranked = sorted(increasing, key=lambda entry: (-entry.contribution, entry.feature))

    return tuple(
        ReasonCode(
            code=code_for(entry),
            feature=entry.feature,
            family=entry.family,
            contribution=entry.contribution,
            direction=entry.direction,
        )
        for entry in ranked[:top_k]
    )
