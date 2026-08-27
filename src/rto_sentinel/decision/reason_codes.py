"""Deterministic reason codes derived from SHAP contributions.

These are produced HERE, in the decision layer, from model attributions - not by
an LLM. The language layer only ever renders an existing code into a sentence.

The distinction matters operationally: reason codes are stable identifiers that
can be counted, filtered, alerted on and audited. "ADDRESS_INCOMPLETE" means the
same thing in every decision log row, forever. A generated sentence does not, and
cannot be aggregated.

STATUS: Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.explanation import ReasonCode
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
}


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
    """
    raise NotImplementedError("Reason-code derivation lands in Phase 2.")
