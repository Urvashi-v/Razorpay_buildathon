"""Address-quality features.

STATUS: Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rto_sentinel.features.base import FeatureFamily

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class AddressQualityFamily(FeatureFamily):
    """Address-quality features.

    SPEC section 04. Genuinely predictive and genuinely fair: a delivery address
    missing a house number really does fail to deliver, and that has nothing to
    do with who the customer is.

    THE RISK TO WATCH: this must not become a literacy proxy. An address written
    in imperfect English is not a risky address. The features below deliberately
    measure *structural completeness* (is there a house number, is the pincode
    consistent with the city) rather than fluency, and the fairness audit checks
    whether they concentrate on tier-3 pincodes without a matching precision."""

    name = "address_quality"

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(self.config.signals)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Feature family AddressQualityFamily lands in Phase 2.")
