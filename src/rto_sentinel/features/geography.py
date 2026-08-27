"""Geography and courier-lane features. HIGHEST FAIRNESS RISK IN THE MODEL.

STATUS: Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rto_sentinel.features.base import FeatureFamily

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class GeographyRouteFamily(FeatureFamily):
    """Geography and courier-lane features. HIGHEST FAIRNESS RISK IN THE MODEL.

    SPEC section 04. A raw per-pincode RTO rate is an income and region proxy,
    and with enough trees it becomes a redlining machine. Three constraints,
    enforced here rather than remembered:

    * Bayesian shrinkage toward the global mean, strength from config.
    * A minimum support threshold; below it the feature is NaN, not a noisy rate
      computed from four orders.
    * Never a top-3 SHAP feature without written justification in REPORT.md.

    If the fairness audit trips its trigger, this is the family that gets pulled
    back - which is why it is isolated behind its own switch."""

    name = "geography_route"

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(self.config.signals)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Feature family GeographyRouteFamily lands in Phase 2.")
