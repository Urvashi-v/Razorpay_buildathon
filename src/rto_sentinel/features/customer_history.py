"""Customer-history features, computed strictly as-of order time.

STATUS: Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rto_sentinel.features.base import FeatureFamily

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class CustomerHistoryFamily(FeatureFamily):
    """Customer-history features, computed strictly as-of order time.

    SPEC section 04. Every column here goes through ``data.asof``; none may be
    computed with a plain groupby, because a plain groupby over the full frame
    includes orders that had not resolved yet.

    THE RISK TO WATCH: new customers have no history, and the model must not
    collapse into "new equals risky". Missing history is encoded as NaN rather
    than 0 so the model can learn the absence as its own state, and the
    new-customer cohort is reported as a separate slice in every evaluation."""

    name = "customer_history"

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(self.config.signals)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Feature family CustomerHistoryFamily lands in Phase 2.")
