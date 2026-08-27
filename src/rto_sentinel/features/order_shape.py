"""Order-shape features: value, basket, discount depth, payment mode.

STATUS: Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rto_sentinel.features.base import FeatureFamily

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


class OrderShapeFamily(FeatureFamily):
    """Order-shape features: value, basket, discount depth, payment mode.

    SPEC section 04. Deep-discount impulse orders are high risk - but that is
    partly the merchant's own doing. The evaluation surfaces discount depth as a
    merchant insight, not only as a customer penalty."""

    name = "order_shape"

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(self.config.signals)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Feature family OrderShapeFamily lands in Phase 2.")
