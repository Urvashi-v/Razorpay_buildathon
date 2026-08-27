"""Rung 1: block COD above a fixed order value.

STATUS: Phase 2-3.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rto_sentinel.models.base import HeuristicModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import pandas as pd

    from rto_sentinel.contracts.risk import ModelCard
    from rto_sentinel.models.base import RiskModel


class BlanketCodBlockModel(HeuristicModel):
    """Rung 1: block COD above a fixed order value.

    What many merchants actually do today. Usually terrible - high recall,
    catastrophic precision, and it kills GMV on exactly the orders worth the
    most. It is on the ladder to show precisely how bad, in rupees, rather than
    to be dismissed rhetorically.

    ``predict_proba`` returns a hard 1.0/0.0 split. That is honest about what the
    rule is: it is not a probability, and its calibration error will say so
    loudly in the report."""

    rung_id = 1
    name = "blanket_cod_block"

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Rung 1 inference lands in Phase 3.")

    def save(self, path: Path, card: ModelCard) -> None:
        raise NotImplementedError("Rung 1 persistence lands in Phase 3.")

    @classmethod
    def load(cls, path: Path) -> tuple[RiskModel, ModelCard]:
        raise NotImplementedError("Rung 1 persistence lands in Phase 3.")
