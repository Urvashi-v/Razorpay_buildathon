"""Rung 0: ship everything, flag nothing.

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


class DoNothingModel(HeuristicModel):
    """Rung 0: ship everything, flag nothing.

    The do-nothing baseline, and the most important number in the whole
    evaluation. It defines the loss the merchant currently absorbs, and every
    later rupee figure is measured against it. A model that "saves money"
    without beating this has saved nothing.

    ``predict_proba`` returns the training-set base rate for every row: a
    constant predictor, which is exactly what doing nothing amounts to."""

    rung_id = 0
    name = "do_nothing"

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Rung 0 inference lands in Phase 3.")

    def save(self, path: Path, card: ModelCard) -> None:
        raise NotImplementedError("Rung 0 persistence lands in Phase 3.")

    @classmethod
    def load(cls, path: Path) -> tuple[RiskModel, ModelCard]:
        raise NotImplementedError("Rung 0 persistence lands in Phase 3.")
