"""Rung 4: LightGBM, isotonically calibrated on the validation fold.

STATUS: Phase 2-3.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rto_sentinel.models.base import RiskModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import pandas as pd

    from rto_sentinel.contracts.risk import ModelCard
    from rto_sentinel.models.base import RiskModel


class LightGbmModel(RiskModel):
    """Rung 4: LightGBM, isotonically calibrated on the validation fold.

    The expected winner. Handles interactions and missingness natively, which
    matters here because "no customer history" is a real state rather than a
    zero, and forcing it to a number would teach the model something false.

    Raw boosting output is not a probability - it is systematically distorted -
    and the entire expected-value layer depends on the score being honest. So
    calibration is not a finishing touch on this rung; it is part of the model.
    See ``models/calibration.py``.

    SHAP contributions come from this rung and feed the reason codes."""

    rung_id = 4
    name = "lightgbm_isotonic"

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError("Rung 4 training lands in Phase 3.")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Rung 4 inference lands in Phase 3.")

    def save(self, path: Path, card: ModelCard) -> None:
        raise NotImplementedError("Rung 4 persistence lands in Phase 3.")

    @classmethod
    def load(cls, path: Path) -> tuple[RiskModel, ModelCard]:
        raise NotImplementedError("Rung 4 persistence lands in Phase 3.")
