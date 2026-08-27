"""Rung 3: logistic regression, monotonic where sensible.

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


class LogisticRegressionModel(RiskModel):
    """Rung 3: logistic regression, monotonic where sensible.

    Interpretable, fast, and calibrated by construction - a genuine shipping
    candidate rather than a strawman. Monotonic constraints on prior RTO rate and
    discount depth encode the direction we are willing to defend: more prior
    returns should never *decrease* predicted risk, whatever the fit finds.

    If this rung wins on net rupees, it ships. That is the ladder working, not
    the ladder failing."""

    rung_id = 3
    name = "logistic_regression"

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError("Rung 3 training lands in Phase 3.")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Rung 3 inference lands in Phase 3.")

    def save(self, path: Path, card: ModelCard) -> None:
        raise NotImplementedError("Rung 3 persistence lands in Phase 3.")

    @classmethod
    def load(cls, path: Path) -> tuple[RiskModel, ModelCard]:
        raise NotImplementedError("Rung 3 persistence lands in Phase 3.")
