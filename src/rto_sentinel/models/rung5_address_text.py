"""Rung 5: rung 4 plus a text encoder over address strings.

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


class LightGbmAddressTextModel(RiskModel):
    """Rung 5: rung 4 plus a text encoder over address strings.

    Attempted only if time permits, and promoted only if it beats rung 4 on NET
    RUPEES - not on AUC. A rung that wins on a ranking metric and loses on the
    cost metric has not earned a place in production.

    Disabled in config by default. The fairness question is sharper here than
    anywhere else in the ladder: an encoder over raw address text can learn
    regional language patterns, which is a protected-attribute proxy by another
    route. If this rung is ever enabled, the fairness audit gates it."""

    rung_id = 5
    name = "lightgbm_address_text"

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError("Rung 5 training lands in Phase 3.")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Rung 5 inference lands in Phase 3.")

    def save(self, path: Path, card: ModelCard) -> None:
        raise NotImplementedError("Rung 5 persistence lands in Phase 3.")

    @classmethod
    def load(cls, path: Path) -> tuple[RiskModel, ModelCard]:
        raise NotImplementedError("Rung 5 persistence lands in Phase 3.")
