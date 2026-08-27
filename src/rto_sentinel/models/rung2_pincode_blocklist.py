"""Rung 2: block the top-decile RTO pincodes.

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


class PincodeBlocklistModel(HeuristicModel):
    """Rung 2: block the top-decile RTO pincodes.

    The common "smart" heuristic, and the one this project most wants to beat -
    not only on rupees but on fairness. Its flag rate by pincode tier is
    reported alongside the model's, which is the clearest way to show what a
    blocklist actually does to tier-3 customers.

    The blocklist is fitted on the training split only, with the same minimum
    support threshold the geography feature family uses, so it is a fair fight."""

    rung_id = 2
    name = "pincode_blocklist"

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Rung 2 inference lands in Phase 3.")

    def save(self, path: Path, card: ModelCard) -> None:
        raise NotImplementedError("Rung 2 persistence lands in Phase 3.")

    @classmethod
    def load(cls, path: Path) -> tuple[RiskModel, ModelCard]:
        raise NotImplementedError("Rung 2 persistence lands in Phase 3.")
