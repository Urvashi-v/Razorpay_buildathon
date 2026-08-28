"""Rung 0: ship everything, flag nothing.

**The question this answers: what happens without intervention?**

The do-nothing baseline, and the most important number in the whole evaluation.
It defines the loss the merchant currently absorbs, and every later rupee figure
is measured against it. A model that "saves money" without beating this has saved
nothing.

WHAT IT PREDICTS, AND WHY THAT IS THE RIGHT CHOICE
==================================================
The training-set base rate, for every row. A constant predictor - which is exactly
what doing nothing amounts to, and it makes the rung honest in two ways.

Its **PR-AUC equals the positive base rate**, which is the floor every other rung
must clear to have demonstrated any ranking ability at all. A rung scoring 0.19
PR-AUC on a book with a 0.18 base rate has learned essentially nothing, and
without this row in the table that would be easy to miss.

Its **ROC-AUC is undefined**, because there is no ranking to score. Reported as
NaN rather than 0.5, because "not defined for this predictor" and "no better than
chance" are different statements.

At any threshold above the base rate it flags nothing, so its flag rate is zero,
its precision is undefined, and its net rupee figure is zero by construction -
the reference point rather than a result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rto_sentinel.models.base import HeuristicModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


class DoNothingModel(HeuristicModel):
    """Predicts the training base rate for every order."""

    rung_id = 0
    name = "do_nothing"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.base_rate_: float = 0.0

    def _fit(self, x: pd.DataFrame, y: pd.Series, context: pd.DataFrame | None) -> None:
        # The one thing doing nothing "learns": how often orders come back.
        self.base_rate_ = float(y.astype(float).mean()) if len(y) else 0.0

    def _predict(self, x: pd.DataFrame, context: pd.DataFrame | None) -> np.ndarray:
        return np.full(len(x), self.base_rate_, dtype="float64")

    def _state(self) -> dict[str, Any]:
        return {**super()._state(), "base_rate": self.base_rate_}

    def _restore(self, state: dict[str, Any]) -> None:
        super()._restore(state)
        self.base_rate_ = state["base_rate"]
