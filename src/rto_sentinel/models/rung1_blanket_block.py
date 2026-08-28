"""Rung 1: block COD above a fixed order value.

**The question this answers: what happens under an extremely aggressive
operational policy?**

This is what many merchants actually do today. It is on the ladder to show
precisely how bad it is, in rupees, rather than to be dismissed rhetorically.

WHY IT IS EXPECTED TO LOSE MONEY
================================
The rule flags every COD order above a value threshold. On a book where COD is
~62% of orders and roughly one in four COD orders returns, that means frictioning
a very large number of customers - and three out of four of them would have
accepted delivery perfectly happily.

Each of those is a false positive with a real cost: abandonment probability times
contribution margin. The rule catches RTOs, but it buys them at a price that
usually exceeds the savings. The comparison table is where that becomes a number
instead of an opinion.

The pathology is also worth naming precisely: the rule is *most* aggressive
exactly where the merchant's margin is largest, because it keys on order value.
It frictions the high-value orders whose abandonment hurts most.

WHAT IT SCORES
==============
A hard 1.0 / 0.0 split, which is honest about what the rule is. It is not a
probability, and its calibration error will say so loudly in the report - a
useful demonstration that ranking metrics and calibration measure different
things.

Because the score is binary, PR-AUC understates it relative to a ranking model:
there is no ordering within the flagged group. That is a real property of the
policy, not an artefact of the evaluation, and the confusion-matrix metrics at
the operating threshold are the fair comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rto_sentinel.data import schema as cols
from rto_sentinel.models.base import HeuristicModel, context_column

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

DEFAULT_VALUE_THRESHOLD_INR = 1500.0


class BlanketCodBlockModel(HeuristicModel):
    """Flags every COD order above ``value_threshold_inr``."""

    rung_id = 1
    name = "blanket_cod_block"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.value_threshold_inr = float(
            self.params.get("value_threshold_inr", DEFAULT_VALUE_THRESHOLD_INR)
        )

    def _fit(self, x: pd.DataFrame, y: pd.Series, context: pd.DataFrame | None) -> None:
        """Nothing to learn. The threshold is a policy choice, not an estimate.

        Deliberately *not* tuned on the training labels. Optimising it would
        turn a blunt operational rule into a fitted model, and the point of this
        rung is to measure what merchants actually run - a round number somebody
        picked in a meeting.
        """
        return None

    def _predict(self, x: pd.DataFrame, context: pd.DataFrame | None) -> np.ndarray:
        is_cod = context_column(context, cols.IS_COD, model_name=self.name).to_numpy(dtype=bool)
        value = context_column(context, cols.ORDER_VALUE_INR, model_name=self.name).to_numpy(
            dtype="float64"
        )
        flagged = is_cod & (value > self.value_threshold_inr)
        return flagged.astype("float64")

    def _state(self) -> dict[str, Any]:
        return {**super()._state(), "value_threshold_inr": self.value_threshold_inr}

    def _restore(self, state: dict[str, Any]) -> None:
        super()._restore(state)
        self.value_threshold_inr = state["value_threshold_inr"]
