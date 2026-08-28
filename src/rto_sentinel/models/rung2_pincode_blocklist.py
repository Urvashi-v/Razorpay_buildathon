"""Rung 2: block the worst-performing pincodes.

**The question this answers: can a simple rule on location-like operational
information perform meaningfully?**

This is the common "smart" heuristic - the one a merchant reaches for after the
blanket block proves too blunt. It is the rung this project most wants to beat,
and not only on rupees.

WHY THIS RUNG USES INFORMATION THE ML MODELS ARE FORBIDDEN
==========================================================
Raw pincode is in ``FORBIDDEN_IN_FEATURES``. No learned rung may touch it,
because with enough trees a per-pincode rate becomes a redlining machine.

This rung uses it anyway, deliberately, because **that is the policy under
examination**. A merchant running a pincode blocklist is using raw pincode. To
measure what that costs - in money, and in flag rate by tier - the baseline has
to be the real thing rather than a sanitised version that would flatter it.

It receives pincode through the ``context`` frame, which is a separate argument
from the design matrix. The learned rungs ignore that argument entirely, and a
test asserts their predictions are unchanged without it.

WHAT "DEFENSIBLE INFORMATION" MEANS HERE
========================================
Three constraints, so the baseline is a fair opponent rather than a strawman:

1. **Fitted on training only.** Rates come from orders that resolved inside the
   training window. A blocklist built on the full dataset would be reading the
   future and would beat the models for the wrong reason.
2. **Minimum support.** A pincode needs at least ``min_support`` resolved
   training orders to be eligible. Without it the blocklist fills with places
   that had three deliveries and two bad ones, which is noise, and it would make
   the baseline look worse than the policy really is.
3. **Top decile by rate.** The standard formulation, taken from config rather
   than tuned.

THE FAIRNESS POINT, STATED IN ADVANCE
=====================================
This rung is expected to concentrate its flags heavily on tier-3 pincodes. That
is not a bug in the baseline; it is what the policy does. Phase 6's fairness
audit reports flag rate and precision by tier for every rung side by side, and
this row is the comparison that gives the model's numbers meaning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rto_sentinel.data import schema as cols
from rto_sentinel.models.base import HeuristicModel, context_column

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

DEFAULT_TOP_DECILE = 0.10
DEFAULT_MIN_SUPPORT = 30


class PincodeBlocklistModel(HeuristicModel):
    """Flags orders shipping to a pincode in the worst decile by training RTO rate."""

    rung_id = 2
    name = "pincode_blocklist"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.top_decile = float(self.params.get("top_decile", DEFAULT_TOP_DECILE))
        self.min_support = int(self.params.get("min_support", DEFAULT_MIN_SUPPORT))
        self.blocklist_: frozenset[str] = frozenset()
        self.n_eligible_pincodes_: int = 0

    def _fit(self, x: pd.DataFrame, y: pd.Series, context: pd.DataFrame | None) -> None:
        """Build the blocklist from training-window outcomes only."""
        pincode = context_column(context, cols.PINCODE, model_name=self.name)

        frame = pd.DataFrame({"pincode": pincode.to_numpy(), "label": y.astype(float).to_numpy()})
        grouped = frame.groupby("pincode", sort=False)["label"].agg(["count", "mean"])
        eligible = grouped[grouped["count"] >= self.min_support]
        self.n_eligible_pincodes_ = len(eligible)

        if eligible.empty:
            # Every pincode is too thin to judge. An empty blocklist is the
            # honest outcome - and it is a real finding about the dataset, not a
            # failure to report. The rung then flags nothing and scores like
            # rung 0, which the comparison table will show.
            self.blocklist_ = frozenset()
            return

        cutoff = eligible["mean"].quantile(1.0 - self.top_decile)
        self.blocklist_ = frozenset(eligible.index[eligible["mean"] >= cutoff].astype(str))

    def _predict(self, x: pd.DataFrame, context: pd.DataFrame | None) -> np.ndarray:
        pincode = context_column(context, cols.PINCODE, model_name=self.name)
        flagged = pincode.astype(str).isin(self.blocklist_).to_numpy(dtype=bool)
        return flagged.astype("float64")

    def _state(self) -> dict[str, Any]:
        return {
            **super()._state(),
            "top_decile": self.top_decile,
            "min_support": self.min_support,
            "blocklist": sorted(self.blocklist_),
            "n_eligible_pincodes": self.n_eligible_pincodes_,
        }

    def _restore(self, state: dict[str, Any]) -> None:
        super()._restore(state)
        self.top_decile = state["top_decile"]
        self.min_support = state["min_support"]
        self.blocklist_ = frozenset(state["blocklist"])
        self.n_eligible_pincodes_ = state["n_eligible_pincodes"]
