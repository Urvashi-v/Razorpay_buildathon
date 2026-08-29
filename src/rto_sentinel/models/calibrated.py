"""The shipped model: a fitted rung wrapped in a fitted calibrator.

WHY THIS IS A WRAPPER AND NOT A FLAG ON THE RUNG
================================================
A calibrated model is a different object from the rung it wraps, because it was
fitted on different data. The rung saw the training split; the calibrator saw the
validation split. Folding the calibrator into the rung would hide that, and the
first thing anyone auditing a probability needs to know is which rows produced
the mapping that made it.

Wrapping also keeps the ladder honest. The Phase 4 rungs stay exactly as they
were measured - uncalibrated, comparable to each other - and the Phase 5 model is
visibly a *composition* of one of them with a transform, rather than a
retroactive edit to a baseline.

WHAT ``predict_proba`` MEANS HERE
=================================
    P(RTO | information available at the moment the order was placed)

Both halves of that sentence are load-bearing. "Probability" is the calibrator's
job and is only true to the extent the reliability diagram says it is. "Available
at order time" is the feature pipeline's job, enforced by the leakage suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rto_sentinel.models.base import NotFittedError, RiskModel
from rto_sentinel.models.calibration import (
    CalibrationError,
    Calibrator,
    restore_calibrator,
)
from rto_sentinel.models.registry import resolve_rung

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from rto_sentinel.contracts.risk import FeatureContribution


class CalibratedModel(RiskModel):
    """A base rung plus the calibrator fitted on the validation split.

    ``rung_id`` and ``name`` follow the wrapped model, so this drops into the
    evaluation harness beside the raw rungs and is measured by the same code.
    """

    rung_id = 4
    name = "calibrated"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.base_: RiskModel | None = None
        self.calibrator_: Calibrator | None = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def wrap(cls, base: RiskModel, calibrator: Calibrator) -> CalibratedModel:
        """Compose an already-fitted rung with an already-fitted calibrator.

        Both must be fitted. Composing an unfitted half would produce an object
        that looks shippable and is not, and the failure would surface as a
        strange probability rather than as an error.
        """
        if not base._fitted:
            msg = f"refusing to wrap unfitted model {base.name}"
            raise NotFittedError(msg)
        if not calibrator.is_fitted:
            msg = f"refusing to wrap {base.name} in an unfitted {calibrator.method} calibrator"
            raise CalibrationError(msg)

        model = cls(dict(base.params))
        model.base_ = base
        model.calibrator_ = calibrator
        model.feature_names_ = base.feature_names_
        model.rung_id = base.rung_id
        model.name = f"{base.name}_{calibrator.method}"
        model._fitted = True
        return model

    @property
    def calibration_method(self) -> str:
        if self.calibrator_ is None:  # pragma: no cover - guarded by construction
            msg = "calibrated model has no calibrator"
            raise CalibrationError(msg)
        return self.calibrator_.method

    @property
    def base_model(self) -> RiskModel:
        if self.base_ is None:  # pragma: no cover - guarded by construction
            msg = "calibrated model has no base model"
            raise NotFittedError(msg)
        return self.base_

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _fit(self, x: pd.DataFrame, y: pd.Series, context: pd.DataFrame | None) -> None:
        """Refused. The two halves are fitted on different splits, by design.

        Accepting a single ``fit(x, y)`` would mean fitting the calibrator on
        whatever split the base model just trained on, which is precisely the
        mistake this module's docstring exists to prevent. Build one with
        :meth:`wrap`.
        """
        msg = (
            "CalibratedModel cannot be fitted in one call: the base model is fitted on "
            "train and the calibrator on validation. Fit them separately and compose "
            "them with CalibratedModel.wrap(base, calibrator)."
        )
        raise NotFittedError(msg)

    def _predict(self, x: pd.DataFrame, context: pd.DataFrame | None) -> np.ndarray:
        raw = self.base_model.predict_proba(x, context)
        if self.calibrator_ is None:  # pragma: no cover - guarded by construction
            msg = "calibrated model has no calibrator"
            raise CalibrationError(msg)
        return np.asarray(self.calibrator_.transform(raw), dtype="float64")

    def predict_raw(self, x: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        """The uncalibrated score, for before/after reliability comparison."""
        return self.base_model.predict_proba(x, context)

    def explain(self, x: pd.DataFrame, top_k: int = 5) -> list[list[FeatureContribution]]:
        """Attributions come from the base model.

        Calibration is a monotone transform of the score, so it reorders nothing
        and changes no feature's relative contribution. Passing the request
        through is the honest answer; recomputing on calibrated output would
        invent attributions for a transform that has no features.
        """
        return self.base_model.explain(x, top_k=top_k)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _state(self) -> dict[str, Any]:
        base = self.base_model
        if self.calibrator_ is None:  # pragma: no cover - guarded by construction
            msg = "calibrated model has no calibrator"
            raise CalibrationError(msg)
        return {
            **super()._state(),
            "base_name": base.name,
            "base_state": base._state(),
            "calibrator": self.calibrator_.state(),
            "rung_id": self.rung_id,
            "name": self.name,
        }

    def _restore(self, state: dict[str, Any]) -> None:
        super()._restore(state)
        base_class = resolve_rung(str(state["base_name"]))
        base = base_class()
        base._restore(state["base_state"])
        self.base_ = base
        self.calibrator_ = restore_calibrator(state["calibrator"])
        self.rung_id = int(state["rung_id"])
        self.name = str(state["name"])
