"""The risk-model interface every rung of the ladder implements.

SPEC section 05: every rung is evaluated identically on the same sealed test set,
against the same cost metric. That is only meaningful if a do-nothing baseline
and a LightGBM present the *same* interface to the evaluation harness - so they
do, and the harness has no idea which is which.

WHAT A MODEL RETURNS AND WHAT IT DOES NOT
=========================================
:meth:`RiskModel.predict_proba` returns probabilities. Nothing here returns a
band, an action, a rupee figure or a threshold. Those belong to
``rto_sentinel.decision``, and keeping them out of this interface is what stops a
model from quietly deciding policy.

THE CONTEXT ARGUMENT, AND WHY IT EXISTS
=======================================
``fit`` and ``predict_proba`` take an optional ``context`` frame alongside the
design matrix. It carries raw operational columns - pincode, payment method,
order value - that the ML rungs are **forbidden** from using as features.

That sounds like a hole in the leakage defences. It is the opposite. Rungs 1 and
2 are not models; they are the operational policies merchants actually run today,
and a blocklist keyed on pincode is exactly what one of them does. Reproducing it
honestly means giving it the raw pincode that ``FORBIDDEN_IN_FEATURES`` withholds
from the learned rungs - because the whole point of that rung is to show, in
rupees and in flag rates by tier, what that policy costs.

The separation is enforced by construction: ``context`` is a different argument
from ``x``, the ML rungs ignore it entirely, and
``tests/unit/test_baselines.py::test_learned_rungs_ignore_context`` checks that
their predictions are unchanged when it is withheld.

CALIBRATION IS NOT HERE
=======================
Every rung in Phase 4 is **uncalibrated**. ``predict_proba`` returns a score in
``[0, 1]`` that ranks orders; for the tree model it is not an honest probability.
Isotonic calibration on the validation fold is Phase 5, and the model card's
``calibration_method`` stays ``None`` until then - which is what stops an
uncalibrated score reaching the decision engine, since that engine refuses one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd

from rto_sentinel.models.artifacts import load_artifact, save_artifact

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rto_sentinel.contracts.risk import FeatureContribution, ModelCard


class NotFittedError(RuntimeError):
    """Raised when a model is asked to predict before it has been fitted."""


class RiskModel(ABC):
    """A rung of the baseline ladder."""

    #: Ladder position from config/models/ladder.yaml. Rung 0 is do-nothing.
    rung_id: int
    #: Stable identifier used in artefacts, reports and the API response.
    name: str

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})
        self.feature_names_: tuple[str, ...] = ()
        self._fitted = False

    # ------------------------------------------------------------------
    # fitting and prediction
    # ------------------------------------------------------------------

    def fit(self, x: pd.DataFrame, y: pd.Series, context: pd.DataFrame | None = None) -> None:
        """Fit on the training split only.

        Implementations must never see validation or test rows here. The
        calibrator is fitted separately, on validation, in Phase 5 - a model that
        calibrates itself on its own training fold is calibrated to its own
        overfitting.
        """
        self.feature_names_ = tuple(x.columns)
        self._fit(x, y, context)
        self._fitted = True

    def predict_proba(self, x: pd.DataFrame, context: pd.DataFrame | None = None) -> np.ndarray:
        """Return P(RTO) for each row, as a 1-D array in ``[0, 1]``.

        Uncalibrated by contract at this layer.
        """
        if not self._fitted:
            msg = f"{self.name} has not been fitted"
            raise NotFittedError(msg)
        if self.feature_names_ and tuple(x.columns) != self.feature_names_:
            expected = list(self.feature_names_)
            msg = (
                f"{self.name} was fitted on a different feature set.\n"
                f"  expected: {expected[:5]}... ({len(expected)} cols)\n"
                f"  received: {list(x.columns)[:5]}... ({len(x.columns)} cols)\n"
                "Training and inference must build the matrix through the same pipeline."
            )
            raise ValueError(msg)

        scores = np.asarray(self._predict(x, context), dtype="float64")
        if scores.shape != (len(x),):
            msg = f"{self.name} returned shape {scores.shape}, expected {(len(x),)}"
            raise ValueError(msg)
        if not np.all(np.isfinite(scores)):
            msg = f"{self.name} produced non-finite scores"
            raise ValueError(msg)
        # Clipping rather than asserting: a tree model can emit 1e-17 below zero
        # through floating point, and failing a whole run for that would be
        # theatre. A genuinely out-of-range score is caught by the range test.
        return np.asarray(np.clip(scores, 0.0, 1.0), dtype="float64")

    @abstractmethod
    def _fit(self, x: pd.DataFrame, y: pd.Series, context: pd.DataFrame | None) -> None:
        """Rung-specific fitting."""

    @abstractmethod
    def _predict(self, x: pd.DataFrame, context: pd.DataFrame | None) -> np.ndarray:
        """Rung-specific scoring."""

    def explain(self, x: pd.DataFrame, top_k: int = 5) -> list[list[FeatureContribution]]:
        """Per-row SHAP contributions, largest absolute magnitude first.

        Default returns empty lists: the heuristic rungs have no SHAP values, and
        the harness must still be able to call this on every rung uniformly.
        """
        return [[] for _ in range(len(x))]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _state(self) -> dict[str, Any]:
        """What must be pickled to restore this model. Overridden where needed."""
        return {"params": self.params, "feature_names": self.feature_names_}

    def _restore(self, state: dict[str, Any]) -> None:
        self.params = state["params"]
        self.feature_names_ = tuple(state["feature_names"])
        self._fitted = True

    def save(self, path: Path, card: ModelCard) -> Path:
        """Persist the fitted model together with its provenance card."""
        if not self._fitted:
            msg = f"refusing to save unfitted model {self.name}"
            raise NotFittedError(msg)
        return save_artifact(path, self._state(), card)

    @classmethod
    def load(cls, path: Path) -> tuple[Self, ModelCard]:
        """Load a persisted model and its card, verifying the checksum.

        Returns ``Self``, so ``LightGbmModel.load(...)`` is typed as a
        ``LightGbmModel`` without every rung restating it. The instance really is
        of that class - ``cls()`` builds it - and the checksum in
        :func:`load_artifact` is what actually guards the file.
        """
        state, card = load_artifact(path)
        model = cls()
        model._restore(state)
        return model, card

    def __repr__(self) -> str:
        return f"{type(self).__name__}(rung={self.rung_id}, name={self.name!r})"


class HeuristicModel(RiskModel):
    """Base for rungs 0-2: operational policies rather than learned models.

    They still implement the full interface so the harness can score them on
    identical footing, and they still have a ``fit`` - rung 2 genuinely learns a
    blocklist from the training split, and rung 0 learns the base rate. What they
    do not do is learn from the design matrix.
    """


def context_column(context: pd.DataFrame | None, column: str, *, model_name: str) -> pd.Series:
    """Fetch a required operational column, with an error that says what is wrong.

    Heuristic rungs depend on raw columns the design matrix deliberately excludes,
    so a missing one is a wiring mistake in the harness rather than a data
    problem, and the message says so.
    """
    if context is None:
        msg = (
            f"{model_name} needs the operational context frame (for {column!r}) but was "
            "given none. Heuristic rungs score on raw order data, not the design matrix."
        )
        raise ValueError(msg)
    if column not in context.columns:
        msg = f"{model_name} needs {column!r} in the context frame; got {list(context.columns)}"
        raise ValueError(msg)
    return context[column]
