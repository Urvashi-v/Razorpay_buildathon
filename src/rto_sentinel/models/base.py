"""The risk-model interface every rung of the ladder implements.

SPEC section 05: every rung is evaluated identically on the same sealed test set,
against the same cost metric. That is only meaningful if a do-nothing baseline
and a calibrated LightGBM present the *same* interface to the evaluation harness
- so they do, and the harness has no idea which is which.

WHAT A MODEL RETURNS AND WHAT IT DOES NOT
-----------------------------------------
:meth:`RiskModel.predict_proba` returns probabilities. Nothing here returns a
band, an action, a rupee figure or a threshold. Those belong to
``rto_sentinel.decision``, and keeping them out of this interface is what stops a
model from quietly deciding policy.

STATUS: Phase 2-3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    import pandas as pd

    from rto_sentinel.contracts.risk import FeatureContribution, ModelCard


class RiskModel(ABC):
    """A rung of the baseline ladder."""

    #: Ladder position from config/models/ladder.yaml. Rung 0 is do-nothing.
    rung_id: int
    #: Stable identifier used in artefacts, reports and the API response.
    name: str

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Fit on the training split only.

        Implementations must never see validation or test rows here. The
        calibrator is fitted separately, on validation, by
        ``models.calibration`` - a model that calibrates itself on its own
        training fold is calibrated to its own overfitting.
        """

    @abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return P(RTO) for each row, as a 1-D array in ``[0, 1]``.

        Uncalibrated by contract at this layer. Calibration is applied by the
        wrapper in ``models.calibration``, and
        :attr:`~rto_sentinel.contracts.risk.RiskScore.calibration_method` records
        which method was used - so an uncalibrated score can never be mistaken
        for a decision-grade one.
        """

    def explain(self, x: pd.DataFrame, top_k: int = 5) -> list[list[FeatureContribution]]:
        """Per-row SHAP contributions, largest absolute magnitude first.

        Default implementation returns empty lists: the heuristic rungs have no
        SHAP values, and the evaluation harness must still be able to call this
        on every rung uniformly.
        """
        return [[] for _ in range(len(x))]

    @abstractmethod
    def save(self, path: Path, card: ModelCard) -> None:
        """Persist the fitted model together with its provenance card."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> tuple[RiskModel, ModelCard]:
        """Load a persisted model and its card."""


class HeuristicModel(RiskModel):
    """Base for rungs 0-2, which have parameters but nothing to fit.

    They still implement the full interface so the harness can score them on
    identical footing. ``fit`` is a no-op by design, not an oversight.
    """

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Heuristics have nothing to learn from the training split."""
        return None
